import argparse
import pickle
from collections import Counter
from typing import TypedDict

import numpy as np
from captum.attr import IntegratedGradients
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


class AttributionResult(TypedDict):
    """Result dictionary from compute_attributions."""
    input_ids: np.ndarray
    tokens: list[str]
    token_importances: np.ndarray


def create_arg_parser() -> argparse.Namespace:
    """Create argument parser matching interpret.py CLI flags."""

    parser = argparse.ArgumentParser(
        description='Generate explanations for a fine-tuned BERT model using Captum',  # noqa: E501
    )

    # Model path
    parser.add_argument(
        'model_path',
        help='The path to the fine-tuned BERT model',
        type=str,
    )

    # Input file
    parser.add_argument(
        'input_file',
        help='The input file to generate explanations for',
        type=str,
    )

    # The k amount of top features
    parser.add_argument(
        '--k',
        help='Number of top tokens per sample',
        type=int,
        default=10,
    )

    parser.add_argument(
        '-o',
        '--output-file',
        help='The output file to save the explanations',
        type=str,
        default='bert_explanations.pkl',
    )

    parser.add_argument(
        '--device',
        help='The device to use for the model',
        type=str,
        default='cuda',
    )

    return parser.parse_args()


def load_model_and_tokenizer(
    model_path: str,
    device: str,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:

    model = AutoModelForSequenceClassification.from_pretrained(  # noqa: E501
        model_path,
        output_attentions=True,
    ).to(device)
    model.eval()

    # TODO: make this configurable
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer


def read_corpus(
    file: str,
) -> tuple[list[str], list[str]]:
    """Read corpus from TSV file."""

    tweets = []
    labels = []

    with open(file, 'r') as in_file:
        for line in in_file.readlines():
            # TSV file so split on tab
            tweet, label = line.strip().split('\t')
            tweets.append(tweet)
            labels.append(label)

    return tweets, labels


def get_token_to_word_mapping(tokens: list[str]) -> list[int]:
    """
    Get the mapping from token index to word index.

    BERT tokenizer splits words into subwords. This function
    maps the token index to the word index.

    E.g.:
        tokens = ['[CLS]', 'play', '##ing', 'is', '[SEP]']
        would give -> [0, 1, 1, 2, 3]  (##ing belongs to word 1)
    """
    word_idx = 0
    token_to_word = []

    for token in tokens:
        if token.startswith('##'):
            # Subword token
            token_to_word.append(word_idx - 1)
        else:
            token_to_word.append(word_idx)
            word_idx += 1

    return token_to_word


def compute_attributions(
    tweet: str,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: str,
    target_class: int,
) -> AttributionResult:
    """
    Compute Integrated Gradients attributions for the input.

    We do this by:
    - Tokenizing the tweet
    - Computing the Integrated Gradients attributions
    - Summing the attributions over the embedding dimension
    - Returning the importances for each token

    """

    # Tokenize the tweet
    encoding = tokenizer(
        tweet,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    token_type_ids = encoding['token_type_ids'].to(device)

    ig = IntegratedGradients(model)
    attributions = ig.attribute(
        inputs=input_ids,
        baselines=None,
        target=target_class,
        additional_forward_args=(attention_mask, token_type_ids),
        n_steps=50,
    )

    token_importances = attributions.sum(
        dim=2,
    ).squeeze(0).cpu().detach().numpy()

    return {
        'input_ids': input_ids.squeeze().cpu().numpy(),
        'tokens': tokenizer.convert_ids_to_tokens(input_ids.squeeze()),
        'token_importances': token_importances,
    }


def aggregate_subwords_to_words(
    token_importances: np.ndarray,
    tokens: list[str],
) -> tuple[list[str], np.ndarray]:
    """
    Aggregate the importances for each subword to the words.

    Captum returns the attributions for each token. This function
    aggregates the importances for each subword to the words.
    """

    token_to_word = get_token_to_word_mapping(tokens)

    # Group by word index
    word_importances: dict[int, list[float]] = {}
    word_tokens: dict[int, str] = {}

    for token, importance, word_idx in zip(
        tokens,
        token_importances,
        token_to_word,
    ):

        # Skip special tokens because they dont
        # have a value for the analysis
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue

        if word_idx not in word_importances:
            word_importances[word_idx] = []
            word_tokens[word_idx] = token.replace('##', '')

        word_importances[word_idx].append(importance)

    # Average the importances for each word
    words = []
    importances = []

    for i in sorted(word_importances.keys()):
        words.append(word_tokens[i])
        importances.append(np.sum(word_importances[i]))

    return words, np.array(importances)


def extract_top_k_words(
    words: list[str],
    importances: np.ndarray,
    k: int = 10,
) -> tuple[dict[str, float], list[str]]:
    """
    Extract the top k words by importance.

    Should match the SVM interpret.py output format for comparison.
    """

    word_importances = {
        word: importance
        for word, importance in zip(words, importances)
    }

    sorted_words = sorted(
        word_importances.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return word_importances, [word for word, _ in sorted_words[:k]]


def create_label_to_class_mapping(labels: list[str]) -> dict[str, int]:
    unique_labels = sorted(set(labels))
    return {label: idx for idx, label in enumerate(unique_labels)}


def main() -> int:

    args = create_arg_parser()

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)

    # Read the corpus
    tweets, labels = read_corpus(args.input_file)
    print(f'Loaded {len(tweets)} tweets and {len(labels)} labels')

    # Processs the test data tweets
    per_sample_importances = []
    top_k_words = []

    label_to_class = create_label_to_class_mapping(labels)


    print(f'DEBUG: Label to class mapping: {label_to_class}')
    print(f'DEBUG: Unique labels in data: {set(labels)}')

    for tweet, label in tqdm(
        zip(tweets, labels),
        total=len(tweets),
        unit='tweet',
    ):

        print(f'DEBUG: {label=}, {label_to_class[label]=}')



        try:
            # Compute the attribution
            attributions = compute_attributions(
                tweet,
                model,
                tokenizer,
                args.device,
                label_to_class[label],
            )

            print(f'DEBUG: tokens={attributions["tokens"]}')
            print(f'DEBUG: token_importances shape={attributions["token_importances"].shape}')
            print(f'DEBUG: token_importances={attributions["token_importances"][:10]}')

            # Aggregate the subwords to the words
            words, importances = aggregate_subwords_to_words(
                attributions['token_importances'],
                attributions['tokens'],
            )

            # Debug: Check aggregation
            print(f'DEBUG: words after aggregation={words}')
            print(f'DEBUG: importances after aggregation={importances}')

            # Extract the top k words
            word_importances, top_k = extract_top_k_words(
                words,
                importances,
                k=args.k,
            )

            print(f'DEBUG: top_k={top_k}')

            per_sample_importances.append(word_importances)
            top_k_words.append(top_k)
        except Exception as e:
            print(f'Error computing attributions for tweet: {tweet}')
            print(f'Error: {e}')
            per_sample_importances.append({})
            top_k_words.append([])

    results = {
        'per_sample_importances': per_sample_importances,
        'top_k_words': top_k_words,
        'tweets': tweets,
        'labels': labels,
        'model_name': args.model_path,
        'k': args.k,
        'model_type': 'BERT',
    }

    # Print the top k words for the full dataset
    print(f'Top {args.k} words for the full dataset:')
    for i, words in enumerate(top_k_words):
        print(f'Tweet {i+1}: {words}')
        print('-' * 80)
        for word in words:
            print(
                f'  - {word:<20} | Importance: {per_sample_importances[i][word]:.4f}',  # noqa: E501
            )
        print('-' * 80)

    token_frequencies = Counter()
    for top_k in top_k_words:
        token_frequencies.update(top_k)

    # Print the top k words sorted by importance
    print(f'Top {args.k} words for the full dataset sorted by importance:')
    for word, frequency in token_frequencies.most_common(args.k):
        print(f'  - {word:<20} | Frequency: {frequency}')

    with open(args.output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f'Saved results to {args.output_file}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
