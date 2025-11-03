import argparse
import json
import pickle
from typing import TypedDict

import numpy as np
import spacy
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm


class AttributionResult(TypedDict):
    """Result from gradient computation."""
    tokens: list[str]
    importances: np.ndarray


MAX_LEN = 50


def create_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate gradient-based explanations for LSTM',
    )

    parser.add_argument(
        'model_dir',
        help='Directory containing saved model and artifacts',
        type=str,
    )

    parser.add_argument(
        'input_file',
        help='TSV file with tweets and labels',
        type=str,
    )

    parser.add_argument(
        '--k',
        help='Number of top tokens per sample',
        type=int,
        default=10,
    )

    parser.add_argument(
        '-o',
        '--output-file',
        help='Output file to save explanations',
        type=str,
        default='lstm_explanations.pkl',
    )

    return parser.parse_args()


def load_model_and_artifacts(model_dir: str):
    """Load LSTM model and all training artifacts."""
    print('Loading model and artifacts...')

    model = tf.keras.models.load_model(
        f'{model_dir}/model.keras',
    )

    with open(f'{model_dir}/word_to_idx.json', 'r') as f:
        word_to_idx = json.load(f)

    with open(f'{model_dir}/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    return model, word_to_idx, label_encoder


def read_corpus(file: str) -> tuple[list[str], list[str]]:
    """Read TSV file with tweets and labels."""
    tweets = []
    labels = []
    with open(file, 'r') as f:
        for line in f:
            tweet, label = line.strip().split('\t')
            tweets.append(tweet)
            labels.append(label)
    return tweets, labels


def spacy_tokenizer(nlp: spacy.language.Language, texts: list[str]) -> list[list[str]]:
    """Tokenize texts using SpaCy (same as training)."""
    all_tokenized_texts = []
    for doc in nlp.pipe(
        texts,
        disable=['parser', 'ner', 'tagger', 'lemmatizer', 'attribute_ruler'],
    ):
        doc_tokens = [token.text for token in doc]
        all_tokenized_texts.append(doc_tokens)
    return all_tokenized_texts


def texts_to_sequences(tokenized_texts: list[list[str]], word_to_idx: dict) -> list[list[int]]:
    """Convert tokenized texts to integer sequences."""
    return [
        [word_to_idx.get(word, word_to_idx.get('<unk>', 1)) for word in text]
        for text in tokenized_texts
    ]


def compute_attributions(
    tweet: str,
    word_to_idx: dict,
    nlp,
    model,
    target_class: int = 1,
) -> AttributionResult:
    """
    Compute gradient-based attributions for LSTM.

    Uses: input × gradient (similar to Captum's InputXGradient)
    """
    # Tokenize
    tokens = spacy_tokenizer(nlp, [tweet])[0]
    token_ids = texts_to_sequences([tokens], word_to_idx)[0]
    token_ids_padded = pad_sequences([token_ids], maxlen=MAX_LEN)[0]

    # Convert to tensor
    input_tensor = tf.constant([token_ids_padded], dtype=tf.int32)

    with tf.GradientTape() as tape:
        tape.watch(input_tensor)

        # Get embedding layer output
        embedding_layer = model.layers[0]  # First layer is Embedding
        embeddings = embedding_layer(input_tensor)

        # Get model output
        logits = model(input_tensor)

    # Compute gradients w.r.t. embeddings
    gradients = tape.gradient(logits[:, target_class], embeddings)

    if gradients is None:
        # Return zeros if gradient computation fails
        return {
            'tokens': tokens,
            'importances': np.zeros(len(tokens)),
        }

    # Importance = |input × gradient| summed across embedding dimension
    importances = (
        tf.abs(embeddings * gradients)
        .numpy()
        .sum(axis=2)  # Sum across embedding dimensions
        .squeeze()
    )

    # Only keep importances for actual tokens (not padding)
    importances = importances[:len(tokens)]

    return {
        'tokens': tokens,
        'importances': importances,
    }


def extract_top_k_words(
    tokens: list[str],
    importances: np.ndarray,
    k: int = 10,
) -> tuple[dict[str, float], list[str]]:
    """Extract top-k important words."""
    word_importances = {
        token: importance
        for token, importance in zip(tokens, importances)
        if token not in ['[CLS]', '[SEP]', '[PAD]']
    }

    sorted_words = sorted(
        word_importances.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return word_importances, [word for word, _ in sorted_words[:k]]


def identify_globally_important_tokens(
    top_k_words_per_sample: list[list[str]],
    n: int = 20,
) -> list[tuple[str, int]]:
    """Get most frequent tokens in top-k."""
    from collections import Counter
    token_frequencies = Counter()
    for top_k in top_k_words_per_sample:
        token_frequencies.update(top_k)
    return token_frequencies.most_common(n)


def main() -> int:
    args = create_arg_parser()

    # Load SpaCy model
    print('Loading SpaCy model...')
    nlp = spacy.load('en_core_web_sm')

    # Load model and artifacts
    model, word_to_idx, label_encoder = load_model_and_artifacts(
        args.model_dir,
    )

    # Load test data
    print('Loading test data...')
    tweets, labels = read_corpus(args.input_file)
    print(f'Loaded {len(tweets)} tweets')

    # Map labels to class indices
    label_to_class = {
        label: idx for idx,
        label in enumerate(label_encoder.classes_)
    }

    # Process all tweets
    per_sample_importances = []
    top_k_words = []

    for tweet, label in tqdm(
        zip(tweets, labels),
        total=len(tweets),
        unit='tweet',
    ):
        try:
            class_idx = label_to_class[label]

            # Compute attributions
            attr_result = compute_attributions(
                tweet,
                word_to_idx,
                nlp,
                model,
                target_class=class_idx,
            )

            # Extract top-k
            word_imp, top_k = extract_top_k_words(
                attr_result['tokens'],
                attr_result['importances'],
                args.k,
            )

            per_sample_importances.append(word_imp)
            top_k_words.append(top_k)

        except Exception as e:
            print(f'Error processing tweet: {e}')
            per_sample_importances.append({})
            top_k_words.append([])

    # Globally important tokens
    global_tokens = identify_globally_important_tokens(top_k_words, n=20)

    print('\n' + '='*70)
    print(f'Top 20 Globally Important Tokens (LSTM)')
    print('='*70)
    for i, (token, freq) in enumerate(global_tokens, 1):
        print(f'  {i:2d}. {token:<20} (frequency: {freq})')

    # Save results
    results = {
        'per_sample_importances': per_sample_importances,
        'top_k_words': top_k_words,
        'global_tokens': global_tokens,
        'tweets': tweets,
        'labels': labels,
        'model_dir': args.model_dir,
        'k': args.k,
        'model_type': 'LSTM',
    }

    with open(args.output_file, 'wb') as f:
        pickle.dump(results, f)

    print(f'Saved results to {args.output_file}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
