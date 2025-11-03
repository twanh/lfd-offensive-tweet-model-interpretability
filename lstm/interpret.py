import argparse
import json
import pickle
from collections import Counter
from typing import TypedDict

import numpy as np
import shap
import spacy
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm


class SHAPResult(TypedDict):
    tokens: list[str]
    shap_values: np.ndarray


MAX_LEN = 50


def create_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate SHAP explanations for LSTM',
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

    print('Loading model and artifacts')

    model = tf.keras.models.load_model(
        f'{model_dir}/model.keras',
    )

    with open(f'{model_dir}/word_to_idx.json', 'r') as f:
        word_to_idx = json.load(f)

    with open(f'{model_dir}/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    with open(f'{model_dir}/training_params.json', 'r') as f:
        training_params = json.load(f)

    return model, word_to_idx, label_encoder, training_params


def read_corpus(file: str) -> tuple[list[str], list[str]]:

    tweets = []
    labels = []
    with open(file, 'r') as f:
        for line in f:
            tweet, label = line.strip().split('\t')

            tweets.append(tweet)
            labels.append(label)

    return tweets, labels


def spacy_tokenizer(
    nlp: spacy.language.Language,
    texts: list[str],
) -> list[list[str]]:

    all_tokenized_texts = []

    for doc in nlp.pipe(
        texts,
        disable=[
            'parser', 'ner', 'tagger',
            'lemmatizer', 'attribute_ruler',
        ],
    ):
        doc_tokens = [token.text for token in doc]
        all_tokenized_texts.append(doc_tokens)
    return all_tokenized_texts


def texts_to_sequences(
    tokenized_texts: list[list[str]],
    word_to_idx: dict,
) -> list[list[int]]:

    return [
        [word_to_idx.get(word, word_to_idx.get('<unk>', 1)) for word in text]
        for text in tokenized_texts
    ]


def create_prediction_function(model):

    def predict(sequences):
        # Ensure sequences are numpy array
        sequences = np.array(sequences)

        # Get model predictions
        predictions = model.predict(sequences, verbose=0)

        return predictions

    return predict


def compute_shap_values(
    tweet: str,
    word_to_idx: dict,
    nlp,
    model,
    background_size: int = 50,
) -> SHAPResult:
    """
    Compute SHAP values for a single tweet.
    """
    # Tokenize using SpaCy (same as training)
    tokens = spacy_tokenizer(nlp, [tweet])[0]

    # Convert to sequences
    token_ids = texts_to_sequences([tokens], word_to_idx)[0]

    # Pad to MAX_LEN
    token_ids_padded = pad_sequences([token_ids], maxlen=MAX_LEN)[0]

    # Create background dataset (random padded sequences)
    background = np.random.randint(
        0, len(word_to_idx), size=(background_size, MAX_LEN),
    )

    # Create SHAP explainer for Keras model
    explainer = shap.DeepExplainer(
        model,
        background,
    )

    # Compute SHAP values
    shap_values = explainer.shap_values(
        np.array([token_ids_padded]),
    )

    shap_values_offensive = shap_values[1][0]

    return {
        'tokens': tokens,
        'shap_values': shap_values_offensive,
    }


def aggregate_tokens_to_words(
    shap_values: np.ndarray,
    tokens: list[str],
) -> tuple[list[str], np.ndarray]:

    importances = np.abs(shap_values[:len(tokens)])
    return tokens, importances


def extract_top_k_words(
    words: list[str],
    importances: np.ndarray,
    k: int = 10,
) -> tuple[dict[str, float], list[str]]:
    """Extract top-k important words."""
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


def identify_globally_important_tokens(
    top_k_words_per_sample: list[list[str]],
    n: int = 20,
) -> list[tuple[str, int]]:

    token_frequencies: Counter[str] = Counter()

    for top_k in top_k_words_per_sample:
        token_frequencies.update(top_k)

    return token_frequencies.most_common(n)


def main() -> int:
    args = create_arg_parser()

    # Load SpaCy model
    print('Loading SpaCy model...')
    nlp = spacy.load('en_core_web_sm')

    # Load model and artifacts
    model, word_to_idx, _, _ = load_model_and_artifacts(
        args.model_dir,
    )

    # Load test data
    print('Loading test data...')
    tweets, labels = read_corpus(args.input_file)
    print(f'Loaded {len(tweets)} tweets')

    # Process all tweets
    per_sample_importances = []
    top_k_words = []

    for tweet, label in tqdm(
        zip(tweets, labels),
        total=len(tweets),
        unit='tweet',
    ):
        try:
            # Compute SHAP
            shap_result = compute_shap_values(
                tweet,
                word_to_idx,
                nlp,
                model,
            )

            # Aggregate to words
            words, importances = aggregate_tokens_to_words(
                shap_result['shap_values'],
                shap_result['tokens'],
            )

            # Extract top-k
            word_imp, top_k = extract_top_k_words(words, importances, args.k)

            per_sample_importances.append(word_imp)
            top_k_words.append(top_k)

        except Exception as e:
            print(f'Error processing tweet: {e}')
            per_sample_importances.append({})
            top_k_words.append([])

    # Globally important tokens
    global_tokens = identify_globally_important_tokens(top_k_words, n=20)

    print('='*70)
    print('Top 20 Globally Important Tokens (LSTM)')
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
