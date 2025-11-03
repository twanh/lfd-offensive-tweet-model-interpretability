import argparse
import pickle

import numpy as np
import tqdm
from train import spacy_tokenizer  # noqa: F401


def create_arg_parser() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model_file',
        help='The trained model file to interpret',
        type=str,
    )

    parser.add_argument(
        'vectorizer_file',
        help='The vectorizer file used during training',
        type=str,
    )

    parser.add_argument(
        'input_file',
        help='The input file to interpret (containing tweets and labels)',
        type=str,
    )

    parser.add_argument(
        '--n',
        help='Number of top features to display',
        type=int,
        default=25,
    )

    parser.add_argument(
        '--k',
        help='Top k tokens per sample',
        type=int,
        default=10,
    )

    parser.add_argument(
        '-o',
        '--output-file',
        help='The output file to save the explanations',
        type=str,
        default='explanations.pkl',
    )

    return parser.parse_args()


def read_corpus(
    file: str,
) -> tuple[list[str], list[str]]:

    tweets = []
    labels = []

    with open(file, 'r') as in_file:

        for line in in_file.readlines():
            # TSV file so split on tab
            tweet, label = line.strip().split('\t')
            tweets.append(tweet)
            labels.append(label)

    return tweets, labels


def extract_per_sample_importances(
    coefs: np.ndarray,
    X_test_vec: np.ndarray,
    feature_names: list[str],
    k: int = 10,
) -> tuple[list[dict], list[list[str]]]:
    """Compute importance for each sample: |w_t| * tfidf_t(s)"""

    # Initalize to save in
    per_sample_importances = []
    top_k_tokens = []

    # Get the absolute coefficients and dense matrix
    abs_coefs = np.abs(coefs)
    X_dense = X_test_vec.toarray()

    # Iterate over the test samples
    for i in tqdm(range(X_dense.shape[0])):

        # Get the TF-IDF values for the sample
        tfidf_values = X_dense[i]
        importances = tfidf_values * abs_coefs

        # Dict: Token -> importance
        # Only keep the tokens with non-zero importance
        sample_token_importances = {
            feature_names[j]: importances[j]
            for j in range(len(feature_names))
            if importances[j] > 0
        }
        per_sample_importances.append(sample_token_importances)

        # Get the top k
        if sample_token_importances:
            sorted_tokens = sorted(
                sample_token_importances.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:k]
            top_k_tokens.append([token for token, _ in sorted_tokens])
        else:
            top_k_tokens.append([])

    return per_sample_importances, top_k_tokens


def main() -> int:

    args = create_arg_parser()

    # Load the model
    with open(args.model_file, 'rb') as f:
        model = pickle.load(f)

    # Load the vectorizer
    with open(args.vectorizer_file, 'rb') as f:
        vectorizer = pickle.load(f)

    # Load the test data
    tweets, labels = read_corpus(args.input_file)
    print(f'Loaded {len(tweets)} tweets and {len(labels)} labels')
    # Note: the vectorizer automatically tokenizes the tweets
    # using the spacy_tokenizer function (which it gets using pickle)
    X_test_vec = vectorizer.transform(tweets)

    # Get the feature names
    feature_names = vectorizer.get_feature_names_out()

    # Get model coefficients
    if hasattr(model, 'coef_'):
        coefs = model.coef_[0].toarray()[0]
    else:
        print('Model does not have coefficients to interpret.')
        return 1

    # Get the per-sample importances and top k tokens
    per_sample_importances, top_k_tokens = extract_per_sample_importances(
        coefs,
        X_test_vec,
        feature_names,
        k=args.k,
    )

    # Print the top k tokens for the full dataset
    print(f'Top {args.k} tokens for the full dataset:')
    for i, tokens in enumerate(top_k_tokens):
        print(f'Tweet {i+1}: {tokens}')
        print('-' * 80)
        for token in tokens:
            print(
                f'  - {token:<20} | Importance: {per_sample_importances[i][token]:.4f}',  # noqa: E501
            )
        print('-' * 80)

    # Save the results
    print(f'Saving results to {args.output_file}')
    results = {
        'per_sample_importances': per_sample_importances,
        'top_k_tokens': top_k_tokens,
        'tweets': tweets,
        'labels': labels,
        'feature_names': feature_names,
        'k': args.k,
        'model_name': args.model_file,
        'n': args.n,
    }

    with open(args.output_file, 'wb') as f:
        pickle.dump(results, f)

    return 0


if __name__ == '__main__':

    raise SystemExit(main())
