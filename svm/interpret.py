import argparse
import pickle

import numpy as np
from train import spacy_tokenizer


def create_arg_parser() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model_file',
        help='The trained model file to interpret',
        type=str
    )

    parser.add_argument(
        'vectorizer_file',
        help='The vectorizer file used during training',
        type=str
    )

    parser.add_argument(
        '--n',
        help='Number of top features to display',
        type=int,
        default=25
    )

    return parser.parse_args()


def main() -> int:

    args = create_arg_parser()

    # Load the model
    with open(args.model_file, 'rb') as f:
        model = pickle.load(f)

    # Load the vectorizer
    with open(args.vectorizer_file, 'rb') as f:
        vectorizer = pickle.load(f)

    feature_names = vectorizer.get_feature_names_out()
    class_labels = model.classes_

    if hasattr(model, 'coef_'):
        coefs = model.coef_[0].toarray()[0]
    else:
        print('Model does not have coefficients to interpret.')
        return 1

    # Top N features for each class
    top_positive_indices = np.argsort(coefs)[-args.n:][::-1]
    print(f'Most indicative features for class: "{class_labels[1]}"')
    for idx in top_positive_indices:
        print(f'  - {feature_names[idx]:<20} (weight: {coefs[idx]:.4f})')

    # Get indices of the N most negative coefficients
    top_negative_indices = np.argsort(coefs)[:args.n]
    print()
    print(f'Most indicative features for class: "{class_labels[0]}"')
    for idx in top_negative_indices:
        print(f'  - {feature_names[idx]:<20} (weight: {coefs[idx]:.4f})')
    # Get top n features

    return 0


if __name__ == '__main__':

    raise SystemExit(main())
