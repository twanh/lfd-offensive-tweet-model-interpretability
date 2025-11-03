import argparse
import pickle
import warnings
from functools import partial
from itertools import product

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')


def create_arg_parser() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'train_file',
        help='The train file',
        type=str,
    )

    parser.add_argument(
        'dev_file',
        help='The dev file',
        type=str,
    )

    parser.add_argument(
        '-t', '--test-file',
        help='The test file to run the evaluation on',
        default=None,
        type=str,
    )

    parser.add_argument(
        '--save-model',
        help='The path to save the model too.',
        default=None,
        type=str,
    )

    parser.add_argument(
        '--grid-search',
        help='Whether to perform grid search',
        action='store_true',
        default=False,
    )

    # Grid search parameters
    parser.add_argument(
        '--C',
        type=float,
        default=1.0,
        help='C for SVM classifiers',
    )
    parser.add_argument(
        '--kernel',
        type=str,
        default='rbf',
        help='Kernel for SVM',
    )
    parser.add_argument(
        '--gamma',
        type=str,
        default='scale',
        help='Gamma for SVM',
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
            # TODO: Tokenize
            tweets.append(tweet)
            labels.append(label)

    return tweets, labels


def spacy_tokenizer(nlp: spacy.language.Language, tweet: str) -> list[str]:

    doc = nlp(tweet)
    return [token.text for token in doc]


def main() -> int:

    args = create_arg_parser()

    grid_params = {
        'C': [0.1, 1.0, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto'],
    }

    X_train, y_train = read_corpus(args.train_file)
    X_dev, y_dev = read_corpus(args.dev_file)

    print('Loaded Data:')
    print(f'Train size: {len(X_train)}')
    print(f'Dev size: {len(X_dev)}')

    # Load spacy model
    nlp = spacy.load('en_core_web_sm')
    print('Loaded spacy model')

    # Create vectorizer with spacy spacy_tokenizer
    tokenizer_func = partial(spacy_tokenizer, nlp)
    vectorizer = TfidfVectorizer(
        tokenizer=tokenizer_func,
    )

    # Vectorize the data
    print('Vectorizing Data')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_dev_vec = vectorizer.transform(X_dev)

    # Initialize model (will be set later)
    fin_model: SVC | None = None

    # Perform grid search if specified
    if args.grid_search:

        best_acc = 0.0
        best_params = {'C': None, 'kernel': None, 'gamma': None}
        best_model = None

        print('Performing Grid Search')
        for C, kernel, gamma in product(  # type: ignore
            grid_params['C'],
            grid_params['kernel'],
            grid_params['gamma'],
        ):
            print(f'Testing C={C}, kernel={kernel}, gamma={gamma}')
            model = SVC(
                C=C,
                kernel=kernel,
                gamma=gamma,
            )

            model.fit(X_train_vec, y_train)
            y_dev_pred = model.predict(X_dev_vec)
            acc = accuracy_score(y_dev, y_dev_pred)

            if acc > best_acc:
                best_acc = acc
                best_params['C'] = C
                best_params['kernel'] = kernel
                best_params['gamma'] = gamma
                best_model = model

        print(f'Best Dev Accuracy: {best_acc:.4f}')
        print(f'Best Parameters: {best_params}')

        fin_model = best_model

    else:
        print('Training with specified parameters')
        print(f'C={args.C}, kernel={args.kernel}, gamma={args.gamma}')

        # Train with specified Parameters
        model = SVC(
            C=args.C,
            kernel=args.kernel,
            gamma=args.gamma,
        )

        model.fit(X_train_vec, y_train)
        y_dev_pred = model.predict(X_dev_vec)
        acc = accuracy_score(y_dev, y_dev_pred)

        print(f'Dev Accuracy: {acc:.4f}')

        # Final model
        fin_model = model

    if args.save_model is not None:
        print('Saving model and vectorizer')
        with open(args.save_model, 'wb') as save_model_file:
            pickle.dump(fin_model, save_model_file)
        print(f'Saved model to {args.save_model}')

        with open(f'vectorizer_{args.save_model}', 'wb') as f:
            pickle.dump(vectorizer, f)
        print(f'Saved vectorizer to vectorizer_{args.save_model}')

    # Run on test set if provided
    if args.test_file is not None:

        X_test, y_test = read_corpus(args.test_file)
        X_test_vec = vectorizer.transform(X_test)

        if fin_model is None:
            raise ValueError('Final model is not trained.')

        y_test_pred = fin_model.predict(X_test_vec)

        test_acc = accuracy_score(y_test, y_test_pred)

        print(f'Test Accuracy: {test_acc:.4f}')
        print('Classification Report:')
        print(classification_report(y_test, y_test_pred))
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, y_test_pred))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
