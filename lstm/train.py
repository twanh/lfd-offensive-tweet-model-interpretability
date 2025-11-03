import argparse
import json
import os
import pickle
import random

import fasttext
import numpy as np
import spacy
import tensorflow as tf
from keras.initializers import Constant
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.metrics import classification_report
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

MAX_LEN = 50  # Maximum sequence length for padding

np.random.seed(43)
tf.random.set_seed(43)
random.seed(43)


def create_arg_parser() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description='Train an LSTM model with SpaCy and FastText.',
    )

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
        '--embeddings',
        default='cc.en.300.bin',
        type=str,
        help='Path to the FastText .bin model file',
    )

    parser.add_argument(
        '--save-model-dir',
        help='The directory to save the model and requirements to',
        default=None,
        type=str,
    )

    parser.add_argument(
        '--grid-search',
        help='Whether to perform grid search',
        action='store_true',
        default=False,
    )

    # Model Hyperparameters (used for single runs or as grid search space)
    parser.add_argument(
        '--optimizer',
        type=str,
        default='Adam',
        choices=['SGD', 'Adam', 'RMSprop'],
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
    )
    parser.add_argument(
        '--lstm-units',
        type=int,
        default=128,
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.2,
    )
    parser.add_argument(
        '--bidirectional',
        action='store_true',
        help='Use a bidirectional LSTM layer',
    )

    # Training Parameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)

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


def spacy_tokenizer(nlp: spacy.language.Language, texts: list[str]) -> list[str]:

    all_tokenized_texts = []
    # Process texts in using nlp.pipe for efficiency
    for doc in nlp.pipe(
            texts,
            disable=[
                'parser', 'ner', 'tagger',
                'lemmatizer', 'attribute_ruler',
            ],
    ):
        doc_tokens = []
        for token in doc:
            doc_tokens.append(token.text)
        all_tokenized_texts.append(doc_tokens)
    return all_tokenized_texts


def build_vocabulary(tokenized_texts: list[str]) -> dict[str, int]:

    vocab = set(word for text in tokenized_texts for word in text)
    word_to_idx = {word: i + 2 for i, word in enumerate(sorted(vocab))}
    word_to_idx['<pad>'] = 0
    word_to_idx['<unk>'] = 1
    return word_to_idx


def get_emb_matrix(
    word_to_idx: dict[str, int],
    ft_model: fasttext.FastText._FastText,
) -> np.ndarray:

    embedding_dim = ft_model.get_word_vector('the').shape[0]
    embedding_matrix = np.zeros((len(word_to_idx), embedding_dim))

    for word, i in word_to_idx.items():
        embedding_matrix[i] = ft_model.get_word_vector(word)

    return embedding_matrix


def create_model(
    num_labels: int,
    emb_matrix: np.ndarray,
    args: argparse.Namespace,
) -> Sequential:

    if args.optimizer == 'SGD':
        optim = tf.keras.optimizers.SGD(
            learning_rate=args.learning_rate,
        )  # type: ignore
    elif args.optimizer == 'Adam':
        optim = tf.keras.optimizers.Adam(
            learning_rate=args.learning_rate,
        )  # type: ignore
    else:
        optim = tf.keras.optimizers.RMSprop(
            learning_rate=args.learning_rate,
        )  # type: ignore

    model = Sequential()

    model.add(
        Embedding(
            input_dim=emb_matrix.shape[0],
            output_dim=emb_matrix.shape[1],
            embeddings_initializer=Constant(emb_matrix),
            input_length=MAX_LEN,
            trainable=False,
        ),
    )

    if args.bidirectional:
        model.add(
            Bidirectional(
                LSTM(
                    units=args.lstm_units,
                    dropout=args.dropout,
                    recurrent_dropout=args.dropout,
                ),
            ),
        )
    else:
        model.add(
            LSTM(
                units=args.lstm_units,
                dropout=args.dropout,
                recurrent_dropout=args.dropout,
            ),
        )

    model.add(Dense(units=num_labels, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optim,
        metrics=['accuracy'],
    )
    return model


def texts_to_sequences(tokenized_texts, word_to_idx):
    """Converts tokenized texts to integer sequences."""
    return [[word_to_idx.get(word, word_to_idx['<unk>']) for word in text] for text in tokenized_texts]


def main():

    args = create_arg_parser()

    print('Loading SpaCy model')
    nlp = spacy.load('en_core_web_sm')
    print(f"Loading FastText model from '{args.embeddings}'")
    ft_model = fasttext.load_model(args.embeddings)

    # Loading data
    print('Loading data')
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)

    # Preparing data
    print('Preparing data')
    X_train_tokens = spacy_tokenizer(nlp, X_train)
    X_dev_tokens = spacy_tokenizer(nlp, X_dev)

    word_to_idx = build_vocabulary(X_train_tokens + X_dev_tokens)
    emb_matrix = get_emb_matrix(word_to_idx, ft_model)

    encoder = LabelBinarizer()
    Y_train_encoded = encoder.fit_transform(Y_train)
    # Use transform, not fit_transform for dev
    Y_dev_encoded = encoder.transform(Y_dev)

    Y_train_int = np.argmax(Y_train_encoded, axis=1) if len(
        Y_train_encoded.shape,
    ) > 1 and Y_train_encoded.shape[1] > 1 else Y_train_encoded.flatten()
    Y_dev_int = np.argmax(Y_dev_encoded, axis=1) if len(
        Y_dev_encoded.shape,
    ) > 1 and Y_dev_encoded.shape[1] > 1 else Y_dev_encoded.flatten()

    num_classes = len(encoder.classes_)
    Y_train_bin = to_categorical(Y_train_int, num_classes=num_classes)
    Y_dev_bin = to_categorical(Y_dev_int, num_classes=num_classes)

    # Convert to padded sequences
    X_train_seq = pad_sequences(
        texts_to_sequences(
            X_train_tokens, word_to_idx,
        ), maxlen=MAX_LEN,
    )
    X_dev_seq = pad_sequences(
        texts_to_sequences(
            X_dev_tokens, word_to_idx,
        ), maxlen=MAX_LEN,
    )

    # Start Training
    final_model = None
    params = {}

    if args.grid_search:
        print('\n--- Starting Grid Search ---')
        param_grid = {
            'lstm_units': [64, 128],
            'learning_rate': [0.01, 0.001],
            'optimizer': ['Adam', 'SGD'],
            'dropout': [0.2, 0.5],
            'bidirectional': [True, False],
        }
        best_accuracy = 0.0
        best_params = None

        for params in ParameterGrid(param_grid):
            print(f'\nTesting parameters: {params}')

            # Create a temporary Namespace to update args for
            # current grid iteration, so we don't modify original args
            temp_args = argparse.Namespace(**vars(args))
            for k, v in params.items():
                setattr(temp_args, k, v)

            model = create_model(len(encoder.classes_), emb_matrix, temp_args)

            callback = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=3, restore_best_weights=True,
            )
            model.fit(
                X_train_seq, Y_train_bin,
                verbose=0, epochs=args.epochs, batch_size=args.batch_size,
                callbacks=[callback], validation_data=(X_dev_seq, Y_dev_bin),
            )
            # Evaluate returns loss, acc
            _, accuracy = model.evaluate(X_dev_seq, Y_dev_bin, verbose=0)
            print(f'Validation Accuracy: {accuracy:.4f}')

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params
                final_model = model

        print(f'\n--- Grid Search Complete ---')
        print(f'Best Validation Accuracy: {best_accuracy:.4f}')
        print(f'Best Hyperparameters: {best_params}')
        params = best_params

    else:
        print('--- Starting Single Training Run ---')
        model = create_model(len(encoder.classes_), emb_matrix, args)
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True,
        )
        model.fit(
            X_train_seq, Y_train_bin,
            verbose=1, epochs=args.epochs, batch_size=args.batch_size,
            callbacks=[callback], validation_data=(X_dev_seq, Y_dev_bin),
        )
        final_model = model
        params = {
            'lstm_units': args.lstm_units,
            'learning_rate': args.learning_rate,
            'optimizer': args.optimizer,
            'dropout': args.dropout,
            'bidirectional': args.bidirectional,
        }

    # Save model and artifacts
    if args.save_model_dir and final_model is not None:
        print(f'Saving model and artifacts to {args.save_model_dir}')
        os.makedirs(args.save_model_dir, exist_ok=True)
        final_model.save(os.path.join(args.save_model_dir, 'model.keras'))
        with open(os.path.join(args.save_model_dir, 'word_to_idx.json'), 'w') as f:
            json.dump(word_to_idx, f)
        with open(os.path.join(args.save_model_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(encoder, f)
        with open(os.path.join(args.save_model_dir, 'training_params.json'), 'w') as f:
            json.dump(params, f)
        print('Artifacts saved successfully.')

    # Evaluate on dev set
    dev_loss, dev_acc = final_model.evaluate(X_dev_seq, Y_dev_bin, verbose=0)
    Y_dev_pred = final_model.predict(X_dev_seq, verbose=0)
    Y_dev_pred_label = np.argmax(Y_dev_pred, axis=1)
    Y_dev_true_label = np.argmax(Y_dev_bin, axis=1)

    dev_report = classification_report(
        Y_dev_true_label,
        Y_dev_pred_label,
        target_names=encoder.classes_,
        digits=4,
    )
    print(f'[DEV] Accuracy: {dev_acc:.4f}')
    print('[DEV] Classification Report:\n' + dev_report)

    # Evaluate on test set (if provided)
    if args.test_file is not None:
        # Load and process test data
        X_test = []
        Y_test = []
        with open(args.test_file, encoding='utf-8') as fin:
            for line in fin:
                if not line.strip() or line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                text, label = parts[0], parts[-1]
                X_test.append(text)
                Y_test.append(label)
        X_test_tok = [spacy_tokenizer(text) for text in X_test]
        X_test_idx = [
            [word_to_idx.get(tok.text.lower(), 1) for tok in doc]
            for doc in X_test_tok
        ]
        X_test_seq = pad_sequences(X_test_idx, maxlen=MAX_LEN, padding='post')
        Y_test_enc = encoder.transform(Y_test)
        if len(Y_test_enc.shape) == 1:
            Y_test_bin = to_categorical(
                Y_test_enc, num_classes=len(encoder.classes_),
            )
        else:
            Y_test_bin = Y_test_enc

        test_loss, test_acc = final_model.evaluate(
            X_test_seq, Y_test_bin, verbose=0,
        )
        Y_test_pred = final_model.predict(X_test_seq, verbose=0)
        Y_test_pred_label = np.argmax(Y_test_pred, axis=1)
        Y_test_true_label = np.argmax(Y_test_bin, axis=1)
        test_report = classification_report(
            Y_test_true_label,
            Y_test_pred_label,
            target_names=encoder.classes_,
            digits=4,
        )
        print(f'[TEST] Accuracy: {test_acc:.4f}')
        print('[TEST] Classification Report:\n' + test_report)


if __name__ == '__main__':
    main()
