import argparse
from typing import Any

import numpy as np
from datasets import Dataset
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import EarlyStoppingCallback
from transformers import EvalPrediction
from transformers import Trainer
from transformers import TrainingArguments


def create_arg_parser() -> argparse.Namespace:
    """Creates the argument parser for command line arguments."""

    parser = argparse.ArgumentParser(
        description='Fine-tune a Transformer with Trainer',
    )

    # Model name
    parser.add_argument(
        'model_name',
        help='The name of the pre-trained model to use',
        type=str,
    )

    # Data arguments
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

    parser.add_argument('--confusion-matrix', default=None)
    parser.add_argument('--max-length', type=int, default=100)

    # Core TrainingArguments
    parser.add_argument('--output-dir', default='./out')
    parser.add_argument('--num-train-epochs', type=int, default=1)
    parser.add_argument('--per-device-train-batch-size', type=int, default=8)
    parser.add_argument('--per-device-eval-batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument(
        '--eval-strategy',
        choices=['no', 'steps', 'epoch'], default='epoch',
    )
    parser.add_argument('--logging-steps', type=int, default=50)
    parser.add_argument('--warmup-ratio', type=float, default=0.0)
    parser.add_argument('--warmup-steps', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument(
        '--early-stopping-patience', type=int,
        default=3, help='Patience for early stopping.',
    )
    parser.add_argument(
        '--metric-for-best-model', type=str,
        default='eval_f1_macro', help='Metric to monitor for best model.',
    )
    parser.add_argument(
        '--load-best-model-at-end', action='store_true',
        help='Load the best model at the end of training.',
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


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_micro = f1_score(labels, preds, average='micro')
    f1_macro = f1_score(labels, preds, average='macro')
    return {'accuracy': acc, 'f1_micro': f1_micro, 'f1_macro': f1_macro}


def prepare_data(
    train_file: str,
    dev_file: str,
    tokenizer: Any,
    max_length: int,
) -> tuple[Dataset, Dataset, LabelEncoder]:

    # Read in the data
    X_train, y_train = read_corpus(train_file)
    X_dev, y_dev = read_corpus(dev_file)

    # Encode the labels
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(
        y_train,
    ).astype(np.int64)  # type:ignore
    y_dev_enc = label_encoder.transform(y_dev).astype(np.int64)

    # Tokenize the data
    tok_train = tokenizer(
        X_train,
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    tok_dev = tokenizer(
        X_dev,
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    # Create datasets
    train_ds = Dataset.from_dict(
        {**tok_train, 'labels': y_train_enc},
    ).with_format('torch')

    dev_ds = Dataset.from_dict(
        {**tok_dev, 'labels': y_dev_enc},
    ).with_format('torch')

    return train_ds, dev_ds, label_encoder


def print_results(metrics: dict[str, float]) -> None:

    # The metric keys will have an "eval_" prefix
    acc = round(metrics['eval_accuracy'] * 100, 1)
    f1_micro = round(metrics['eval_f1_micro'] * 100, 1)
    f1_macro = round(metrics['eval_f1_macro'] * 100, 1)

    print(metrics)

    print('\n\nFinal metrics:\n')
    print(f'Accuracy: {acc}')
    print(f'Micro F1: {f1_micro}')
    print(f'Macro F1: {f1_macro}')


def save_confusion_matrix(
    cm: Any,
    label_encoder: LabelEncoder,
    filename: str,
) -> None:

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))

    # Get the label names to show in the matrix
    labels = label_encoder.classes_
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', xticks_rotation='vertical', values_format='d')
    plt.tight_layout()

    # Save the figure
    plt.savefig(filename)
    plt.close()


def main() -> int:

    args = create_arg_parser()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_ds, dev_ds, label_encoder = prepare_data(
        args.train_file,
        args.dev_file,
        tokenizer,
        args.max_length,
    )

    num_labels = len(label_encoder.classes_)  # type: ignore

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
    )

    # Create the training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy=args.eval_strategy,
        save_strategy=args.eval_strategy,
        logging_steps=args.logging_steps,
        warmup_ratio=args.warmup_ratio if args.warmup_steps == 0 else 0.0,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        report_to=[],
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        save_total_limit=2,
    )

    # Add early stopping
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience,
    )

    # Create the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
    )

    # Start training_args
    trainer.train()
    metrics = trainer.evaluate()
    print_results(metrics)

    if args.test_file is not None:

        X_test, Y_test = read_corpus(args.test_file)
        y_test_ids = label_encoder.transform(Y_test).astype(np.int64)

        tok_test = tokenizer(
            X_test, padding=True, truncation=True,
            max_length=args.max_length,
        )

        test_ds = Dataset.from_dict(
            {**tok_test, 'labels': y_test_ids},
        ).with_format('torch')

        # Get predictions
        test_metrics = trainer.evaluate(eval_dataset=test_ds)

        print('\nTest set results:')
        print_results(test_metrics)

        # If specified, create and save the confusion matrix image
        if args.confusion_matrix is not None:
            # Print confusion matrix for the test file
            preds_output = trainer.predict(test_ds)
            preds = np.argmax(preds_output.predictions, axis=-1)
            cm = confusion_matrix(y_test_ids, preds)
            print('Confusion Matrix:')
            print(cm)
            save_confusion_matrix(cm, label_encoder, args.confusion_matrix)
            print(f'Saving confusion matrix to {args.confusion_matrix}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
