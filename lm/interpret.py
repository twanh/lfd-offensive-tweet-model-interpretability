import argparse
from collections import defaultdict

import torch
from BERT_explainability.modules.BERT.BertForSequenceClassification import BertForSequenceClassification
from BERT_explainability.modules.BERT.ExplanationGenerator import Generator
from captum.attr import visualization
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import BertTokenizer


def create_arg_parser() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description='Generate explanations for a fine-tuned BERT model',
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

    # Output file for explanations
    # parser.add_argument(
    #     'output_file',
    #     help='The output file to save the explanations',
    #     type=str,
    # )

    # The N amount of top features
    parser.add_argument(
        '--n',
        help='Number of top features to display',
        type=int,
        default=10,
    )

    parser.add_argument(
        '--min-count',
        help='Minimum count for a feature to be considered',
        type=int,
        default=10,
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


def main() -> int:

    args = create_arg_parser()

    model = BertForSequenceClassification.from_pretrained(
        args.model_path,
    ).to('cuda')
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    explainer = Generator(model)

    classications = ['OFF', 'NOT']
    off_indx = classications.index('OFF')
    not_indx = classications.index('NOT')

    tweets, labels = read_corpus(args.input_file)

    global_off_scores = defaultdict(float)
    global_off_counts = defaultdict(int)

    global_not_scores = defaultdict(float)
    global_not_counts = defaultdict(int)

    special_tokens = [
        tokenizer.cls_token,
        tokenizer.sep_token, tokenizer.pad_token,
    ]

    for tweet in tqdm(tweets, unit='tweet'):

        encoding = tokenizer(tweet, return_tensors='pt')
        input_ids = encoding['input_ids'].to('cuda')
        attention_mask = encoding['attention_mask'].to('cuda')

        # Get scores for OFF class
        expl_off = explainer.generate_LRP(
            input_ids=input_ids,
            attention_mask=attention_mask,
            index=off_indx,
            start_layer=0,
        )[0]

        # Get scores for NOT class
        expl_not = explainer.generate_LRP(
            input_ids=input_ids,
            attention_mask=attention_mask,
            index=not_indx,
            start_layer=0,
        )[0]

        # Add scores to global counts
        for i, token in enumerate(tokens):
            # Skip special tokens
            if token in special_tokens:
                continue

            global_off_scores[token] += expl_off[i].item()
            global_off_counts[token] += 1

            global_not_scores[token] += expl_not[i].item()
            global_not_counts[token] += 1

    # Compute average scores
    print('Calculating average scores')
    avg_off_scores = {}
    for token, total_score in global_off_scores.items():
        count = global_off_counts[token]
        if count >= args.min_count:
            avg_off_scores[token] = total_score / count

    # Calculate average "NOT" scores
    avg_not_scores = {}
    for token, total_score in global_not_scores.items():
        count = global_not_counts[token]
        if count >= args.min_count:
            avg_not_scores[token] = total_score / count

    # Sort to get Top N
    sorted_off = sorted(
        avg_off_scores.items(),
        key=lambda x: x[1], reverse=True,
    )
    sorted_not = sorted(
        avg_not_scores.items(),
        key=lambda x: x[1], reverse=True,
    )

    print('\n' + '='*62)
    print(f'GLOBAL FEATURE IMPORTANCE (Min. Count: {args.min_count})')
    print('='*62 + '\n')

    print(f"{'Top N for OFF (Class 0)':<30} | {'Top N for NOT (Class 1)':<30}")
    print(f"{'-'*28:<30} | {'-'*28:<30}")

    for i in range(args.n):
        # Get OFF token/score
        try:
            tok_off, score_off = sorted_off[i]
            off_str = f'{i+1: >2}. {tok_off:<15} ({score_off:.4f})'
        except IndexError:
            off_str = ''

        # Get NOT token/score
        try:
            tok_not, score_not = sorted_not[i]
            not_str = f'{i+1: >2}. {tok_not:<15} ({score_not:.4f})'
        except IndexError:
            not_str = ''

        print(f'{off_str:<30} | {not_str:<30}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
