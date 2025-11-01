import argparse
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from captum.attr import IntegratedGradients
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


def create_arg_parser() -> argparse.Namespace:
    """Create argument parser matching interpret.py CLI flags."""

    parser = argparse.ArgumentParser(
        description='Generate explanations for a fine-tuned BERT model using Captum',
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


class BertWrapper(torch.nn.Module):
    """Wrapper for BERT model to work with Captum."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass that returns logits for the target class."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.logits


def compute_attributions(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_class: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Integrated Gradients attributions for the input."""

    wrapped_model = BertWrapper(model).to('cuda')
    wrapped_model.eval()

    # Create baseline (all zeros/padding tokens)
    baseline_ids = torch.zeros_like(input_ids).to('cuda')

    # Initialize IntegratedGradients
    ig = IntegratedGradients(wrapped_model)

    # Compute attributions
    attributions = ig.attribute(
        input_ids,
        baselines=baseline_ids,
        target=target_class,
        additional_forward_args=(attention_mask,),
        n_steps=50,
    )

    return attributions, input_ids


def analyze_feature_contributions(
    attributions: torch.Tensor,
    tokens: list[str],
    attention_mask: torch.Tensor,
    n: int = 10,
) -> dict[str, Any]:
    """Analyze which features contribute most towards classification."""

    # Sum attributions across embedding dimension
    attr_scores = attributions.sum(dim=2).squeeze(0).cpu().detach().numpy()
    attention = attention_mask.squeeze(0).cpu().numpy()

    # Filter out padding tokens
    valid_indices = attention == 1
    valid_tokens = [tokens[i] for i in range(len(tokens)) if valid_indices[i]]
    valid_attr = attr_scores[valid_indices]

    # Get top contributing features
    top_indices = np.argsort(valid_attr)[::-1][:n]
    top_features = [
        {
            'token': valid_tokens[idx],
            'attribution': float(valid_attr[idx]),
            'index': int(idx),
        }
        for idx in top_indices
    ]

    return {
        'top_features': top_features,
        'all_attributions': dict(zip(valid_tokens, valid_attr.tolist())),
    }


def analyze_context_usage(
    attributions: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict[str, Any]:
    """Analyze how much context is used by the model."""

    attr_scores = attributions.sum(dim=2).squeeze(0).cpu().detach().numpy()
    attention = attention_mask.squeeze(0).cpu().numpy()

    # Filter out padding tokens
    valid_attr = attr_scores[attention == 1]

    # Calculate metrics
    total_attribution = np.abs(valid_attr).sum()
    num_tokens = len(valid_attr)

    # Calculate how much of the context contributes significantly
    # (tokens with attribution > 5% of total)
    threshold = total_attribution * 0.05
    significant_tokens = np.sum(np.abs(valid_attr) > threshold)
    context_usage_ratio = significant_tokens / num_tokens if num_tokens > 0 else 0

    # Calculate entropy (measure of concentration)
    # Higher entropy = more distributed, lower entropy = more concentrated
    abs_attr = np.abs(valid_attr)
    abs_attr = abs_attr + 1e-10  # Avoid log(0)
    normalized = abs_attr / abs_attr.sum()
    entropy = -np.sum(normalized * np.log(normalized))
    max_entropy = np.log(num_tokens) if num_tokens > 0 else 0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    return {
        'total_tokens': int(num_tokens),
        'significant_tokens': int(significant_tokens),
        'context_usage_ratio': float(context_usage_ratio),
        'entropy': float(entropy),
        'normalized_entropy': float(normalized_entropy),
        'concentration': 'distributed' if normalized_entropy > 0.7 else 'focused',
    }


def analyze_steering_away_features(
    attributions: torch.Tensor,
    tokens: list[str],
    attention_mask: torch.Tensor,
    n: int = 10,
) -> list[dict[str, Any]]:
    """Analyze which features steer away from classification."""

    attr_scores = attributions.sum(dim=2).squeeze(0).cpu().detach().numpy()
    attention = attention_mask.squeeze(0).cpu().numpy()

    # Filter out padding tokens
    valid_indices = attention == 1
    valid_tokens = [tokens[i] for i in range(len(tokens)) if valid_indices[i]]
    valid_attr = attr_scores[valid_indices]

    # Get features with negative attributions (steering away)
    negative_indices = np.where(valid_attr < 0)[0]
    if len(negative_indices) == 0:
        return []

    negative_attr = valid_attr[negative_indices]
    top_negative_indices = negative_indices[np.argsort(negative_attr)][:n]

    steering_away = [
        {
            'token': valid_tokens[idx],
            'attribution': float(valid_attr[idx]),
            'index': int(idx),
        }
        for idx in top_negative_indices
    ]

    return steering_away


def analyze_feature_combinations(
    attributions: torch.Tensor,
    tokens: list[str],
    attention_mask: torch.Tensor,
    n: int = 10,
    window_size: int = 3,
) -> list[dict[str, Any]]:
    """Analyze combinations of features that steer towards classification."""

    attr_scores = attributions.sum(dim=2).squeeze(0).cpu().detach().numpy()
    attention = attention_mask.squeeze(0).cpu().numpy()

    # Filter out padding tokens
    valid_indices = attention == 1
    valid_tokens = [tokens[i] for i in range(len(tokens)) if valid_indices[i]]
    valid_attr = attr_scores[valid_indices]

    # Calculate combination scores for n-grams
    combination_scores = []

    for i in range(len(valid_tokens) - window_size + 1):
        # Get tokens in window
        window_tokens = valid_tokens[i:i + window_size]
        window_attr = valid_attr[i:i + window_size]

        # Calculate combined score (sum of positive attributions)
        positive_attr = window_attr[window_attr > 0]
        if len(positive_attr) > 0:
            combination_score = positive_attr.sum()
            combination_scores.append({
                'tokens': window_tokens,
                'score': float(combination_score),
                'start_index': int(i),
            })

    # Sort by score and return top N
    combination_scores.sort(key=lambda x: x['score'], reverse=True)
    return combination_scores[:n]


def aggregate_statistics(
    all_results: list[dict[str, Any]],
    min_count: int = 10,
) -> dict[str, Any]:
    """Aggregate statistics across all tweets."""

    # Aggregate top features
    feature_counts = defaultdict(lambda: {'count': 0, 'total_attr': 0.0})
    steering_away_counts = defaultdict(lambda: {'count': 0, 'total_attr': 0.0})
    combination_counts = defaultdict(lambda: {'count': 0, 'total_score': 0.0})

    context_usage_ratios = []
    entropies = []

    for result in all_results:
        # Aggregate top features
        for feat in result.get('top_features', []):
            token = feat['token']
            feature_counts[token]['count'] += 1
            feature_counts[token]['total_attr'] += feat['attribution']

        # Aggregate steering away features
        for feat in result.get('steering_away', []):
            token = feat['token']
            steering_away_counts[token]['count'] += 1
            steering_away_counts[token]['total_attr'] += abs(
                feat['attribution'],
            )

        # Aggregate combinations
        for combo in result.get('combinations', []):
            combo_key = ' '.join(combo['tokens'])
            combination_counts[combo_key]['count'] += 1
            combination_counts[combo_key]['total_score'] += combo['score']

        # Aggregate context metrics
        if 'context' in result:
            context_usage_ratios.append(
                result['context']['context_usage_ratio'],
            )
            entropies.append(result['context']['entropy'])

    # Calculate averages
    top_features_avg = [
        {
            'token': token,
            'avg_attribution': data['total_attr'] / data['count'],
            'count': data['count'],
        }
        for token, data in feature_counts.items()
        if data['count'] >= min_count
    ]
    top_features_avg.sort(key=lambda x: x['avg_attribution'], reverse=True)

    steering_away_avg = [
        {
            'token': token,
            'avg_attribution': data['total_attr'] / data['count'],
            'count': data['count'],
        }
        for token, data in steering_away_counts.items()
        if data['count'] >= min_count
    ]
    steering_away_avg.sort(key=lambda x: x['avg_attribution'], reverse=True)

    combinations_avg = [
        {
            'tokens': key,
            'avg_score': data['total_score'] / data['count'],
            'count': data['count'],
        }
        for key, data in combination_counts.items()
        if data['count'] >= min_count
    ]
    combinations_avg.sort(key=lambda x: x['avg_score'], reverse=True)

    avg_context_usage = np.mean(
        context_usage_ratios,
    ) if context_usage_ratios else 0.0
    avg_entropy = np.mean(entropies) if entropies else 0.0

    return {
        'top_features': top_features_avg,
        'steering_away_features': steering_away_avg,
        'combinations': combinations_avg,
        'avg_context_usage': float(avg_context_usage),
        'avg_entropy': float(avg_entropy),
    }


def print_results(
    result: dict[str, Any],
    tweet: str,
    predicted_class: str,
) -> None:
    """Print interpretation results for a single tweet."""

    print('\n' + '=' * 80)
    print(f'Tweet: {tweet}')
    print(f'Predicted Class: {predicted_class}')
    print('=' * 80)

    # Top contributing features
    print('\nTop Contributing Features:')
    print('-' * 80)
    for i, feat in enumerate(result.get('top_features', [])[:10], 1):
        print(
            f'{i:2d}. {feat["token"]:20s} | Attribution: {feat["attribution"]:10.6f}',
        )

    # Context usage
    if 'context' in result:
        ctx = result['context']
        print('\nContext Usage Analysis:')
        print('-' * 80)
        print(f'Total tokens: {ctx["total_tokens"]}')
        print(f'Significant tokens: {ctx["significant_tokens"]}')
        print(f'Context usage ratio: {ctx["context_usage_ratio"]:.2%}')
        print(
            f'Concentration: {ctx["concentration"]} (entropy: {ctx["entropy"]:.3f})',
        )

    # Steering away features
    steering_away = result.get('steering_away', [])
    if steering_away:
        print('\nFeatures Steering Away from Classification:')
        print('-' * 80)
        for i, feat in enumerate(steering_away[:10], 1):
            print(
                f'{i:2d}. {feat["token"]:20s} | Attribution: {feat["attribution"]:10.6f}',
            )

    # Feature combinations
    combinations = result.get('combinations', [])
    if combinations:
        print('\nTop Feature Combinations:')
        print('-' * 80)
        for i, combo in enumerate(combinations[:10], 1):
            tokens_str = ' '.join(combo['tokens'])
            print(
                f'{i:2d}. [{tokens_str}] | Combined Score: {combo["score"]:10.6f}',
            )


def print_aggregated_results(
    aggregated: dict[str, Any],
    n: int = 10,
) -> None:
    """Print aggregated statistics across all tweets."""

    print('\n' + '=' * 80)
    print('Aggregated Statistics Across All Tweets')
    print('=' * 80)

    # Top features
    print('\nMost Frequently Contributing Features:')
    print('-' * 80)
    for i, feat in enumerate(aggregated['top_features'][:n], 1):
        print(
            f'{i:2d}. {feat["token"]:20s} | Avg Attribution: {feat["avg_attribution"]:10.6f} | '
            f'Count: {feat["count"]}',
        )

    # Steering away features
    if aggregated['steering_away_features']:
        print('\nMost Frequently Steering Away Features:')
        print('-' * 80)
        for i, feat in enumerate(aggregated['steering_away_features'][:n], 1):
            print(
                f'{i:2d}. {feat["token"]:20s} | Avg Attribution: {feat["avg_attribution"]:10.6f} | '
                f'Count: {feat["count"]}',
            )

    # Combinations
    if aggregated['combinations']:
        print('\nMost Effective Feature Combinations:')
        print('-' * 80)
        for i, combo in enumerate(aggregated['combinations'][:n], 1):
            print(
                f'{i:2d}. [{combo["tokens"]}] | Avg Score: {combo["avg_score"]:10.6f} | '
                f'Count: {combo["count"]}',
            )

    # Context usage
    print('\nOverall Context Usage:')
    print('-' * 80)
    print(
        f'Average context usage ratio: {aggregated["avg_context_usage"]:.2%}',
    )
    print(f'Average entropy: {aggregated["avg_entropy"]:.3f}')


def main() -> int:
    """Main function to run interpretability analysis."""

    args = create_arg_parser()

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
    ).to('cuda')
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    tweets, labels = read_corpus('bert-base-uncased')

    classications = ['NOT', 'OFF']

    all_results = []

    # Process each tweet
    for tweet, true_label in tqdm(zip(tweets, labels), total=len(tweets), unit='tweet'):
        # Tokenize
        encoding = tokenizer(
            tweet,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = encoding['input_ids'].to('cuda')
        attention_mask = encoding['attention_mask'].to('cuda')

        # Get prediction
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(dim=1).item()
            predicted_class = classications[predicted_class_idx]

        # Compute attributions for the predicted class
        attributions, _ = compute_attributions(
            model,
            input_ids,
            attention_mask,
            predicted_class_idx,
        )

        # Get tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids.flatten())

        # Analyze features
        feature_analysis = analyze_feature_contributions(
            attributions,
            tokens,
            attention_mask,
            n=args.n,
        )

        # Analyze context usage
        context_analysis = analyze_context_usage(attributions, attention_mask)

        # Analyze steering away features
        steering_away = analyze_steering_away_features(
            attributions,
            tokens,
            attention_mask,
            n=args.n,
        )

        # Analyze feature combinations
        combinations = analyze_feature_combinations(
            attributions,
            tokens,
            attention_mask,
            n=args.n,
            window_size=3,
        )

        result = {
            'tweet': tweet,
            'predicted_class': predicted_class,
            'true_label': true_label,
            'top_features': feature_analysis['top_features'],
            'context': context_analysis,
            'steering_away': steering_away,
            'combinations': combinations,
        }

        all_results.append(result)

    # Aggregate statistics
    aggregated = aggregate_statistics(all_results, min_count=args.min_count)

    # Print aggregated results
    print_aggregated_results(aggregated, n=args.n)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
