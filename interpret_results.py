import argparse
import pickle
from collections import Counter
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr


def create_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compare feature alignment across SVM, LSTM, BERT',
    )

    parser.add_argument(
        'svm_results',
        help='SVM interpretability results pickle',
        type=str,
    )

    parser.add_argument(
        'lstm_results',
        help='LSTM interpretability results pickle',
        type=str,
    )

    parser.add_argument(
        'bert_results',
        help='BERT interpretability results pickle',
        type=str,
    )

    parser.add_argument(
        '--k',
        help='Top-k for agreement computation',
        type=int,
        default=10,
    )

    parser.add_argument(
        '-o',
        '--output',
        help='Output file for comparison results',
        type=str,
        default='comparison_results.pkl',
    )

    return parser.parse_args()


def load_results(svm_file: str, lstm_file: str, bert_file: str) -> dict:

    print('Loading results')

    with open(svm_file, 'rb') as f:
        svm_results = pickle.load(f)

    with open(lstm_file, 'rb') as f:
        lstm_results = pickle.load(f)

    with open(bert_file, 'rb') as f:
        bert_results = pickle.load(f)

    return {
        'SVM': svm_results,
        'LSTM': lstm_results,
        'BERT': bert_results,
    }


def compute_jaccard_similarity(set1: set, set2: set) -> float:

    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    if len(set1) == 0 or len(set2) == 0:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def compute_intersection_ratio(set1: set, set2: set, k: int) -> float:

    if k == 0:
        return 0.0
    intersection = len(set1 & set2)
    return intersection / k


def compute_rank_correlation(
    tokens_a: list[str],
    tokens_b: list[str],
) -> float:

    all_tokens = set(tokens_a + tokens_b)

    if len(all_tokens) < 2:
        return np.nan

    # Create rank mappings
    ranks_a = {token: i for i, token in enumerate(tokens_a)}
    ranks_b = {token: i for i, token in enumerate(tokens_b)}

    max_rank_a = len(tokens_a)
    max_rank_b = len(tokens_b)

    ranks_a_full = [ranks_a.get(token, max_rank_a) for token in all_tokens]
    ranks_b_full = [ranks_b.get(token, max_rank_b) for token in all_tokens]

    corr, _ = spearmanr(ranks_a_full, ranks_b_full)
    return corr if not np.isnan(corr) else 0.0  # type: ignore


def compute_pairwise_agreement(
    results: dict,
    k: int,
    label_mask: np.ndarray | None = None,
) -> dict:

    metrics = {}

    # Get top-k tokens for each model
    top_k_per_model = {}
    for model_name, result in results.items():
        # Note: I did not use consistent names
        if 'top_k_tokens' in result:
            top_k_tokens = result['top_k_tokens']
        elif 'top_k_words' in result:
            top_k_tokens = result['top_k_words']
        else:
            raise ValueError(f'No top-k tokens found in {model_name} results')

        # Apply label mask if provided
        if label_mask is not None:
            top_k_tokens = [
                t for t, mask in zip(
                    top_k_tokens, label_mask,
                ) if mask
            ]

        top_k_per_model[model_name] = top_k_tokens

    # Compute pairwise agreements
    model_pairs = [
        ('SVM', 'LSTM'),
        ('SVM', 'BERT'),
        ('LSTM', 'BERT'),
    ]

    for model_a, model_b in model_pairs:
        tokens_a = top_k_per_model[model_a]
        tokens_b = top_k_per_model[model_b]

        # Convert to sets for agreement
        token_sets_a = [set(tokens) for tokens in tokens_a]
        token_sets_b = [set(tokens) for tokens in tokens_b]

        # Intersection ratios
        intersections = [
            compute_intersection_ratio(token_sets_a[i], token_sets_b[i], k)
            for i in range(len(tokens_a))
        ]

        # Rank correlations
        correlations = [
            compute_rank_correlation(tokens_a[i], tokens_b[i])
            for i in range(len(tokens_a))
        ]
        correlations = [c for c in correlations if not np.isnan(c)]

        pair_name = f'{model_a}-{model_b}'
        metrics[pair_name] = {
            'mean_intersection': np.mean(intersections),
            'std_intersection': np.std(intersections),
            'mean_rank_corr': np.mean(correlations) if correlations else 0.0,
            'std_rank_corr': np.std(correlations) if correlations else 0.0,
            'n_samples': len(tokens_a),
        }

    return metrics


def compare_globally_important(results: dict) -> dict:
    """Compare globally important tokens, normalized by frequency."""

    comparison = {}

    for model_name, result in results.items():
        # Count token frequencies
        top_k_key = 'top_k_words' if 'top_k_words' in result else 'top_k_tokens'  # noqa: E501
        token_freq: Counter[str] = Counter()
        for top_k in result[top_k_key]:
            token_freq.update(top_k)

        # Aggregate importance
        token_importance_sum: dict[str, float] = defaultdict(float)
        token_count: dict[str, int] = defaultdict(int)

        for imp_dict in result['per_sample_importances']:
            for token, importance in imp_dict.items():
                token_importance_sum[token] += importance
                token_count[token] += 1

        # Compute normalized importance: avg_importance / frequency
        token_normalized = {}
        for token in token_importance_sum:
            avg_importance = token_importance_sum[token] / token_count[token]
            frequency = token_freq[token]
            # Normalize: importance per occurrence
            # +1 to avoid division by 0
            normalized = avg_importance / (frequency + 1)
            token_normalized[token] = normalized

        # Sort by normalized importance
        sorted_tokens = sorted(
            token_normalized.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        tokens_list = [token for token, _ in sorted_tokens[:20]]
        comparison[model_name] = tokens_list

    # Compute overlaps
    svm_set = set(comparison['SVM'])
    lstm_set = set(comparison['LSTM'])
    bert_set = set(comparison['BERT'])

    return {
        'SVM': comparison['SVM'],
        'LSTM': comparison['LSTM'],
        'BERT': comparison['BERT'],
        'overlap_all_three': list(svm_set & lstm_set & bert_set),
        'overlap_SVM_LSTM': list(svm_set & lstm_set),
        'overlap_SVM_BERT': list(svm_set & bert_set),
        'overlap_LSTM_BERT': list(lstm_set & bert_set),
    }


def main() -> int:
    args = create_arg_parser()

    # Load results
    results = load_results(
        args.svm_results, args.lstm_results, args.bert_results,
    )

    # Verify all have same tweets/labels
    tweets = results['SVM']['tweets']
    labels = results['SVM']['labels']

    print(f'\nComparing {len(tweets)} test samples')
    print(f'k={args.k}')

    # ===== OVERALL AGREEMENT =====
    print('\n' + '='*70)
    print('OVERALL AGREEMENT (ALL SAMPLES)')
    print('='*70)

    overall_metrics = compute_pairwise_agreement(results, args.k)

    for pair, metrics in overall_metrics.items():
        print(f'\n{pair}:')
        print(
            f'\tIntersection@{args.k}: {metrics["mean_intersection"]:.4f} ± {metrics["std_intersection"]:.4f}',  # noqa: E501
        )
        print(
            f'\tRank Correlation: {metrics["mean_rank_corr"]:.4f} ± {metrics["std_rank_corr"]:.4f}',  # noqa: E501
        )

    print('='*70)
    print('PER-LABEL AGREEMENT')
    print('='*70)

    per_label_metrics = {}

    for label in sorted(set(labels)):
        label_mask = np.array(labels) == label
        label_count = np.sum(label_mask)

        print(f'\n{label} (n={label_count}):')
        label_metrics = compute_pairwise_agreement(results, args.k, label_mask)  # noqa: E501
        per_label_metrics[label] = label_metrics

        for pair, metrics in label_metrics.items():
            print(f'\t{pair}:')
            print(
                f'\t\tIntersection@{args.k}: {metrics["mean_intersection"]:.4f} ± {metrics["std_intersection"]:.4f}',  # noqa: E501
            )
            print(
                f'\t\tRank Correlation: {metrics["mean_rank_corr"]:.4f} ± {metrics["std_rank_corr"]:.4f}',  # noqa: E501
            )

    print('\n' + '='*70)
    print('GLOBALLY IMPORTANT TOKENS (TOP 20)')
    print('='*70)

    global_comparison = compare_globally_important(results)

    print()
    print('SVM top 20:')
    for i, token in enumerate(global_comparison['SVM'], 1):
        print(f'\t{i:2d}. {token}')

    print()
    print('LSTM top 20:')
    for i, token in enumerate(global_comparison['LSTM'], 1):
        print(f'\t{i:2d}. {token}')

    print()
    print('BERT top 20:')
    for i, token in enumerate(global_comparison['BERT'], 1):
        print(f'\t{i:2d}. {token}')

    print('\n' + '-'*70)
    print('Token Overlap:')
    print(
        f'\tAll three models: {len(global_comparison["overlap_all_three"])} tokens',  # noqa: E501
    )
    print(f'\t\t{global_comparison["overlap_all_three"]}')
    print(f'\tSVM & LSTM: {len(global_comparison["overlap_SVM_LSTM"])} tokens')
    print(f'\tSVM & BERT: {len(global_comparison["overlap_SVM_BERT"])} tokens')
    print(
        f'\tLSTM & BERT: {len(global_comparison["overlap_LSTM_BERT"])} tokens',
    )

    comparison_results = {
        'overall_metrics': overall_metrics,
        'per_label_metrics': per_label_metrics,
        'global_comparison': global_comparison,
        'k': args.k,
        'n_samples': len(tweets),
    }

    with open(args.output, 'wb') as f:
        pickle.dump(comparison_results, f)

    print(f'Comparison results saved to {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
