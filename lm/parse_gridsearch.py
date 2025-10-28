#!/usr/bin/env python3
import re
from pathlib import Path
from pprint import pprint

# --- Configuration ---
# !!! UPDATE THIS PATH to your grid search directory
BASE_DIR = Path('/scratch/s4781619/lfd_final/grid_search/')
# --- End Configuration ---


def parse_config(content: str) -> dict:
    """Parses the '--- Grid Search Run ---' block."""
    config = {}
    try:
        config['lr'] = re.search(r'Learning Rate: (.*)', content).group(1)
        config['batch_size'] = int(
            re.search(r'Batch Size: (.*)', content).group(1),
        )
        config['max_length'] = int(
            re.search(r'Max Length: (.*)', content).group(1),
        )
        config['patience'] = int(
            re.search(r'Patience: (.*)', content).group(1),
        )
    except AttributeError:
        # This will happen if a file is incomplete
        return None
    return config


def parse_metrics(content: str) -> tuple[dict, dict] | tuple[None, None]:
    """
    Parses the two 'Final metrics:' blocks.
    The first is DEV, the second is TEST.
    """

    # This regex finds all "Final metrics" blocks
    metrics_regex = re.compile(
        r'Final metrics:\n\n'
        r'Accuracy: (.*)\n'
        r'Micro F1: (.*)\n'
        r'Macro F1: (.*)',
    )

    all_matches = metrics_regex.findall(content)

    # We expect exactly two matches: [dev_metrics, test_metrics]
    if len(all_matches) < 2:
        return None, None  # Job likely failed before finishing

    dev_match = all_matches[0]
    test_match = all_matches[1]

    dev_metrics = {
        'accuracy': float(dev_match[0]),
        'f1_micro': float(dev_match[1]),
        'f1_macro': float(dev_match[2]),
    }

    test_metrics = {
        'accuracy': float(test_match[0]),
        'f1_micro': float(test_match[1]),
        'f1_macro': float(test_match[2]),
    }

    return dev_metrics, test_metrics


def main():
    """
    Main function to find, parse, sort, and print results.
    """
    all_results = []

    # Find all 'results.txt' files in subdirectories
    result_files = list(BASE_DIR.glob('*/results.txt'))

    if not result_files:
        print(f"Error: No 'results.txt' files found in {BASE_DIR}")
        return

    print(f'Found {len(result_files)} result files. Parsing...\n')

    for result_file in result_files:
        run_name = result_file.parent.name
        content = result_file.read_text()

        config = parse_config(content)
        dev_metrics, test_metrics = parse_metrics(content)

        if config and dev_metrics and test_metrics:
            all_results.append({
                'run': run_name,
                'config': config,
                'dev_metrics': dev_metrics,
                'test_metrics': test_metrics,
            })
        else:
            print(f'--- Skipping incomplete or failed run: {run_name} ---')

    # Sort results by test F1-Macro, from highest to lowest
    sorted_results = sorted(
        all_results,
        key=lambda x: x['test_metrics'].get('f1_macro', 0),
        reverse=True,
    )

    # Print the sorted results
    print('\n--- Grid Search Results (Sorted by Test F1-Macro) ---')
    for run in sorted_results:
        cfg = run['config']
        dev = run['dev_metrics']
        test = run['test_metrics']

        print(f'\n=======================================================')
        print(f"RUN: {run['run']}")
        print(
            f"Config: LR={cfg['lr']}, BS={cfg['batch_size']}, Len={cfg['max_length']}, Pat={cfg['patience']}",
        )
        print('-------------------------------------------------------')
        print(
            f"  DEV Set  -> F1-Macro: {dev['f1_macro']:<6} | Accuracy: {dev['accuracy']:<6}",
        )
        print(
            f"  TEST Set -> F1-Macro: {test['f1_macro']:<6} | Accuracy: {test['accuracy']:<6}",
        )
        print(f'=======================================================')


if __name__ == '__main__':
    main()
