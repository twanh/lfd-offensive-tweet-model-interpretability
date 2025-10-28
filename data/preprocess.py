import argparse

import emoji


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

    parser = argparse.ArgumentParser(
        description='Preprocess tweet dataset.',
    )

    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input TSV file containing tweets and labels.',
    )

    parser.add_argument(
        'output_file',
        type=str,
        help='Path to output TSV file for preprocessed tweets and labels.',
    )

    args = parser.parse_args()

    # Read the corpus
    tweets, labels = read_corpus(args.input_file)

    clean_tweets = []

    for tweet in tweets:
        # Lowercase the tweet
        tweet = tweet.lower()
        clean_tweets.append(tweet)

    # Write the preprocessed tweets and labels to output file
    with open(args.output_file, 'w') as out_file:
        for tweet, label in zip(clean_tweets, labels):
            out_file.write(f'{tweet}\t{label}\n')

    return 0


if __name__ == '__main__':

    raise SystemExit(main())
