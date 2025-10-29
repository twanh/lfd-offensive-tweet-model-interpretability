import argparse

import torch
from BERT_explainability.modules.BERT.BertForSequenceClassification import BertForSequenceClassification
from BERT_explainability.modules.BERT.ExplanationGenerator import Generator
from captum.attr import visualization
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    explainer = Generator(model)

    classications = ['OFF', 'NOT']

    tweets, labels = read_corpus(args.input_file)

    for tweet, label in zip(tweets, labels):

        true_class = classications.index(label)

        encoding = tokenizer(tweet, return_tensors='pt')
        input_ids = encoding['input_ids'].to('cuda')
        attention_mask = encoding['attention_mask'].to('cuda')

        expl = explainer.generate_LRP(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_layer=0,
        )[0]
        # Normalize scores
        expl = (expl - expl.min()) / (expl.max() - expl.min())

        # Get the model classifcation
        output = torch.nn.functional.softmax(
            model(input_ids=input_ids, attention_mask=attention_mask)[0],
            dim=-1,
        )
        predicted_class = torch.argmax(output, dim=-1).item()

        class_name = classications[predicted_class]
        tokens = tokenizer.convert_ids_to_tokens(
            input_ids.flatten(),
        )
        print(f'Tweet: {tweet}')
        print(f'True class: {label}, Predicted class: {class_name}')
        print('Top explanations:')
        print([(tokens[i], expl[i].item()) for i in range(len(tokens))])

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
