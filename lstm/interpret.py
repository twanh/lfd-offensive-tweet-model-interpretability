import argparse
import json
import os
from collections import defaultdict
from typing import Any, Union

import numpy as np
import spacy
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 50  # Maximum sequence length for padding


def create_arg_parser() -> argparse.Namespace:
    """Create argument parser matching interpret.py CLI flags."""

    parser = argparse.ArgumentParser(
        description='Generate explanations for a fine-tuned LSTM model using SHAP-inspired attribution',
    )

    # Model path
    parser.add_argument(
        'model_path',
        help='The path to the fine-tuned LSTM model directory',
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


def spacy_tokenizer(nlp: spacy.language.Language, texts: list[str]) -> list[list[str]]:
    """Tokenize texts using SpaCy."""

    all_tokenized_texts = []
    # Process texts using nlp.pipe for efficiency
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


def texts_to_sequences(tokenized_texts: list[list[str]], word_to_idx: dict[str, int]) -> list[list[int]]:
    """Converts tokenized texts to integer sequences."""
    return [[word_to_idx.get(word, word_to_idx['<unk>']) for word in text] for text in tokenized_texts]


class LSTMWrapper:
    """Wrapper for LSTM model to work with SHAP."""
    
    def __init__(
        self,
        model: tf.keras.Model,
        word_to_idx: dict[str, int],
        nlp: spacy.language.Language,
        label_encoder: Any,
    ):
        self.model = model
        self.word_to_idx = word_to_idx
        self.nlp = nlp
        self.label_encoder = label_encoder
        self.idx_to_word = {idx: word for word, idx in word_to_idx.items()}
        
    def __call__(self, texts: Union[list[str], np.ndarray, str]) -> np.ndarray:
        """Process texts and return predictions.
        
        Args:
            texts: List of text strings or numpy array of strings
            
        Returns:
            Array of logits for each class
        """
        # Handle different input types
        if isinstance(texts, np.ndarray):
            # Handle case where SHAP passes numpy array
            if texts.dtype == object or texts.dtype.kind == 'U':
                texts = [str(text) for text in texts]
            else:
                texts = [str(text.item()) if text.size == 1 else str(text) for text in texts]
        elif isinstance(texts, str):
            # Handle single string input
            texts = [texts]
        
        # Ensure texts is a list
        if not isinstance(texts, list):
            texts = list(texts)
        
        # Tokenize
        tokenized = spacy_tokenizer(self.nlp, texts)
        
        # Convert to sequences
        sequences = texts_to_sequences(tokenized, self.word_to_idx)
        
        # Pad sequences
        padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
        
        # Get predictions
        predictions = self.model.predict(padded, verbose=0)
        
        # Return logits (convert from probabilities to logits for better SHAP explanations)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        logits = np.log(predictions / (1 - predictions + epsilon))
        
        return logits


class SpacyTokenizerWrapper:
    """Wrapper for SpaCy tokenizer to work with SHAP Text masker.
    
    SHAP expects certain attributes and methods from tokenizers.
    """
    
    def __init__(self, nlp: spacy.language.Language):
        self.nlp = nlp
        # Add special tokens map that SHAP expects
        self.special_tokens_map = {}
        
    def __call__(self, text: str) -> list[str]:
        """Tokenize text and return list of tokens."""
        if not text or text.strip() == "":
            # Return empty list for empty string
            return []
        doc = self.nlp(text, disable=['parser', 'ner', 'tagger', 'lemmatizer', 'attribute_ruler'])
        return [token.text for token in doc]
    
    def encode(self, text: str) -> list[str]:
        """Encode text (alias for __call__ for SHAP compatibility)."""
        return self.__call__(text)
    
    def decode(self, tokens: list[str]) -> str:
        """Decode tokens back to text."""
        return " ".join(tokens)


def analyze_feature_contributions(
    all_shap_values: list,
    tokenized_tweets: list,
    tweets: list[str],
    class_idx: int,
    n: int = 10,
) -> list[dict[str, Any]]:
    """Analyze which features contribute most towards classification."""
    
    results = []
    
    for i, (tweet, tokens, shap_vals) in enumerate(zip(tweets, tokenized_tweets, all_shap_values)):
        # Create feature analysis
        feature_scores = []
        for j, (token, value) in enumerate(zip(tokens, shap_vals)):
            if token.strip():  # Skip empty tokens
                # Get the SHAP value for the specified class
                class_value = value[class_idx] if len(value) > class_idx else 0.0
                feature_scores.append({
                    'token': token,
                    'shap_value': float(class_value),
                    'index': int(j),
                })
        
        # Sort by absolute SHAP value and get top N
        feature_scores.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        top_features = feature_scores[:n]
        
        results.append({
            'tweet': tweet,
            'top_features': top_features,
            'all_features': feature_scores,
        })
    
    return results


def aggregate_statistics(
    all_results: list[dict[str, Any]],
    min_count: int = 10,
) -> dict[str, Any]:
    """Aggregate statistics across all tweets."""

    # Aggregate top features
    feature_counts = defaultdict(lambda: {'count': 0, 'total_shap': 0.0, 'total_abs_shap': 0.0})
    
    for result in all_results:
        # Aggregate top features
        for feat in result.get('top_features', []):
            token = feat['token']
            feature_counts[token]['count'] += 1
            feature_counts[token]['total_shap'] += feat['shap_value']
            feature_counts[token]['total_abs_shap'] += abs(feat['shap_value'])

    # Calculate averages
    top_features_avg = [
        {
            'token': token,
            'avg_shap_value': data['total_shap'] / data['count'],
            'avg_abs_shap_value': data['total_abs_shap'] / data['count'],
            'count': data['count'],
        }
        for token, data in feature_counts.items()
        if data['count'] >= min_count
    ]
    top_features_avg.sort(key=lambda x: x['avg_abs_shap_value'], reverse=True)

    return {
        'top_features': top_features_avg,
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
            f'{i:2d}. {feat["token"]:20s} | SHAP Value: {feat["shap_value"]:10.6f}',
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
            f'{i:2d}. {feat["token"]:20s} | Avg |SHAP|: {feat["avg_abs_shap_value"]:10.6f} | '
            f'Avg SHAP: {feat["avg_shap_value"]:10.6f} | Count: {feat["count"]}',
        )


def main() -> int:
    """Main function to run interpretability analysis."""

    args = create_arg_parser()

    # Load SpaCy model
    print('Loading SpaCy model...')
    nlp = spacy.load('en_core_web_sm')

    # Load model and artifacts
    print(f'Loading model from {args.model_path}...')
    model_path = os.path.join(args.model_path, 'model.keras')
    word_to_idx_path = os.path.join(args.model_path, 'word_to_idx.json')
    label_encoder_path = os.path.join(args.model_path, 'label_encoder.pkl')
    
    if not os.path.exists(model_path):
        print(f'Error: Model file not found at {model_path}')
        return 1
    
    model = tf.keras.models.load_model(model_path)
    
    # Load word_to_idx
    with open(word_to_idx_path, 'r') as f:
        word_to_idx = json.load(f)
    
    # Load label encoder
    import pickle
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Create wrapper
    model_wrapper = LSTMWrapper(model, word_to_idx, nlp, label_encoder)
    
    # Read tweets
    tweets, labels = read_corpus(args.input_file)
    
    # Limit to a reasonable number for SHAP (can be slow for many samples)
    max_samples = min(100, len(tweets))  # Process up to 100 samples
    if len(tweets) > max_samples:
        print(f'Note: Processing first {max_samples} samples (SHAP can be slow for many samples)')
        tweets = tweets[:max_samples]
        labels = labels[:max_samples]
    
    # Get class names
    class_names = label_encoder.classes_.tolist()
    
    print(f'\nProcessing {len(tweets)} tweets...')
    
    # For LSTM, we'll use a simpler approach with SHAP's Partition explainer
    # Create a wrapper that works with pre-tokenized inputs
    print('Creating SHAP explainer...')
    
    # Tokenize all tweets first
    print('Tokenizing tweets...')
    tokenized_tweets = []
    for tweet in tqdm(tweets, desc="Tokenizing"):
        doc = nlp(tweet, disable=['parser', 'ner', 'tagger', 'lemmatizer', 'attribute_ruler'])
        tokens = [token.text for token in doc]
        tokenized_tweets.append(tokens)
    
    # Create a wrapper that accepts token lists
    def model_wrapper_tokens(token_lists):
        """Wrapper that accepts lists of tokens instead of raw text."""
        texts = []
        for tokens in token_lists:
            # Join tokens back into text
            if isinstance(tokens, np.ndarray):
                tokens = tokens.tolist()
            if isinstance(tokens, list):
                text = " ".join([str(t) for t in tokens if t])
            else:
                text = str(tokens)
            texts.append(text)
        return model_wrapper(texts)
    
    # Use a simpler masker approach - create a custom masker for token lists
    # We'll compute SHAP values by iteratively masking tokens
    print('Computing SHAP values manually (this may take a while)...')
    print('Note: This process evaluates the model many times per sample.')
    
    all_shap_values = []
    base_values = []
    
    for idx, (tweet, tokens) in enumerate(tqdm(zip(tweets, tokenized_tweets), total=len(tweets), desc="Computing SHAP")):
        if len(tokens) == 0:
            continue
            
        # Get base prediction (with all tokens)
        full_pred = model_wrapper([tweet])[0]
        
        # Get prediction with empty input (baseline)
        empty_pred = model_wrapper([""])[0]
        base_values.append(empty_pred)
        
        # Compute SHAP values for each token
        token_shap_values = []
        for i in range(len(tokens)):
            # Create masked version (remove this token)
            masked_tokens = tokens[:i] + tokens[i+1:]
            masked_text = " ".join(masked_tokens) if masked_tokens else ""
            
            # Get prediction without this token
            masked_pred = model_wrapper([masked_text])[0]
            
            # SHAP value is approximately the difference
            shap_val = full_pred - masked_pred
            token_shap_values.append(shap_val)
        
        all_shap_values.append(token_shap_values)
    
    # Convert to a format similar to SHAP Explanation objects
    # We'll create a simple structure that our analysis functions can use
    class SimpleExplanation:
        def __init__(self, values, data, base_values):
            self.values = values
            self.data = data
            self.base_values = base_values
            
        def __getitem__(self, idx):
            if isinstance(idx, int):
                return SimpleExplanation([self.values[idx]], [self.data[idx]], [self.base_values[idx]])
            elif isinstance(idx, slice):
                return SimpleExplanation(self.values[idx], self.data[idx], self.base_values[idx])
            elif isinstance(idx, tuple):
                # Handle multi-dimensional indexing
                item = self
                for i in idx:
                    if isinstance(i, int):
                        item = item._get_single(i)
                    elif i == Ellipsis:
                        continue
                    else:
                        item = item._get_slice(i)
                return item
            return self
        
        def _get_single(self, idx):
            return self
        
        def _get_slice(self, idx):
            return self
        
        @property
        def shape(self):
            if not self.values:
                return (0, 0, 0)
            max_tokens = max(len(v) for v in self.values)
            num_classes = len(self.values[0][0]) if self.values and self.values[0] else 2
            return (len(self.values), max_tokens, num_classes)
    
    shap_values = SimpleExplanation(all_shap_values, tokenized_tweets, base_values)
    
    print(f'SHAP values shape: {shap_values.shape}')
    print(f'Class names: {class_names}')
    
    all_results = []
    
    # Process each tweet
    for i, (tweet, true_label) in enumerate(tqdm(zip(tweets, labels), total=len(tweets), unit='tweet')):
        # Get prediction
        pred_logits = model_wrapper([tweet])
        predicted_class_idx = np.argmax(pred_logits[0])
        predicted_class = class_names[predicted_class_idx]
        
        # Analyze features for the predicted class
        feature_analysis = analyze_feature_contributions(
            [all_shap_values[i]],
            [tokenized_tweets[i]],
            [tweet],
            predicted_class_idx,
            n=args.n,
        )
        
        result = {
            'tweet': tweet,
            'predicted_class': predicted_class,
            'true_label': true_label,
            'top_features': feature_analysis[0]['top_features'] if feature_analysis else [],
        }
        
        all_results.append(result)
        
        # Print results for this tweet
        print_results(result, tweet, predicted_class)
    
    # Aggregate statistics
    aggregated = aggregate_statistics(all_results, min_count=args.min_count)
    
    # Print aggregated results
    print_aggregated_results(aggregated, n=args.n)
    
    # Note: Custom visualizations would go here
    # Since we're using a simplified SHAP computation, we don't have the full
    # SHAP library visualization capabilities. The aggregated statistics above
    # provide the key insights about feature importance.
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

