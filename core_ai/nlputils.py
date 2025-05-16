# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

 # src/nlputils.py
import nltk
import os
import logging
import random
import re
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, FreqDist, bigrams, trigrams  # Add missing imports
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # Add missing import

# Project directory structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
LOG_DIR = os.path.join(BASE_DIR, 'log')
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# Ensure directories exist
for directory in [SRC_DIR, LOG_DIR, RAW_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'mathemagician.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download necessary NLTK data packages
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    logger.info("NLTK data packages downloaded successfully")
except Exception as e:
    logger.error(f"Error downloading NLTK data: {e}")


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into words using NLTK's word_tokenize.

    Args:
        text: The input text.

    Returns:
        List[str]: A list of word tokens.
    """
    try:
        tokens = word_tokenize(text)
        logger.info(f"Tokenized text into {len(tokens)} tokens")
        return tokens
    except Exception as e:
        logger.error(f"Error tokenizing text: {e}")
        return []


def sentence_tokenize(text: str) -> List[str]:
    """
    Tokenize text into sentences using NLTK's sent_tokenize.

    Args:
        text: The input text.

    Returns:
        List[str]: A list of sentence tokens.
    """
    try:
        sentences = sent_tokenize(text)
        logger.info(f"Tokenized text into {len(sentences)} sentences")
        return sentences
    except Exception as e:
        logger.error(f"Error sentence tokenizing text: {e}")
        return []


def remove_stopwords(
        tokens: List[str],
        language: str = 'english') -> List[str]:
    """
    Remove stopwords from a list of tokens.

    Args:
        tokens: List of tokens.
        language: Language of stopwords (default: 'english').

    Returns:
        List[str]: Tokens with stopwords removed.
    """
    try:
        stop_words = set(stopwords.words(language))
        filtered = [token for token in tokens if token.lower()
                    not in stop_words]
        logger.info(f"Removed {len(tokens) - len(filtered)} stopwords")
        return filtered
    except Exception as e:
        logger.error(f"Error removing stopwords: {e}")
        return tokens


def stem_tokens(tokens: List[str], stemmer_type: str = 'porter') -> List[str]:
    """
    Stem tokens using a specified stemmer.

    Args:
        tokens: List of tokens.
        stemmer_type: Type of stemmer ('porter' supported currently).

    Returns:
        List[str]: List of stemmed tokens.
    """
    try:
        if stemmer_type.lower() == 'porter':
            stemmer = PorterStemmer()
        else:
            raise ValueError(f"Unsupported stemmer type: {stemmer_type}")
        stemmed = [stemmer.stem(token) for token in tokens]
        logger.info(f"Stemmed {len(tokens)} tokens")
        return stemmed
    except Exception as e:
        logger.error(f"Error stemming tokens: {e}")
        return tokens


def lemmatize_tokens(tokens: List[str],
                     pos_tags: Optional[List[Tuple[str,
                                                   str]]] = None) -> List[str]:
    """
    Lemmatize tokens using WordNetLemmatizer with optional POS tags.

    Args:
        tokens: List of tokens.
        pos_tags: Optional list of (token, POS tag) tuples for context-aware lemmatization.

    Returns:
        List[str]: List of lemmatized tokens.
    """
    try:
        lemmatizer = WordNetLemmatizer()
        if pos_tags and len(pos_tags) == len(tokens):
            def get_wordnet_pos(tag: str) -> str:
                if tag.startswith('J'):
                    return wordnet.ADJ
                elif tag.startswith('V'):
                    return wordnet.VERB
                elif tag.startswith('N'):
                    return wordnet.NOUN
                elif tag.startswith('R'):
                    return wordnet.ADV
                return wordnet.NOUN
            lemmatized = [
                lemmatizer.lemmatize(
                    token,
                    get_wordnet_pos(tag)) for token,
                tag in pos_tags]
        else:
            lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
        logger.info(f"Lemmatized {len(tokens)} tokens")
        return lemmatized
    except Exception as e:
        logger.error(f"Error lemmatizing tokens: {e}")
        return tokens


def pos_tag_tokens(tokens: List[str]) -> List[Tuple[str, str]]:
    """
    Tag tokens with their part-of-speech.

    Args:
        tokens: List of tokens.

    Returns:
        List[Tuple[str, str]]: List of (token, POS tag) tuples.
    """
    try:
        pos_tags = pos_tag(tokens)
        logger.info(f"POS tagged {len(tokens)} tokens")
        return pos_tags
    except Exception as e:
        logger.error(f"Error POS tagging tokens: {e}")
        return [(token, 'NN') for token in tokens]  # Default to noun


def frequency_distribution(tokens: List[str]) -> FreqDist:
    """
    Compute the frequency distribution of tokens.

    Args:
        tokens: List of tokens.

    Returns:
        FreqDist: Frequency distribution object.
    """
    try:
        freq_dist = FreqDist(tokens)
        logger.info(
            f"Computed frequency distribution for {
                len(tokens)} tokens")
        return freq_dist
    except Exception as e:
        logger.error(f"Error computing frequency distribution: {e}")
        return FreqDist()


def extract_ngrams(tokens: List[str], n: int = 2) -> List[Tuple[str, ...]]:
    """
    Extract n-grams from a list of tokens.

    Args:
        tokens: List of tokens.
        n: Size of n-grams (e.g., 2 for bigrams, 3 for trigrams).

    Returns:
        List[Tuple[str, ...]]: List of n-gram tuples.
    """
    try:
        if n == 2:
            ngrams = list(bigrams(tokens))
        elif n == 3:
            ngrams = list(trigrams(tokens))
        else:
            ngrams = list(nltk.ngrams(tokens, n))
        logger.info(f"Extracted {len(ngrams)} {n}-grams")
        return ngrams
    except Exception as e:
        logger.error(f"Error extracting {n}-grams: {e}")
        return []


def sentiment_analysis(text: str) -> Dict[str, float]:
    """
    Perform sentiment analysis using VADER.

    Args:
        text: Input text.

    Returns:
        Dict[str, float]: Sentiment scores (neg, neu, pos, compound).
    """
    try:
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(text)
        logger.info(
            f"Sentiment analysis completed: compound score = {
                scores['compound']:.4f}")
        return scores
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}


def preprocess_text(
        text: str,
        remove_stop: bool = True,
        lemmatize: bool = True,
        stem: bool = False) -> List[str]:
    """
    Preprocess text with tokenization, stopword removal, and optional stemming/lemmatization.

    Args:
        text: Input text.
        remove_stop: Whether to remove stopwords.
        lemmatize: Whether to lemmatize tokens.
        stem: Whether to stem tokens (overrides lemmatize if True).

    Returns:
        List[str]: Processed tokens.
    """
    tokens = tokenize(text)
    if remove_stop:
        tokens = remove_stopwords(tokens)
    if stem:
        tokens = stem_tokens(tokens)
    elif lemmatize:
        pos_tags = pos_tag_tokens(tokens)
        tokens = lemmatize_tokens(tokens, pos_tags)
    return tokens


def load_text_from_raw(filename: str) -> Optional[str]:
    """
    Load text content from a raw data file in data/raw.

    Args:
        filename: Filename relative to data/raw.

    Returns:
        str: Text content, or None if failed.
    """
    filepath = os.path.join(RAW_DIR, filename)
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
            text = ' '.join(df.astype(str).agg(' '.join, axis=1))
        elif filename.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            raise ValueError(f"Unsupported file type: {filename}")
        logger.info(f"Loaded text from {filepath}")
        return text
    except Exception as e:
        logger.error(f"Error loading text from {filepath}: {e}")
        return None


def process_raw_data() -> Dict[str, List[str]]:
    """
    Process all text files in data/raw and return tokenized results.

    Returns:
        Dict[str, List[str]]: Mapping of filenames to processed tokens.
    """
    results = {}
    for filename in os.listdir(RAW_DIR):
        text = load_text_from_raw(filename)
        if text:
            tokens = preprocess_text(text)
            results[filename] = tokens
    return results


def save_tokens(tokens: List[str], filename: str) -> bool:
    """
    Save tokens to a file in outputs directory.

    Args:
        tokens: List of tokens.
        filename: Destination filename (relative to outputs).

    Returns:
        bool: True if successful, False otherwise.
    """
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(tokens))
        logger.info(f"Saved tokens to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving tokens to {filepath}: {e}")
        return False


def extract_keywords(
        tokens: List[str], top_n: int = 5) -> List[Tuple[str, int]]:
    """
    Extract top N keywords based on frequency.

    Args:
        tokens: List of tokens.
        top_n: Number of keywords to return.

    Returns:
        List[Tuple[str, int]]: List of (keyword, frequency) tuples.
    """
    freq_dist = frequency_distribution(tokens)
    keywords = freq_dist.most_common(top_n)
    logger.info(f"Extracted {len(keywords)} keywords")
    return keywords


def dangerous_sentiment_analysis(text: str) -> float:
    """
    Experimental sentiment analysis with risky amplification (dangerous AI theme).

    Args:
        text: Input text.

    Returns:
        float: Amplified compound sentiment score.
    """
    try:
        scores = sentiment_analysis(text)
        amplified = scores['compound'] * \
            random.uniform(1, 10)  # Risky amplification
        logger.warning(
            f"Dangerous sentiment analysis: amplified score = {
                amplified:.4f}")
        return amplified
    except Exception as e:
        logger.error(f"Error in dangerous sentiment analysis: {e}")
        return 0.0


def main():
    """Demonstrate NLP utilities with sample text and raw data."""
    sample_text = "Solve x^2 + 3x + 5 = 0 and then compute the derivative."

    # Basic processing
    tokens = tokenize(sample_text)
    print("Tokens:", tokens)

    filtered = remove_stopwords(tokens)
    print("Filtered Tokens:", filtered)

    stemmed = stem_tokens(filtered)
    print("Stemmed Tokens:", stemmed)

    pos_tags = pos_tag_tokens(tokens)
    lemmatized = lemmatize_tokens(tokens, pos_tags)
    print("Lemmatized Tokens:", lemmatized)

    print("POS Tags:", pos_tags)

    freq_dist = frequency_distribution(tokens)
    print("Frequency Distribution:", list(freq_dist.items()))

    # N-grams
    bigrams_list = extract_ngrams(tokens, 2)
    print("Bigrams:", bigrams_list)

    # Sentiment
    sentiment = sentiment_analysis(sample_text)
    print("Sentiment Scores:", sentiment)

    # Preprocess and save
    processed = preprocess_text(sample_text)
    save_tokens(processed, "processed_sample.txt")

    # Raw data processing
    raw_results = process_raw_data()
    for filename, tokens in raw_results.items():
        print(f"\nProcessed {filename}: {tokens[:10]}... ({len(tokens)} total)")
        keywords = extract_keywords(tokens)
        print(f"Top Keywords in {filename}: {keywords}")

    # Dangerous mode demo
    dangerous_score = dangerous_sentiment_analysis(sample_text)
    print(f"Dangerous Sentiment Score: {dangerous_score:.4f}")


if __name__ == "__main__":
    main()

# Additional utilities


def clean_text(text: str) -> str:
    """Remove special characters and extra whitespace."""
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    text = ' '.join(text.split())
    return text


def batch_process_texts(texts: List[str]) -> Dict[str, List[str]]:
    """Process multiple texts in batch."""
    return {f"text_{i}": preprocess_text(text) for i, text in enumerate(texts)}


def export_freq_dist(freq_dist: FreqDist, filename: str) -> bool:
    """Export frequency distribution to a CSV file."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        pd.DataFrame(
            freq_dist.items(),
            columns=[
                'Token',
                'Frequency']).to_csv(
            filepath,
            index=False)
        logger.info(f"Frequency distribution exported to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error exporting frequency distribution: {e}")
        return False