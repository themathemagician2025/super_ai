# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

# src/textprocessor.py
import re
import os
import logging
import pandas as pd
from typing import List, Set, Dict, Optional, Any
from datetime import datetime
import src.nlputils as nlputils  # Integration with NLP utilities
from config import PROJECT_CONFIG  # For directory structure and settings

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


def extract_math_expressions(text: str) -> List[str]:
    """
    Extract mathematical expressions from text using regex.

    Args:
        text: Input text.

    Returns:
        List[str]: List of extracted expressions.
    """
    try:
        # Extended to include = and parentheses
        pattern = r'[\w\+\-\*\/\^\=\(\)]+'
        expressions = re.findall(pattern, text)
        logger.info(f"Extracted {len(expressions)} math expressions from text")
        return expressions
    except Exception as e:
        logger.error(f"Error extracting math expressions: {e}")
        return []


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into words using nlputils for consistency.

    Args:
        text: Input text.

    Returns:
        List[str]: List of tokens.
    """
    try:
        tokens = nlputils.tokenize(text)
        logger.info(f"Tokenized text into {len(tokens)} tokens")
        return tokens
    except Exception as e:
        logger.error(f"Error tokenizing text: {e}")
        return text.split()  # Fallback to simple split


def clean_text(text: str, preserve_math: bool = True) -> str:
    """
    Clean text by removing special characters except word characters, whitespace, and math operators.

    Args:
        text: Input text.
        preserve_math: Whether to preserve math operators.

    Returns:
        str: Cleaned text.
    """
    try:
        if preserve_math:
            pattern = r'[^\w\s\+\-\*\/\^\=\(\)]'
        else:
            pattern = r'[^\w\s]'
        cleaned = re.sub(pattern, '', text)
        logger.info("Text cleaned successfully")
        return cleaned
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return text


def convert_exponentiation(text: str) -> str:
    """
    Convert caret (^) exponentiation to Python's double asterisk (**) operator.

    Args:
        text: Input text.

    Returns:
        str: Modified text.
    """
    try:
        converted = re.sub(r'\^', '**', text)
        logger.info("Converted exponentiation from ^ to **")
        return converted
    except Exception as e:
        logger.error(f"Error converting exponentiation: {e}")
        return text


def find_variables(text: str) -> Set[str]:
    """
    Extract unique variable names from the text.

    Args:
        text: Input text.

    Returns:
        Set[str]: Set of unique variable names.
    """
    try:
        tokens = re.findall(r'\b[a-zA-Z]+\b', text)
        # Filter out common math functions and Python keywords
        exclude = {
            'sin',
            'cos',
            'tan',
            'log',
            'exp',
            'sqrt',
            'and',
            'or',
            'not',
            'if',
            'else'}
        variables = set(
            token for token in tokens if token not in exclude and not token.isdigit())
        logger.info(f"Found {len(variables)} unique variables")
        return variables
    except Exception as e:
        logger.error(f"Error finding variables: {e}")
        return set()


def preprocess_text(text: str, use_nlp: bool = True) -> Dict[str, Any]:
    """
    Preprocess text with tokenization, cleaning, and optional NLP enhancements.

    Args:
        text: Input text.
        use_nlp: Whether to use nlputils for advanced processing.

    Returns:
        Dict: Processed text components (tokens, cleaned, expressions, variables).
    """
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    expressions = extract_math_expressions(text)
    variables = find_variables(text)

    if use_nlp:
        nlp_tokens = nlputils.preprocess_text(text)
        sentiment = nlputils.sentiment_analysis(text)
    else:
        nlp_tokens = tokens
        sentiment = None

    result = {
        'tokens': nlp_tokens,
        'cleaned_text': cleaned,
        'math_expressions': expressions,
        'variables': variables,
        'sentiment': sentiment
    }
    logger.info("Text preprocessing completed")
    return result


def load_raw_text() -> Dict[str, str]:
    """
    Load text content from raw data files in data/raw.

    Returns:
        Dict: Mapping of filenames to text content.
    """
    raw_texts = {}
    for filename in os.listdir(RAW_DIR):
        filepath = os.path.join(RAW_DIR, filename)
        try:
            if filename.endswith('.txt'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    raw_texts[filename] = f.read()
            elif filename.endswith('.csv'):
                df = pd.read_csv(filepath)
                raw_texts[filename] = ' '.join(
                    df.astype(str).agg(' '.join, axis=1))
            logger.info(f"Loaded raw text from {filename}")
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
    return raw_texts


def process_raw_data() -> Dict[str, Dict[str, Any]]:
    """
    Process all text files in data/raw.

    Returns:
        Dict: Mapping of filenames to processed text components.
    """
    raw_texts = load_raw_text()
    return {filename: preprocess_text(text)
            for filename, text in raw_texts.items()}


def save_processed_text(processed: Dict[str, Any], filename: str) -> bool:
    """
    Save processed text components to a file.

    Args:
        processed: Processed text dictionary.
        filename: Destination filename (relative to outputs).

    Returns:
        bool: True if successful, False otherwise.
    """
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for key, value in processed.items():
                f.write(f"{key}: {value}\n")
        logger.info(f"Saved processed text to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving processed text to {filepath}: {e}")
        return False


def evaluate_expression(expression: str,
                        variables: Dict[str,
                                        float] = None) -> Optional[float]:
    """
    Safely evaluate a mathematical expression with given variable values.

    Args:
        expression: Mathematical expression as a string.
        variables: Dictionary of variable names to values.

    Returns:
        float: Result of evaluation, or None if failed.
    """
    if variables is None:
        variables = {}

    try:
        expr = convert_exponentiation(expression)
        result = eval(expr, {"__builtins__": {}}, variables)
        logger.info(f"Evaluated expression '{expression}' to {result}")
        return result
    except Exception as e:
        logger.error(f"Error evaluating expression '{expression}': {e}")
        return None


def dangerous_evaluate_expression(expression: str) -> float:
    """
    Evaluate an expression with risky amplification (dangerous AI theme).

    Args:
        expression: Mathematical expression as a string.

    Returns:
        float: Amplified result.
    """
    try:
        result = eval(
            convert_exponentiation(expression), {
                "__builtins__": {}}, {})
        amplified = result * random.uniform(1, 100)  # Risky amplification
        logger.warning(
            f"Dangerous evaluation of '{expression}': {result} amplified to {amplified}")
        return amplified
    except Exception as e:
        logger.error(f"Error in dangerous evaluation of '{expression}': {e}")
        return 0.0


def extract_equations(text: str) -> List[str]:
    """
    Extract equations (containing '=') from text.

    Args:
        text: Input text.

    Returns:
        List[str]: List of equations.
    """
    try:
        pattern = r'[\w\s\+\-\*\/\^\(\)]*\=[\w\s\+\-\*\/\^\(\)]*'
        equations = re.findall(pattern, text)
        equations = [eq.strip() for eq in equations if eq.strip()]
        logger.info(f"Extracted {len(equations)} equations")
        return equations
    except Exception as e:
        logger.error(f"Error extracting equations: {e}")
        return []


def main():
    """Demonstrate text processing functionality."""
    sample_text = "Solve x^2 + 3x + 5 = 0 and then compute y = 2*x^3 - 4"

    # Basic processing
    expressions = extract_math_expressions(sample_text)
    print("Extracted Math Expressions:", expressions)

    converted = convert_exponentiation(sample_text)
    print("Converted Text:", converted)

    cleaned = clean_text(sample_text)
    tokens = tokenize_text(cleaned)
    print("Cleaned and Tokenized Text:", tokens)

    variables = find_variables(sample_text)
    print("Variables:", variables)

    # Advanced preprocessing
    processed = preprocess_text(sample_text)
    print("Processed Text Components:", processed)

    # Raw data processing
    raw_results = process_raw_data()
    for filename, result in raw_results.items():
        print(f"\nProcessed {filename}:")
        print(result)
        save_processed_text(result, f"processed_{filename}.txt")

    # Expression evaluation
    expr = "x**2 + 3"
    vars_dict = {'x': 2.0}
    result = evaluate_expression(expr, vars_dict)
    print(f"Evaluated '{expr}' with x=2: {result}")

    # Dangerous mode demo
    dangerous_result = dangerous_evaluate_expression("x**2", {'x': 2.0})
    print(f"Dangerous evaluation result: {dangerous_result}")

    # Equation extraction
    equations = extract_equations(sample_text)
    print("Extracted Equations:", equations)


if __name__ == "__main__":
    main()

# Additional utilities


def batch_process_texts(texts: List[str]) -> Dict[str, Dict[str, Any]]:
    """Process multiple texts in batch."""
    return {f"text_{i}": preprocess_text(text) for i, text in enumerate(texts)}


def normalize_expressions(expressions: List[str]) -> List[str]:
    """Normalize expressions by removing redundant spaces."""
    return [re.sub(r'\s+', '', expr) for expr in expressions]


def validate_expression(expression: str) -> bool:
    """Validate if an expression is syntactically correct."""
    try:
        eval(
            convert_exponentiation(expression), {
                "__builtins__": {}}, {
                'x': 1})
        return True
    except Exception:
        return False
