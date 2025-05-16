#!/usr/bin/env python3
# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
NLP Processor Module

This module provides natural language processing capabilities
using spaCy and Hugging Face transformers.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("nlp_processor")

# Define models directory
MODELS_DIR = os.environ.get("MODELS_DIR", "data/nlp_models")

class NLPProcessor:
    """NLP Processor for text analysis and sentiment prediction"""

    def __init__(self, models_dir: str = MODELS_DIR, use_gpu: bool = False):
        """
        Initialize the NLP processor

        Args:
            models_dir: Directory to store/load models
            use_gpu: Whether to use GPU for inference (if available)
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)

        # Set device
        self.device = 0 if use_gpu and self._is_gpu_available() else -1
        logger.info(f"Using device: {'GPU' if self.device >= 0 else 'CPU'}")

        # Initialize models
        self.nlp = None  # spaCy model
        self.sentiment_analyzer = None  # Transformer sentiment analysis
        self.news_classifier = None  # Financial news classifier

        # Load models
        self._load_models()

    def _is_gpu_available(self) -> bool:
        """Check if GPU is available for inference"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_models(self) -> None:
        """Load NLP models"""
        try:
            # Load spaCy model
            logger.info("Loading spaCy model...")
            try:
                self.nlp = spacy.load("en_core_web_md")
                logger.info("Loaded spaCy model: en_core_web_md")
            except OSError:
                logger.warning("spaCy model not found. Downloading it...")
                spacy.cli.download("en_core_web_md")
                self.nlp = spacy.load("en_core_web_md")

            # Load sentiment analysis model
            logger.info("Loading sentiment analysis model...")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=self.device
            )
            logger.info("Loaded sentiment analysis model")

            # Load or initialize financial news classifier
            # In a real application, you would fine-tune this on financial data
            # Here we're using a pre-trained model for demonstration
            logger.info("Loading financial news classifier...")
            self.news_classifier = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                device=self.device
            )
            logger.info("Loaded financial news classifier")

        except Exception as e:
            logger.error(f"Error loading NLP models: {str(e)}")
            raise

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text using spaCy

        Args:
            text: Text to analyze

        Returns:
            Dictionary with analysis results
        """
        if not self.nlp:
            return {"error": "spaCy model not loaded"}

        try:
            # Process text with spaCy
            doc = self.nlp(text)

            # Extract named entities
            entities = [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                }
                for ent in doc.ents
            ]

            # Extract noun phrases
            noun_phrases = [chunk.text for chunk in doc.noun_chunks]

            # Extract tokens with POS and dependencies
            tokens = [
                {
                    "text": token.text,
                    "lemma": token.lemma_,
                    "pos": token.pos_,
                    "tag": token.tag_,
                    "dep": token.dep_,
                    "is_stop": token.is_stop
                }
                for token in doc
            ]

            # Extract sentences
            sentences = [sent.text for sent in doc.sents]

            return {
                "entities": entities,
                "noun_phrases": noun_phrases,
                "tokens": tokens,
                "sentences": sentences,
                "text_length": len(text),
                "word_count": len(doc)
            }

        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            return {"error": f"Analysis error: {str(e)}"}

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment analysis results
        """
        if not self.sentiment_analyzer:
            return {"error": "Sentiment analysis model not loaded"}

        try:
            # Analyze sentiment
            result = self.sentiment_analyzer(text)[0]

            # Extract result
            sentiment = result["label"]
            score = result["score"]

            # Convert sentiment to numeric score (-1 to 1)
            if sentiment == "POSITIVE":
                sentiment_score = score
            else:
                sentiment_score = -score

            return {
                "sentiment": sentiment.lower(),
                "score": score,
                "sentiment_score": sentiment_score,
                "text": text[:100] + "..." if len(text) > 100 else text
            }

        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {"error": f"Sentiment analysis error: {str(e)}"}

    def classify_financial_news(self, text: str) -> Dict[str, Any]:
        """
        Classify financial news text

        Args:
            text: Financial news text to classify

        Returns:
            Dictionary with classification results
        """
        if not self.news_classifier:
            return {"error": "Financial news classifier not loaded"}

        try:
            # Classify financial news
            result = self.news_classifier(text)[0]

            # Extract result
            label = result["label"]
            score = result["score"]

            # Map FinBERT labels to more descriptive ones
            label_mapping = {
                "positive": "bullish",
                "negative": "bearish",
                "neutral": "neutral"
            }

            market_sentiment = label_mapping.get(label.lower(), label.lower())

            return {
                "market_sentiment": market_sentiment,
                "confidence": score,
                "label": label,
                "text": text[:100] + "..." if len(text) > 100 else text
            }

        except Exception as e:
            logger.error(f"Error classifying financial news: {str(e)}")
            return {"error": f"Classification error: {str(e)}"}

    def extract_financial_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract financial entities from text

        Args:
            text: Text to analyze

        Returns:
            Dictionary with extracted financial entities
        """
        if not self.nlp:
            return {"error": "spaCy model not loaded"}

        try:
            # Process text with spaCy
            doc = self.nlp(text)

            # Extract entities
            entities = []
            for ent in doc.ents:
                # Filter for financial-related entities
                if ent.label_ in ["ORG", "MONEY", "PERCENT", "CARDINAL", "GPE"]:
                    entities.append({
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char
                    })

            # Look for currency symbols and pairs
            currency_pattern = r'\b[A-Z]{3}/[A-Z]{3}\b|\$|€|£|¥'
            currency_matches = []

            # Extract additional financial terms using token patterns
            financial_terms = []
            financial_keywords = [
                "stock", "market", "investor", "bond", "trade", "price",
                "forex", "currency", "exchange", "rate", "volatility",
                "bull", "bear", "trend", "rally", "crash", "recession"
            ]

            for token in doc:
                if token.lemma_.lower() in financial_keywords:
                    financial_terms.append({
                        "text": token.text,
                        "lemma": token.lemma_,
                        "start": token.idx,
                        "end": token.idx + len(token.text)
                    })

            return {
                "financial_entities": entities,
                "financial_terms": financial_terms,
                "text": text[:100] + "..." if len(text) > 100 else text
            }

        except Exception as e:
            logger.error(f"Error extracting financial entities: {str(e)}")
            return {"error": f"Extraction error: {str(e)}"}

    def predict_market_impact(self, news_text: str) -> Dict[str, Any]:
        """
        Predict potential market impact from news text

        Args:
            news_text: Financial news text

        Returns:
            Dictionary with market impact prediction
        """
        try:
            # Analyze sentiment first
            sentiment_result = self.analyze_sentiment(news_text)

            # Classify financial news
            classification_result = self.classify_financial_news(news_text)

            # Extract entities
            entities_result = self.extract_financial_entities(news_text)

            # Process text with spaCy for additional analysis
            doc = self.nlp(news_text)

            # Calculate a composite impact score (-1 to 1)
            sentiment_score = sentiment_result.get("sentiment_score", 0)

            # Adjust score based on classification
            classification_confidence = classification_result.get("confidence", 0.5)
            classification_label = classification_result.get("market_sentiment", "neutral")

            # Calculate intensity from text
            # (Using simplified heuristics - in production, this would be more sophisticated)
            intensity_words = {
                "high": ["surge", "soar", "jump", "explode", "crash", "plummet", "collapse"],
                "medium": ["rise", "increase", "decline", "decrease", "fall", "drop"],
                "low": ["edge", "slight", "minor", "small", "little"]
            }

            intensity_score = 0.5  # Default medium intensity
            for token in doc:
                if token.lemma_.lower() in intensity_words["high"]:
                    intensity_score = 0.9
                    break
                elif token.lemma_.lower() in intensity_words["medium"]:
                    intensity_score = 0.6
                elif token.lemma_.lower() in intensity_words["low"]:
                    intensity_score = 0.3

            # Combine factors for final impact prediction
            if classification_label == "bullish":
                impact_score = sentiment_score * classification_confidence * intensity_score
            elif classification_label == "bearish":
                impact_score = -sentiment_score * classification_confidence * intensity_score
            else:
                impact_score = sentiment_score * 0.3 * intensity_score  # Reduced impact for neutral

            # Determine impacted markets based on entities
            impacted_markets = []
            for entity in entities_result.get("financial_entities", []):
                if entity["label"] == "ORG":
                    impacted_markets.append({
                        "entity": entity["text"],
                        "type": "company"
                    })
                elif entity["label"] == "GPE":  # Countries/regions
                    impacted_markets.append({
                        "entity": entity["text"],
                        "type": "region"
                    })

            # Determine timeframe of impact
            timeframe_words = {
                "short_term": ["today", "now", "immediate", "current", "short-term"],
                "medium_term": ["week", "month", "quarter", "mid-term"],
                "long_term": ["year", "decade", "long-term", "future"]
            }

            timeframe = "medium_term"  # Default
            for token in doc:
                if token.lemma_.lower() in timeframe_words["short_term"]:
                    timeframe = "short_term"
                    break
                elif token.lemma_.lower() in timeframe_words["long_term"]:
                    timeframe = "long_term"

            # Generate impact summary
            if abs(impact_score) < 0.3:
                impact_level = "minimal"
            elif abs(impact_score) < 0.6:
                impact_level = "moderate"
            else:
                impact_level = "significant"

            impact_direction = "positive" if impact_score > 0 else "negative"

            return {
                "impact_score": round(impact_score, 2),
                "impact_summary": f"{impact_level} {impact_direction} impact",
                "market_sentiment": classification_label,
                "confidence": classification_result.get("confidence"),
                "impacted_markets": impacted_markets,
                "timeframe": timeframe,
                "intensity": list(intensity_words.keys())[int(intensity_score * 3) - 1],
                "key_entities": [e["text"] for e in entities_result.get("financial_entities", [])[:5]]
            }

        except Exception as e:
            logger.error(f"Error predicting market impact: {str(e)}")
            return {"error": f"Prediction error: {str(e)}"}

    def batch_analyze_news(self, news_items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Analyze a batch of news items

        Args:
            news_items: List of news items with 'title' and 'content' keys

        Returns:
            List of analysis results
        """
        results = []

        for item in news_items:
            title = item.get("title", "")
            content = item.get("content", "")

            # Combine title and content for analysis, giving more weight to title
            analysis_text = f"{title} {title} {content}"

            # Analyze sentiment
            sentiment = self.analyze_sentiment(analysis_text)

            # Classify news
            classification = self.classify_financial_news(analysis_text)

            # Predict market impact
            impact = self.predict_market_impact(analysis_text)

            # Combine results
            result = {
                "title": title,
                "content_summary": content[:200] + "..." if len(content) > 200 else content,
                "sentiment": sentiment.get("sentiment"),
                "sentiment_score": sentiment.get("sentiment_score"),
                "market_sentiment": classification.get("market_sentiment"),
                "impact_score": impact.get("impact_score"),
                "impact_summary": impact.get("impact_summary"),
                "key_entities": impact.get("key_entities", []),
                "source": item.get("source"),
                "url": item.get("url"),
                "timestamp": item.get("timestamp") or datetime.now().isoformat()
            }

            results.append(result)

        return results

# Test code for local debugging
if __name__ == "__main__":
    # Create processor
    processor = NLPProcessor()

    # Test text analysis
    print("\nTesting text analysis...")
    text = "Apple Inc. is planning to invest $10 billion in U.S. data centers. The investment will create 20,000 jobs."
    result = processor.analyze_text(text)
    print(f"Text analysis result: {result}")

    # Test sentiment analysis
    print("\nTesting sentiment analysis...")
    texts = [
        "The company reported excellent earnings, exceeding all analyst expectations.",
        "The stock market crashed today, with the Dow losing over 1000 points.",
        "Interest rates remained unchanged, as expected by most economists."
    ]

    for text in texts:
        result = processor.analyze_sentiment(text)
        print(f"Sentiment: {result}")

    # Test financial news classification
    print("\nTesting financial news classification...")
    news_texts = [
        "Bitcoin surged to new all-time highs, breaking the $70,000 barrier for the first time.",
        "Inflation concerns grow as consumer prices increase by 5% year-over-year.",
        "The Federal Reserve announced they will maintain the current interest rate policy."
    ]

    for text in news_texts:
        result = processor.classify_financial_news(text)
        print(f"Classification: {result}")

    # Test entity extraction
    print("\nTesting financial entity extraction...")
    text = "Microsoft and Apple stocks fell after the Fed raised interest rates by 0.5%. EUR/USD dropped to 1.05."
    result = processor.extract_financial_entities(text)
    print(f"Extracted entities: {result}")

    # Test market impact prediction
    print("\nTesting market impact prediction...")
    news_text = "BREAKING: Federal Reserve announces surprise 0.75% interest rate hike to combat soaring inflation. Markets expected to react strongly to this aggressive move."
    result = processor.predict_market_impact(news_text)
    print(f"Market impact prediction: {result}")

    # Test batch analysis
    print("\nTesting batch analysis...")
    news_batch = [
        {
            "title": "Tech stocks surge on AI optimism",
            "content": "Technology stocks surged today as investors showed renewed optimism about artificial intelligence developments. NVIDIA and AMD led the gains.",
            "source": "Financial Times",
            "url": "https://www.ft.com/example"
        },
        {
            "title": "Oil prices plummet on demand concerns",
            "content": "Crude oil prices fell sharply today as global demand concerns intensified. WTI crude dropped below $70 per barrel.",
            "source": "Bloomberg",
            "url": "https://www.bloomberg.com/example"
        }
    ]

    results = processor.batch_analyze_news(news_batch)
    for i, result in enumerate(results):
        print(f"\nBatch analysis result {i+1}:")
        print(f"  Title: {result['title']}")
        print(f"  Sentiment: {result['sentiment']} ({result['sentiment_score']})")
        print(f"  Market sentiment: {result['market_sentiment']}")
        print(f"  Impact: {result['impact_summary']} ({result['impact_score']})")
        print(f"  Key entities: {', '.join(result['key_entities'])}")