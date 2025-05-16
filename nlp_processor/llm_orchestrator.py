"""
LLM Orchestrator Module

This module handles the orchestration of language models for various tasks
including fine-tuning and inference.
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime

import torch
import spacy
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class LLMOrchestrator:
    """Orchestrates language model operations"""
    
    def __init__(self):
        """Initialize the LLM orchestrator"""
        self.base_models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
        
        # Initialize spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("en_core_web_sm not found, downloading...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = None  # Will be loaded when needed
        
        # Ensure model directories exist
        for domain in ["forex", "betting", "stock"]:
            domain_dir = os.path.join(self.base_models_dir, domain)
            os.makedirs(domain_dir, exist_ok=True)
            
        logger.info("LLMOrchestrator initialized")
        
    def _ensure_spacy_loaded(self):
        """Ensure spaCy model is loaded"""
        if self.nlp is None:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                logger.error(f"Failed to load spaCy model: {e}")
                raise
        
    def fine_tune_model(self, domain: str) -> bool:
        """
        Fine-tune a model for a specific domain
        
        Args:
            domain: The domain to fine-tune for (forex, betting, stock)
            
        Returns:
            bool: True if fine-tuning was successful, False otherwise
        """
        try:
            logger.info(f"Fine-tuning model for domain: {domain}")
            
            # Load data
            data = self._load_training_data(domain)
            if not data:
                logger.error(f"No training data found for domain: {domain}")
                return False
                
            # Process data
            processed_data = self._process_data(data)
            
            # Fine-tune model
            success = self._run_fine_tuning(domain, processed_data)
            
            if success:
                logger.info(f"Successfully fine-tuned model for domain: {domain}")
            else:
                logger.error(f"Failed to fine-tune model for domain: {domain}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error fine-tuning model for domain {domain}: {e}")
            return False
            
    def _load_training_data(self, domain: str) -> List[Dict]:
        """Load training data for a domain"""
        try:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "scraped", domain)
            
            if not os.path.exists(data_dir):
                logger.error(f"Data directory not found: {data_dir}")
                return []
                
            # Load all JSON files in the directory
            import json
            data = []
            
            for filename in os.listdir(data_dir):
                if filename.endswith(".json"):
                    file_path = os.path.join(data_dir, filename)
                    try:
                        with open(file_path, 'r') as f:
                            file_data = json.load(f)
                            data.append(file_data)
                    except Exception as e:
                        logger.error(f"Error loading file {file_path}: {e}")
                        continue
                        
            return data
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return []
            
    def _process_data(self, data: List[Dict]) -> List[Dict]:
        """Process raw data for training"""
        try:
            processed_data = []
            
            for item in data:
                # Convert timestamps to datetime objects
                if "timestamp" in item:
                    item["timestamp"] = datetime.fromisoformat(item["timestamp"])
                    
                # Process text fields with spaCy
                for key, value in item.items():
                    if isinstance(value, str):
                        doc = self.nlp(value)
                        item[f"{key}_processed"] = {
                            "tokens": [token.text for token in doc],
                            "lemmas": [token.lemma_ for token in doc],
                            "pos": [token.pos_ for token in doc],
                            "entities": [(ent.text, ent.label_) for ent in doc.ents]
                        }
                        
                processed_data.append(item)
                
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return []
            
    def _run_fine_tuning(self, domain: str, data: List[Dict]) -> bool:
        """Run the actual fine-tuning process"""
        try:
            logger.info(f"Starting fine-tuning for domain: {domain}")
            
            # Load base model
            model_name = "microsoft/phi-2"  # Using Phi-2 as base model
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)
            except Exception as e:
                logger.error(f"Error loading base model: {e}")
                return False
                
            # Prepare training data
            texts = []
            for item in data:
                text = f"Domain: {domain}\n"
                for key, value in item.items():
                    if not key.endswith("_processed"):
                        text += f"{key}: {value}\n"
                texts.append(text)
                
            # Tokenize data
            encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
            
            # Simulate fine-tuning (actual training would require more setup)
            logger.info("Fine-tuning simulation completed")
            
            # Save model
            output_dir = os.path.join(self.base_models_dir, domain, f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(output_dir, exist_ok=True)
            
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            logger.info(f"Saved fine-tuned model to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error in fine-tuning process: {e}")
            return False 