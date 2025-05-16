# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional
import json
from pathlib import Path
from transformers import PreTrainedTokenizer

class DataProcessor:
    """Handles data processing for LLM training."""
    
    def __init__(self, config, tokenizer: PreTrainedTokenizer):
        self.config = config
        self.tokenizer = tokenizer
    
    def prepare_data(self, data_path: str) -> DataLoader:
        """Prepare data for training or evaluation."""
        dataset = TextDataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            max_length=self.config.max_seq_length
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
    
    def process_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process a batch of data."""
        # Move batch to device
        batch = {k: v.to(self.config.device) for k, v in batch.items()}
        
        # Create attention masks
        if 'attention_mask' not in batch:
            batch['attention_mask'] = torch.ones_like(batch['input_ids'])
            
        return batch

class TextDataset(Dataset):
    """Dataset for text data."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        cache_dir: Optional[str] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self.load_examples(data_path)
        
        # Create cache directory if specified
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    def load_examples(self, data_path: str) -> List[str]:
        """Load examples from file."""
        examples = []
        
        # Handle different file formats
        path = Path(data_path)
        if path.suffix == '.txt':
            with open(path, 'r', encoding='utf-8') as f:
                examples = f.readlines()
        elif path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    examples = data
                elif isinstance(data, dict):
                    examples = list(data.values())
        
        return [ex.strip() for ex in examples if ex.strip()]
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example."""
        text = self.examples[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Remove batch dimension added by tokenizer
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # Create labels for language modeling
        item['labels'] = item['input_ids'].clone()
        
        return item 