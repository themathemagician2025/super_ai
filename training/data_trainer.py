import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pytesseract
from pdf2image import convert_from_path
import docx
import cv2
from typing import Dict, List, Any, Union
import sqlite3
from datetime import datetime, date
import warnings
from tqdm import tqdm
import concurrent.futures
from functools import partial

from ..utils.cache_manager import CacheManager
from ..utils.monitoring import MetricsManager, PerformanceMonitor

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects"""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class MultiModalDataProcessor:
    """Process different types of data files for training"""
    
    def __init__(self, data_dir: str = "data", max_workers: int = 4):
        self.data_dir = Path(data_dir)
        self.processed_data = []
        self.max_workers = max_workers
        
        # Initialize cache manager
        self.cache_manager = CacheManager(self.data_dir / "processed")
        
        # Initialize metrics
        self.metrics = MetricsManager()
        self.performance_monitor = PerformanceMonitor(self.metrics)
        
    @MetricsManager.track_processing("pdf")
    def process_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract text and images from PDF files"""
        logger.info(f"Processing PDF: {file_path}")
        try:
            # Convert PDF to images
            images = convert_from_path(str(file_path))
            
            results = []
            for i, image in enumerate(images):
                # Extract text using OCR
                text = pytesseract.image_to_string(image)
                
                # Save image temporarily for feature extraction
                temp_img_path = f"temp_page_{i}.jpg"
                image.save(temp_img_path)
                
                # Extract image features
                img_features = self.extract_image_features(temp_img_path)
                
                results.append({
                    'page': i,
                    'text': text,
                    'image_features': img_features,
                    'source': str(file_path)
                })
                
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
                
            return results
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            self.cache_manager.mark_file_processed(file_path, status='ERROR', error_msg=str(e))
            return []

    @MetricsManager.track_processing("docx")
    def process_docx(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract text from Word documents"""
        logger.info(f"Processing DOCX: {file_path}")
        try:
            doc = docx.Document(file_path)
            results = [{
                'text': paragraph.text,
                'source': str(file_path)
            } for paragraph in doc.paragraphs if paragraph.text.strip()]
            self.cache_manager.mark_file_processed(file_path, status='SUCCESS')
            return results
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path}: {str(e)}")
            self.cache_manager.mark_file_processed(file_path, status='ERROR', error_msg=str(e))
            return []

    @MetricsManager.track_processing("csv")
    def process_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process CSV files"""
        logger.info(f"Processing CSV: {file_path}")
        try:
            df = pd.read_csv(file_path)
            results = [{
                'data': row.to_dict(),
                'source': str(file_path)
            } for _, row in df.iterrows()]
            self.cache_manager.mark_file_processed(file_path, status='SUCCESS')
            return results
        except Exception as e:
            logger.error(f"Error processing CSV {file_path}: {str(e)}")
            self.cache_manager.mark_file_processed(file_path, status='ERROR', error_msg=str(e))
            return []

    @MetricsManager.track_processing("json")
    def process_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process JSON files"""
        logger.info(f"Processing JSON: {file_path}")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            results = [{
                'data': data,
                'source': str(file_path)
            }]
            self.cache_manager.mark_file_processed(file_path, status='SUCCESS')
            return results
        except Exception as e:
            logger.error(f"Error processing JSON {file_path}: {str(e)}")
            self.cache_manager.mark_file_processed(file_path, status='ERROR', error_msg=str(e))
            return []

    @MetricsManager.track_processing("image")
    def process_image(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process image files"""
        logger.info(f"Processing Image: {file_path}")
        try:
            # Extract text using OCR
            text = pytesseract.image_to_string(Image.open(file_path))
            
            # Extract image features
            img_features = self.extract_image_features(str(file_path))
            
            results = [{
                'text': text,
                'image_features': img_features,
                'source': str(file_path)
            }]
            self.cache_manager.mark_file_processed(file_path, status='SUCCESS')
            return results
        except Exception as e:
            logger.error(f"Error processing Image {file_path}: {str(e)}")
            self.cache_manager.mark_file_processed(file_path, status='ERROR', error_msg=str(e))
            return []

    @MetricsManager.track_processing("sav")
    def process_sav(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process SPSS .sav files"""
        logger.info(f"Processing SAV: {file_path}")
        try:
            import pyreadstat
            df, meta = pyreadstat.read_sav(file_path)
            results = [{
                'data': row.to_dict(),
                'metadata': {
                    'variable_labels': meta.variable_value_labels if hasattr(meta, 'variable_value_labels') else {},
                    'column_labels': meta.column_labels if hasattr(meta, 'column_labels') else {},
                    'column_names': meta.column_names if hasattr(meta, 'column_names') else []
                },
                'source': str(file_path)
            } for _, row in df.iterrows()]
            self.cache_manager.mark_file_processed(file_path, status='SUCCESS')
            return results
        except Exception as e:
            logger.error(f"Error processing SAV {file_path}: {str(e)}")
            self.cache_manager.mark_file_processed(file_path, status='ERROR', error_msg=str(e))
            return []

    def extract_image_features(self, image_path: str) -> np.ndarray:
        """Extract features from images using OpenCV"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return np.array([])
            
            # Resize for consistency
            img = cv2.resize(img, (224, 224))
            
            # Extract basic features
            features = []
            
            # Color histogram
            for i in range(3):
                hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                features.extend(hist.flatten())
            
            # Edge detection
            edges = cv2.Canny(img, 100, 200)
            edge_features = edges.flatten()
            
            return np.concatenate([np.array(features), edge_features])
        except Exception as e:
            logger.error(f"Error extracting image features: {str(e)}")
            return np.array([])

    def process_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a single file based on its type"""
        if not self.cache_manager.should_process_file(file_path):
            self.metrics.record_cache_hit(file_path.suffix)
            return []
            
        ext = file_path.suffix.lower()
        processors = {
            '.pdf': self.process_pdf,
            '.docx': self.process_docx,
            '.csv': self.process_csv,
            '.json': self.process_json,
            '.jpg': self.process_image,
            '.jpeg': self.process_image,
            '.png': self.process_image,
            '.sav': self.process_sav
        }
        
        processor = processors.get(ext)
        if processor:
            return processor(file_path)
        return []

    def process_all_files(self) -> None:
        """Process all supported files in the data directory using parallel processing"""
        logger.info("Starting to process all files...")
        self.performance_monitor.start()
        
        # Get list of files to process
        files_to_process = []
        for file_path in self.data_dir.rglob("*"):
            if not file_path.is_file():
                continue
                
            # Skip cache and temporary files
            if any(skip_dir in str(file_path) for skip_dir in ['__pycache__', '.git', '.pytest_cache']):
                continue
                
            if self.cache_manager.should_process_file(file_path):
                files_to_process.append(file_path)
        
        self.performance_monitor.checkpoint("file_discovery")
        logger.info(f"Found {len(files_to_process)} files to process")
        
        # Process files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_file, file_path): file_path 
                for file_path in files_to_process
            }
            
            for future in tqdm(
                concurrent.futures.as_completed(future_to_file),
                total=len(files_to_process),
                desc="Processing files"
            ):
                file_path = future_to_file[future]
                try:
                    data = future.result()
                    if data:
                        self.processed_data.extend(data)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
        
        self.performance_monitor.checkpoint("file_processing")
        
        # Push final metrics
        try:
            cache_stats = self.cache_manager.get_cache_stats()
            self.metrics.update_cache_size(cache_stats['total_files'])
            self.metrics.push_metrics()
        except Exception as e:
            logger.error(f"Error pushing metrics: {str(e)}")
        
        self.performance_monitor.checkpoint("metrics_update")
        
        # Log performance results
        performance_results = self.performance_monitor.end()
        logger.info(f"Processing completed in {performance_results['total_duration']:.2f} seconds")
        logger.info(f"Performance checkpoints: {performance_results['checkpoints']}")

class MultiModalDataset(Dataset):
    """Dataset class for training"""
    
    def __init__(self, data: List[Dict[str, Any]], tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Process text
        text = item.get('text', '')
        if not text and 'data' in item:
            text = json.dumps(item['data'], cls=DateTimeEncoder)
        
        tokens = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Process image features if available
        image_features = torch.tensor(item.get('image_features', [0.0]))
        
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'image_features': image_features
        }

class MultiModalTrainer:
    """Trainer class for the multi-modal AI system"""
    
    def __init__(self, data_processor: MultiModalDataProcessor):
        self.data_processor = data_processor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=1  # Regression task
        ).to(self.device)
        
    def train(self, batch_size: int = 16, epochs: int = 5):
        """Train the model on processed data"""
        logger.info("Starting training process")
        
        # Process all files
        self.data_processor.process_all_files()
        
        if not self.data_processor.processed_data:
            raise ValueError("No data was processed. Please check the data directory and file permissions.")
        
        logger.info(f"Processed {len(self.data_processor.processed_data)} items")
        
        # Create dataset and dataloader
        dataset = MultiModalDataset(self.data_processor.processed_data, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                image_features = batch['image_features'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                optimizer.step()
            
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            self._save_checkpoint(epoch, avg_loss)
    
    def _save_checkpoint(self, epoch: int, loss: float):
        """Save model checkpoint"""
        checkpoint_dir = Path("data/checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"model_epoch_{epoch}_loss_{loss:.4f}.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")

def main():
    """Main function to run the training process"""
    try:
        # Initialize processor and trainer
        data_processor = MultiModalDataProcessor()
        trainer = MultiModalTrainer(data_processor)
        
        # Start training
        trainer.train()
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 