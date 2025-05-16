# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import logging
from pathlib import Path
from tqdm import tqdm
import wandb
from typing import Optional, Dict, Any

from .model import TransformerLLM
from .data_processor import DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMTrainer:
    """Manages the training process for the LLM."""
    
    def __init__(
        self,
        model: TransformerLLM,
        data_processor: DataProcessor,
        config: Any,
        use_wandb: bool = True
    ):
        self.model = model
        self.data_processor = data_processor
        self.config = config
        self.use_wandb = use_wandb
        
        # Move model to device
        self.model.to(self.config.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(project="llm-training", config=vars(config))
    
    def _setup_optimizer(self) -> AdamW:
        """Setup optimizer with weight decay."""
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        return AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
    
    def _setup_scheduler(self) -> LambdaLR:
        """Setup learning rate scheduler with warmup."""
        def lr_lambda(current_step: int):
            if current_step < self.config.warmup_steps:
                return float(current_step) / float(max(1, self.config.warmup_steps))
            return 1.0
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def save_checkpoint(self, epoch: int, step: int, loss: float):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint-{epoch}-{step}.pt"
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch'], checkpoint['step']
    
    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: Optional[torch.utils.data.DataLoader] = None,
        resume_from: Optional[str] = None
    ):
        """Train the model."""
        # Resume from checkpoint if specified
        start_epoch = 0
        global_step = 0
        if resume_from:
            start_epoch, global_step = self.load_checkpoint(resume_from)
        
        # Training loop
        self.model.train()
        for epoch in range(start_epoch, self.config.max_epochs):
            epoch_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
            
            for step, batch in enumerate(progress_bar):
                # Process batch
                batch = self.data_processor.process_batch(batch)
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs[1]  # model returns (logits, loss)
                
                # Backward pass
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                # Update weights
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # Log progress
                    epoch_loss += loss.item()
                    progress_bar.set_postfix({'loss': loss.item()})
                    
                    if self.use_wandb:
                        wandb.log({
                            'loss': loss.item(),
                            'learning_rate': self.scheduler.get_last_lr()[0],
                            'epoch': epoch,
                            'step': global_step
                        })
                    
                    # Save checkpoint
                    if global_step % self.config.save_steps == 0:
                        self.save_checkpoint(epoch, global_step, loss.item())
                    
                    # Evaluate if requested
                    if eval_dataloader and global_step % self.config.eval_steps == 0:
                        eval_loss = self.evaluate(eval_dataloader)
                        self.model.train()
                        
                        if self.use_wandb:
                            wandb.log({'eval_loss': eval_loss})
            
            # Log epoch metrics
            epoch_loss = epoch_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch} average loss: {epoch_loss:.4f}")
    
    @torch.no_grad()
    def evaluate(self, eval_dataloader: torch.utils.data.DataLoader) -> float:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = self.data_processor.process_batch(batch)
            outputs = self.model(**batch)
            loss = outputs[1]
            total_loss += loss.item()
        
        avg_loss = total_loss / len(eval_dataloader)
        logger.info(f"Evaluation loss: {avg_loss:.4f}")
        
        return avg_loss 