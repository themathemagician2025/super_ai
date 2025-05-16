# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    """Configuration for LLM training."""
    
    # Model architecture
    vocab_size: int = 50257
    max_position_embeddings: int = 1024
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 5e-5
    max_epochs: int = 10
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Data processing
    max_seq_length: int = 512
    mask_probability: float = 0.15
    
    # Optimization
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    
    # Hardware
    device: str = "cuda"  # or "cpu"
    num_workers: int = 4
    
    # Logging & Checkpointing
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 500
    
    # Paths
    train_data_path: Optional[str] = None
    eval_data_path: Optional[str] = None
    output_dir: str = "outputs/llm_training"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.num_attention_heads > 0, "Number of attention heads must be positive"
        assert self.hidden_size % self.num_attention_heads == 0, "Hidden size must be divisible by number of attention heads"
        assert self.max_seq_length <= self.max_position_embeddings, "Max sequence length cannot exceed position embeddings" 