# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import argparse
from pathlib import Path
from transformers import AutoTokenizer

from .config import TrainingConfig
from .model import TransformerLLM
from .data_processor import DataProcessor
from .trainer import LLMTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train a transformer LLM from scratch')
    
    # Data arguments
    parser.add_argument('--train_data', type=str, required=True,
                      help='Path to training data file')
    parser.add_argument('--eval_data', type=str,
                      help='Path to evaluation data file')
    
    # Model arguments
    parser.add_argument('--vocab_size', type=int, default=50257,
                      help='Vocabulary size')
    parser.add_argument('--hidden_size', type=int, default=768,
                      help='Hidden size of transformer')
    parser.add_argument('--num_layers', type=int, default=12,
                      help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=12,
                      help='Number of attention heads')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                      help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=10,
                      help='Number of training epochs')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                      help='Number of warmup steps')
    parser.add_argument('--output_dir', type=str, default='outputs/llm_training',
                      help='Output directory for checkpoints and logs')
    parser.add_argument('--resume_from', type=str,
                      help='Resume training from checkpoint')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--no_wandb', action='store_true',
                      help='Disable Weights & Biases logging')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create config
    config = TrainingConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        warmup_steps=args.warmup_steps,
        output_dir=str(output_dir)
    )
    
    # Initialize tokenizer
    # Using GPT-2 tokenizer as default, but you can use any other pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    # Create model and data processor
    model = TransformerLLM(config)
    data_processor = DataProcessor(config, tokenizer)
    
    # Prepare data
    train_dataloader = data_processor.prepare_data(args.train_data)
    eval_dataloader = None
    if args.eval_data:
        eval_dataloader = data_processor.prepare_data(args.eval_data)
    
    # Create trainer
    trainer = LLMTrainer(
        model=model,
        data_processor=data_processor,
        config=config,
        use_wandb=not args.no_wandb
    )
    
    # Train model
    trainer.train(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        resume_from=args.resume_from
    )

if __name__ == '__main__':
    main() 