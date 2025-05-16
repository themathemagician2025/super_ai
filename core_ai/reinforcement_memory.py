"""
Reinforcement Memory Module for Prediction Engine

Implements:
- Reinforcement learning feedback loop
- Experience replay memory
- Memory-based learning
- Failed case storage and retraining
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
from collections import deque
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class ExperienceMemory:
    """Stores and manages prediction experiences"""
    
    def __init__(self, max_size: int = 10000):
        self.memory = deque(maxlen=max_size)
        self.failed_cases = deque(maxlen=max_size // 2)
        self.success_threshold = 0.9
        
    def add_experience(self, 
                      features: np.ndarray,
                      prediction: float,
                      actual: float,
                      metadata: Dict[str, Any]):
        """Add new experience to memory"""
        error = abs(prediction - actual)
        success = error < self.success_threshold
        
        experience = {
            'features': features,
            'prediction': prediction,
            'actual': actual,
            'error': error,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata
        }
        
        self.memory.append(experience)
        
        if not success:
            self.failed_cases.append(experience)
            
    def get_failed_cases(self) -> List[Dict[str, Any]]:
        """Get failed prediction cases for retraining"""
        return list(self.failed_cases)
        
    def get_training_batch(self, 
                          batch_size: int,
                          failed_case_ratio: float = 0.3) -> List[Dict[str, Any]]:
        """Get mixed batch of successful and failed cases"""
        n_failed = int(batch_size * failed_case_ratio)
        n_success = batch_size - n_failed
        
        failed_sample = self._sample_experiences(self.failed_cases, n_failed)
        success_sample = self._sample_experiences(
            [x for x in self.memory if x['success']], 
            n_success
        )
        
        return failed_sample + success_sample
        
    def _sample_experiences(self, 
                          experiences: List[Dict[str, Any]], 
                          n: int) -> List[Dict[str, Any]]:
        """Sample n experiences from list"""
        if not experiences or n <= 0:
            return []
        indices = np.random.choice(len(experiences), min(n, len(experiences)))
        return [experiences[i] for i in indices]
        
    def get_performance_stats(self) -> Dict[str, float]:
        """Calculate performance statistics"""
        if not self.memory:
            return {}
            
        errors = [exp['error'] for exp in self.memory]
        success_rate = sum(exp['success'] for exp in self.memory) / len(self.memory)
        
        return {
            'mean_error': np.mean(errors),
            'max_error': np.max(errors),
            'success_rate': success_rate,
            'failed_cases_count': len(self.failed_cases)
        }

class ReinforcementTrainer:
    """Implements reinforcement learning for prediction models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory = ExperienceMemory(
            max_size=config.get('memory_size', 10000)
        )
        self.reward_scale = config.get('reward_scale', 1.0)
        self.punishment_scale = config.get('punishment_scale', 1.5)
        
    def calculate_reward(self, 
                        prediction: float,
                        actual: float,
                        metadata: Dict[str, Any]) -> float:
        """Calculate reward/punishment based on prediction accuracy"""
        error = abs(prediction - actual)
        
        # Base reward calculation
        if error < 0.01:  # Highly accurate
            base_reward = 1.0
        elif error < 0.05:  # Good
            base_reward = 0.5
        elif error < 0.1:  # Acceptable
            base_reward = 0.1
        else:  # Poor
            base_reward = -0.5
            
        # Apply scaling
        if base_reward > 0:
            reward = base_reward * self.reward_scale
        else:
            reward = base_reward * self.punishment_scale
            
        # Adjust based on prediction context
        if metadata.get('high_confidence', False):
            reward *= 1.2  # Bonus for high confidence predictions
        if metadata.get('high_risk', False):
            reward *= 1.5  # Higher stakes for risky predictions
            
        return reward
        
    def update_model(self, 
                    model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    batch: List[Dict[str, Any]]) -> float:
        """Update model using reinforcement learning"""
        if not batch:
            return 0.0
            
        total_loss = 0.0
        model.train()
        
        for experience in batch:
            features = torch.FloatTensor(experience['features'])
            prediction = model(features)
            reward = self.calculate_reward(
                prediction.item(),
                experience['actual'],
                experience['metadata']
            )
            
            # Calculate loss with reward weighting
            loss = -reward * torch.log(prediction)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(batch)
        
    def train_on_failed_cases(self,
                             model: nn.Module,
                             optimizer: torch.optim.Optimizer,
                             n_epochs: int = 5) -> Dict[str, float]:
        """Retrain specifically on failed cases"""
        failed_cases = self.memory.get_failed_cases()
        if not failed_cases:
            return {'loss': 0.0, 'cases_improved': 0}
            
        improved_cases = 0
        total_loss = 0.0
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for case in failed_cases:
                features = torch.FloatTensor(case['features'])
                
                # Forward pass
                model.train()
                prediction = model(features)
                
                # Calculate new error
                new_error = abs(prediction.item() - case['actual'])
                
                if new_error < case['error']:
                    improved_cases += 1
                    
                # Calculate loss
                reward = self.calculate_reward(
                    prediction.item(),
                    case['actual'],
                    case['metadata']
                )
                loss = -reward * torch.log(prediction)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            total_loss += epoch_loss / len(failed_cases)
            
        return {
            'loss': total_loss / n_epochs,
            'cases_improved': improved_cases
        }

class AdaptiveReinforcementEngine:
    """Manages reinforcement learning and memory adaptation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.trainer = ReinforcementTrainer(config)
        self.memory = self.trainer.memory
        self.adaptation_threshold = config.get('adaptation_threshold', 0.8)
        
    def adapt_model(self,
                   model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   features: np.ndarray,
                   prediction: float,
                   actual: float,
                   metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt model based on prediction results"""
        # Store experience
        self.memory.add_experience(features, prediction, actual, metadata)
        
        # Get performance stats
        stats = self.memory.get_performance_stats()
        
        # Check if adaptation needed
        if stats['success_rate'] < self.adaptation_threshold:
            # Get training batch
            batch = self.memory.get_training_batch(
                batch_size=self.config.get('batch_size', 32)
            )
            
            # Update model
            loss = self.trainer.update_model(model, optimizer, batch)
            
            # Retrain on failed cases if needed
            if len(self.memory.failed_cases) > 10:
                failed_stats = self.trainer.train_on_failed_cases(
                    model,
                    optimizer,
                    n_epochs=self.config.get('failed_case_epochs', 5)
                )
                stats.update({
                    'failed_case_loss': failed_stats['loss'],
                    'improved_cases': failed_stats['cases_improved']
                })
                
            stats.update({'adaptation_loss': loss})
            
        return stats

if __name__ == "__main__":
    # Example usage
    config = {
        'memory_size': 10000,
        'reward_scale': 1.0,
        'punishment_scale': 1.5,
        'adaptation_threshold': 0.8,
        'batch_size': 32,
        'failed_case_epochs': 5
    }
    
    engine = AdaptiveReinforcementEngine(config)
    
    # Create dummy model and optimizer
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1),
        nn.Sigmoid()
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Example prediction
    features = np.random.randn(10)
    prediction = 0.7
    actual = 0.8
    metadata = {'high_confidence': True}
    
    # Adapt model
    stats = engine.adapt_model(
        model,
        optimizer,
        features,
        prediction,
        actual,
        metadata
    )
    print("Adaptation stats:", stats) 