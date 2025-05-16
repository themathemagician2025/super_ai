import torch
from typing import Dict, List, Optional, Tuple
import numpy as np
from ..engine.starcoder_engine import StarCoderEngine
from ...utils.monitoring import MetricsManager
import asyncio
import json

class ArchitectureManager:
    def __init__(self, starcoder: StarCoderEngine):
        self.starcoder = starcoder
        self.metrics = MetricsManager()
        self.evolution_history = []
        
    async def evolve_layer(self, layer_config: Dict, performance: Dict) -> Dict:
        current = json.dumps(layer_config, indent=2)
        evolved = self.starcoder.evolve_architecture(current, performance)
        return json.loads(evolved)
        
    async def optimize_architecture(self, model_config: Dict, metrics: Dict) -> Dict:
        evolved_config = model_config.copy()
        tasks = []
        for layer_name, layer in model_config.items():
            if isinstance(layer, dict):
                task = self.evolve_layer(layer, metrics)
                tasks.append((layer_name, task))
        results = await asyncio.gather(*[task for _, task in tasks])
        for (layer_name, _), result in zip(tasks, results):
            evolved_config[layer_name] = result
        return evolved_config

    def validate_architecture(self, config: Dict) -> bool:
        try:
            self._build_temp_model(config)
            return True
        except Exception:
            return False

    def _build_temp_model(self, config: Dict) -> torch.nn.Module:
        layers = []
        for layer_config in config.values():
            if isinstance(layer_config, dict):
                layer_type = layer_config.get('type')
                if layer_type == 'Linear':
                    layers.append(torch.nn.Linear(
                        layer_config['in_features'],
                        layer_config['out_features']
                    ))
                elif layer_type == 'Conv2d':
                    layers.append(torch.nn.Conv2d(
                        layer_config['in_channels'],
                        layer_config['out_channels'],
                        layer_config['kernel_size']
                    ))
        return torch.nn.Sequential(*layers)

    async def evolve_with_constraints(
        self,
        model_config: Dict,
        metrics: Dict,
        constraints: Dict
    ) -> Dict:
        evolved = await self.optimize_architecture(model_config, metrics)
        while not self._meets_constraints(evolved, constraints):
            evolved = await self.optimize_architecture(evolved, metrics)
        return evolved

    def _meets_constraints(self, config: Dict, constraints: Dict) -> bool:
        total_params = sum(
            np.prod([p.numel() for p in self._build_temp_model(config).parameters()])
        )
        return total_params <= constraints.get('max_params', float('inf'))

    def track_evolution(self, original: Dict, evolved: Dict, metrics: Dict):
        self.evolution_history.append({
            'original': original,
            'evolved': evolved,
            'metrics': metrics,
            'improvement': self._calculate_improvement(metrics)
        })

    def _calculate_improvement(self, metrics: Dict) -> float:
        if 'accuracy' in metrics and 'latency' in metrics:
            return metrics['accuracy'] / metrics['latency']
        return 0.0

    def get_best_architecture(self) -> Optional[Dict]:
        if not self.evolution_history:
            return None
        return max(
            self.evolution_history,
            key=lambda x: x['improvement']
        )['evolved'] 