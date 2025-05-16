"""
Self-Modification Manager for Super AI System

This module handles autonomous code evolution using StarCoder2 and Ollama models.
It implements secure code generation, testing, and deployment with version control.
"""

import os
import sys
import json
import time
import logging
import importlib
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import httpx
import ast
import asyncio
import torch
import optuna
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import accuracy_score
import ollama

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class OllamaModelManager:
    """Manages interactions with Ollama models"""
    
    def __init__(self):
        self.models = {
            'starcoder': 'starcoder2:15b',
            'llama': 'llama2:13b',
            'phi': 'phi:latest'
        }
        self.base_url = "http://localhost:11434/api"
        
    async def generate_code(self, prompt: str, model: str = 'starcoder') -> str:
        """Generate code using specified Ollama model"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/generate",
                    json={
                        "model": self.models[model],
                        "prompt": prompt,
                        "temperature": 0.1,  # Low temperature for code generation
                        "max_tokens": 2000
                    }
                )
                return response.json()['response']
        except Exception as e:
            logger.error(f"Error generating code with {model}: {e}")
            return None

class CodeValidator:
    """Validates generated code for syntax and security"""
    
    @staticmethod
    def validate_syntax(code: str) -> bool:
        """Check if generated code has valid Python syntax"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
            
    @staticmethod
    def check_security(code: str) -> Tuple[bool, str]:
        """Check for potentially dangerous operations"""
        forbidden = ['os.system', 'subprocess.call', 'eval(', 'exec(']
        for item in forbidden:
            if item in code:
                return False, f"Forbidden operation found: {item}"
        return True, "Code passed security check"

class VersionControl:
    """Manages code versions and checkpoints"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.versions_dir = self.base_dir / 'versions'
        self.versions_dir.mkdir(exist_ok=True)
        
    def save_version(self, code: str, metadata: Dict) -> str:
        """Save a new code version with metadata"""
        version_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        version_path = self.versions_dir / f"v_{version_id}"
        version_path.mkdir(exist_ok=True)
        
        # Save code
        with open(version_path / 'code.py', 'w') as f:
            f.write(code)
            
        # Save metadata
        with open(version_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return version_id
        
    def rollback(self, version_id: str) -> Optional[str]:
        """Rollback to a previous version"""
        version_path = self.versions_dir / f"v_{version_id}"
        if not version_path.exists():
            return None
            
        with open(version_path / 'code.py', 'r') as f:
            return f.read()

class SelfModificationManager:
    """Main engine for self-modifying code capabilities"""
    
    def __init__(self, config_path: str = 'config/self_improvement.json'):
        self.config = self._load_config(config_path)
        self.modifications = []
        self.performance_metrics = {}
        self.modification_history = []
        self.experience_bank = {}
        self.current_accuracy = 0.0
        self.accuracy_target = 0.95
        self.llm = ollama.Client()
        self.setup_gpu_support()
        self.setup_adversarial_testing()
        
    def setup_gpu_support(self):
        """Configure multi-GPU support for parallel processing"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            logger.info(f'Running on {self.num_gpus} GPUs')
        else:
            self.num_gpus = 0
            logger.info('Running on CPU')

    def setup_adversarial_testing(self):
        """Initialize adversarial testing components"""
        self.adversarial_config = {
            'test_frequency': 3600,  # Run tests hourly
            'min_confidence': 0.8,
            'max_retries': 3
        }

    async def apply_modification(self, modification: Dict[str, Any]) -> bool:
        """Apply a code modification with validation and rollback"""
        repo = git.Repo('.')
        branch_name = f'self-improve-{datetime.now().strftime("%Y%m%d-%H%M%S")}'

        try:
            # Create branch for modification
            current = repo.create_head(branch_name)
            current.checkout()

            # Apply changes
            success = await self._safe_modify_code(modification)
            if not success:
                raise Exception('Modification failed validation')

            # Run tests and measure performance
            if await self.validate_changes():
                repo.index.add('*')
                repo.index.commit(f'Self-improvement: {modification["purpose"]}\n\nMetrics: {self.current_accuracy}')
                return True
            else:
                raise Exception('Validation failed')

        except Exception as e:
            logger.error(f'Failed to apply modification: {e}')
            repo.git.reset('--hard', 'HEAD')
            return False

        finally:
            repo.heads.main.checkout()

    async def analyze_module(self, module_path: str) -> Dict[str, Any]:
        """Analyze a module for potential improvements"""
        with open(module_path, 'r') as f:
            code = f.read()
            
        # Use StarCoder to analyze code quality
        analysis_prompt = f"Analyze this Python code for potential improvements:\n{code}"
        analysis = await self.ollama.generate_code(analysis_prompt)
        
        return {
            'module': module_path,
            'suggestions': analysis,
            'timestamp': datetime.now().isoformat()
        }
        
    async def generate_improved_code(self, original_code: str, improvements: str) -> str:
        """Generate improved version of code using StarCoder"""
        prompt = f'''
        Original code:
        {original_code}
        
        Required improvements:
        {improvements}
        
        Generate an improved version that:
        1. Maintains all existing functionality
        2. Implements the suggested improvements
        3. Follows best practices and patterns
        4. Includes comprehensive error handling
        5. Adds performance optimizations
        '''
        
        return await self.ollama.generate_code(prompt)
        
    async def test_modifications(self, code: str, test_suite: str) -> Tuple[bool, float]:
        """Test modified code in sandbox environment"""
        # Create temporary test environment
        test_dir = Path('test_sandbox')
        test_dir.mkdir(exist_ok=True)
        
        try:
            # Save code to test file
            test_file = test_dir / 'test_module.py'
            with open(test_file, 'w') as f:
                f.write(code)
                
            # Run tests
            result = subprocess.run(
                ['python', '-m', 'pytest', test_suite],
                cwd=test_dir,
                capture_output=True,
                text=True
            )
            
            # Calculate test coverage and success rate
            coverage = 0.95  # Placeholder - implement actual coverage calculation
            success_rate = 0.99 if result.returncode == 0 else 0.0
            
            return result.returncode == 0, min(coverage, success_rate)
            
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(test_dir)
            
    async def deploy_modification(self, module_path: str, new_code: str, metadata: Dict):
        """Deploy verified code modifications"""
        # Save backup
        version_id = self.version_control.save_version(
            new_code,
            {**metadata, 'timestamp': datetime.now().isoformat()}
        )
        
        # Deploy new code
        with open(module_path, 'w') as f:
            f.write(new_code)
            
        # Reload module if it's imported
        module_name = Path(module_path).stem
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
            
        logger.info(f"Deployed new version {version_id} for {module_path}")
        
    async def improvement_cycle(self):
        """Run continuous improvement cycle"""
        while True:
            try:
                # Scan for modules to improve
                for module in Path('super_ai').rglob('*.py'):
                    # Analyze current module
                    analysis = await self.analyze_module(str(module))
                    
                    if analysis['suggestions']:
                        # Generate improved code
                        new_code = await self.generate_improved_code(
                            original_code=open(module, 'r').read(),
                            improvements=analysis['suggestions']
                        )
                        
                        # Validate syntax and security
                        if not self.validator.validate_syntax(new_code):
                            logger.warning(f"Invalid syntax in generated code for {module}")
                            continue
                            
                        is_safe, msg = self.validator.check_security(new_code)
                        if not is_safe:
                            logger.warning(f"Security check failed for {module}: {msg}")
                            continue
                            
                        # Test modifications
                        success, accuracy = await self.test_modifications(
                            new_code,
                            test_suite=f"tests/test_{module.stem}.py"
                        )
                        
                        if success and accuracy >= self.performance_threshold:
                            # Deploy improvements
                            await self.deploy_modification(
                                str(module),
                                new_code,
                                metadata={
                                    'accuracy': accuracy,
                                    'changes': analysis['suggestions']
                                }
                            )
                            
                # Sleep between cycles
                await asyncio.sleep(3600)  # 1 hour between cycles
                
            except Exception as e:
                logger.error(f"Error in improvement cycle: {e}")
                await asyncio.sleep(300)  # 5 min backoff on error

if __name__ == "__main__":
    import asyncio
    
    engine = SelfModificationEngine()
    asyncio.run(engine.improvement_cycle()) 