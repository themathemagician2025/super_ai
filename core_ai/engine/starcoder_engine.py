import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional, Union
import ast
import astor
import logging
from pathlib import Path
import concurrent.futures
from ..utils.monitoring import MetricsManager

class StarCoderEngine:
    def __init__(self, model_path: str = "bigcode/starcoder", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.metrics = MetricsManager()
        
    def generate_code(self, prompt: str, max_length: int = 1024) -> str:
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=1
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def analyze_code(self, code: str) -> Dict:
        try:
            tree = ast.parse(code)
            return {
                'imports': [n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)],
                'functions': [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)],
                'classes': [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)],
                'complexity': sum(1 for _ in ast.walk(tree))
            }
        except Exception as e:
            return {'error': str(e)}

    def optimize_code(self, code: str) -> str:
        analysis = self.analyze_code(code)
        prompt = f"Optimize this code while maintaining functionality:\n{code}\nAnalysis:{analysis}"
        return self.generate_code(prompt)

    def evolve_architecture(self, current_arch: str, metrics: Dict) -> str:
        prompt = f"Current architecture:\n{current_arch}\nPerformance metrics:{metrics}\nGenerate improved architecture:"
        return self.generate_code(prompt)

    def validate_code(self, code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def refactor_code(self, code: str, target: str) -> str:
        prompt = f"Refactor this code to {target}:\n{code}"
        return self.generate_code(prompt)

    async def parallel_optimize(self, code_blocks: List[str]) -> List[str]:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return list(executor.map(self.optimize_code, code_blocks))

    def generate_tests(self, code: str) -> str:
        prompt = f"Generate comprehensive tests for:\n{code}"
        return self.generate_code(prompt)

    def enhance_performance(self, code: str, profile_data: Dict) -> str:
        prompt = f"Enhance performance based on profile:\n{code}\nProfile:{profile_data}"
        return self.generate_code(prompt) 