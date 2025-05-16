# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

# --- imports ---
import os
import time
import importlib
import logging
from datetime import datetime
from pathlib import Path
import ast
import inspect
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import json
import sqlite3
import subprocess
import sys
try:
    import docker
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "docker"])
    import docker

try:
    import ollama
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ollama"])
    import ollama

import pytest
from contextlib import contextmanager

# Configure logging
log_dir = 'monitoring_logging/logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'log_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- LLM Orchestrator ---
class LLMOrchestrator:
    def __init__(self):
        self.models = {
            "code": "starcoder2:15b",    # For code generation
            "complex_nlp": "llama3.1:8b", # For log analysis, detailed responses
            "fast_nlp": "phi:latest"      # For quick NLP tasks
        }
        self.cache = LLMCache()

    def generate_code(self, prompt: str, task_type: str = "code") -> str:
        cached_response = self.cache.get_cached_response(prompt)
        if cached_response:
            logger.info(f"Retrieved cached code response: {prompt[:50]}...")
            return cached_response

        model = self.models.get(task_type, "starcoder2:15b")
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={"temperature": 0.7, "max_tokens": 2000}
        )
        result = response["text"]
        self.cache.cache_response(prompt, result)
        logger.info(f"Generated code with {model}: {result[:50]}...")
        return result

    def analyze_logs(self, log_content: str, task_type: str = "complex_nlp") -> Dict[str, Any]:
        prompt = f"""
        Analyze the following logs to identify performance bottlenecks, errors, and warnings. Provide a JSON response with:
        - summary: Key issues and their impact.
        - recommendations: Actionable changes with code snippets.
        - priority: High, Medium, Low.

        Logs:
        {log_content}
        """
        model = self.models.get(task_type, "llama3.1:8b")
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={"temperature": 0.5, "max_tokens": 2000}
        )
        result = json.loads(response["text"])
        logger.info(f"Log analysis completed with {model}: {result['summary']}")
        return result

    def refine_response(self, initial_response: str, context: str, task_type: str = "fast_nlp") -> Dict[str, str]:
        prompt = f"""
        Refine the following response to improve accuracy, clarity, and relevance for the SUPER AI PROJECT. Return JSON with refined_response and changes_made.

        Initial Response:
        {initial_response}

        Context:
        {context}
        """
        model = self.models.get(task_type, "phi:latest")
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={"temperature": 0.5, "max_tokens": 2000}
        )
        result = json.loads(response["text"])
        logger.info(f"Refined response with {model}: {result['refined_response'][:50]}...")
        return result

class LLMCache:
    def __init__(self, db_path: str = "data/cache.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS response_cache (prompt_hash TEXT PRIMARY KEY, response TEXT)")

    def cache_response(self, prompt: str, response: str):
        prompt_hash = hash(prompt)
        self.conn.execute("INSERT OR REPLACE INTO response_cache (prompt_hash, response) VALUES (?, ?)",
                         (prompt_hash, response))
        self.conn.commit()

    def get_cached_response(self, prompt: str) -> Optional[str]:
        prompt_hash = hash(prompt)
        cursor = self.conn.execute("SELECT response FROM response_cache WHERE prompt_hash = ?", (prompt_hash,))
        result = cursor.fetchone()
        return result[0] if result else None

# --- AI Assessment ---
class AIAssessment:
    def __init__(self):
        self.metrics = {}
        self.llm = LLMOrchestrator()

    def assess_performance(self, predictions: List[float], actual: List[float]) -> bool:
        correct = sum(1 for p, a in zip(predictions, actual) if abs(p - a) < 0.1)
        accuracy = correct / len(predictions) if predictions else 0.0
        self.metrics['accuracy'] = accuracy

        # Incorporate profit/loss feedback
        profit_loss = sum(a - p for p, a in zip(predictions, actual))  # Simplified
        self.metrics['profit_loss'] = profit_loss

        logger.info(f"Assessed performance: accuracy={accuracy}, profit_loss={profit_loss}")
        return accuracy >= 0.75 and profit_loss >= 0.0

    def assess_response_time(self, start_time: datetime) -> bool:
        elapsed = (datetime.now() - start_time).total_seconds()
        self.metrics['response_time'] = elapsed
        logger.info(f"Assessed response time: {elapsed}s")
        return elapsed < 2.0

    def assess_code_quality(self, analyzer: 'CodeAnalyzer') -> bool:
        analysis = analyzer.analyze_code()
        self.metrics['code_quality'] = analysis
        logger.info(f"Assessed code quality: {analysis}")

        # Use Llama3.1 to analyze code quality
        code_snippet = ast.unparse(analyzer.tree)
        prompt = f"Analyze the following code for quality and suggest improvements:\n{code_snippet}"
        suggestions = self.llm.generate_code(prompt, task_type="complex_nlp")
        self.metrics['code_quality_suggestions'] = suggestions
        return analysis['lines'] < 1000 and 'errors' not in suggestions.lower()

# --- Simulated Ground Truth ---
def get_actual_results() -> List[float]:
    return [1.0, 0.0, 1.0, 1.0, 0.0]  # Simulated for evaluation

# --- Code Modifier ---
class CodeModifier:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.backup_dir = root_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        self.modification_history = []
        self.llm = LLMOrchestrator()
        self.docker_client = docker.from_env()
        logger.info("Code modifier initialized")

    def modify_algorithm(self, file_path: Path, performance_metrics: Dict[str, float]) -> bool:
        try:
            # Create backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"{file_path.stem}_{timestamp}.py"
            self._create_backup(file_path, backup_path)

            # Read and parse code
            with open(file_path, 'r') as f:
                original_code = f.read()
            tree = ast.parse(original_code)

            # Analyze code
            analyzer = CodeAnalyzer(tree)
            modifications = self._determine_modifications(analyzer, performance_metrics)

            # Generate new code with StarCoder2
            prompt = f"""
            Rewrite the following Python code to improve performance based on these metrics: {performance_metrics}.
            Optimize for accuracy, response time, and code quality. Ensure PEP 8 compliance and include comments.

            Original Code:
            {original_code}

            Modifications Needed:
            {json.dumps(modifications, indent=2)}
            """
            new_code = self.llm.generate_code(prompt, task_type="code")

            # Validate in sandbox
            if not self._validate_in_sandbox(file_path, new_code):
                logger.error("Sandbox validation failed")
                self._restore_backup(backup_path, file_path)
                return False

            # Write modified code
            with open(file_path, 'w') as f:
                f.write(new_code)

            # Generate explainable change log with Llama3.1
            change_log = self._generate_change_log(original_code, new_code, performance_metrics)
            self.modification_history.append({
                'timestamp': timestamp,
                'file': file_path,
                'metrics': performance_metrics,
                'modifications': modifications,
                'change_log': change_log
            })

            logger.info(f"Successfully modified {file_path}")
            return True

        except Exception as e:
            logger.error(f"Code modification failed: {str(e)}")
            self._restore_backup(backup_path, file_path)
            return False

    def _determine_modifications(self, analyzer: 'CodeAnalyzer', metrics: Dict[str, float]) -> Dict[str, Any]:
        modifications = {}
        if metrics.get('accuracy', 0) < 0.8:
            modifications['learning_rate'] = 'increase'
            modifications['model_architecture'] = 'add_lstm_layer'
        if metrics.get('profit_loss', 0) < 0.0:
            modifications['feature_engineering'] = 'add_volatility_features'
        if metrics.get('response_time', float('inf')) > 2.0:
            modifications['batch_size'] = 'decrease'
        for func_name, stats in analyzer.functions.items():
            if stats['complexity'] > 10:
                modifications[func_name] = 'refactor'
            if stats['lines'] > 50:
                modifications[func_name] = 'split'
        return modifications

    def _create_backup(self, source: Path, backup: Path):
        import shutil
        shutil.copy2(source, backup)
        logger.info(f"Created backup: {backup}")

    def _restore_backup(self, backup: Path, target: Path):
        import shutil
        if backup.exists():
            shutil.copy2(backup, target)
            logger.info(f"Restored backup: {backup}")

    def _validate_in_sandbox(self, file_path: Path, new_code: str) -> bool:
        try:
            # Create temporary directory
            temp_dir = Path("/tmp/sandbox")
            temp_dir.mkdir(exist_ok=True)
            temp_file = temp_dir / file_path.name
            with open(temp_file, 'w') as f:
                f.write(new_code)

            # Run unit tests in Docker container
            container = self.docker_client.containers.run(
                image="python:3.11-slim",
                command=f"bash -c 'pip install pytest && pytest {temp_file}'",
                volumes={str(temp_dir): {'bind': '/app', 'mode': 'rw'}},
                detach=True
            )
            result = container.wait()
            logs = container.logs().decode()
            container.remove()

            if result['StatusCode'] != 0:
                logger.error(f"Sandbox validation failed: {logs}")
                return False

            logger.info("Sandbox validation passed")
            return True

        except Exception as e:
            logger.error(f"Sandbox validation failed: {str(e)}")
            return False

    def _generate_change_log(self, original_code: str, new_code: str, metrics: Dict[str, float]) -> str:
        prompt = f"""
        Generate an explainable change log comparing the original and new code. Explain the reasons for changes based on performance metrics: {metrics}.
        Original Code:
        {original_code[:1000]}...
        New Code:
        {new_code[:1000]}...
        """
        return self.llm.generate_code(prompt, task_type="complex_nlp")

# --- Code Analyzer ---
class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self, tree):
        self.functions = {}
        self.classes = {}
        self.complexity = {}
        self.tree = tree
        self.visit(tree)

    def visit_FunctionDef(self, node):
        self.functions[node.name] = {
            'args': len(node.args.args),
            'lines': len(node.body),
            'complexity': self._calculate_complexity(node)
        }
        self.generic_visit(node)

    def _calculate_complexity(self, node) -> int:
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try)):
                complexity += 1
        return complexity

    def analyze_code(self) -> Dict[str, Any]:
        return {
            'lines': sum(node.end_lineno - node.lineno for node in ast.walk(self.tree) if hasattr(node, 'lineno')),
            'functions': len(self.functions),
            'complexity': sum(stats['complexity'] for stats in self.functions.values())
        }

# --- Code Transformer ---
class CodeTransformer(ast.NodeTransformer):
    def __init__(self, modifications: Dict[str, Any]):
        self.modifications = modifications

    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in self.modifications:
                if self.modifications[name] == 'increase':
                    if isinstance(node.value, ast.Num):
                        node.value.n *= 1.5
                elif self.modifications[name] == 'decrease':
                    if isinstance(node.value, ast.Num):
                        node.value.n *= 0.75
        return node

# --- Self-Modification Engine ---
class SelfModificationEngine:
    def __init__(self, config_path: Path = Path("config/ai_config.json")):
        self.config_path = config_path
        self.backup_dir = Path("backups")
        self.backup_dir.mkdir(exist_ok=True)
        self.load_config()
        self.code_modifier = CodeModifier(Path(__file__).parent)
        self.llm = LLMOrchestrator()
        self.weights = np.random.rand(self.config["input_size"])
        self.modification_history = []
        self.file_index = {}
        self.dependency_graph = {}
        self._build_file_index()
        logger.info("Self-modification engine initialized")

    def _build_file_index(self):
        root = Path(__file__).parent.parent
        for file in root.rglob("*.py"):
            with open(file, 'r') as f:
                content = f.read()
            tree = ast.parse(content)
            imports = self._extract_imports(tree)
            self.file_index[file] = {
                'last_modified': file.stat().st_mtime,
                'imports': imports,
                'loc': len(content.splitlines()),
                'modules': self._extract_modules(tree)
            }
        self._build_dependency_graph()
        logger.info(f"Indexed {len(self.file_index)} files")

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(n.name for n in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
        return imports

    def _extract_modules(self, tree: ast.AST) -> List[str]:
        return [node.name for node in ast.walk(tree)
                if isinstance(node, (ast.ClassDef, ast.FunctionDef))]

    def _build_dependency_graph(self):
        for file, info in self.file_index.items():
            self.dependency_graph[file] = []
            for imp in info['imports']:
                for other_file in self.file_index:
                    if imp in self.file_index[other_file]['modules']:
                        self.dependency_graph[file].append(other_file)

    def load_config(self):
        default_config = {
            "input_size": 50,
            "learning_rate": 0.01,
            "threshold": 0.5,
            "lstm_units": 128,
            "dense_units": 64,
            "dropout_rate": 0.2
        }
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                self.config = default_config
                self.save_config()
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            self.config = default_config

    def save_config(self):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        logger.info("Configuration saved successfully")

    def modify_architecture(self, performance_metrics: Dict[str, float]) -> bool:
        modified = False
        if performance_metrics.get('accuracy', 0) < 0.8:
            self.config["lstm_units"] = int(self.config["lstm_units"] * 1.2)
            self.config["learning_rate"] *= 1.1
            modified = True
        if performance_metrics.get('profit_loss', 0) < 0.0:
            self.config["dense_units"] = int(self.config["dense_units"] * 1.2)
            self.config["dropout_rate"] = min(0.5, self.config["dropout_rate"] + 0.05)
            modified = True
        if modified:
            self.save_config()
            logger.info("Architecture modified based on performance")
        return modified

    def rewrite_code(self, file_path: Path, performance_metrics: Dict[str, float]) -> bool:
        return self.code_modifier.modify_algorithm(file_path, performance_metrics)

    def evaluate_performance(self, predictions: List[float], targets: List[float]) -> Dict[str, float]:
        metrics = {}
        correct = sum(1 for p, t in zip(predictions, targets) if abs(p - t) < self.config['threshold'])
        metrics['accuracy'] = correct / len(predictions) if predictions else 0.0
        metrics['loss'] = np.mean([(p - t) ** 2 for p, t in zip(predictions, targets)])
        metrics['profit_loss'] = sum(t - p for p, t in zip(predictions, targets))  # Simplified
        logger.info(f"Performance metrics: {metrics}")
        return metrics

    def implement_meta_learning(self) -> bool:
        try:
            # Use StarCoder2 to generate a meta-learning algorithm
            prompt = """
            Generate a Python function implementing a meta-learning algorithm (e.g., MAML) to optimize the learning process for the SUPER AI PROJECT. The function should adapt hyperparameters based on performance metrics and support multi-domain predictions (Forex, sports, stocks).

            Requirements:
            - Use TensorFlow or PyTorch.
            - Include comments explaining the logic.
            - Optimize for accuracy and adaptability.
            """
            meta_learning_code = self.llm.generate_code(prompt, task_type="code")

            # Save to a new module
            meta_learning_path = Path("prediction_engines/meta_learning.py")
            with open(meta_learning_path, 'w') as f:
                f.write(meta_learning_code)

            # Validate in sandbox
            if not self.code_modifier._validate_in_sandbox(meta_learning_path, meta_learning_code):
                logger.error("Meta-learning code validation failed")
                return False

            # Dynamically load module
            spec = importlib.util.spec_from_file_location("meta_learning", meta_learning_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            logger.info("Meta-learning algorithm implemented")
            return True

        except Exception as e:
            logger.error(f"Meta-learning implementation failed: {str(e)}")
            return False

    def behavior_adaptation_loop(self):
        """Main loop for behavior adaptation"""
        while True:
            try:
                # Simulate predictions
                predictions = [0.9, 0.1, 0.8, 0.7, 0.2]  # Replace with actual model predictions
                targets = get_actual_results()
                metrics = self.evaluate_performance(predictions, targets)

                # Analyze logs
                log_file = max(Path(log_dir).glob("log_*.log"), key=lambda p: p.stat().st_mtime)
                with open(log_file, 'r') as f:
                    log_content = f.read()
                log_analysis = self.llm.analyze_logs(log_content)

                # Modify code based on metrics and log analysis
                for file_path in self.file_index:
                    if self.rewrite_code(file_path, metrics):
                        logger.info(f"Adapted code in {file_path}")
                        # Dynamically reload modified module
                        module_name = file_path.stem
                        if module_name in sys.modules:
                            importlib.reload(sys.modules[module_name])

                # Implement meta-learning if accuracy is low
                if metrics['accuracy'] < 0.8:
                    self.implement_meta_learning()

                # Refine user response
                initial_response = f"System performance: {metrics}"
                context = "Provide a clear, user-friendly summary of system performance and recent adaptations."
                refined_response = self.llm.refine_response(initial_response, context)
                logger.info(f"Refined user response: {refined_response['refined_response']}")

            except Exception as e:
                logger.error(f"Error in behavior adaptation loop: {e}")
                time.sleep(60)  # Wait before retrying

# --- Main Execution ---
if __name__ == "__main__":
    engine = SelfModificationEngine()
    engine.behavior_adaptation_loop()