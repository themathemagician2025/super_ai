import ast
import astor
from typing import Dict, List, Optional, Set
from pathlib import Path
import asyncio
from ..engine.starcoder_engine import StarCoderEngine
from ...utils.monitoring import MetricsManager
import logging

class CodeEvolution:
    def __init__(self, starcoder: StarCoderEngine, workspace: Path):
        self.starcoder = starcoder
        self.workspace = workspace
        self.metrics = MetricsManager()
        self.modified_files = set()
        
    async def evolve_file(self, file_path: Path) -> bool:
        try:
            with open(file_path, 'r') as f:
                code = f.read()
            analysis = self.starcoder.analyze_code(code)
            optimized = self.starcoder.optimize_code(code)
            if self.validate_evolution(code, optimized, analysis):
                with open(file_path, 'w') as f:
                    f.write(optimized)
                self.modified_files.add(file_path)
                return True
            return False
        except Exception as e:
            logging.error(f"Evolution failed for {file_path}: {e}")
            return False

    def validate_evolution(self, original: str, evolved: str, analysis: Dict) -> bool:
        if not self.starcoder.validate_code(evolved):
            return False
        orig_tree = ast.parse(original)
        evol_tree = ast.parse(evolved)
        return (
            self._compare_signatures(orig_tree, evol_tree) and
            self._verify_imports(evol_tree, analysis['imports']) and
            self._check_complexity(evol_tree, analysis['complexity'])
        )

    def _compare_signatures(self, orig_tree: ast.AST, evol_tree: ast.AST) -> bool:
        orig_funcs = {n.name: self._get_signature(n) for n in ast.walk(orig_tree) 
                     if isinstance(n, ast.FunctionDef)}
        evol_funcs = {n.name: self._get_signature(n) for n in ast.walk(evol_tree)
                     if isinstance(n, ast.FunctionDef)}
        return orig_funcs == evol_funcs

    def _get_signature(self, node: ast.FunctionDef) -> tuple:
        return (
            [arg.arg for arg in node.args.args],
            [arg.arg for arg in node.args.kwonlyargs],
            bool(node.args.vararg),
            bool(node.args.kwarg)
        )

    def _verify_imports(self, tree: ast.AST, required_imports: List[str]) -> bool:
        imports = {n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)}
        return all(imp in imports for imp in required_imports)

    def _check_complexity(self, tree: ast.AST, max_complexity: int) -> bool:
        return sum(1 for _ in ast.walk(tree)) <= max_complexity * 1.2

    async def evolve_module(self, module_path: Path) -> Set[Path]:
        if not module_path.is_dir():
            return set()
        tasks = []
        for file in module_path.rglob('*.py'):
            if file.name != '__init__.py':
                tasks.append(self.evolve_file(file))
        results = await asyncio.gather(*tasks)
        return {file for file, success in zip(module_path.rglob('*.py'), results) if success}

    async def evolve_codebase(self, exclude_patterns: List[str] = None) -> Dict[str, int]:
        exclude_patterns = exclude_patterns or ['__pycache__', '.git', 'venv']
        stats = {'evolved': 0, 'failed': 0, 'skipped': 0}
        
        for file in self.workspace.rglob('*.py'):
            if any(pattern in str(file) for pattern in exclude_patterns):
                stats['skipped'] += 1
                continue
            success = await self.evolve_file(file)
            if success:
                stats['evolved'] += 1
            else:
                stats['failed'] += 1
        
        return stats

    def rollback_evolution(self, file_path: Path) -> bool:
        if file_path not in self.modified_files:
            return False
        try:
            with open(file_path, 'r') as f:
                evolved = f.read()
            original = self.starcoder.refactor_code(evolved, "original style")
            with open(file_path, 'w') as f:
                f.write(original)
            self.modified_files.remove(file_path)
            return True
        except Exception:
            return False

    async def continuous_evolution(self, interval_seconds: int = 3600):
        while True:
            stats = await self.evolve_codebase()
            self.metrics.update_system_metrics({
                'evolved_files': stats['evolved'],
                'failed_evolutions': stats['failed']
            })
            await asyncio.sleep(interval_seconds) 