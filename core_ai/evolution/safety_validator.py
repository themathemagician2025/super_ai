import ast
from typing import Dict, List, Set, Optional
import re
from pathlib import Path
import logging
from ..engine.starcoder_engine import StarCoderEngine

class SafetyValidator:
    def __init__(self):
        self.unsafe_patterns = {
            'system_calls': r'os\.system|subprocess\..*|eval|exec',
            'file_ops': r'open\s*\(.*,\s*[\'"]w[\'"]\)|remove|rmdir|delete',
            'network': r'socket\..*|urllib\..*|requests\..*',
            'imports': {'os', 'subprocess', 'socket', 'urllib', 'requests'}
        }
        self.allowed_mutations = {
            'optimization': {'loop_unrolling', 'vectorization', 'caching'},
            'refactoring': {'rename', 'extract_method', 'inline_method'},
            'architecture': {'add_layer', 'remove_layer', 'adjust_params'}
        }
        
    def validate_code_safety(self, code: str) -> tuple[bool, Optional[str]]:
        try:
            tree = ast.parse(code)
            checks = [
                self._check_dangerous_calls(tree),
                self._check_file_operations(tree),
                self._check_imports(tree),
                self._check_network_access(tree)
            ]
            for check, message in checks:
                if not check:
                    return False, message
            return True, None
        except Exception as e:
            return False, f"Validation failed: {str(e)}"

    def _check_dangerous_calls(self, tree: ast.AST) -> tuple[bool, Optional[str]]:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in {'eval', 'exec'}:
                        return False, f"Dangerous call detected: {node.func.id}"
                elif isinstance(node.func, ast.Attribute):
                    call = f"{node.func.value.id}.{node.func.attr}"
                    if re.match(self.unsafe_patterns['system_calls'], call):
                        return False, f"Dangerous system call detected: {call}"
        return True, None

    def _check_file_operations(self, tree: ast.AST) -> tuple[bool, Optional[str]]:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'open':
                    for kw in node.keywords:
                        if kw.arg == 'mode' and 'w' in kw.value.s:
                            return False, "Write file operation detected"
        return True, None

    def _check_imports(self, tree: ast.AST) -> tuple[bool, Optional[str]]:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    if name.name in self.unsafe_patterns['imports']:
                        return False, f"Unsafe import detected: {name.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module in self.unsafe_patterns['imports']:
                    return False, f"Unsafe import from detected: {node.module}"
        return True, None

    def _check_network_access(self, tree: ast.AST) -> tuple[bool, Optional[str]]:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    call = f"{node.func.value.id}.{node.func.attr}"
                    if re.match(self.unsafe_patterns['network'], call):
                        return False, f"Network access detected: {call}"
        return True, None

    def validate_mutation(self, original: str, mutated: str) -> tuple[bool, Optional[str]]:
        orig_tree = ast.parse(original)
        mut_tree = ast.parse(mutated)
        
        checks = [
            self._check_structure_preservation(orig_tree, mut_tree),
            self._check_mutation_type(orig_tree, mut_tree),
            self._check_complexity_bounds(mut_tree),
            self._check_naming_conventions(mut_tree)
        ]
        
        for check, message in checks:
            if not check:
                return False, message
        return True, None

    def _check_structure_preservation(
        self, orig: ast.AST, mutated: ast.AST
    ) -> tuple[bool, Optional[str]]:
        orig_classes = {n.name for n in ast.walk(orig) if isinstance(n, ast.ClassDef)}
        mut_classes = {n.name for n in ast.walk(mutated) if isinstance(n, ast.ClassDef)}
        
        if not orig_classes.issubset(mut_classes):
            return False, "Critical class definitions were removed"
        return True, None

    def _check_mutation_type(
        self, orig: ast.AST, mutated: ast.AST
    ) -> tuple[bool, Optional[str]]:
        changes = self._detect_changes(orig, mutated)
        if not changes.intersection(
            *self.allowed_mutations.values()
        ):
            return False, "Unrecognized mutation type"
        return True, None

    def _check_complexity_bounds(self, tree: ast.AST) -> tuple[bool, Optional[str]]:
        complexity = sum(1 for _ in ast.walk(tree))
        if complexity > 1000:  # Arbitrary limit
            return False, "Mutation resulted in excessive complexity"
        return True, None

    def _check_naming_conventions(self, tree: ast.AST) -> tuple[bool, Optional[str]]:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', node.name):
                    return False, f"Invalid name convention: {node.name}"
        return True, None

    def _detect_changes(self, orig: ast.AST, mutated: ast.AST) -> Set[str]:
        changes = set()
        
        orig_funcs = {n.name: n for n in ast.walk(orig) 
                     if isinstance(n, ast.FunctionDef)}
        mut_funcs = {n.name: n for n in ast.walk(mutated)
                    if isinstance(n, ast.FunctionDef)}
        
        for name, func in mut_funcs.items():
            if name in orig_funcs:
                if ast.dump(func) != ast.dump(orig_funcs[name]):
                    changes.add('optimization')
            else:
                changes.add('refactoring')
        
        return changes 