# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Mutation Strategies

This module provides various strategies for mutating Python code
to generate evolved variants with potentially improved characteristics.
"""

import ast
import re
import random
import logging
from typing import Dict, List, Tuple, Any, Optional, Union

logger = logging.getLogger(__name__)

class MutationStrategies:
    """
    Implements various strategies for code mutation to evolve program variants.
    
    Includes:
    - Genetic mutations (point mutations, crossovers)
    - Rule-based transformations (optimizations, patterns)
    - Template-based generation
    """
    
    def __init__(self):
        """Initialize the mutation strategies module"""
        self.genetic_mutations = {
            'rename_variables': 0.3,
            'modify_docstrings': 0.2,
            'inline_function': 0.15,
            'extract_function': 0.15,
            'modify_control_flow': 0.1,
            'change_data_structure': 0.1
        }
        
        self.rule_transformations = {
            'optimize_loops': 0.2,
            'add_error_handling': 0.3,
            'optimize_conditionals': 0.15,
            'use_faster_methods': 0.25,
            'reduce_memory_usage': 0.1
        }
        
        self.code_templates = {
            'performance_patterns': [
                "# Template for optimized iteration",
                "# Template for error handling",
                "# Template for resource management"
            ]
        }
        
        # Results cache to avoid redundant processing
        self.cache = {}
        
    def apply_genetic_mutations(self, code: str, tree: ast.Module) -> str:
        """
        Apply genetic mutation strategies to evolve code.
        
        Args:
            code: Original source code
            tree: AST of the original code
            
        Returns:
            Mutated code
        """
        # Clone the code (this is our starting point)
        mutated_code = code
        
        # Choose 1-3 mutation operations to apply
        num_mutations = random.randint(1, 3)
        operations = random.choices(
            list(self.genetic_mutations.keys()),
            weights=list(self.genetic_mutations.values()),
            k=num_mutations
        )
        
        # Apply each mutation
        for op in operations:
            try:
                if op == 'rename_variables':
                    mutated_code = self._rename_variables(mutated_code)
                elif op == 'modify_docstrings':
                    mutated_code = self._modify_docstrings(mutated_code)
                elif op == 'inline_function':
                    mutated_code = self._inline_function(mutated_code)
                elif op == 'extract_function':
                    mutated_code = self._extract_function(mutated_code)
                elif op == 'modify_control_flow':
                    mutated_code = self._modify_control_flow(mutated_code)
                elif op == 'change_data_structure':
                    mutated_code = self._change_data_structure(mutated_code)
            except Exception as e:
                logger.warning(f"Mutation {op} failed: {e}")
                
        return mutated_code
    
    def apply_rule_based_transformations(self, code: str, tree: ast.Module) -> str:
        """
        Apply rule-based transformations to optimize code.
        
        Args:
            code: Original source code
            tree: AST of the original code
            
        Returns:
            Transformed code
        """
        # Clone the code
        transformed_code = code
        
        # Choose 1-2 transformations to apply
        num_transforms = random.randint(1, 2)
        operations = random.choices(
            list(self.rule_transformations.keys()),
            weights=list(self.rule_transformations.values()),
            k=num_transforms
        )
        
        # Apply each transformation
        for op in operations:
            try:
                if op == 'optimize_loops':
                    transformed_code = self._optimize_loops(transformed_code)
                elif op == 'add_error_handling':
                    transformed_code = self._add_error_handling(transformed_code)
                elif op == 'optimize_conditionals':
                    transformed_code = self._optimize_conditionals(transformed_code)
                elif op == 'use_faster_methods':
                    transformed_code = self._use_faster_methods(transformed_code)
                elif op == 'reduce_memory_usage':
                    transformed_code = self._reduce_memory_usage(transformed_code)
            except Exception as e:
                logger.warning(f"Transformation {op} failed: {e}")
                
        return transformed_code
    
    def apply_template_based_generation(self, code: str, tree: ast.Module) -> str:
        """
        Apply template-based code generation to improve code.
        
        Args:
            code: Original source code
            tree: AST of the original code
            
        Returns:
            Generated code based on templates
        """
        # This is a simplified version - in practice, this would use more sophisticated
        # template matching and code generation
        
        # For now, we just return a slightly modified version of the original code
        lines = code.split('\n')
        
        # Add some performance annotations and comments
        for i in range(len(lines)):
            # Add performance hint comments at strategic locations
            if 'for ' in lines[i] and i > 0:
                lines[i] = "# Performance optimization: consider using comprehension\n" + lines[i]
            elif 'def ' in lines[i] and i > 0:
                lines[i] = "# Performance: cache frequently called functions\n" + lines[i]
                
        return '\n'.join(lines)
    
    # Genetic mutation implementations
    def _rename_variables(self, code: str) -> str:
        """Rename variables to improve clarity"""
        # Simple implementation: find variable assignments and rename them
        # This is a placeholder - real implementation would use AST transformation
        patterns = [
            (r'\btemp\b', 'temporary_value'),
            (r'\bi\b(?!\s*[=,\]])(?!\s*in\b)', 'index'),
            (r'\bj\b(?!\s*[=,\]])(?!\s*in\b)', 'sub_index'),
            (r'\bx\b(?!\s*[=,\]])(?!\s*in\b)', 'value'),
            (r'\by\b(?!\s*[=,\]])(?!\s*in\b)', 'result'),
        ]
        
        # Apply a random subset of the patterns
        chosen_patterns = random.sample(patterns, k=min(2, len(patterns)))
        result = code
        
        for pattern, replacement in chosen_patterns:
            result = re.sub(pattern, replacement, result)
            
        return result
    
    def _modify_docstrings(self, code: str) -> str:
        """Improve docstrings with more detailed information"""
        # Find existing docstrings and enhance them
        improved_code = code
        
        # Simple regex-based approach (will be replaced with AST in real impl)
        docstring_pattern = r'"""(.*?)"""'
        
        def enhance_docstring(match):
            docstring = match.group(1)
            # Add more detailed information
            if len(docstring.strip()) > 0:
                enhanced = f'"""{docstring}\n\nReturns:\n    Result of the operation\n"""'
                return enhanced
            return match.group(0)
            
        improved_code = re.sub(docstring_pattern, enhance_docstring, improved_code, flags=re.DOTALL)
        return improved_code
    
    def _inline_function(self, code: str) -> str:
        """Inline a small function for performance"""
        # This is a placeholder - real implementation would use AST
        return code
    
    def _extract_function(self, code: str) -> str:
        """Extract repeated code into a function"""
        # This is a placeholder - real implementation would use AST
        return code
    
    def _modify_control_flow(self, code: str) -> str:
        """Modify control flow for potential optimization"""
        # This is a placeholder - real implementation would use AST
        return code
    
    def _change_data_structure(self, code: str) -> str:
        """Change data structure for better performance"""
        # Simple replacement patterns - real implementation would be AST-based
        patterns = [
            (r'list\(\)', 'list()  # Changed from list to faster data structure when appropriate'),
            (r'dict\(\)', 'dict()  # Consider defaultdict for auto-initialization of values'),
            (r'\.append\(', '.append(  # Consider using a deque for frequent appends/pops'),
        ]
        
        result = code
        # Apply a random pattern
        if patterns:
            pattern, replacement = random.choice(patterns)
            result = re.sub(pattern, replacement, result)
            
        return result
    
    # Rule-based transformation implementations
    def _optimize_loops(self, code: str) -> str:
        """Optimize loops for better performance"""
        # Simple pattern-based optimization
        patterns = [
            # Convert range(len(x)) to enumerate
            (r'for\s+(\w+)\s+in\s+range\(len\((\w+)\)\):', r'for \1, value in enumerate(\2):'),
            # Use list comp instead of loop+append
            (r'(\w+)\s*=\s*\[\]\s*\n\s*for\s+(\w+)\s+in\s+(\w+)\s*:\s*\n\s*\1\.append\((\w+)\)', 
             r'\1 = [\4 for \2 in \3]'),
        ]
        
        result = code
        # Apply a random pattern if it matches
        if patterns:
            pattern, replacement = random.choice(patterns)
            result = re.sub(pattern, replacement, result)
            
        return result
    
    def _add_error_handling(self, code: str) -> str:
        """Add error handling around risky operations"""
        # Look for operations that might need error handling
        patterns = [
            # File operations
            (r'(with\s+open\(.+\)\s+as\s+\w+:)', 
             r'try:\n    \1\nexcept IOError as e:\n    logger.error(f"File operation failed: {e}")'),
            # Dictionary access
            (r'(\w+)\[(\w+)\]', r'\1.get(\2, None)'),
        ]
        
        result = code
        # Apply a random pattern if it matches
        if patterns:
            pattern, replacement = random.choice(patterns)
            # Only apply once to avoid nested try-excepts
            result = re.sub(pattern, replacement, result, count=1)
            
        return result
    
    def _optimize_conditionals(self, code: str) -> str:
        """Optimize conditional expressions"""
        # Simple optimizations
        patterns = [
            # Combine multiple ifs on same variable
            (r'if\s+(\w+)\s*==\s*(.+):\s*\n\s*(.+)\s*\n\s*if\s+\1\s*==\s*(.+):\s*\n\s*(.+)',
             r'if \1 == \2:\n    \3\nelif \1 == \4:\n    \5'),
        ]
        
        result = code
        # Apply a random pattern if it matches
        if patterns:
            pattern, replacement = random.choice(patterns)
            result = re.sub(pattern, replacement, result)
            
        return result
    
    def _use_faster_methods(self, code: str) -> str:
        """Replace methods with faster alternatives"""
        # Simple replacements
        patterns = [
            # Use join instead of + for string concatenation
            (r'(\w+)\s*\+\s*(\w+)\s*\+\s*(\w+)', r"''.join([\1, \2, \3])"),
        ]
        
        result = code
        # Apply a random pattern if it matches
        if patterns:
            pattern, replacement = random.choice(patterns)
            result = re.sub(pattern, replacement, result)
            
        return result
    
    def _reduce_memory_usage(self, code: str) -> str:
        """Modify code to reduce memory usage"""
        # Simple memory optimizations
        patterns = [
            # Use generators instead of lists for iteration
            (r'for\s+(\w+)\s+in\s+list\((.+)\):', r'for \1 in \2:'),
        ]
        
        result = code
        # Apply a random pattern if it matches
        if patterns:
            pattern, replacement = random.choice(patterns)
            result = re.sub(pattern, replacement, result)
            
        return result 