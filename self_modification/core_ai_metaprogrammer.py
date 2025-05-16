#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Meta-programming module for SUPER AI PROJECT that enables runtime code modification,
self-healing capabilities, and architectural evolution using Ollama models.
"""

import inspect
import os
import sys
import types
import ast
import logging
import importlib
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional, Union, Tuple

# Ensure ollama is available
try:
    import ollama
except ImportError:
    raise ImportError("Ollama package is required. Install with: pip install ollama")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.getenv("LOG_DIR", "/app/logs") + "/metaprogrammer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("core_ai_metaprogrammer")

class CodeAnalyzer:
    """Analyzes Python code to understand its structure and dependencies."""
    
    @staticmethod
    def parse_module(module_path: str) -> ast.Module:
        """Parse a Python module to its AST representation."""
        with open(module_path, 'r', encoding='utf-8') as file:
            return ast.parse(file.read(), filename=module_path)
    
    @staticmethod
    def get_imported_modules(module_ast: ast.Module) -> List[str]:
        """Extract all imported modules from an AST."""
        imports = []
        for node in ast.walk(module_ast):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return imports
    
    @staticmethod
    def get_functions(module_ast: ast.Module) -> List[str]:
        """Extract all function names from an AST."""
        return [node.name for node in ast.walk(module_ast) 
                if isinstance(node, ast.FunctionDef)]
    
    @staticmethod
    def get_classes(module_ast: ast.Module) -> List[str]:
        """Extract all class names from an AST."""
        return [node.name for node in ast.walk(module_ast) 
                if isinstance(node, ast.ClassDef)]
    
    @staticmethod
    def analyze_module_dependencies(module_path: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of a module's structure and dependencies.
        
        Returns:
            Dict containing imports, functions, classes, and complexity metrics
        """
        module_ast = CodeAnalyzer.parse_module(module_path)
        
        # Basic structure
        imports = CodeAnalyzer.get_imported_modules(module_ast)
        functions = CodeAnalyzer.get_functions(module_ast)
        classes = CodeAnalyzer.get_classes(module_ast)
        
        # Calculate cyclomatic complexity (basic version)
        complexity = 0
        for node in ast.walk(module_ast):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                complexity += 1
        
        return {
            "imports": imports,
            "functions": functions,
            "classes": classes,
            "complexity": complexity,
            "loc": len(module_ast.body),  # Rough estimate of lines of code
        }


class CodeRewriter:
    """Rewrites Python code using Ollama models to improve or fix it."""
    
    def __init__(self, model: str = "llama3.1:8b"):
        """
        Initialize the code rewriter with specified Ollama model.
        
        Args:
            model: Name of the Ollama model to use for code generation
        """
        self.model = model
        self.temperature = 0.2  # Keep temperature low for code generation
        self.max_tokens = 4000  # Allow for extensive code generation
        
    def rewrite_function(self, function: Callable, goal: str) -> str:
        """
        Rewrite a Python function to achieve a specific goal.
        
        Args:
            function: The function object to rewrite
            goal: Description of what the rewritten function should achieve
            
        Returns:
            String containing the rewritten function code
        """
        source_code = inspect.getsource(function)
        prompt = f"""
        You are an expert Python developer tasked with rewriting and improving the following function.
        
        GOAL: {goal}
        
        CURRENT FUNCTION:
        ```python
        {source_code}
        ```
        
        Please rewrite the function to better achieve the stated goal while maintaining its core functionality.
        Make the function more efficient, robust, and maintainable.
        Your response should contain ONLY the improved function code, nothing else.
        Include detailed docstrings and type annotations where appropriate.
        """
        
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
        )
        
        # Extract function code from response
        code = response["text"].strip()
        if "```python" in code:
            # Extract code from markdown code blocks if present
            code = code.split("```python")[1].split("```")[0].strip()
        
        return code
    
    def extend_class(self, class_obj: type, new_features: List[str]) -> str:
        """
        Extend a Python class with new methods and features.
        
        Args:
            class_obj: The class object to extend
            new_features: List of descriptions of new features to add
            
        Returns:
            String containing the extended class code
        """
        source_code = inspect.getsource(class_obj)
        features_text = "\n".join([f"- {feature}" for feature in new_features])
        
        prompt = f"""
        You are an expert Python developer tasked with extending the following class.
        
        CURRENT CLASS:
        ```python
        {source_code}
        ```
        
        ADD THE FOLLOWING FEATURES:
        {features_text}
        
        Please extend this class to include the requested features while maintaining compatibility with existing code.
        Your response should contain ONLY the improved class code, nothing else.
        Include detailed docstrings and type annotations for all new methods.
        """
        
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
        )
        
        # Extract class code from response
        code = response["text"].strip()
        if "```python" in code:
            # Extract code from markdown code blocks if present
            code = code.split("```python")[1].split("```")[0].strip()
        
        return code
    
    def rewrite_module(self, module_path: str, goals: List[str]) -> str:
        """
        Rewrite an entire Python module to achieve specific goals.
        
        Args:
            module_path: Path to the module file
            goals: List of goals the rewritten module should achieve
            
        Returns:
            String containing the rewritten module code
        """
        with open(module_path, 'r', encoding='utf-8') as file:
            source_code = file.read()
        
        goals_text = "\n".join([f"- {goal}" for goal in goals])
        
        prompt = f"""
        You are an expert Python developer tasked with rewriting and improving the following module.
        
        GOALS:
        {goals_text}
        
        CURRENT MODULE:
        ```python
        {source_code}
        ```
        
        Please rewrite this module to achieve the stated goals while maintaining its core functionality.
        Increase code verbosity and add detailed explanations through comments.
        Your response should contain ONLY the improved module code, nothing else.
        """
        
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
        )
        
        # Extract module code from response
        code = response["text"].strip()
        if "```python" in code:
            # Extract code from markdown code blocks if present
            code = code.split("```python")[1].split("```")[0].strip()
        
        return code


class SelfHealingEngine:
    """
    Engine that detects and fixes issues in the codebase automatically.
    Uses LLM analysis to identify problems and generate solutions.
    """
    
    def __init__(self, 
                 model_complex: str = "llama3.1:8b", 
                 model_fast: str = "phi:latest"):
        """
        Initialize the self-healing engine.
        
        Args:
            model_complex: Model for complex reasoning and code generation
            model_fast: Model for quick analysis and simpler tasks
        """
        self.complex_model = model_complex
        self.fast_model = model_fast
        self.code_rewriter = CodeRewriter(model=model_complex)
        self.fix_attempts = {}  # Track fix attempts to avoid infinite loops
        
    def analyze_exception(self, exc_info: Tuple) -> Dict[str, Any]:
        """
        Analyze an exception to determine its cause and potential fixes.
        
        Args:
            exc_info: Exception information tuple from sys.exc_info()
            
        Returns:
            Dict containing analysis and suggested fixes
        """
        exc_type, exc_value, exc_traceback = exc_info
        
        # Format exception information
        tb_str = ''.join(traceback.format_tb(exc_traceback))
        exc_message = str(exc_value)
        
        prompt = f"""
        Analyze this Python exception and provide a detailed diagnosis and fix recommendations.
        
        EXCEPTION TYPE: {exc_type.__name__}
        EXCEPTION MESSAGE: {exc_message}
        TRACEBACK:
        {tb_str}
        
        Return your analysis as JSON with the following structure:
        {{
            "root_cause": "Brief description of the root cause",
            "severity": "high/medium/low",
            "affected_files": ["list of affected file paths"],
            "suggested_fixes": [
                {{
                    "file_path": "path/to/file.py",
                    "line_number": 42,
                    "original_code": "problematic code",
                    "fixed_code": "corrected code",
                    "explanation": "Why this fix works"
                }}
            ]
        }}
        """
        
        try:
            response = ollama.generate(
                model=self.complex_model,
                prompt=prompt,
                options={"temperature": 0.2, "max_tokens": 2000}
            )
            
            # Extract and parse JSON from response
            text = response["text"].strip()
            # Find JSON in the response (between curly braces)
            json_str = text[text.find('{'):text.rfind('}')+1]
            return json.loads(json_str)
            
        except Exception as e:
            logger.error(f"Error analyzing exception: {e}")
            # Fallback basic analysis
            return {
                "root_cause": f"Unknown error: {exc_message}",
                "severity": "high",
                "affected_files": [],
                "suggested_fixes": []
            }
    
    def apply_fixes(self, analysis: Dict[str, Any]) -> bool:
        """
        Apply suggested fixes from an exception analysis.
        
        Args:
            analysis: Exception analysis from analyze_exception()
            
        Returns:
            Boolean indicating if fixes were successfully applied
        """
        if not analysis.get("suggested_fixes"):
            logger.warning("No suggested fixes found in analysis")
            return False
        
        success = True
        for fix in analysis["suggested_fixes"]:
            file_path = fix.get("file_path")
            if not file_path or not os.path.exists(file_path):
                logger.warning(f"Cannot find file: {file_path}")
                success = False
                continue
                
            # Check if we've tried this fix before to prevent loops
            fix_key = f"{file_path}:{fix.get('line_number')}"
            if fix_key in self.fix_attempts:
                self.fix_attempts[fix_key] += 1
                if self.fix_attempts[fix_key] > 3:
                    logger.warning(f"Exceeded maximum fix attempts for {fix_key}")
                    success = False
                    continue
            else:
                self.fix_attempts[fix_key] = 1
            
            try:
                # Read current file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Apply the fix
                if fix.get("original_code") and fix.get("fixed_code"):
                    new_content = content.replace(
                        fix["original_code"], 
                        fix["fixed_code"]
                    )
                    
                    # Write the fixed content
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                        
                    logger.info(f"Applied fix to {file_path}")
                    
                    # Update any loaded modules
                    if file_path.endswith('.py'):
                        module_name = os.path.basename(file_path)[:-3]
                        if module_name in sys.modules:
                            try:
                                importlib.reload(sys.modules[module_name])
                                logger.info(f"Reloaded module: {module_name}")
                            except Exception as e:
                                logger.error(f"Failed to reload module {module_name}: {e}")
                                success = False
                
            except Exception as e:
                logger.error(f"Error applying fix to {file_path}: {e}")
                success = False
        
        return success

    def auto_fix_exception(self, exc_info: Tuple) -> bool:
        """
        Automatically analyze and fix an exception.
        
        Args:
            exc_info: Exception information tuple from sys.exc_info()
            
        Returns:
            Boolean indicating if the exception was successfully fixed
        """
        analysis = self.analyze_exception(exc_info)
        logger.info(f"Exception analysis: {json.dumps(analysis, indent=2)}")
        
        if analysis["severity"] == "high":
            logger.warning(f"High severity issue detected: {analysis['root_cause']}")
        
        return self.apply_fixes(analysis)
    
    def active_monitoring(self):
        """Start active monitoring for exceptions and issues."""
        # Store original excepthook
        original_excepthook = sys.excepthook
        
        # Define custom exception handler
        def custom_excepthook(exc_type, exc_value, exc_traceback):
            # First, call original to maintain log output
            original_excepthook(exc_type, exc_value, exc_traceback)
            
            # Then attempt auto-fix
            logger.info(f"Attempting to auto-fix exception: {exc_type.__name__}")
            exc_info = (exc_type, exc_value, exc_traceback)
            if self.auto_fix_exception(exc_info):
                logger.info("Successfully applied fixes")
            else:
                logger.warning("Failed to apply fixes automatically")
        
        # Set custom excepthook
        sys.excepthook = custom_excepthook
        logger.info("Self-healing monitoring activated")


class ResponseAmplifier:
    """
    Amplifies and improves responses from the AI system using Ollama models.
    Makes responses more verbose, detailed, and informative.
    """
    
    def __init__(self, 
                 primary_model: str = "llama3.1:8b", 
                 fallback_model: str = "phi:latest",
                 min_words: int = 100):
        """
        Initialize the response amplifier.
        
        Args:
            primary_model: Primary Ollama model for response amplification
            fallback_model: Fallback model if primary fails
            min_words: Minimum word count for amplified responses
        """
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.min_words = min_words
        self.call_stats = {
            "total_calls": 0,
            "successful_amplifications": 0,
            "fallback_used": 0,
            "average_amplification_ratio": 0
        }
    
    def amplify_response(self, 
                        original_response: str, 
                        user_query: str,
                        context: Optional[str] = None) -> str:
        """
        Amplify a response to make it more detailed and informative.
        
        Args:
            original_response: The original response to amplify
            user_query: The user query that generated the original response
            context: Additional context to guide amplification
            
        Returns:
            Amplified response text
        """
        self.call_stats["total_calls"] += 1
        
        # Prepare context
        context_text = f"\nADDITIONAL CONTEXT:\n{context}" if context else ""
        
        prompt = f"""
        You are an expert at enhancing AI responses to make them more detailed, comprehensive, and insightful.
        
        USER QUERY:
        {user_query}
        
        ORIGINAL RESPONSE:
        {original_response}
        {context_text}
        
        Please enhance and amplify this response to be more comprehensive, with these requirements:
        1. The response must be at least {self.min_words} words
        2. Add technical depth and sophistication
        3. Include practical examples where relevant
        4. Maintain a professional but approachable tone
        5. Organize information logically with clear structure
        6. Expand explanations of complex concepts
        
        Your enhanced response should build upon the original, not contradict it.
        PROVIDE ONLY THE ENHANCED RESPONSE, NO INTRODUCTION OR EXPLANATION.
        """
        
        try:
            # Try with primary model first
            response = ollama.generate(
                model=self.primary_model,
                prompt=prompt,
                options={"temperature": 0.7, "max_tokens": 4000}
            )
            amplified_text = response["text"].strip()
            
        except Exception as e:
            logger.warning(f"Primary model failed: {e}. Falling back to {self.fallback_model}")
            self.call_stats["fallback_used"] += 1
            
            try:
                # Fallback to secondary model
                response = ollama.generate(
                    model=self.fallback_model,
                    prompt=prompt,
                    options={"temperature": 0.7, "max_tokens": 3000}
                )
                amplified_text = response["text"].strip()
                
            except Exception as e2:
                logger.error(f"Fallback model also failed: {e2}. Returning original response.")
                return original_response
        
        # Calculate amplification ratio
        original_words = len(original_response.split())
        amplified_words = len(amplified_text.split())
        
        # Update stats
        if amplified_words > original_words:
            self.call_stats["successful_amplifications"] += 1
            # Calculate running average of amplification ratio
            current_ratio = amplified_words / max(original_words, 1)
            prev_avg = self.call_stats["average_amplification_ratio"]
            prev_count = self.call_stats["successful_amplifications"] - 1
            if prev_count > 0:
                new_avg = (prev_avg * prev_count + current_ratio) / self.call_stats["successful_amplifications"]
                self.call_stats["average_amplification_ratio"] = new_avg
            else:
                self.call_stats["average_amplification_ratio"] = current_ratio
        
        logger.info(f"Response amplified: {original_words} → {amplified_words} words")
        
        return amplified_text


class MetaProgrammer:
    """
    Main class that orchestrates the entire self-modifying AI system.
    Integrates code analysis, rewriting, self-healing, and response amplification.
    """
    
    def __init__(self, 
                 code_base_path: str,
                 complex_model: str = "llama3.1:8b", 
                 fast_model: str = "phi:latest"):
        """
        Initialize the MetaProgrammer.
        
        Args:
            code_base_path: Root path of the codebase
            complex_model: Model name for complex tasks
            fast_model: Model name for faster, simpler tasks
        """
        self.code_base_path = code_base_path
        self.complex_model = complex_model
        self.fast_model = fast_model
        
        # Initialize components
        self.analyzer = CodeAnalyzer()
        self.rewriter = CodeRewriter(model=complex_model)
        self.self_healing = SelfHealingEngine(
            model_complex=complex_model,
            model_fast=fast_model
        )
        self.response_amplifier = ResponseAmplifier(
            primary_model=complex_model,
            fallback_model=fast_model
        )
        
        # Meta information
        self.modification_history = []
        self.performance_metrics = {}
        
    def enable_self_healing(self):
        """Enable automatic exception handling and self-healing."""
        self.self_healing.active_monitoring()
    
    def find_modules_to_optimize(self) -> List[Dict[str, Any]]:
        """
        Find modules in the codebase that would benefit from optimization.
        
        Returns:
            List of modules with optimization opportunities
        """
        optimization_candidates = []
        
        for root, _, files in os.walk(self.code_base_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    
                    # Skip already optimized files and test files
                    if "test" in file or file.startswith("_"):
                        continue
                    
                    try:
                        # Analyze the module
                        analysis = self.analyzer.analyze_module_dependencies(file_path)
                        
                        # Consider high-complexity modules as candidates
                        if analysis["complexity"] > 10 or analysis["loc"] > 200:
                            optimization_candidates.append({
                                "path": file_path,
                                "complexity": analysis["complexity"],
                                "loc": analysis["loc"],
                                "classes": len(analysis["classes"]),
                                "functions": len(analysis["functions"]),
                            })
                    except Exception as e:
                        logger.warning(f"Error analyzing {file_path}: {e}")
        
        # Sort by complexity (highest first)
        return sorted(optimization_candidates, 
                      key=lambda x: (x["complexity"], x["loc"]), 
                      reverse=True)
    
    def optimize_module(self, module_path: str) -> bool:
        """
        Optimize a single module by rewriting it.
        
        Args:
            module_path: Path to the module file
            
        Returns:
            Boolean indicating if optimization was successful
        """
        try:
            logger.info(f"Optimizing module: {module_path}")
            
            # Create backup
            backup_path = f"{module_path}.bak"
            with open(module_path, 'r', encoding='utf-8') as src:
                with open(backup_path, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
            
            # Define optimization goals
            goals = [
                "Increase code verbosity and detailed comments",
                "Improve error handling and robustness",
                "Optimize for performance where possible",
                "Add self-monitoring capabilities"
            ]
            
            # Rewrite the module
            new_code = self.rewriter.rewrite_module(module_path, goals)
            
            # Save the optimized module
            with open(module_path, 'w', encoding='utf-8') as f:
                f.write(new_code)
            
            # Record the modification
            self.modification_history.append({
                "timestamp": time.time(),
                "module": module_path,
                "action": "optimize",
                "backup_path": backup_path
            })
            
            logger.info(f"Successfully optimized module: {module_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to optimize module {module_path}: {e}")
            # Restore from backup if it exists
            if os.path.exists(backup_path):
                with open(backup_path, 'r', encoding='utf-8') as src:
                    with open(module_path, 'w', encoding='utf-8') as dst:
                        dst.write(src.read())
                logger.info(f"Restored {module_path} from backup")
            return False
    
    def auto_optimize_codebase(self, max_modules: int = 5) -> Dict[str, Any]:
        """
        Automatically find and optimize modules in the codebase.
        
        Args:
            max_modules: Maximum number of modules to optimize
            
        Returns:
            Summary of optimization results
        """
        candidates = self.find_modules_to_optimize()
        logger.info(f"Found {len(candidates)} optimization candidates")
        
        results = {
            "total_candidates": len(candidates),
            "attempted": 0,
            "successful": 0,
            "failed": 0,
            "optimized_modules": []
        }
        
        # Optimize the top candidates
        for i, candidate in enumerate(candidates[:max_modules]):
            results["attempted"] += 1
            module_path = candidate["path"]
            
            if self.optimize_module(module_path):
                results["successful"] += 1
                results["optimized_modules"].append(module_path)
            else:
                results["failed"] += 1
        
        logger.info(f"Optimization summary: {results['successful']}/{results['attempted']} modules optimized")
        return results
    
    def amplify_ai_response(self, 
                           original_response: str, 
                           user_query: str, 
                           context: Optional[str] = None) -> str:
        """
        Amplify an AI response to make it more detailed and verbose.
        
        Args:
            original_response: Original response from the AI
            user_query: User query that generated the response
            context: Additional context for amplification
            
        Returns:
            Amplified response
        """
        try:
            return self.response_amplifier.amplify_response(
                original_response=original_response,
                user_query=user_query,
                context=context
            )
        except Exception as e:
            logger.error(f"Error amplifying response: {e}")
            return original_response
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate a performance report of the metaprogramming system.
        
        Returns:
            Dict containing performance metrics and statistics
        """
        # Get amplifier stats
        amplifier_stats = self.response_amplifier.call_stats
        
        # Count modifications
        modifications = len(self.modification_history)
        
        # System performance
        return {
            "response_amplification": {
                "total_responses_processed": amplifier_stats["total_calls"],
                "successfully_amplified": amplifier_stats["successful_amplifications"],
                "fallback_used": amplifier_stats["fallback_used"],
                "average_amplification_ratio": amplifier_stats["average_amplification_ratio"]
            },
            "code_modifications": {
                "total_modifications": modifications,
                "last_modified": self.modification_history[-1]["timestamp"] if modifications > 0 else None,
                "modules_modified": [m["module"] for m in self.modification_history]
            },
            "self_healing": {
                "fixes_attempted": len(self.self_healing.fix_attempts),
                "unique_issues_addressed": len(set(self.self_healing.fix_attempts.keys()))
            }
        }


# Function to initialize and return the metaprogrammer
def initialize_metaprogrammer(code_base_path: str = None) -> MetaProgrammer:
    """
    Initialize the MetaProgrammer with the correct codebase path.
    
    Args:
        code_base_path: Path to the codebase. If None, will try to detect automatically.
        
    Returns:
        Configured MetaProgrammer instance
    """
    if code_base_path is None:
        # Try to detect the codebase path
        current_file = Path(__file__)
        # Assume the codebase root is 3 levels up from this file
        code_base_path = str(current_file.parent.parent.parent)
    
    logger.info(f"Initializing MetaProgrammer with codebase path: {code_base_path}")
    
    metaprogrammer = MetaProgrammer(
        code_base_path=code_base_path,
        complex_model="llama3.1:8b",
        fast_model="phi:latest"
    )
    
    return metaprogrammer


if __name__ == "__main__":
    # When run as a script, initialize and enable self-healing
    meta = initialize_metaprogrammer()
    meta.enable_self_healing()
    logger.info("MetaProgrammer initialized and self-healing enabled")
    
    # Run auto-optimization
    optimization_results = meta.auto_optimize_codebase(max_modules=3)
    logger.info(f"Auto-optimization complete: {optimization_results}") 