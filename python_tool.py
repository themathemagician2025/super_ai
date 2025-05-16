# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Python Tool Module

Provides secure Python code execution capabilities.
"""

import ast
import sys
import io
import traceback
import logging
import math
import re
import json
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

class PythonTool:
    def __init__(self, max_execution_time: int = 5):
        """
        Initialize the Python execution tool.

        Args:
            max_execution_time: Maximum execution time in seconds
        """
        self.max_execution_time = max_execution_time
        self.allowed_modules = {
            'math', 'random', 'datetime', 'json', 're', 'collections',
            'itertools', 'functools', 'operator', 'statistics', 'decimal',
            'fractions', 'string', 'calendar'
        }

    def execute(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code securely.

        Args:
            code: Python code to execute

        Returns:
            Dict containing execution results and output
        """
        # Check for unsafe imports
        if not self._is_code_safe(code):
            return {
                "status": "error",
                "error": "Code contains potentially unsafe operations",
                "output": None,
                "result": None
            }

        # Capture stdout
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output

        result = None
        error = None

        try:
            # Execute the code
            compiled_code = compile(code, "<string>", "exec")
            local_vars = {}
            global_vars = self._get_safe_globals()

            exec(compiled_code, global_vars, local_vars)

            # Look for a result variable
            if "result" in local_vars:
                result = local_vars["result"]

        except Exception as e:
            error = traceback.format_exc()
            logger.error(f"Error executing Python code: {str(e)}")

        finally:
            # Restore stdout
            sys.stdout = old_stdout

        output = redirected_output.getvalue()

        return {
            "status": "error" if error else "success",
            "error": error,
            "output": output,
            "result": self._serialize_result(result)
        }

    def _get_safe_globals(self) -> Dict[str, Any]:
        """Create a dictionary of safe globals for execution."""
        safe_globals = {
            'math': math,
            're': re,
            'json': json,
        }

        # Add safe math functions directly
        for name in dir(math):
            if not name.startswith('_'):
                safe_globals[name] = getattr(math, name)

        return safe_globals

    def _is_code_safe(self, code: str) -> bool:
        """
        Check if the code is safe to execute.

        Args:
            code: Python code to check

        Returns:
            True if the code is safe, False otherwise
        """
        try:
            # Parse the code into an AST
            tree = ast.parse(code)

            # Check for unsafe operations
            for node in ast.walk(tree):
                # Check for imports
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    # For Import nodes
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            if name.name.split('.')[0] not in self.allowed_modules:
                                logger.warning(f"Unsafe import detected: {name.name}")
                                return False

                    # For ImportFrom nodes
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and node.module.split('.')[0] not in self.allowed_modules:
                            logger.warning(f"Unsafe import from detected: {node.module}")
                            return False

                # Check for exec or eval calls
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in ['exec', 'eval', '__import__']:
                        logger.warning(f"Unsafe function call detected: {node.func.id}")
                        return False

                # Check for attribute access that might be risky
                elif isinstance(node, ast.Attribute):
                    if node.attr.startswith('__') and node.attr.endswith('__'):
                        logger.warning(f"Potentially unsafe dunder method access: {node.attr}")
                        return False

                    # Check for system/file operations
                    if node.attr in ['system', 'popen', 'spawn', 'open', 'read', 'write']:
                        full_path = self._get_attribute_path(node)
                        if full_path in ['os.system', 'os.popen', 'subprocess.run', 'subprocess.Popen',
                                        'open', 'file.open', 'file.write', 'file.read']:
                            logger.warning(f"Unsafe operation detected: {full_path}")
                            return False

            return True

        except SyntaxError as e:
            logger.error(f"Syntax error in code: {str(e)}")
            return False

    def _get_attribute_path(self, node: ast.Attribute) -> str:
        """
        Recursively build the full attribute path (e.g., 'os.path.join').

        Args:
            node: AST attribute node

        Returns:
            String representation of the attribute path
        """
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            return f"{self._get_attribute_path(node.value)}.{node.attr}"
        return node.attr

    def _serialize_result(self, result: Any) -> Any:
        """
        Serialize the result to ensure it can be converted to JSON.

        Args:
            result: Result to serialize

        Returns:
            JSON-serializable version of the result
        """
        if result is None:
            return None

        # Handle common types
        if isinstance(result, (int, float, str, bool, type(None))):
            return result

        # Handle lists and dictionaries recursively
        if isinstance(result, list):
            return [self._serialize_result(item) for item in result]

        if isinstance(result, dict):
            return {str(k): self._serialize_result(v) for k, v in result.items()}

        # Handle sets
        if isinstance(result, set):
            return list(result)

        # Convert other types to string
        return str(result)

# Create an instance for use in tool_router
python_tool = PythonTool()

def python_tool_handler(user_input: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle Python execution requests.

    Args:
        user_input: Parsed user input
        context: Session context

    Returns:
        Execution result
    """
    # Extract the code from the user input
    raw_input = user_input.get("raw_input", "")

    # Try to find a code block in the input
    code_match = re.search(r"```python\s+(.*?)\s+```", raw_input, re.DOTALL)
    if code_match:
        code = code_match.group(1)
    else:
        # If no code block, try to extract code based on heuristics
        lines = raw_input.strip().split('\n')
        if len(lines) > 1:
            # Multiple lines - treat as code
            code = raw_input
        else:
            # Single line - add a result variable
            code = f"result = {raw_input}"

    # Execute the code
    execution_result = python_tool.execute(code)

    return {
        "title": "Python Execution Result",
        "status": execution_result["status"],
        "content": execution_result["output"] if execution_result["output"] else "No output",
        "result": execution_result["result"],
        "error": execution_result["error"]
    }

if __name__ == "__main__":
    # Test the Python tool
    test_code = """
import math
x = 10
y = math.sqrt(x)
result = y * 2
print(f"The square root of {x} is {y}")
"""

    tool = PythonTool()
    result = tool.execute(test_code)
    print(json.dumps(result, indent=2))
