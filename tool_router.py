# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Tool Router Module

Selects the appropriate model or tool to handle user requests.
"""

import importlib
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ToolRouter:
    def __init__(self):
        """Initialize the tool router."""
        self.tools = {}
        self.intent_mappings = {
            "predict": ["forex_predictor", "sports_predictor"],
            "analyze": ["data_analyzer", "performance_analyzer"],
            "search": ["search_tool", "database_searcher", "web_searcher"],
            "help": ["documentation_helper"],
            "execute": ["python_tool"],
            "remember": ["memory_tool"]
        }

    def register_tool(self, name: str, tool_func: Callable, description: str = "") -> None:
        """
        Register a tool with the router.

        Args:
            name: Unique name for the tool
            tool_func: Callable function implementing the tool
            description: Human-readable description of the tool
        """
        self.tools[name] = {
            "function": tool_func,
            "description": description,
            "usage_count": 0
        }
        logger.info(f"Registered tool: {name}")

    def select_tool(self, user_input: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        Select the most appropriate tool based on user input and context.

        Args:
            user_input: Parsed user input
            context: Session context

        Returns:
            Name of the selected tool
        """
        intent = user_input.get("intent", "unknown")
        raw_input = user_input.get("raw_input", "").lower()

        # Check for tool-specific keywords first
        if "python" in raw_input and any(kw in raw_input for kw in ["calculate", "compute", "solve", "execute"]):
            return "python_tool" if "python_tool" in self.tools else "general_assistant"

        if any(kw in raw_input for kw in ["search", "find", "look up", "google"]):
            return "search_tool" if "search_tool" in self.tools else "general_assistant"

        if any(kw in raw_input for kw in ["remember", "recall", "history", "memory"]):
            return "memory_tool" if "memory_tool" in self.tools else "general_assistant"

        # Check if there's a direct mapping from intent
        candidate_tools = self.intent_mappings.get(intent, [])

        # Filter to only available tools
        available_candidates = [name for name in candidate_tools if name in self.tools]

        if available_candidates:
            # For now, just pick the first available tool for this intent
            # This can be enhanced with more sophisticated selection logic
            selected_tool = available_candidates[0]
        else:
            # Fallback to default tool
            selected_tool = "general_assistant"

        logger.info(f"Selected tool {selected_tool} for intent {intent}")
        return selected_tool

    def execute_tool(self, tool_name: str, user_input: Dict[str, Any],
                    context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the selected tool.

        Args:
            tool_name: Name of the tool to execute
            user_input: Parsed user input
            context: Session context

        Returns:
            Tool's response
        """
        if tool_name not in self.tools:
            logger.error(f"Tool {tool_name} not found")
            return {
                "error": f"Tool {tool_name} not available",
                "status": "error"
            }

        tool = self.tools[tool_name]
        tool["usage_count"] += 1

        try:
            # Execute the tool function
            logger.info(f"Executing tool: {tool_name}")
            response = tool["function"](user_input, context)
            return response
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            return {
                "error": f"Tool execution failed: {str(e)}",
                "status": "error"
            }

    def process_request(self, user_input: Dict[str, Any],
                       context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request by selecting and executing the appropriate tool.

        Args:
            user_input: Parsed user input
            context: Session context

        Returns:
            Response from the executed tool
        """
        # Select the tool
        tool_name = self.select_tool(user_input, context)

        # Execute the tool
        response = self.execute_tool(tool_name, user_input, context)

        # Add metadata to response
        response["tool_used"] = tool_name
        response["status"] = response.get("status", "success")

        return response

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get information about all available tools.

        Returns:
            List of tool information dictionaries
        """
        return [
            {
                "name": name,
                "description": info["description"],
                "usage_count": info["usage_count"]
            }
            for name, info in self.tools.items()
        ]

# Import tool handlers
try:
    from .python_tool import python_tool_handler
    from .search_tool import search_tool_handler
    from .memory_tool import memory_tool_handler
except ImportError:
    logger.warning("Could not import one or more tool handlers - they may need to be installed")
