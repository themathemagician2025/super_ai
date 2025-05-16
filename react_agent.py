# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
ReAct Agent Module

Provides autonomous reasoning, action, and self-correction capabilities.
"""

import re
import logging
import json
import time
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class ReactAgent:
    """Agent that can Think, Act, and Reflect autonomously in a ReAct pattern."""

    def __init__(self, tool_router=None, max_iterations: int = 3, confidence_threshold: float = 0.7):
        """
        Initialize the ReAct agent.

        Args:
            tool_router: The tool router component
            max_iterations: Maximum number of self-correction iterations
            confidence_threshold: Threshold for confidence scoring
        """
        self.tool_router = tool_router
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.uncertainty_patterns = [
            r"I'm not sure",
            r"I don't know",
            r"I'm uncertain",
            r"cannot determine",
            r"cannot find",
            r"unclear",
            r"don't have (?:enough|sufficient) information",
            r"would need more information",
            r"unable to provide"
        ]
        self.confidence_factors = {
            "specific_answer": 0.3,       # Contains specific data points
            "uncertainty_marker": -0.4,   # Contains uncertainty language
            "citation": 0.2,              # Cites sources
            "qualified_language": -0.15,  # Uses qualifiers (may, might, could)
            "direct_answer": 0.25,        # Directly answers the question
            "irrelevant": -0.5            # Response doesn't address the query
        }

    def autonomous_process(self, user_input: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a user query with autonomous reasoning and self-correction.

        Args:
            user_input: Parsed user input
            context: Session context

        Returns:
            Processed response with potential self-corrections
        """
        raw_input = user_input.get("raw_input", "")
        logger.info(f"Starting autonomous processing for: {raw_input}")

        # Initial processing using default approach
        response = self._execute_basic_response(user_input, context)

        # Track iterations for self-correction
        iterations = 0
        response_history = [response]

        # Check if the response is uncertain or low confidence
        while (iterations < self.max_iterations and
               (self._detect_uncertainty(response) or
                self._calculate_confidence(response, user_input) < self.confidence_threshold)):

            logger.info(f"Low confidence detected, iteration {iterations+1}/{self.max_iterations}")

            # Add reasoning trace to context
            context["reasoning_trace"] = context.get("reasoning_trace", [])
            context["reasoning_trace"].append({
                "iteration": iterations + 1,
                "response": response
            })

            # Try to improve through search if we detect uncertainty
            if self._detect_uncertainty(response):
                search_response = self._try_search_for_answer(user_input, context)
                if search_response and search_response.get("status") == "success":
                    # Update response with search results
                    new_response = self._enhance_with_search(response, search_response)
                    response_history.append(new_response)
                    response = new_response

            # Try running a Python computation if it seems like a calculation
            if self._looks_like_computation(raw_input) and "python_tool" in self.tool_router.tools:
                compute_response = self._try_computation(user_input, context)
                if compute_response and compute_response.get("status") == "success":
                    # Update response with computation results
                    new_response = self._enhance_with_computation(response, compute_response)
                    response_history.append(new_response)
                    response = new_response

            # If there's still low confidence, try a different approach
            if self._calculate_confidence(response, user_input) < self.confidence_threshold:
                new_response = self._try_alternative_approach(user_input, context, iterations)
                response_history.append(new_response)
                response = new_response

            iterations += 1

        # Add the reasoning trace and response history to the final response
        response["autonomous_processing"] = {
            "iterations": iterations,
            "response_history": response_history if iterations > 0 else None,
            "confidence": self._calculate_confidence(response, user_input)
        }

        if iterations > 0:
            logger.info(f"Completed autonomous processing after {iterations} iterations")

        return response

    def _execute_basic_response(self, user_input: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the basic response using the tool router."""
        if self.tool_router:
            return self.tool_router.process_request(user_input, context)
        else:
            # Fallback if no tool router is available
            return {
                "content": "I'm unable to process this request without the tool router.",
                "status": "error"
            }

    def _detect_uncertainty(self, response: Dict[str, Any]) -> bool:
        """Detect if the response indicates uncertainty."""
        content = response.get("content", "")
        if not content or not isinstance(content, str):
            return False

        # Check for uncertainty patterns
        for pattern in self.uncertainty_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                logger.info(f"Detected uncertainty pattern: {pattern}")
                return True

        return False

    def _calculate_confidence(self, response: Dict[str, Any], user_input: Dict[str, Any]) -> float:
        """Calculate a confidence score for the response."""
        content = response.get("content", "")
        if not content or not isinstance(content, str):
            return 0.0

        # Start with a base confidence
        confidence = 0.5
        raw_input = user_input.get("raw_input", "").lower()

        # Check for specific factors
        if any(re.search(pattern, content, re.IGNORECASE) for pattern in self.uncertainty_patterns):
            confidence += self.confidence_factors["uncertainty_marker"]

        if re.search(r"\d+(?:\.\d+)?", content):  # Contains numbers
            confidence += self.confidence_factors["specific_answer"]

        if re.search(r"(?:according to|source|reference|cite)", content, re.IGNORECASE):
            confidence += self.confidence_factors["citation"]

        if re.search(r"(?:may|might|could|possibly|perhaps)", content, re.IGNORECASE):
            confidence += self.confidence_factors["qualified_language"]

        # Check if response contains key terms from the query
        key_terms = self._extract_key_terms(raw_input)
        if key_terms:
            matches = sum(1 for term in key_terms if term.lower() in content.lower())
            if matches / len(key_terms) > 0.5:
                confidence += self.confidence_factors["direct_answer"]
            else:
                confidence += self.confidence_factors["irrelevant"]

        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from the input text."""
        # Remove common words and keep only substantive terms
        stop_words = {
            "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
            "be", "been", "being", "in", "on", "at", "to", "for", "with",
            "by", "about", "like", "through", "over", "before", "after",
            "between", "under", "above", "of", "from", "up", "down", "into",
            "would", "could", "should", "will", "can", "may", "might",
            "i", "you", "he", "she", "it", "we", "they", "who", "what",
            "when", "where", "why", "how", "all", "any", "both", "each",
            "do", "does", "did", "have", "has", "had", "not"
        }

        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if word not in stop_words and len(word) > 2]

    def _try_search_for_answer(self, user_input: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Attempt to search for information to answer the query."""
        if not self.tool_router or "search_tool" not in self.tool_router.tools:
            return None

        # Create a search-specific user input by modifying the intent
        search_input = user_input.copy()
        search_input["intent"] = "search"

        # Execute the search tool
        logger.info("Attempting to search for additional information")
        search_result = self.tool_router.execute_tool("search_tool", search_input, context)
        return search_result

    def _enhance_with_search(self, original_response: Dict[str, Any], search_response: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance the original response with search results."""
        enhanced_response = original_response.copy()

        # Extract search results
        search_items = search_response.get("items", [])
        if not search_items:
            return original_response

        # Build an enhanced content
        original_content = original_response.get("content", "")

        # Create a summary of the search results
        search_summary = "After searching for more information, I found:\n\n"
        for i, item in enumerate(search_items[:3]):  # Limit to top 3 results
            search_summary += f"{i+1}. {item.get('title', 'Untitled')}\n"
            search_summary += f"   {item.get('snippet', 'No description available')}\n\n"

        # Add a revised answer if the original showed uncertainty
        if self._detect_uncertainty(original_response):
            search_summary += "\nBased on this information, I can now provide a better answer:\n"
            # We'd ideally use the LLM here to generate a better answer based on search results
            # For now, we'll just append the search results

        enhanced_response["content"] = original_content + "\n\n" + search_summary

        return enhanced_response

    def _looks_like_computation(self, text: str) -> bool:
        """Check if the input looks like a mathematical computation."""
        # Check for mathematical operators and digits
        math_patterns = [
            r'\d+\s*[\+\-\*\/\^%]\s*\d+',  # Basic operations: 5 + 3, 10-2
            r'(?:calc|calculate|compute|solve|evaluate|find|what is)\s+[\d\+\-\*\/\^\(\)\s]+',  # Explicit calc requests
            r'square root|log|sin|cos|tan|mean|average|median|mode|variance|standard deviation'  # Math functions
        ]

        return any(re.search(pattern, text, re.IGNORECASE) for pattern in math_patterns)

    def _try_computation(self, user_input: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Attempt to solve a computational query."""
        if not self.tool_router or "python_tool" not in self.tool_router.tools:
            return None

        # Create a computation-specific user input
        compute_input = user_input.copy()
        compute_input["intent"] = "execute"

        # Execute the python tool
        logger.info("Attempting to solve through computation")
        compute_result = self.tool_router.execute_tool("python_tool", compute_input, context)
        return compute_result

    def _enhance_with_computation(self, original_response: Dict[str, Any], compute_response: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance the original response with computation results."""
        enhanced_response = original_response.copy()

        # Extract computation results
        result = compute_response.get("result")
        output = compute_response.get("content")

        if result is not None or output:
            # Build an enhanced content
            original_content = original_response.get("content", "")

            computation_summary = "\nI've performed a calculation to solve this:\n\n"
            if output and output != "No output":
                computation_summary += f"Output: {output}\n"
            if result is not None:
                computation_summary += f"Result: {result}\n"

            enhanced_response["content"] = original_content + computation_summary

        return enhanced_response

    def _try_alternative_approach(self, user_input: Dict[str, Any], context: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        """Try an alternative approach based on the current iteration."""
        # Different strategies for different iterations
        if iteration == 0:
            # First try: attempt to decompose the problem
            return self._decompose_problem(user_input, context)
        elif iteration == 1:
            # Second try: reformulate the query
            return self._reformulate_query(user_input, context)
        else:
            # Final try: be honest about limitations
            return {
                "content": "After several attempts, I'm unable to provide a satisfactory answer to your question. " +
                           "Could you please clarify or provide additional information?",
                "status": "partial"
            }

    def _decompose_problem(self, user_input: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Decompose the problem into smaller parts."""
        raw_input = user_input.get("raw_input", "")

        # Simple heuristic to break down a complex query
        parts = re.split(r'(?:and|,|\?|;)\s+', raw_input)
        if len(parts) > 1:
            # We have multiple parts to the query
            responses = []

            for i, part in enumerate(parts):
                if part.strip():
                    sub_input = user_input.copy()
                    sub_input["raw_input"] = part.strip()

                    logger.info(f"Processing sub-question {i+1}: {part.strip()}")

                    sub_response = self._execute_basic_response(sub_input, context)
                    responses.append({
                        "question": part.strip(),
                        "answer": sub_response.get("content", "No answer available")
                    })

            # Combine the responses
            combined_content = "I'll break this down into parts:\n\n"
            for i, resp in enumerate(responses):
                combined_content += f"Part {i+1}: {resp['question']}\n"
                combined_content += f"Answer: {resp['answer']}\n\n"

            return {
                "content": combined_content,
                "status": "success",
                "decomposed": True
            }
        else:
            # Couldn't decompose, return a basic response
            return self._execute_basic_response(user_input, context)

    def _reformulate_query(self, user_input: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Reformulate the query to get a better response."""
        raw_input = user_input.get("raw_input", "")

        # Simple heuristic to reformulate the query
        # In a real system, you'd use LLM for this
        reformulation = None

        if "how" in raw_input.lower():
            reformulation = raw_input.replace("how", "what steps").replace("?", "") + " process?"
        elif "what is" in raw_input.lower():
            reformulation = raw_input.replace("what is", "define").replace("?", "")
        elif "why" in raw_input.lower():
            reformulation = raw_input.replace("why", "what causes").replace("?", "")

        if reformulation:
            reformulated_input = user_input.copy()
            reformulated_input["raw_input"] = reformulation

            logger.info(f"Reformulated query: {reformulation}")

            response = self._execute_basic_response(reformulated_input, context)
            response["content"] = f"After rephrasing your question as '{reformulation}':\n\n{response.get('content', '')}"
            response["reformulated"] = True

            return response
        else:
            # Couldn't reformulate, return a basic response
            return self._execute_basic_response(user_input, context)

# Create an instance for use in SuperAI
react_agent = ReactAgent()

def initialize_react_agent(tool_router):
    """Initialize the ReAct agent with the tool router."""
    global react_agent
    react_agent = ReactAgent(tool_router)
    return react_agent
