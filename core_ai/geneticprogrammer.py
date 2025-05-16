# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Genetic Programmer Module

This module provides interfaces for code generation and evolution using genetic programming
techniques. It complements geneticprogramming.py by focusing on the "programmer" aspects,
with an emphasis on code generation rather than algorithm development.
"""

import os
import random
import logging
import numpy as np
import inspect
from typing import List, Dict, Any, Callable, Optional, Tuple, Union
import math
import re
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(SRC_DIR, "generated_code")
os.makedirs(CODE_DIR, exist_ok=True)

class GeneticProgrammer:
    """
    GeneticProgrammer uses genetic programming techniques to evolve code solutions
    to programming tasks through generations of mutation and selection.
    """

    def __init__(self,
                population_size: int = 100,
                mutation_rate: float = 0.1,
                crossover_rate: float = 0.8,
                elite_size: int = 5,
                max_generations: int = 50,
                programming_language: str = "python"):
        """
        Initialize the genetic programmer.

        Args:
            population_size: Size of the code population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_size: Number of top individuals to preserve
            max_generations: Maximum number of generations to evolve
            programming_language: Target programming language
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.max_generations = max_generations
        self.programming_language = programming_language.lower()

        # Code metrics and templates
        self.code_patterns = self._initialize_code_patterns()
        self.code_templates = self._initialize_code_templates()
        self.population = []
        self.best_solution = None
        self.best_fitness = 0.0
        self.generation = 0
        self.fitness_history = []

        logger.info(f"GeneticProgrammer initialized for {programming_language} with "
                   f"population size {population_size}")

    def _initialize_code_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize code patterns for different languages."""
        patterns = {
            "python": {
                "function": r"def\s+(\w+)\s*\((.*?)\)\s*:",
                "class": r"class\s+(\w+)(?:\((.*?)\))?\s*:",
                "variable": r"(\w+)\s*=\s*(.*)",
                "import": r"(?:from\s+([\w.]+)\s+)?import\s+([\w.*]+)(?:\s+as\s+(\w+))?",
                "loop": r"(?:for|while)\s+(.*?):",
                "if_condition": r"if\s+(.*?):",
                "comment": r"#\s*(.*)",
                "indent": 4
            },
            "javascript": {
                "function": r"function\s+(\w+)\s*\((.*?)\)\s*\{",
                "class": r"class\s+(\w+)(?:\s+extends\s+(\w+))?\s*\{",
                "variable": r"(?:var|let|const)\s+(\w+)\s*=\s*(.*?);",
                "import": r"import\s+(?:\{\s*(.*?)\s*\}|\s*(\w+))\s+from\s+['\"](.+?)['\"];",
                "loop": r"(?:for|while)\s*\((.*?)\)",
                "if_condition": r"if\s*\((.*?)\)",
                "comment": r"\/\/\s*(.*)",
                "indent": 2
            }
        }
        return patterns

    def _initialize_code_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize code templates for different languages."""
        templates = {
            "python": {
                "function": "def {name}({params}):\n{indent}{body}\n",
                "class": "class {name}({base}):\n{indent}{body}\n",
                "variable": "{name} = {value}\n",
                "if": "if {condition}:\n{indent}{body}\n",
                "for": "for {var} in {iterable}:\n{indent}{body}\n",
                "while": "while {condition}:\n{indent}{body}\n",
                "import": "from {module} import {names}\n",
                "comment": "# {text}\n",
                "try_except": "try:\n{indent}{try_body}\nexcept {exception}:\n{indent}{except_body}\n",
                "return": "return {value}\n"
            },
            "javascript": {
                "function": "function {name}({params}) {{\n{indent}{body}}}\n",
                "class": "class {name}{extends} {{\n{indent}{body}}}\n",
                "variable": "{const} {name} = {value};\n",
                "if": "if ({condition}) {{\n{indent}{body}}}\n",
                "for": "for ({init}; {condition}; {update}) {{\n{indent}{body}}}\n",
                "foreach": "for (const {var} of {iterable}) {{\n{indent}{body}}}\n",
                "while": "while ({condition}) {{\n{indent}{body}}}\n",
                "import": "import {{{names}}} from '{module}';\n",
                "comment": "// {text}\n",
                "try_catch": "try {{\n{indent}{try_body}\n}} catch ({exception}) {{\n{indent}{catch_body}\n}}\n",
                "return": "return {value};\n"
            }
        }
        return templates

    def generate_initial_population(self, problem_statement: str) -> List[str]:
        """
        Generate initial population of code snippets based on problem statement.

        Args:
            problem_statement: Description of the problem to solve

        Returns:
            List of code snippets as strings
        """
        logger.info(f"Generating initial population for problem: {problem_statement[:50]}...")

        # Analyze problem for keywords and structure
        keywords = self._extract_keywords(problem_statement)

        population = []
        for _ in range(self.population_size):
            # Generate code with varying complexity
            complexity = random.randint(1, 5)
            code = self._generate_code_snippet(keywords, complexity)
            population.append(code)

        self.population = population
        logger.info(f"Generated initial population of {len(population)} code snippets")
        return population

    def _extract_keywords(self, problem_statement: str) -> Dict[str, List[str]]:
        """Extract relevant keywords from problem statement."""
        keywords = {
            "functions": [],
            "variables": [],
            "operations": [],
            "data_structures": [],
            "conditionals": []
        }

        # Simple keyword extraction based on common programming terms
        if re.search(r'\b(?:list|array|sequence)\b', problem_statement, re.I):
            keywords["data_structures"].append("list")

        if re.search(r'\b(?:dictionary|map|hash)\b', problem_statement, re.I):
            keywords["data_structures"].append("dict")

        if re.search(r'\b(?:loop|iterate|for each)\b', problem_statement, re.I):
            keywords["operations"].append("loop")

        if re.search(r'\b(?:if|check|condition|when)\b', problem_statement, re.I):
            keywords["conditionals"].append("if")

        if re.search(r'\b(?:function|method|routine)\b', problem_statement, re.I):
            keywords["functions"].append("function")

        if re.search(r'\b(?:class|object|instance)\b', problem_statement, re.I):
            keywords["functions"].append("class")

        # Extract variable name candidates
        var_matches = re.findall(r'\b([a-z][a-z_]*)\b', problem_statement, re.I)
        keywords["variables"] = list(set(var_matches))[:5]  # Limit to 5 unique variables

        return keywords

    def _generate_code_snippet(self, keywords: Dict[str, List[str]], complexity: int) -> str:
        """Generate a code snippet based on keywords and desired complexity."""
        lang = self.programming_language
        templates = self.code_templates.get(lang, self.code_templates["python"])

        # Start with imports or header
        code = ""
        if complexity > 1 and lang == "python":
            imports = ["random", "math", "os", "sys", "time", "re", "json"]
            selected_imports = random.sample(imports, min(complexity, len(imports)))
            for imp in selected_imports:
                code += templates["import"].format(module=imp, names="*")
            code += "\n"

        # Generate main function
        func_name = random.choice(["solve_problem", "process_data", "calculate_result",
                                  "find_solution", "optimize_answer"]) if not keywords["functions"] else random.choice(keywords["functions"])

        # Generate parameters
        num_params = random.randint(1, max(1, complexity))
        params = []
        for i in range(num_params):
            if keywords["variables"] and i < len(keywords["variables"]):
                params.append(keywords["variables"][i])
            else:
                params.append(f"param{i+1}")

        # Generate function body based on complexity
        body = ""
        if "list" in keywords["data_structures"]:
            var_name = random.choice(params) if params else "data"
            body += templates["variable"].format(name="result", value=f"[]\n")
            body += templates["for"].format(
                var="item",
                iterable=var_name,
                body=templates["if"].format(
                    condition="item > 0",
                    body=templates["variable"].format(name="processed", value="item * 2\n") +
                         templates["comment"].format(text="Process positive items\n") +
                         "result.append(processed)\n",
                    indent=" " * (templates.get("indent", 4) * 2)
                ),
                indent=" " * templates.get("indent", 4)
            )
        else:
            # Basic calculation
            body += templates["variable"].format(name="result", value="0\n")
            body += templates["for"].format(
                var="i",
                iterable=f"range({random.randint(5, 10)})",
                body=templates["variable"].format(name="result", value=f"result + i\n"),
                indent=" " * templates.get("indent", 4)
            )

        # Add return statement
        body += templates["return"].format(value="result")

        # Format full function
        function_code = templates["function"].format(
            name=func_name,
            params=", ".join(params),
            body=body,
            indent=" " * templates.get("indent", 4)
        )

        code += function_code

        # Add test code for higher complexity
        if complexity > 3:
            code += "\n\n"
            if lang == "python":
                code += "if __name__ == \"__main__\":\n"
                code += f"    {func_name}({', '.join(['10'] * len(params))})\n"
            elif lang == "javascript":
                code += f"{func_name}({', '.join(['10'] * len(params))});\n"

        return code

    def evolve(self, fitness_function: Callable[[str], float], generations: Optional[int] = None) -> str:
        """
        Evolve the population to find better code solutions.

        Args:
            fitness_function: Function to evaluate fitness of a code snippet
            generations: Number of generations to evolve, or use self.max_generations if None

        Returns:
            The best code solution found
        """
        if not self.population:
            raise ValueError("Population not initialized. Call generate_initial_population first.")

        max_gen = generations if generations is not None else self.max_generations
        logger.info(f"Starting evolution for {max_gen} generations...")

        for gen in range(max_gen):
            self.generation = gen + 1

            # Evaluate fitness
            fitness_scores = []
            for code in self.population:
                try:
                    fitness = fitness_function(code)
                except Exception as e:
                    logger.warning(f"Fitness evaluation failed: {e}")
                    fitness = 0.0
                fitness_scores.append(fitness)

            # Find best solution
            max_fitness_idx = fitness_scores.index(max(fitness_scores))
            if fitness_scores[max_fitness_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[max_fitness_idx]
                self.best_solution = self.population[max_fitness_idx]

            self.fitness_history.append(max(fitness_scores))

            logger.info(f"Generation {gen+1}/{max_gen}: Best fitness = {self.best_fitness:.4f}")

            # Create new population
            new_population = []

            # Elitism: Keep best solutions
            elites_idx = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:self.elite_size]
            for idx in elites_idx:
                new_population.append(self.population[idx])

            # Fill rest with crossover and mutation
            while len(new_population) < self.population_size:
                if random.random() < self.crossover_rate and len(self.population) > 1:
                    # Select parents using tournament selection
                    parent1 = self._tournament_selection(fitness_scores)
                    parent2 = self._tournament_selection(fitness_scores)

                    # Create child through crossover
                    child = self._crossover(self.population[parent1], self.population[parent2])
                else:
                    # Just select one parent
                    parent = self._tournament_selection(fitness_scores)
                    child = self.population[parent]

                # Apply mutation
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)

                new_population.append(child)

            self.population = new_population

        logger.info(f"Evolution completed. Best fitness: {self.best_fitness:.4f}")
        return self.best_solution

    def _tournament_selection(self, fitness_scores: List[float], tournament_size: int = 3) -> int:
        """Select an individual using tournament selection."""
        indices = random.sample(range(len(fitness_scores)), min(tournament_size, len(fitness_scores)))
        tournament_fitness = [fitness_scores[i] for i in indices]
        return indices[tournament_fitness.index(max(tournament_fitness))]

    def _crossover(self, code1: str, code2: str) -> str:
        """Perform crossover between two code snippets."""
        # Split into lines
        lines1 = code1.split('\n')
        lines2 = code2.split('\n')

        # Choose crossover points
        point1 = random.randint(1, max(1, len(lines1) - 1))
        point2 = random.randint(1, max(1, len(lines2) - 1))

        # Create child
        child_lines = lines1[:point1] + lines2[point2:]

        return '\n'.join(child_lines)

    def _mutate(self, code: str) -> str:
        """Mutate a code snippet."""
        lines = code.split('\n')

        # Choose mutation type
        mutation_type = random.choice(['add', 'delete', 'replace', 'modify'])

        if mutation_type == 'add' and lines:
            # Add a new line
            position = random.randint(0, len(lines))
            new_line = self._generate_random_code_line()
            lines.insert(position, new_line)

        elif mutation_type == 'delete' and len(lines) > 1:
            # Delete a line
            position = random.randint(0, len(lines) - 1)
            del lines[position]

        elif mutation_type == 'replace' and lines:
            # Replace a line
            position = random.randint(0, len(lines) - 1)
            lines[position] = self._generate_random_code_line()

        elif mutation_type == 'modify' and lines:
            # Modify a line
            position = random.randint(0, len(lines) - 1)
            if lines[position].strip():  # Not empty line
                # Simple modifications
                if '=' in lines[position]:
                    # Change value in assignment
                    parts = lines[position].split('=')
                    if len(parts) > 1:
                        try:
                            value = float(parts[1].strip())
                            new_value = value * random.uniform(0.5, 2.0)
                            lines[position] = f"{parts[0]}= {new_value}"
                        except ValueError:
                            # Not a number, leave unchanged
                            pass
                elif 'range(' in lines[position]:
                    # Change range value
                    match = re.search(r'range\((\d+)\)', lines[position])
                    if match:
                        old_val = int(match.group(1))
                        new_val = max(1, old_val + random.randint(-2, 2))
                        lines[position] = lines[position].replace(f"range({old_val})", f"range({new_val})")

        return '\n'.join(lines)

    def _generate_random_code_line(self) -> str:
        """Generate a random line of code."""
        lang = self.programming_language
        templates = self.code_templates.get(lang, self.code_templates["python"])

        line_types = ["variable", "comment", "if"]
        line_type = random.choice(line_types)

        if line_type == "variable":
            var_name = f"var_{random.randint(1, 100)}"
            value = random.choice([f"{random.randint(1, 100)}", f"'{chr(random.randint(97, 122))}'", "[]", "{}"])
            if lang == "javascript":
                return templates["variable"].format(const="let", name=var_name, value=value)
            return templates["variable"].format(name=var_name, value=value)

        elif line_type == "comment":
            comments = ["TODO: Improve this later", "Fix this algorithm", "Check for edge cases",
                        "Optimize performance", "Handle error cases"]
            return templates["comment"].format(text=random.choice(comments))

        elif line_type == "if":
            conditions = ["x > 0", "i < 10", "value != 0", "status == 'active'"]
            if lang == "javascript":
                return f"if ({random.choice(conditions)}) {{}}"
            return f"if {random.choice(conditions)}:"

        return ""  # Fallback

    def save_code(self, code: str, filename: Optional[str] = None) -> str:
        """
        Save the code to a file.

        Args:
            code: Code to save
            filename: Optional filename, or generate one if None

        Returns:
            Path to the saved file
        """
        if filename is None:
            extension = ".py" if self.programming_language == "python" else ".js"
            filename = f"genetic_solution_{self.generation}_{int(self.best_fitness * 100)}{extension}"

        filepath = os.path.join(CODE_DIR, filename)
        with open(filepath, 'w') as f:
            f.write(code)

        logger.info(f"Saved code to {filepath}")
        return filepath

    def save_evolution_history(self, filename: str = "evolution_history.pkl") -> str:
        """
        Save the evolution history.

        Args:
            filename: Filename for the history data

        Returns:
            Path to the saved file
        """
        history = {
            "fitness_history": self.fitness_history,
            "best_fitness": self.best_fitness,
            "best_solution": self.best_solution,
            "generations": self.generation,
            "language": self.programming_language
        }

        filepath = os.path.join(CODE_DIR, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(history, f)

        logger.info(f"Saved evolution history to {filepath}")
        return filepath

def test_fitness_function(code: str) -> float:
    """
    Simple test fitness function that evaluates code quality.

    Args:
        code: Code to evaluate

    Returns:
        Fitness score between 0 and 1
    """
    score = 0.0

    # Check for syntax errors
    try:
        compile(code, "<string>", "exec")
        score += 0.5  # Base score for valid syntax
    except SyntaxError:
        return 0.0  # Invalid syntax gets 0

    # Check for good practices
    if "def " in code:
        score += 0.1  # Has functions

    if "#" in code or "///" in code:
        score += 0.1  # Has comments

    if "if " in code:
        score += 0.1  # Has conditionals

    if "for " in code or "while " in code:
        score += 0.1  # Has loops

    if "try" in code and ("except" in code or "catch" in code):
        score += 0.1  # Has error handling

    # Length penalty for overly complex solutions
    lines = len(code.split('\n'))
    if lines > 50:
        score -= 0.1 * (lines / 50 - 1)  # Penalty for very long code

    return max(0.0, min(1.0, score))  # Ensure between 0 and 1

def main():
    """Test the GeneticProgrammer module."""
    problem = "Create a function to find the sum of even numbers in a list"

    # Initialize and test GeneticProgrammer
    gp = GeneticProgrammer(population_size=20, max_generations=10)
    gp.generate_initial_population(problem)
    best_code = gp.evolve(test_fitness_function)

    print(f"Best solution (fitness: {gp.best_fitness:.4f}):")
    print(best_code)

    # Save the solution
    gp.save_code(best_code)
    gp.save_evolution_history()

if __name__ == "__main__":
    main()
