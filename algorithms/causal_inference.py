# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class CausalRelationship:
    cause: str
    effect: str
    strength: float  # -1 to 1 scale
    confidence: float  # 0 to 1 scale
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class CausalInferenceEngine:
    """
    Engine for discovering and analyzing causal relationships in data
    beyond simple correlations
    """

    def __init__(self):
        self.causal_graph = {}  # Graph representation of causal relationships
        self.discovered_relationships = []  # History of discovered relationships
        self.confidence_threshold = 0.65  # Minimum confidence for a relationship
        self.min_observations = 20  # Minimum observations for inference
        self.interventions_history = []  # History of interventions and outcomes

    def discover_relationships(self, data: pd.DataFrame) -> List[CausalRelationship]:
        """
        Discover causal relationships from observational data
        using causal discovery algorithms
        """
        logger.info(f"Discovering causal relationships from {len(data)} observations")

        # In a full implementation, this would use algorithms like:
        # - PC algorithm
        # - FCI algorithm
        # - GES (Greedy Equivalence Search)

        # Example simplified implementation
        relationships = []
        variables = data.columns.tolist()

        # For demonstration, we'll implement a simplified approach
        # that looks for conditional independence patterns
        for cause in variables:
            for effect in variables:
                if cause != effect:
                    # Check if there's a potential causal relationship
                    relationship = self._test_causal_relationship(data, cause, effect)
                    if relationship:
                        relationships.append(relationship)
                        # Update the causal graph
                        if cause not in self.causal_graph:
                            self.causal_graph[cause] = []
                        self.causal_graph[cause].append((effect, relationship.strength))

        # Store the discovered relationships
        self.discovered_relationships.extend(relationships)

        return relationships

    def _test_causal_relationship(self, data: pd.DataFrame, cause: str, effect: str) -> Optional[CausalRelationship]:
        """Test if there's a causal relationship between cause and effect"""
        if len(data) < self.min_observations:
            logger.warning(f"Insufficient observations ({len(data)}) for causal inference")
            return None

        # Simplified causal testing using conditional probabilities
        # In a real implementation, this would use more sophisticated methods

        # Mock implementation for demonstration
        # In reality, we would perform tests like:
        # 1. Check for conditional independence given potential confounders
        # 2. Apply do-calculus rules
        # 3. Use instrumental variables if available

        # For demonstration, we'll generate a mock relationship with random strength
        if np.random.random() > 0.3:  # 70% chance of finding a relationship
            strength = np.random.uniform(-0.8, 0.8)
            confidence = np.random.uniform(0.6, 0.95)

            if confidence >= self.confidence_threshold:
                return CausalRelationship(
                    cause=cause,
                    effect=effect,
                    strength=strength,
                    confidence=confidence
                )

        return None

    def do_calculus(self, data: pd.DataFrame, intervention: str, target: str,
                   intervention_value: Any) -> Dict[str, Any]:
        """
        Apply do-calculus to predict the effect of an intervention
        P(target | do(intervention = intervention_value))
        """
        logger.info(f"Calculating effect of do({intervention}={intervention_value}) on {target}")

        # In a real implementation, this would apply Pearl's do-calculus rules
        # to estimate the causal effect

        # For demonstration, we'll implement a simplified approach
        result = {
            "intervention": intervention,
            "intervention_value": intervention_value,
            "target": target,
            "estimated_effect": None,
            "confidence": 0.0
        }

        # Check if we have established a causal relationship
        relationship = None
        for rel in self.discovered_relationships:
            if rel.cause == intervention and rel.effect == target:
                relationship = rel
                break

        if relationship:
            # We have an established relationship, use it to estimate effect
            # Simplified calculation for demonstration
            baseline = data[target].mean()
            effect_size = relationship.strength

            # Calculate the estimated effect (simplified)
            # In a real implementation, this would be much more sophisticated
            deviation = abs(intervention_value - data[intervention].mean()) / data[intervention].std()
            estimated_effect = baseline + (effect_size * deviation * data[target].std())

            result["estimated_effect"] = estimated_effect
            result["confidence"] = relationship.confidence

            # Record this intervention for learning
            self.interventions_history.append({
                "timestamp": time.time(),
                "intervention": intervention,
                "intervention_value": intervention_value,
                "target": target,
                "estimated_effect": estimated_effect,
                "confidence": relationship.confidence
            })

        return result

    def counterfactual_analysis(self, data: pd.DataFrame, factual_row: pd.Series,
                               intervention: str, intervention_value: Any) -> Dict[str, Any]:
        """
        Perform counterfactual analysis: What would have happened if X had been different?
        """
        logger.info(f"Performing counterfactual analysis for {intervention}={intervention_value}")

        # In a real implementation, this would use structural equation models
        # and abduction-action-prediction steps

        # For demonstration, we'll create a simplified counterfactual
        counterfactual = factual_row.copy()
        counterfactual[intervention] = intervention_value

        # Calculate predicted changes for all causally downstream variables
        changes = {}
        affected_variables = self._get_descendants(intervention)

        for variable in affected_variables:
            # Find the relationship
            relationship = None
            for rel in self.discovered_relationships:
                if rel.cause == intervention and rel.effect == variable:
                    relationship = rel
                    break

            if relationship:
                # Calculate the counterfactual value (simplified)
                original_value = factual_row[variable]
                deviation = abs(intervention_value - factual_row[intervention]) / data[intervention].std()
                new_value = original_value + (relationship.strength * deviation * data[variable].std())

                counterfactual[variable] = new_value
                changes[variable] = {
                    "original": original_value,
                    "counterfactual": new_value,
                    "difference": new_value - original_value,
                    "confidence": relationship.confidence
                }

        return {
            "original": factual_row.to_dict(),
            "counterfactual": counterfactual.to_dict(),
            "changes": changes,
            "intervention": {
                "variable": intervention,
                "original_value": factual_row[intervention],
                "counterfactual_value": intervention_value
            }
        }

    def _get_descendants(self, node: str) -> List[str]:
        """Get all descendants of a node in the causal graph"""
        if node not in self.causal_graph:
            return []

        descendants = []
        queue = [node]
        visited = set()

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue

            visited.add(current)

            if current in self.causal_graph:
                for child, _ in self.causal_graph[current]:
                    descendants.append(child)
                    queue.append(child)

        return list(set(descendants))  # Remove duplicates

    def explain_relationship(self, cause: str, effect: str) -> Dict[str, Any]:
        """
        Generate a multi-level explanation of a causal relationship
        """
        # Find the relationship
        relationship = None
        for rel in self.discovered_relationships:
            if rel.cause == cause and rel.effect == effect:
                relationship = rel
                break

        if not relationship:
            return {
                "simple": f"No known causal relationship between {cause} and {effect}",
                "detailed": f"We haven't discovered any statistically significant causal relationship between {cause} and {effect}",
                "technical": f"No causal link established between {cause} and {effect}. Consider collecting more data or checking for confounding variables."
            }

        # Generate explanations at different levels
        strength_desc = "strong positive" if relationship.strength > 0.5 else \
                        "moderate positive" if relationship.strength > 0.1 else \
                        "weak positive" if relationship.strength > 0 else \
                        "strong negative" if relationship.strength < -0.5 else \
                        "moderate negative" if relationship.strength < -0.1 else \
                        "weak negative"

        confidence_desc = "high" if relationship.confidence > 0.8 else \
                          "moderate" if relationship.confidence > 0.7 else \
                          "limited"

        simple = f"{cause} {relationship.strength > 0 and 'increases' or 'decreases'} {effect}"

        detailed = (f"There is a {strength_desc} causal relationship between {cause} and {effect} "
                   f"with {confidence_desc} confidence ({relationship.confidence:.0%}). "
                   f"When {cause} changes, it directly affects {effect}.")

        technical = (f"Causal relationship: {cause} → {effect}\n"
                    f"Strength: {relationship.strength:.3f}\n"
                    f"Confidence: {relationship.confidence:.3f}\n"
                    f"Discovered: {time.ctime(relationship.timestamp)}")

        # Check if there are mediating variables
        mediators = self._find_mediators(cause, effect)
        if mediators:
            mediators_str = ", ".join(mediators)
            detailed += f" This effect is mediated through {mediators_str}."
            technical += f"\nMediators: {mediators_str}"

        # Check if there are confounding variables
        confounders = self._find_confounders(cause, effect)
        if confounders:
            confounders_str = ", ".join(confounders)
            detailed += f" We've accounted for confounding variables such as {confounders_str}."
            technical += f"\nControlled confounders: {confounders_str}"

        return {
            "simple": simple,
            "detailed": detailed,
            "technical": technical
        }

    def _find_mediators(self, cause: str, effect: str) -> List[str]:
        """Find mediating variables between cause and effect"""
        # In a real implementation, this would analyze the causal graph
        # to find all paths from cause to effect
        return []  # Placeholder

    def _find_confounders(self, cause: str, effect: str) -> List[str]:
        """Find confounding variables that affect both cause and effect"""
        # In a real implementation, this would analyze the causal graph
        # to find common causes
        return []  # Placeholder
