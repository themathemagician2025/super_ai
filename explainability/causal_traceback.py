# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import logging
import json
import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class CausalTraceback:
    """
    Provides detailed explanations of predictions with causal tracebacks,
    showing how and why each prediction was made with multi-layered explanations
    """

    def __init__(self, config_path: Path = Path("config/explainability.yaml")):
        self.config_path = config_path
        self.explainability_level = "detailed"  # basic, detailed, expert
        self.visualization_enabled = True
        self.explanations_history = []
        self.causal_graphs = {}
        self.feature_importance = {}
        self.explanation_templates = {}

        # Load configuration if it exists
        self._load_config()
        logger.info("Causal Traceback Engine initialized")

    def _load_config(self):
        """Load configuration settings if available"""
        try:
            if self.config_path.exists():
                import yaml
                with open(self.config_path, 'r') as file:
                    config = yaml.safe_load(file)
                    if config:
                        self.explainability_level = config.get('explainability_level', 'detailed')
                        self.visualization_enabled = config.get('visualization_enabled', True)
                        self.explanation_templates = config.get('explanation_templates', {})
                        logger.info(f"Loaded explainability configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")

    def explain_prediction(self,
                          prediction_data: Dict[str, Any],
                          model_info: Dict[str, Any],
                          audience: str = "general") -> Dict[str, Any]:
        """
        Generate a comprehensive explanation for a prediction with causal tracebacks

        Parameters:
        - prediction_data: Contains the prediction results and input features
        - model_info: Information about the model that made the prediction
        - audience: Target audience for the explanation (general, technical, regulatory)

        Returns:
        - A multi-layered explanation of the prediction
        """
        try:
            # Extract key information
            prediction_id = prediction_data.get("id", str(datetime.datetime.now().timestamp()))
            prediction_value = prediction_data.get("prediction")
            confidence = prediction_data.get("confidence", 0.0)
            features = prediction_data.get("features", {})
            domain = model_info.get("domain", "general")

            # Generate causal graph for this prediction
            causal_graph = self._generate_causal_graph(features, prediction_value, model_info)

            # Store the causal graph for future reference
            self.causal_graphs[prediction_id] = causal_graph

            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(features, causal_graph)
            self.feature_importance[prediction_id] = feature_importance

            # Generate the explanation based on audience level
            explanation = self._format_explanation(
                prediction_id=prediction_id,
                prediction_value=prediction_value,
                confidence=confidence,
                feature_importance=feature_importance,
                causal_graph=causal_graph,
                audience=audience,
                domain=domain
            )

            # Visualize the explanation if enabled
            if self.visualization_enabled:
                viz_path = self._visualize_causal_graph(causal_graph, prediction_id)
                explanation["visualization_path"] = str(viz_path)

            # Store the explanation in history
            self.explanations_history.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "prediction_id": prediction_id,
                "explanation": explanation
            })

            return explanation

        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return {
                "error": f"Failed to generate explanation: {str(e)}",
                "prediction_id": prediction_data.get("id", "unknown")
            }

    def _generate_causal_graph(self,
                              features: Dict[str, Any],
                              prediction: Any,
                              model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a causal graph showing relationships between features and prediction"""
        # In a real implementation, this would use actual causal discovery algorithms
        # or extract causal relationships from the model

        causal_graph = {
            "nodes": [],
            "edges": [],
            "root_causes": []
        }

        # Add outcome node
        outcome_name = model_info.get("target_variable", "prediction")
        causal_graph["nodes"].append({
            "id": "outcome",
            "name": outcome_name,
            "type": "outcome",
            "value": prediction
        })

        # Process each feature as a potential causal factor
        for i, (feature_name, feature_value) in enumerate(features.items()):
            # Add feature node
            node_id = f"feature_{i}"
            causal_graph["nodes"].append({
                "id": node_id,
                "name": feature_name,
                "type": "feature",
                "value": feature_value
            })

            # Simulate causal strength (in a real system, this would be calculated)
            # based on model coefficients, SHAP values, or other methods
            causal_strength = np.random.random() * 0.9 + 0.1  # Random value between 0.1 and 1.0

            # Add edge from feature to outcome
            causal_graph["edges"].append({
                "source": node_id,
                "target": "outcome",
                "strength": causal_strength,
                "type": "causal"
            })

            # Randomly designate some features as root causes
            if np.random.random() > 0.7:  # 30% chance of being a root cause
                causal_graph["root_causes"].append({
                    "feature": feature_name,
                    "strength": causal_strength,
                    "explanation": f"Strong influence on {outcome_name}"
                })

        # Add interactions between features (for more complex causal graphs)
        if len(features) > 1:
            for i in range(len(features) - 1):
                if np.random.random() > 0.7:  # 30% chance of interaction
                    source_id = f"feature_{i}"
                    target_id = f"feature_{i+1}"

                    # Add interaction edge
                    causal_graph["edges"].append({
                        "source": source_id,
                        "target": target_id,
                        "strength": np.random.random() * 0.5,
                        "type": "interaction"
                    })

        return causal_graph

    def _calculate_feature_importance(self,
                                     features: Dict[str, Any],
                                     causal_graph: Dict[str, Any]) -> Dict[str, float]:
        """Calculate feature importance based on the causal graph"""
        feature_importance = {}

        # Extract importance from causal graph edges
        for edge in causal_graph["edges"]:
            if edge["target"] == "outcome" and edge["type"] == "causal":
                # Find the source node
                source_id = edge["source"]
                source_node = next((node for node in causal_graph["nodes"]
                                   if node["id"] == source_id), None)

                if source_node:
                    feature_name = source_node["name"]
                    feature_importance[feature_name] = edge["strength"]

        return feature_importance

    def _format_explanation(self,
                           prediction_id: str,
                           prediction_value: Any,
                           confidence: float,
                           feature_importance: Dict[str, float],
                           causal_graph: Dict[str, Any],
                           audience: str,
                           domain: str) -> Dict[str, Any]:
        """Format the explanation based on the audience level"""
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(),
                                key=lambda x: x[1], reverse=True)

        # Basic explanation for general audience
        basic_explanation = {
            "summary": f"Prediction: {prediction_value} with {confidence:.1%} confidence",
            "key_factors": [{"name": name, "importance": f"{importance:.1%}"}
                           for name, importance in sorted_features[:3]]
        }

        # Add domain-specific contextual explanation if available
        context_template = self.explanation_templates.get(domain, {}).get("context",
                          "This prediction was based on analysis of {num_factors} factors.")
        basic_explanation["context"] = context_template.format(
            num_factors=len(feature_importance),
            prediction=prediction_value,
            confidence=f"{confidence:.1%}"
        )

        # For general audience, return basic explanation
        if audience == "general" or self.explainability_level == "basic":
            return {
                "prediction_id": prediction_id,
                "level": "basic",
                "explanation": basic_explanation
            }

        # Detailed explanation adds causal relationships
        detailed_explanation = {
            **basic_explanation,
            "causal_factors": [
                {
                    "name": cause["feature"],
                    "impact": f"{cause['strength']:.1%}",
                    "explanation": cause["explanation"]
                }
                for cause in causal_graph["root_causes"]
            ],
            "confidence_factors": self._explain_confidence(confidence, causal_graph)
        }

        # For technical audience or detailed level, add more information
        if audience == "technical" or self.explainability_level == "detailed":
            return {
                "prediction_id": prediction_id,
                "level": "detailed",
                "explanation": detailed_explanation
            }

        # Expert/regulatory explanation includes everything
        expert_explanation = {
            **detailed_explanation,
            "feature_importance": [
                {"name": name, "importance": importance}
                for name, importance in sorted_features
            ],
            "causal_graph": {
                "nodes": len(causal_graph["nodes"]),
                "edges": len(causal_graph["edges"]),
                "complexity": self._calculate_graph_complexity(causal_graph)
            },
            "methodology": {
                "model_type": model_info.get("model_type", "unknown"),
                "causal_discovery_method": model_info.get("causal_method", "correlation analysis"),
                "data_points": model_info.get("training_samples", "unknown")
            }
        }

        return {
            "prediction_id": prediction_id,
            "level": "expert",
            "explanation": expert_explanation
        }

    def _explain_confidence(self, confidence: float, causal_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate explanation for the confidence level"""
        confidence_explanation = []

        # In a real system, this would analyze the actual sources of uncertainty
        if confidence > 0.8:
            confidence_explanation.append({
                "factor": "Strong causal evidence",
                "impact": "high",
                "description": "Clear causal relationships were identified"
            })
        elif confidence > 0.5:
            confidence_explanation.append({
                "factor": "Moderate causal evidence",
                "impact": "medium",
                "description": "Some causal relationships were identified but with uncertainty"
            })
        else:
            confidence_explanation.append({
                "factor": "Weak causal evidence",
                "impact": "low",
                "description": "Limited causal relationships were identified"
            })

        # Add information about data quality if available
        data_quality = np.random.random()  # Simulated data quality score
        confidence_explanation.append({
            "factor": "Data quality",
            "impact": "high" if data_quality > 0.7 else "medium" if data_quality > 0.4 else "low",
            "description": f"Input data quality affects confidence"
        })

        return confidence_explanation

    def _calculate_graph_complexity(self, causal_graph: Dict[str, Any]) -> float:
        """Calculate the complexity of the causal graph"""
        num_nodes = len(causal_graph["nodes"])
        num_edges = len(causal_graph["edges"])

        if num_nodes <= 1:
            return 0.0

        # Calculate density of the graph
        max_possible_edges = num_nodes * (num_nodes - 1) / 2
        density = num_edges / max_possible_edges if max_possible_edges > 0 else 0

        # Calculate average degree
        avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0

        # Combine metrics into a complexity score
        complexity = (density * 0.5) + (min(avg_degree / 10, 0.5))
        return min(complexity, 1.0)  # Cap at 1.0

    def _visualize_causal_graph(self, causal_graph: Dict[str, Any], prediction_id: str) -> Path:
        """Create a visualization of the causal graph"""
        try:
            # Create a NetworkX graph
            G = nx.DiGraph()

            # Add nodes
            for node in causal_graph["nodes"]:
                G.add_node(node["id"],
                          name=node["name"],
                          type=node["type"],
                          value=node["value"])

            # Add edges
            for edge in causal_graph["edges"]:
                G.add_edge(edge["source"],
                          edge["target"],
                          weight=edge["strength"],
                          type=edge["type"])

            # Create the visualization
            plt.figure(figsize=(10, 8))
            pos = nx.spring_layout(G)

            # Draw nodes
            node_colors = ['#ff7f0e' if G.nodes[n]['type'] == 'outcome' else '#1f77b4'
                          for n in G.nodes]
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)

            # Draw edges with width proportional to strength
            edge_widths = [G[u][v]['weight'] * 5 for u, v in G.edges]
            nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7,
                                  edge_color='#2ca02c')

            # Draw labels
            labels = {n: G.nodes[n]['name'] for n in G.nodes}
            nx.draw_networkx_labels(G, pos, labels, font_size=12)

            # Save the figure
            output_dir = Path("visualizations")
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"causal_graph_{prediction_id}.png"
            plt.savefig(output_file)
            plt.close()

            return output_file

        except Exception as e:
            logger.error(f"Error visualizing causal graph: {str(e)}")
            return Path("visualizations/error.png")

    def get_historical_explanation(self, prediction_id: str) -> Dict[str, Any]:
        """Retrieve a historical explanation by prediction ID"""
        for entry in self.explanations_history:
            if entry["prediction_id"] == prediction_id:
                return entry["explanation"]

        return {"error": f"No explanation found for prediction {prediction_id}"}

    def compare_explanations(self, prediction_ids: List[str]) -> Dict[str, Any]:
        """Compare explanations between multiple predictions to identify patterns"""
        explanations = []

        for pred_id in prediction_ids:
            explanation = self.get_historical_explanation(pred_id)
            if "error" not in explanation:
                explanations.append(explanation)

        if not explanations:
            return {"error": "No valid explanations found for comparison"}

        # Analysis of common factors across predictions
        common_factors = self._find_common_factors(explanations)

        # Compare confidence levels
        confidence_comparison = self._compare_confidence(explanations)

        return {
            "num_explanations": len(explanations),
            "common_factors": common_factors,
            "confidence_comparison": confidence_comparison,
            "pattern_strength": self._calculate_pattern_strength(explanations)
        }

    def _find_common_factors(self, explanations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find common causal factors across multiple explanations"""
        # In a real implementation, this would do sophisticated analysis
        # For now, just a simple implementation to demonstrate the concept
        factor_counts = {}

        for explanation in explanations:
            exp_data = explanation.get("explanation", {})
            for factor in exp_data.get("key_factors", []):
                factor_name = factor.get("name")
                if factor_name:
                    factor_counts[factor_name] = factor_counts.get(factor_name, 0) + 1

        # Return factors that appear in more than one explanation
        common = [
            {"name": name, "occurrence": count, "prevalence": count / len(explanations)}
            for name, count in factor_counts.items()
            if count > 1
        ]

        return sorted(common, key=lambda x: x["occurrence"], reverse=True)

    def _compare_confidence(self, explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare confidence levels across explanations"""
        confidences = []

        for explanation in explanations:
            exp_data = explanation.get("explanation", {})
            summary = exp_data.get("summary", "")

            # Extract confidence from summary (simplified)
            import re
            confidence_match = re.search(r"(\d+(?:\.\d+)?)%", summary)
            if confidence_match:
                confidences.append(float(confidence_match.group(1)) / 100)

        if not confidences:
            return {"error": "Could not extract confidence values"}

        return {
            "average": sum(confidences) / len(confidences),
            "min": min(confidences),
            "max": max(confidences),
            "range": max(confidences) - min(confidences),
            "consistency": 1.0 - (np.std(confidences) if len(confidences) > 1 else 0)
        }

    def _calculate_pattern_strength(self, explanations: List[Dict[str, Any]]) -> float:
        """Calculate the strength of patterns across explanations"""
        # In a real implementation, this would be much more sophisticated
        # This is just a placeholder to show the concept

        if len(explanations) < 2:
            return 0.0

        # Look at common factors
        common_factors = self._find_common_factors(explanations)
        num_common = len(common_factors)

        # More common factors = stronger pattern
        max_possible = min(5, len(explanations))  # Cap at 5 to avoid over-emphasis
        pattern_strength = min(num_common / max_possible, 1.0) if max_possible > 0 else 0.0

        return pattern_strength
