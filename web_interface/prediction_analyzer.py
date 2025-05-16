# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Prediction Analyzer Module

This module handles contextual analysis for athlete predictions, evaluating multiple
factors including form, fitness, surface compatibility, head-to-head records,
and mental resilience to generate comprehensive match predictions.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import random  # For demo purposes only - would use actual ML model in production

logger = logging.getLogger("prediction_analyzer")

class PredictionAnalyzer:
    """
    Analyzes athlete data to generate contextual match predictions.

    This class evaluates multiple factors to create comprehensive match predictions
    including player form, fitness levels, tactical approaches, surface compatibility,
    and mental resilience under pressure.
    """

    def __init__(self):
        """Initialize the prediction analyzer."""
        # Factor weights for different prediction aspects (would be ML-trained in production)
        self.factor_weights = {
            "recent_form": 0.25,
            "fitness": 0.20,
            "head_to_head": 0.15,
            "surface_compatibility": 0.15,
            "mental_resilience": 0.15,
            "tactical_analysis": 0.10
        }

    def analyze_matchup(self, athlete1_data: Dict[str, Any],
                        athlete2_data: Dict[str, Any],
                        context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze a matchup between two athletes and generate a prediction.

        Args:
            athlete1_data: Complete profile and metrics for the first athlete
            athlete2_data: Complete profile and metrics for the second athlete
            context: Optional contextual information like surface, tournament, etc.

        Returns:
            A dictionary containing the detailed prediction analysis
        """
        if not context:
            context = {}

        # Extract key metrics for both athletes
        athlete1_metrics = self._extract_key_metrics(athlete1_data)
        athlete2_metrics = self._extract_key_metrics(athlete2_data)

        # Perform factor-based analysis
        analysis = {
            "recent_form": self._analyze_recent_form(athlete1_metrics, athlete2_metrics),
            "fitness": self._analyze_fitness(athlete1_metrics, athlete2_metrics),
            "head_to_head": self._analyze_head_to_head(athlete1_data, athlete2_data),
            "surface_compatibility": self._analyze_surface_compatibility(
                athlete1_metrics, athlete2_metrics, context.get("surface")
            ),
            "mental_resilience": self._analyze_mental_resilience(athlete1_metrics, athlete2_metrics),
            "tactical_analysis": self._analyze_tactical_matchup(athlete1_metrics, athlete2_metrics)
        }

        # Calculate overall prediction
        prediction_result = self._calculate_prediction(analysis, athlete1_data, athlete2_data)

        # Structure the response
        return {
            "prediction": prediction_result,
            "factors": analysis,
            "athlete1": athlete1_data.get("name", "Unknown Athlete 1"),
            "athlete2": athlete2_data.get("name", "Unknown Athlete 2"),
            "context": context,
            "confidence_level": prediction_result.get("confidence_level", 0.5)
        }

    def _extract_key_metrics(self, athlete_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key performance metrics from athlete data.

        Args:
            athlete_data: Complete athlete profile and metrics

        Returns:
            Dictionary of key metrics for prediction analysis
        """
        # The keys we want to extract from the athlete data
        metrics = {}

        # Recent form (last 5-10 matches)
        metrics["recent_matches"] = athlete_data.get("recent_matches", [])
        metrics["win_loss_last_10"] = athlete_data.get("win_loss_last_10", (0, 0))
        metrics["recent_win_percentage"] = athlete_data.get("recent_win_percentage", 0.5)

        # Fitness indicators
        metrics["injury_status"] = athlete_data.get("injury_status", "healthy")
        metrics["fatigue_level"] = athlete_data.get("fatigue_level", 0.0)
        metrics["recovery_score"] = athlete_data.get("recovery_score", 100.0)

        # Surface performance
        metrics["surface_win_percentages"] = athlete_data.get("surface_win_percentages", {})

        # Mental resilience metrics
        metrics["tiebreak_win_percentage"] = athlete_data.get("tiebreak_win_percentage", 0.5)
        metrics["deciding_set_win_percentage"] = athlete_data.get("deciding_set_win_percentage", 0.5)
        metrics["comeback_wins"] = athlete_data.get("comeback_wins", 0)

        # Tactical metrics
        metrics["style_of_play"] = athlete_data.get("style_of_play", "all-court")
        metrics["strengths"] = athlete_data.get("strengths", [])
        metrics["weaknesses"] = athlete_data.get("weaknesses", [])

        return metrics

    def _analyze_recent_form(self, athlete1_metrics: Dict[str, Any],
                             athlete2_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the recent form of both athletes.

        Args:
            athlete1_metrics: Key metrics for the first athlete
            athlete2_metrics: Key metrics for the second athlete

        Returns:
            Analysis of recent form comparison
        """
        # Extract recent win percentages
        athlete1_recent_win_pct = athlete1_metrics.get("recent_win_percentage", 0.5)
        athlete2_recent_win_pct = athlete2_metrics.get("recent_win_percentage", 0.5)

        # Calculate form difference
        form_difference = athlete1_recent_win_pct - athlete2_recent_win_pct

        # Determine who has better form
        if form_difference > 0.15:
            advantage = "athlete1"
            advantage_level = "significant"
        elif form_difference > 0.05:
            advantage = "athlete1"
            advantage_level = "moderate"
        elif form_difference < -0.15:
            advantage = "athlete2"
            advantage_level = "significant"
        elif form_difference < -0.05:
            advantage = "athlete2"
            advantage_level = "moderate"
        else:
            advantage = "neutral"
            advantage_level = "minimal"

        return {
            "advantage": advantage,
            "advantage_level": advantage_level,
            "athlete1_recent_win_percentage": athlete1_recent_win_pct,
            "athlete2_recent_win_percentage": athlete2_recent_win_pct,
            "form_difference": abs(form_difference)
        }

    def _analyze_fitness(self, athlete1_metrics: Dict[str, Any],
                        athlete2_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the fitness levels of both athletes.

        Args:
            athlete1_metrics: Key metrics for the first athlete
            athlete2_metrics: Key metrics for the second athlete

        Returns:
            Analysis of fitness comparison
        """
        # Extract fitness indicators
        athlete1_injury = athlete1_metrics.get("injury_status", "healthy")
        athlete2_injury = athlete2_metrics.get("injury_status", "healthy")

        athlete1_fatigue = athlete1_metrics.get("fatigue_level", 0.0)
        athlete2_fatigue = athlete2_metrics.get("fatigue_level", 0.0)

        athlete1_recovery = athlete1_metrics.get("recovery_score", 100.0)
        athlete2_recovery = athlete2_metrics.get("recovery_score", 100.0)

        # Calculate overall fitness score (lower is better)
        # Injury status affects score significantly
        injury_factor_1 = 0.0 if athlete1_injury == "healthy" else (0.3 if athlete1_injury == "minor" else 0.7)
        injury_factor_2 = 0.0 if athlete2_injury == "healthy" else (0.3 if athlete2_injury == "minor" else 0.7)

        athlete1_fitness_score = athlete1_fatigue * 0.4 + (100 - athlete1_recovery) * 0.3 + injury_factor_1 * 0.3
        athlete2_fitness_score = athlete2_fatigue * 0.4 + (100 - athlete2_recovery) * 0.3 + injury_factor_2 * 0.3

        # Determine fitness advantage
        fitness_difference = athlete2_fitness_score - athlete1_fitness_score

        if fitness_difference > 20:
            advantage = "athlete1"
            advantage_level = "significant"
        elif fitness_difference > 10:
            advantage = "athlete1"
            advantage_level = "moderate"
        elif fitness_difference < -20:
            advantage = "athlete2"
            advantage_level = "significant"
        elif fitness_difference < -10:
            advantage = "athlete2"
            advantage_level = "moderate"
        else:
            advantage = "neutral"
            advantage_level = "minimal"

        return {
            "advantage": advantage,
            "advantage_level": advantage_level,
            "athlete1_fitness_score": 100 - athlete1_fitness_score,  # Convert to 100-based score (higher is better)
            "athlete2_fitness_score": 100 - athlete2_fitness_score,
            "athlete1_injury_status": athlete1_injury,
            "athlete2_injury_status": athlete2_injury,
            "fitness_difference": abs(fitness_difference)
        }

    def _analyze_head_to_head(self, athlete1_data: Dict[str, Any],
                             athlete2_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the head-to-head record between the athletes.

        Args:
            athlete1_data: Complete profile for the first athlete
            athlete2_data: Complete profile for the second athlete

        Returns:
            Analysis of head-to-head comparison
        """
        # Extract head-to-head data
        h2h_data = athlete1_data.get("head_to_head", {}).get(athlete2_data.get("id", ""), {})

        # If no data, check the reverse direction
        if not h2h_data:
            h2h_data = athlete2_data.get("head_to_head", {}).get(athlete1_data.get("id", ""), {})
            if h2h_data:
                # Reverse the wins/losses since we're looking from athlete2's perspective
                h2h_data = {
                    "wins": h2h_data.get("losses", 0),
                    "losses": h2h_data.get("wins", 0),
                    "matches": h2h_data.get("matches", [])
                }

        # Default if no head-to-head data exists
        wins = h2h_data.get("wins", 0)
        losses = h2h_data.get("losses", 0)
        total_matches = wins + losses

        if total_matches == 0:
            return {
                "advantage": "neutral",
                "advantage_level": "no data",
                "athlete1_wins": 0,
                "athlete2_wins": 0,
                "total_matches": 0,
                "h2h_ratio": 0.0
            }

        # Calculate head-to-head ratio
        h2h_ratio = wins / total_matches if total_matches > 0 else 0.5

        # Determine advantage
        if h2h_ratio > 0.7 and total_matches >= 3:
            advantage = "athlete1"
            advantage_level = "significant"
        elif h2h_ratio > 0.6:
            advantage = "athlete1"
            advantage_level = "moderate"
        elif h2h_ratio < 0.3 and total_matches >= 3:
            advantage = "athlete2"
            advantage_level = "significant"
        elif h2h_ratio < 0.4:
            advantage = "athlete2"
            advantage_level = "moderate"
        else:
            advantage = "neutral"
            advantage_level = "minimal"

        return {
            "advantage": advantage,
            "advantage_level": advantage_level,
            "athlete1_wins": wins,
            "athlete2_wins": losses,
            "total_matches": total_matches,
            "h2h_ratio": h2h_ratio
        }

    def _analyze_surface_compatibility(self, athlete1_metrics: Dict[str, Any],
                                      athlete2_metrics: Dict[str, Any],
                                      surface: Optional[str]) -> Dict[str, Any]:
        """
        Analyze how each athlete performs on the specified surface.

        Args:
            athlete1_metrics: Key metrics for the first athlete
            athlete2_metrics: Key metrics for the second athlete
            surface: The playing surface (e.g., "clay", "grass", "hard")

        Returns:
            Analysis of surface compatibility
        """
        # Default surface if none specified
        if not surface:
            surface = "hard"

        # Get surface win percentages for both athletes
        athlete1_surface_pcts = athlete1_metrics.get("surface_win_percentages", {})
        athlete2_surface_pcts = athlete2_metrics.get("surface_win_percentages", {})

        # Get win percentage on this surface
        athlete1_surface_win_pct = athlete1_surface_pcts.get(surface, 0.5)
        athlete2_surface_win_pct = athlete2_surface_pcts.get(surface, 0.5)

        # Calculate surface advantage
        surface_difference = athlete1_surface_win_pct - athlete2_surface_win_pct

        if surface_difference > 0.15:
            advantage = "athlete1"
            advantage_level = "significant"
        elif surface_difference > 0.05:
            advantage = "athlete1"
            advantage_level = "moderate"
        elif surface_difference < -0.15:
            advantage = "athlete2"
            advantage_level = "significant"
        elif surface_difference < -0.05:
            advantage = "athlete2"
            advantage_level = "moderate"
        else:
            advantage = "neutral"
            advantage_level = "minimal"

        return {
            "advantage": advantage,
            "advantage_level": advantage_level,
            "surface": surface,
            "athlete1_surface_win_percentage": athlete1_surface_win_pct,
            "athlete2_surface_win_percentage": athlete2_surface_win_pct,
            "surface_difference": abs(surface_difference)
        }

    def _analyze_mental_resilience(self, athlete1_metrics: Dict[str, Any],
                                  athlete2_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the mental resilience of both athletes under pressure.

        Args:
            athlete1_metrics: Key metrics for the first athlete
            athlete2_metrics: Key metrics for the second athlete

        Returns:
            Analysis of mental resilience comparison
        """
        # Extract mental resilience metrics
        athlete1_tiebreak_pct = athlete1_metrics.get("tiebreak_win_percentage", 0.5)
        athlete2_tiebreak_pct = athlete2_metrics.get("tiebreak_win_percentage", 0.5)

        athlete1_deciding_set_pct = athlete1_metrics.get("deciding_set_win_percentage", 0.5)
        athlete2_deciding_set_pct = athlete2_metrics.get("deciding_set_win_percentage", 0.5)

        # Calculate combined mental strength score
        athlete1_mental_score = athlete1_tiebreak_pct * 0.4 + athlete1_deciding_set_pct * 0.6
        athlete2_mental_score = athlete2_tiebreak_pct * 0.4 + athlete2_deciding_set_pct * 0.6

        mental_difference = athlete1_mental_score - athlete2_mental_score

        if mental_difference > 0.15:
            advantage = "athlete1"
            advantage_level = "significant"
        elif mental_difference > 0.05:
            advantage = "athlete1"
            advantage_level = "moderate"
        elif mental_difference < -0.15:
            advantage = "athlete2"
            advantage_level = "significant"
        elif mental_difference < -0.05:
            advantage = "athlete2"
            advantage_level = "moderate"
        else:
            advantage = "neutral"
            advantage_level = "minimal"

        return {
            "advantage": advantage,
            "advantage_level": advantage_level,
            "athlete1_mental_score": athlete1_mental_score,
            "athlete2_mental_score": athlete2_mental_score,
            "athlete1_tiebreak_win_percentage": athlete1_tiebreak_pct,
            "athlete2_tiebreak_win_percentage": athlete2_tiebreak_pct,
            "athlete1_deciding_set_win_percentage": athlete1_deciding_set_pct,
            "athlete2_deciding_set_win_percentage": athlete2_deciding_set_pct,
            "mental_difference": abs(mental_difference)
        }

    def _analyze_tactical_matchup(self, athlete1_metrics: Dict[str, Any],
                                 athlete2_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the tactical matchup between the athletes' playing styles.

        Args:
            athlete1_metrics: Key metrics for the first athlete
            athlete2_metrics: Key metrics for the second athlete

        Returns:
            Analysis of tactical matchup
        """
        # Extract tactical information
        athlete1_style = athlete1_metrics.get("style_of_play", "all-court")
        athlete2_style = athlete2_metrics.get("style_of_play", "all-court")

        athlete1_strengths = set(athlete1_metrics.get("strengths", []))
        athlete2_weaknesses = set(athlete2_metrics.get("weaknesses", []))

        athlete2_strengths = set(athlete2_metrics.get("strengths", []))
        athlete1_weaknesses = set(athlete1_metrics.get("weaknesses", []))

        # Check for style mismatches
        style_advantage = self._determine_style_advantage(athlete1_style, athlete2_style)

        # Check for strength-weakness exploitation
        athlete1_exploits = athlete1_strengths.intersection(athlete2_weaknesses)
        athlete2_exploits = athlete2_strengths.intersection(athlete1_weaknesses)

        # Calculate tactical advantage
        athlete1_tactical_score = 0.5  # Neutral starting point
        athlete2_tactical_score = 0.5

        # Adjust for style advantage
        if style_advantage == "athlete1":
            athlete1_tactical_score += 0.15
            athlete2_tactical_score -= 0.15
        elif style_advantage == "athlete2":
            athlete1_tactical_score -= 0.15
            athlete2_tactical_score += 0.15

        # Adjust for strength-weakness exploitation
        athlete1_tactical_score += len(athlete1_exploits) * 0.05
        athlete2_tactical_score += len(athlete2_exploits) * 0.05

        # Normalize scores to ensure they're between 0 and 1
        athlete1_tactical_score = max(0.0, min(1.0, athlete1_tactical_score))
        athlete2_tactical_score = max(0.0, min(1.0, athlete2_tactical_score))

        tactical_difference = athlete1_tactical_score - athlete2_tactical_score

        if tactical_difference > 0.15:
            advantage = "athlete1"
            advantage_level = "significant"
        elif tactical_difference > 0.05:
            advantage = "athlete1"
            advantage_level = "moderate"
        elif tactical_difference < -0.15:
            advantage = "athlete2"
            advantage_level = "significant"
        elif tactical_difference < -0.05:
            advantage = "athlete2"
            advantage_level = "moderate"
        else:
            advantage = "neutral"
            advantage_level = "minimal"

        return {
            "advantage": advantage,
            "advantage_level": advantage_level,
            "athlete1_style": athlete1_style,
            "athlete2_style": athlete2_style,
            "athlete1_tactical_score": athlete1_tactical_score,
            "athlete2_tactical_score": athlete2_tactical_score,
            "athlete1_exploits": list(athlete1_exploits),
            "athlete2_exploits": list(athlete2_exploits),
            "style_advantage": style_advantage,
            "tactical_difference": abs(tactical_difference)
        }

    def _determine_style_advantage(self, style1: str, style2: str) -> str:
        """
        Determine if one playing style has an advantage over another.

        Args:
            style1: Playing style of athlete 1
            style2: Playing style of athlete 2

        Returns:
            "athlete1", "athlete2", or "neutral" indicating style advantage
        """
        # Style advantage mapping (greatly simplified)
        style_advantages = {
            "aggressive_baseliner": {"defensive": "athlete1", "serve_and_volley": "athlete2"},
            "defensive": {"serve_and_volley": "athlete1", "aggressive_baseliner": "athlete2"},
            "serve_and_volley": {"aggressive_baseliner": "athlete1", "defensive": "athlete2"},
            "all-court": {}  # All-court players adapt to other styles
        }

        # Check for advantage
        if style1 in style_advantages and style2 in style_advantages.get(style1, {}):
            return style_advantages[style1][style2]

        return "neutral"

    def _calculate_prediction(self, analysis: Dict[str, Dict[str, Any]],
                             athlete1_data: Dict[str, Any],
                             athlete2_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate the final prediction based on factor analysis.

        Args:
            analysis: Complete factor analysis for the matchup
            athlete1_data: Complete profile for the first athlete
            athlete2_data: Complete profile for the second athlete

        Returns:
            Final prediction result
        """
        # Initialize scores
        athlete1_score = 0.0
        athlete2_score = 0.0

        # Calculate weighted factor scores
        for factor, factor_analysis in analysis.items():
            weight = self.factor_weights.get(factor, 0.1)

            # Convert advantage to score
            advantage = factor_analysis.get("advantage", "neutral")
            advantage_level = factor_analysis.get("advantage_level", "minimal")

            # Skip factors with no data
            if advantage_level == "no data":
                continue

            # Calculate score based on advantage level
            factor_score = 0.0
            if advantage_level == "significant":
                factor_score = 0.9
            elif advantage_level == "moderate":
                factor_score = 0.7
            elif advantage_level == "minimal":
                factor_score = 0.55

            # Apply score to the right athlete
            if advantage == "athlete1":
                athlete1_score += factor_score * weight
                athlete2_score += (1 - factor_score) * weight
            elif advantage == "athlete2":
                athlete1_score += (1 - factor_score) * weight
                athlete2_score += factor_score * weight
            else:
                # Neutral advantage
                athlete1_score += 0.5 * weight
                athlete2_score += 0.5 * weight

        # Normalize scores to add up to 1.0
        total = athlete1_score + athlete2_score
        if total > 0:
            athlete1_score /= total
            athlete2_score /= total
        else:
            athlete1_score = athlete2_score = 0.5

        # Determine winner
        if athlete1_score > athlete2_score:
            winner = "athlete1"
            win_probability = athlete1_score
        else:
            winner = "athlete2"
            win_probability = athlete2_score

        # Calculate confidence level
        score_difference = abs(athlete1_score - athlete2_score)
        if score_difference > 0.3:
            confidence_level = "high"
        elif score_difference > 0.15:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        return {
            "winner": winner,
            "winner_name": athlete1_data.get("name") if winner == "athlete1" else athlete2_data.get("name"),
            "athlete1_win_probability": athlete1_score,
            "athlete2_win_probability": athlete2_score,
            "confidence_level": confidence_level,
            "score_difference": score_difference
        }

def analyze_prediction(athlete1_data: Dict[str, Any],
                      athlete2_data: Dict[str, Any],
                      context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Analyze a matchup between two athletes and generate a prediction.

    Args:
        athlete1_data: Complete profile and metrics for the first athlete
        athlete2_data: Complete profile and metrics for the second athlete
        context: Optional contextual information like surface, tournament, etc.

    Returns:
        A dictionary containing the detailed prediction analysis
    """
    analyzer = PredictionAnalyzer()
    return analyzer.analyze_matchup(athlete1_data, athlete2_data, context)


# Example usage
if __name__ == "__main__":
    # This would be real athlete data in production
    demo_athlete1 = {
        "id": "player1",
        "name": "Player One",
        "recent_win_percentage": 0.75,
        "injury_status": "healthy",
        "surface_win_percentages": {"hard": 0.8, "clay": 0.6, "grass": 0.7},
        "tiebreak_win_percentage": 0.65,
        "deciding_set_win_percentage": 0.6,
        "style_of_play": "aggressive_baseliner",
        "strengths": ["forehand", "serving"],
        "weaknesses": ["backhand"]
    }

    demo_athlete2 = {
        "id": "player2",
        "name": "Player Two",
        "recent_win_percentage": 0.65,
        "injury_status": "minor",
        "surface_win_percentages": {"hard": 0.7, "clay": 0.75, "grass": 0.5},
        "tiebreak_win_percentage": 0.55,
        "deciding_set_win_percentage": 0.5,
        "style_of_play": "defensive",
        "strengths": ["backhand", "movement"],
        "weaknesses": ["serving"]
    }

    context = {"surface": "hard", "tournament": "grand slam"}

    result = analyze_prediction(demo_athlete1, demo_athlete2, context)
    print(f"Prediction: {result['prediction']['winner_name']} will win")
    print(f"Confidence: {result['prediction']['confidence_level']}")
    print(f"Win probabilities: {result['prediction']['athlete1_win_probability']:.2f} vs {result['prediction']['athlete2_win_probability']:.2f}")
    print("\nFactor Analysis:")
    for factor, analysis in result['factors'].items():
        print(f"- {factor}: {analysis['advantage']} has {analysis['advantage_level']} advantage")
