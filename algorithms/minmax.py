# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import time
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class GameState:
    board: List[any]
    score: float
    depth: int
    causal_factors: Dict[str, float] = None
    history: List[Dict] = None

    def __post_init__(self):
        if self.causal_factors is None:
            self.causal_factors = {}
        if self.history is None:
            self.history = []

class MinMaxSearcher:
    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self.decision_history = []
        self.explanation_mode = "detailed"  # simple, detailed, technical
        self.error_counter = {}
        self.performance_stats = {"calls": 0, "cache_hits": 0, "pruning_events": 0}
        self.transposition_table = {}
        self.self_correction_active = True

    def alpha_beta_search(self, state: GameState, alpha: float, beta: float,
                         maximizing: bool) -> Tuple[float, Dict[str, Any]]:
        """
        Enhanced alpha-beta search with explainability and self-correction
        Returns: (value, explanation)
        """
        self.performance_stats["calls"] += 1

        # Check if we've seen this state before (transposition table)
        state_hash = str(state.board) + str(state.depth)
        if state_hash in self.transposition_table:
            self.performance_stats["cache_hits"] += 1
            return self.transposition_table[state_hash]

        if state.depth >= self.max_depth or self._is_terminal(state):
            evaluation = self._evaluate(state)
            explanation = {"type": "leaf", "value": evaluation}
            return evaluation, explanation

        if maximizing:
            value = float('-inf')
            best_explanation = {}
            for move in self._get_possible_moves(state):
                new_state = self._apply_move(state, move)

                move_value, move_explanation = self.alpha_beta_search(new_state, alpha, beta, False)

                if move_value > value:
                    value = move_value
                    best_explanation = {
                        "move": move,
                        "value": value,
                        "sub_explanation": move_explanation,
                        "type": "max"
                    }

                alpha = max(alpha, value)
                if alpha >= beta:
                    self.performance_stats["pruning_events"] += 1
                    break

            # Cache the result
            self.transposition_table[state_hash] = (value, best_explanation)
            return value, best_explanation
        else:
            value = float('inf')
            best_explanation = {}
            for move in self._get_possible_moves(state):
                new_state = self._apply_move(state, move)

                move_value, move_explanation = self.alpha_beta_search(new_state, alpha, beta, True)

                if move_value < value:
                    value = move_value
                    best_explanation = {
                        "move": move,
                        "value": value,
                        "sub_explanation": move_explanation,
                        "type": "min"
                    }

                beta = min(beta, value)
                if alpha >= beta:
                    self.performance_stats["pruning_events"] += 1
                    break

            # Cache the result
            self.transposition_table[state_hash] = (value, best_explanation)
            return value, best_explanation

    def explain_decision(self, explanation: Dict[str, Any]) -> str:
        """Generate human-readable explanation for a decision"""
        if explanation.get("type") == "leaf":
            return f"Position evaluated with score {explanation.get('value')}"

        elif explanation.get("type") == "max":
            return f"Selected move {explanation.get('move')} to maximize score to {explanation.get('value')}"

        elif explanation.get("type") == "min":
            return f"Selected move {explanation.get('move')} to minimize opponent's score to {explanation.get('value')}"

        return "Unknown decision type"

    def counterfactual_analysis(self, state: GameState, alternative_move: Any) -> Dict[str, Any]:
        """What would happen if we made a different move?"""
        # Create a new state with the alternative move
        new_state = self._apply_move(state, alternative_move)

        # Analyze this alternative path
        value, explanation = self.alpha_beta_search(new_state, float('-inf'), float('inf'), False)

        return {
            "alternative_move": alternative_move,
            "counterfactual_value": value,
            "explanation": explanation
        }

    def detect_and_fix_issues(self) -> Dict[str, Any]:
        """Self-debugging capability"""
        if not self.self_correction_active:
            return {"status": "inactive"}

        issues = {"detected": 0, "fixed": 0, "details": []}

        # Check for performance issues
        if self.performance_stats["calls"] > 10000 and self.performance_stats["cache_hits"] / self.performance_stats["calls"] < 0.4:
            issues["detected"] += 1
            # Fix: Reset transposition table if it's not providing good hit ratio
            self.transposition_table = {}
            issues["fixed"] += 1

        return issues
