# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

@dataclass
class GameState:
    """Represents a game state for MinMax search"""
    features: np.ndarray
    depth: int
    score: float = 0.0

class MinMaxSearcher:
    """MinMax search with Alpha-Beta pruning for game analysis"""
    
    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self.nodes_explored = 0
        logger.info(f"MinMax searcher initialized with depth {max_depth}")

    def search(self, state: GameState) -> Tuple[float, List[int]]:
        """Perform alpha-beta search from current state"""
        self.nodes_explored = 0
        alpha = float('-inf')
        beta = float('inf')
        
        best_value = float('-inf')
        best_move = None
        
        for move in self._get_valid_moves(state):
            new_state = self._apply_move(state, move)
            value = self._alpha_beta(new_state, alpha, beta, False)
            
            if value > best_value:
                best_value = value
                best_move = move
                
            alpha = max(alpha, best_value)
            if alpha >= beta:
                break
                
        logger.info(f"Search complete. Nodes explored: {self.nodes_explored}")
        return best_value, best_move

    def _alpha_beta(self, state: GameState, alpha: float, beta: float, 
                   maximizing: bool) -> float:
        """Alpha-beta pruning implementation"""
        self.nodes_explored += 1
        
        if state.depth >= self.max_depth or self._is_terminal(state):
            return self._evaluate_position(state)
            
        if maximizing:
            value = float('-inf')
            for move in self._get_valid_moves(state):
                new_state = self._apply_move(state, move)
                value = max(value, self._alpha_beta(new_state, alpha, beta, False))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = float('inf')
            for move in self._get_valid_moves(state):
                new_state = self._apply_move(state, move)
                value = min(value, self._alpha_beta(new_state, alpha, beta, True))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value