# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

@dataclass
class GameState:
    board: List[any]
    score: float
    depth: int

class MinMaxSearcher:
    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        
    def alpha_beta_search(self, state: GameState, alpha: float, beta: float, 
                         maximizing: bool) -> float:
        if state.depth >= self.max_depth or self._is_terminal(state):
            return self._evaluate(state)
            
        if maximizing:
            value = float('-inf')
            for move in self._get_possible_moves(state):
                new_state = self._apply_move(state, move)
                value = max(value, self.alpha_beta_search(new_state, alpha, beta, False))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = float('inf')
            for move in self._get_possible_moves(state):
                new_state = self._apply_move(state, move)
                value = min(value, self.alpha_beta_search(new_state, alpha, beta, True))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value