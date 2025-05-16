# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import logging
from typing import List, Set, Dict, Any
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class State:
    """Represents current state in state space"""
    features: Dict[str, Any]
    score: float = 0.0

@dataclass
class Operator:
    """Represents action operator"""
    name: str
    preconditions: Set[str]
    add_list: Set[str]
    del_list: Set[str]

class RationalAgent:
    """Implements rational agent with state space search"""
    
    def __init__(self):
        self.operators = self._initialize_operators()
        self.op_stack = deque()
        self.state_history = []
        logger.info("Rational agent initialized")

    def _initialize_operators(self) -> List[Operator]:
        """Initialize available operators"""
        return [
            Operator(
                name="analyze_market",
                preconditions={"has_data"},
                add_list={"market_analyzed"},
                del_list=set()
            ),
            Operator(
                name="generate_prediction",
                preconditions={"market_analyzed"},
                add_list={"prediction_ready"},
                del_list=set()
            ),
            # Add more operators
        ]

    def choose_goal(self, goals: Set[str], state: State) -> str:
        for goal in goals:
            if goal not in state.features:
                return goal
        return None

    def select_operator(self, goal: str, state: State) -> Operator:
        for op in self.operators:
            if goal in op.add_list:
                if op.preconditions.issubset(state.features):
                    self.op_stack.append(op)
                    return op
        return None

    def apply_operator(self, operator: Operator, state: State) -> State:
        """Apply operator to current state"""
        try:
            new_state = State(features=state.features.copy())
            
            # Remove deleted features
            for feature in operator.del_list:
                new_state.features.pop(feature, None)
                
            # Add new features
            for feature in operator.add_list:
                new_state.features[feature] = True
                
            self.state_history.append(new_state)
            logger.info(f"Applied operator: {operator.name}")
            return new_state
            
        except Exception as e:
            logger.error(f"Operator application failed: {str(e)}")
            raise