# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import logging
from dataclasses import dataclass
from collections import deque
from typing import Set, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class State:
    """State representation for rational agent"""
    features: Dict[str, Any]
    score: float = 0.0

@dataclass
class Operator:
    """Action operator for state transformations"""
    name: str
    preconditions: Set[str]
    add_list: Set[str]
    del_list: Set[str]

class RationalAgent:
    """Implements means-ends analysis with state space search"""
    
    def __init__(self):
        self.op_stack = deque()
        self.state_history = []
        self.operators = self._initialize_operators()
        logger.info("Rational agent initialized")

    def process_goal(self, goal: str, current_state: State) -> List[Operator]:
        """Process goal using means-ends analysis"""
        try:
            plan = []
            working_state = current_state
            
            while not self._goal_achieved(goal, working_state):
                operator = self._select_operator(goal, working_state)
                if operator:
                    # Apply operator and update state
                    working_state = self._apply_operator(operator, working_state)
                    plan.append(operator)
                    logger.info(f"Applied operator: {operator.name}")
                else:
                    logger.warning(f"No operator found for goal: {goal}")
                    break
            
            return plan
            
        except Exception as e:
            logger.error(f"Goal processing failed: {str(e)}")
            raise

    def _select_operator(self, goal: str, state: State) -> Operator:
        """Select appropriate operator for goal"""
        for op in self.operators:
            if goal in op.add_list and op.preconditions.issubset(state.features):
                return op
        return None

    def _apply_operator(self, operator: Operator, state: State) -> State:
        """Apply operator to current state"""
        new_state = State(features=state.features.copy())
        
        # Remove deleted features
        for feature in operator.del_list:
            new_state.features.pop(feature, None)
            
        # Add new features
        for feature in operator.add_list:
            new_state.features[feature] = True
            
        self.state_history.append(new_state)
        return new_state