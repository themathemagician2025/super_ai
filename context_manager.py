"""
Context Manager Module

Tracks session memory and conversation history.
"""

import time
from typing import Dict, List, Any, Optional
from collections import deque

class ContextManager:
    def __init__(self, max_history: int = 10):
        """
        Initialize the context manager.

        Args:
            max_history: Maximum number of interactions to store
        """
        self.max_history = max_history
        self.sessions = {}  # Dictionary to store session data by session_id

    def create_session(self, session_id: str) -> None:
        """
        Create a new session context.

        Args:
            session_id: Unique identifier for the session
        """
        self.sessions[session_id] = {
            "created_at": time.time(),
            "last_active": time.time(),
            "history": deque(maxlen=self.max_history),
            "entities": {},  # Persistent entities across the session
            "preferences": {},  # User preferences
            "stats": {
                "total_interactions": 0,
                "intents": {}  # Count of each intent type
            }
        }

    def add_interaction(self, session_id: str, user_input: Dict[str, Any],
                       system_response: Dict[str, Any]) -> None:
        """
        Add an interaction to the session history.

        Args:
            session_id: Session identifier
            user_input: Parsed user input
            system_response: System response
        """
        if session_id not in self.sessions:
            self.create_session(session_id)

        session = self.sessions[session_id]
        session["last_active"] = time.time()

        # Update statistics
        session["stats"]["total_interactions"] += 1
        intent = user_input.get("intent", "unknown")
        session["stats"]["intents"][intent] = session["stats"]["intents"].get(intent, 0) + 1

        # Add to history
        interaction = {
            "timestamp": time.time(),
            "user_input": user_input,
            "system_response": system_response
        }
        session["history"].append(interaction)

        # Update persistent entities
        if "entities" in user_input and user_input["entities"]:
            for entity_type, entities in user_input["entities"].items():
                if entities:  # Only update if not empty
                    session["entities"][entity_type] = entities

    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """
        Get the full context for a session.

        Args:
            session_id: Session identifier

        Returns:
            Dict containing session context
        """
        if session_id not in self.sessions:
            self.create_session(session_id)

        return self.sessions[session_id]

    def get_history(self, session_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get session history.

        Args:
            session_id: Session identifier
            limit: Maximum number of history items to return

        Returns:
            List of interaction records
        """
        if session_id not in self.sessions:
            return []

        history = list(self.sessions[session_id]["history"])
        if limit is not None:
            history = history[-limit:]

        return history

    def clear_session(self, session_id: str) -> bool:
        """
        Clear a session from memory.

        Args:
            session_id: Session identifier

        Returns:
            True if session was cleared, False if session did not exist
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------
