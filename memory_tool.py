# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Memory Tool Module

Provides long-term memory storage using ChromaDB.
"""

import os
import logging
import json
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

class MemoryTool:
    """Tool for storing and retrieving conversation history using ChromaDB."""

    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the memory tool with ChromaDB.

        Args:
            persist_directory: Directory to store ChromaDB data
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        # Create client with persistence
        try:
            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=persist_directory
            ))

            # Create collections for different memory types
            self.conversation_collection = self._get_or_create_collection("conversations")
            self.entities_collection = self._get_or_create_collection("entities")
            self.facts_collection = self._get_or_create_collection("facts")

            logger.info(f"ChromaDB initialized with persistence at {persist_directory}")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            # Fallback to in-memory client
            self.client = chromadb.Client()
            self.conversation_collection = self._get_or_create_collection("conversations")
            self.entities_collection = self._get_or_create_collection("entities")
            self.facts_collection = self._get_or_create_collection("facts")
            logger.warning("Using in-memory ChromaDB client as fallback")

    def _get_or_create_collection(self, name: str):
        """
        Get or create a ChromaDB collection.

        Args:
            name: Collection name

        Returns:
            ChromaDB collection
        """
        try:
            return self.client.get_or_create_collection(name=name)
        except Exception as e:
            logger.error(f"Error creating collection {name}: {str(e)}")
            raise

    def store_conversation(self,
                         session_id: str,
                         user_input: str,
                         system_response: str,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a conversation turn in ChromaDB.

        Args:
            session_id: Session identifier
            user_input: User's input text
            system_response: System's response text
            metadata: Additional metadata

        Returns:
            Record ID of the stored conversation
        """
        try:
            # Create default metadata if none provided
            if metadata is None:
                metadata = {}

            # Add timestamp and session_id to metadata
            metadata.update({
                "timestamp": time.time(),
                "session_id": session_id
            })

            # Combine input and response for embedding
            combined_text = f"User: {user_input}\nSystem: {system_response}"

            # Generate a unique ID for this conversation turn
            record_id = str(uuid.uuid4())

            # Add to collection
            self.conversation_collection.add(
                documents=[combined_text],
                metadatas=[metadata],
                ids=[record_id]
            )

            logger.info(f"Stored conversation with ID {record_id} for session {session_id}")
            return record_id

        except Exception as e:
            logger.error(f"Error storing conversation: {str(e)}")
            return ""

    def store_entity(self,
                   entity_type: str,
                   entity_name: str,
                   entity_data: Dict[str, Any],
                   session_id: Optional[str] = None) -> str:
        """
        Store entity information in ChromaDB.

        Args:
            entity_type: Type of entity (person, place, etc.)
            entity_name: Name of the entity
            entity_data: Data about the entity
            session_id: Optional session identifier

        Returns:
            Record ID of the stored entity
        """
        try:
            # Serialize entity data to string
            entity_json = json.dumps(entity_data)

            # Create metadata
            metadata = {
                "entity_type": entity_type,
                "entity_name": entity_name,
                "timestamp": time.time()
            }

            if session_id:
                metadata["session_id"] = session_id

            # Generate a unique ID or use entity name
            record_id = f"{entity_type}_{entity_name}_{uuid.uuid4()}"

            # Add to collection
            self.entities_collection.add(
                documents=[entity_json],
                metadatas=[metadata],
                ids=[record_id]
            )

            logger.info(f"Stored entity {entity_name} of type {entity_type} with ID {record_id}")
            return record_id

        except Exception as e:
            logger.error(f"Error storing entity: {str(e)}")
            return ""

    def store_fact(self,
                 fact: str,
                 source: Optional[str] = None,
                 entities: Optional[List[str]] = None,
                 session_id: Optional[str] = None) -> str:
        """
        Store a factual statement in ChromaDB.

        Args:
            fact: The factual statement
            source: Source of the fact
            entities: Related entities
            session_id: Optional session identifier

        Returns:
            Record ID of the stored fact
        """
        try:
            # Create metadata
            metadata = {
                "timestamp": time.time()
            }

            if source:
                metadata["source"] = source

            if entities:
                metadata["entities"] = json.dumps(entities)

            if session_id:
                metadata["session_id"] = session_id

            # Generate a unique ID
            record_id = str(uuid.uuid4())

            # Add to collection
            self.facts_collection.add(
                documents=[fact],
                metadatas=[metadata],
                ids=[record_id]
            )

            logger.info(f"Stored fact with ID {record_id}")
            return record_id

        except Exception as e:
            logger.error(f"Error storing fact: {str(e)}")
            return ""

    def query_conversations(self,
                          query: str,
                          session_id: Optional[str] = None,
                          limit: int = 5) -> List[Dict[str, Any]]:
        """
        Query conversation history.

        Args:
            query: Search query
            session_id: Optional session filter
            limit: Maximum number of results

        Returns:
            List of conversation matches
        """
        try:
            # Prepare where clause if session_id is provided
            where_clause = {"session_id": session_id} if session_id else None

            # Query the collection
            results = self.conversation_collection.query(
                query_texts=[query],
                n_results=limit,
                where=where_clause
            )

            # Format results
            formatted_results = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    formatted_results.append({
                        "text": doc,
                        "metadata": metadata,
                        "id": results["ids"][0][i] if results["ids"] else "",
                        "distance": results["distances"][0][i] if "distances" in results and results["distances"] else None
                    })

            logger.info(f"Found {len(formatted_results)} conversation matches for query '{query}'")
            return formatted_results

        except Exception as e:
            logger.error(f"Error querying conversations: {str(e)}")
            return []

    def query_entities(self,
                     query: str,
                     entity_type: Optional[str] = None,
                     limit: int = 5) -> List[Dict[str, Any]]:
        """
        Query stored entities.

        Args:
            query: Search query
            entity_type: Optional entity type filter
            limit: Maximum number of results

        Returns:
            List of entity matches
        """
        try:
            # Prepare where clause if entity_type is provided
            where_clause = {"entity_type": entity_type} if entity_type else None

            # Query the collection
            results = self.entities_collection.query(
                query_texts=[query],
                n_results=limit,
                where=where_clause
            )

            # Format results
            formatted_results = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}

                    try:
                        # Parse the entity data from JSON
                        entity_data = json.loads(doc)
                    except json.JSONDecodeError:
                        entity_data = {}

                    formatted_results.append({
                        "entity_data": entity_data,
                        "metadata": metadata,
                        "id": results["ids"][0][i] if results["ids"] else "",
                        "distance": results["distances"][0][i] if "distances" in results and results["distances"] else None
                    })

            logger.info(f"Found {len(formatted_results)} entity matches for query '{query}'")
            return formatted_results

        except Exception as e:
            logger.error(f"Error querying entities: {str(e)}")
            return []

    def get_session_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get conversation history for a specific session.

        Args:
            session_id: Session identifier
            limit: Maximum number of results

        Returns:
            List of conversation turns
        """
        try:
            # Query conversations for this session
            results = self.conversation_collection.get(
                where={"session_id": session_id},
                limit=limit
            )

            # Format results
            formatted_results = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"]):
                    metadata = results["metadatas"][i] if results["metadatas"] else {}
                    formatted_results.append({
                        "text": doc,
                        "metadata": metadata,
                        "id": results["ids"][i] if results["ids"] else ""
                    })

            # Sort by timestamp if available
            formatted_results.sort(
                key=lambda x: x.get("metadata", {}).get("timestamp", 0)
            )

            logger.info(f"Retrieved {len(formatted_results)} history items for session {session_id}")
            return formatted_results

        except Exception as e:
            logger.error(f"Error getting session history: {str(e)}")
            return []

# Create an instance for use in tool_router
memory_tool = MemoryTool()

def memory_tool_handler(user_input: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle memory-related requests.

    Args:
        user_input: Parsed user input
        context: Session context

    Returns:
        Memory operation result
    """
    raw_input = user_input.get("raw_input", "")
    session_id = user_input.get("session_id", "")

    # Extract intent from context
    intent = user_input.get("intent", "unknown")

    # Store current interaction in memory
    if "raw_response" in context:
        system_response = context["raw_response"].get("content", "")
        memory_tool.store_conversation(
            session_id,
            raw_input,
            system_response
        )

    # Handle memory-specific intents
    if "remember" in raw_input.lower() or "recall" in raw_input.lower():
        # Query conversation history
        query = raw_input.replace("remember", "").replace("recall", "").strip()
        results = memory_tool.query_conversations(query, session_id=session_id)

        return {
            "title": "Memory Recall",
            "status": "success",
            "items": results
        }

    # Default: just store the conversation and return success
    return {
        "status": "success",
        "content": "Memory updated"
    }

if __name__ == "__main__":
    # Test the memory tool
    tool = MemoryTool()
    session_id = str(uuid.uuid4())

    # Store a test conversation
    tool.store_conversation(
        session_id,
        "What is the weather today?",
        "The weather is sunny with a high of 75°F."
    )

    # Query the conversation
    results = tool.query_conversations("weather", session_id)
    print(json.dumps(results, indent=2))
