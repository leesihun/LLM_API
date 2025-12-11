from typing import List, Optional, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from backend.storage.conversation_store import conversation_store
from backend.models.schemas import ChatMessage

class ConversationLoader:
    """
    Centralized conversation history loading utilities.
    """

    @staticmethod
    def load_as_dicts(
        session_id: Optional[str],
        limit: int = 10
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Load conversation as list of dicts.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of message dictionaries or None
        """
        if not session_id:
            return None

        messages = conversation_store.get_messages(session_id, limit=limit)
        if not messages:
            return None

        return [
            {
                'role': msg.role,
                'content': msg.content,
                'timestamp': msg.timestamp.isoformat() if msg.timestamp else ""
            }
            for msg in messages
        ]

    @staticmethod
    def load_as_langchain_messages(
        session_id: Optional[str],
        limit: int = 1000
    ) -> List[BaseMessage]:
        """
        Load conversation as LangChain message objects.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of LangChain BaseMessage objects
        """
        if not session_id:
            return []

        messages = conversation_store.get_messages(session_id, limit=limit)

        result = []
        for msg in messages:
            if msg.role == "user":
                result.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                result.append(AIMessage(content=msg.content))

        return result

