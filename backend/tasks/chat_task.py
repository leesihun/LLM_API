"""
Normal Chat Task
Simple chat with or without conversation memory
"""

from typing import List, Optional, AsyncIterator
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from backend.config.settings import settings
from backend.models.schemas import ChatMessage
from backend.storage.conversation_store import conversation_store


class ChatTask:
    """Handles normal chat conversations"""

    def __init__(self):
        from backend.utils.llm_factory import LLMFactory
        self.llm = LLMFactory.create_llm()

    async def execute(
        self,
        messages: List[ChatMessage],
        session_id: Optional[str] = None,
        use_memory: bool = True
    ) -> str:
        """
        Execute a simple chat interaction

        Args:
            messages: List of chat messages (current conversation)
            session_id: Optional session ID for conversation history
            use_memory: Whether to use conversation memory

        Returns:
            AI response text
        """
        # Build conversation context
        conversation = self._build_conversation(messages, session_id, use_memory)

        # Generate response
        response = await self.llm.ainvoke(conversation)

        return response.content

    def _build_conversation(
        self,
        messages: List[ChatMessage],
        session_id: Optional[str] = None,
        use_memory: bool = True
    ) -> List:
        """Build conversation context from messages and history"""
        conversation = []

        # Add conversation history if using memory and session exists
        if use_memory and session_id:
            history = conversation_store.get_messages(session_id, limit=1000)

            for msg in history:
                if msg.role == "user":
                    conversation.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    conversation.append(AIMessage(content=msg.content))

        # Add current messages
        for msg in messages:
            if msg.role == "user":
                conversation.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                conversation.append(AIMessage(content=msg.content))

        return conversation


# Global chat task instance
chat_task = ChatTask()
