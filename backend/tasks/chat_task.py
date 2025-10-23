"""
Normal Chat Task
Simple chat with or without conversation memory
"""

from typing import List, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from backend.config.settings import settings
from backend.models.schemas import ChatMessage
from backend.storage.conversation_store import conversation_store


class ChatTask:
    """Handles normal chat conversations"""

    def __init__(self):
        import httpx
        # Use AsyncClient for async operations
        async_client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.ollama_timeout / 1000, connect=60.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )

        self.llm = ChatOllama(
            base_url=settings.ollama_host,
            model=settings.ollama_model,
            temperature=settings.ollama_temperature,
            num_ctx=settings.ollama_num_ctx,
            top_p=settings.ollama_top_p,
            top_k=settings.ollama_top_k,
            timeout=settings.ollama_timeout / 1000,  # Convert ms to seconds
            async_client=async_client
        )

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
        conversation = []

        # Add conversation history if using memory and session exists
        if use_memory and session_id:
            history = conversation_store.get_messages(session_id, limit=10)

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
            elif msg.role == "system":
                conversation.insert(0, SystemMessage(content=msg.content))

        # Generate response
        response = await self.llm.ainvoke(conversation)

        return response.content


# Global chat task instance
chat_task = ChatTask()
