"""
LLM Manager Utility
==================
Consolidates LLM initialization patterns across agents.

Eliminates duplicate code for user_id-based LLM creation and component initialization.
"""

from typing import Optional
from backend.utils.llm_factory import LLMFactory
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class LLMManager:
    """Manages LLM instances with user_id tracking and component initialization."""

    def __init__(self, user_id: str = "default"):
        """
        Initialize LLM manager.

        Args:
            user_id: Initial user ID
        """
        self.user_id = user_id
        self.llm = LLMFactory.create_llm(user_id=user_id)

    def ensure_user_id(self, user_id: str):
        """
        Ensure LLM is created/updated for the given user_id.

        Args:
            user_id: User ID to ensure LLM is configured for
        """
        if user_id != self.user_id:
            self.user_id = user_id
            self.llm = LLMFactory.create_llm(user_id=user_id)
            logger.debug(f"[LLMManager] Updated LLM for user_id: {user_id}")


class SimpleLLMManager:
    """Simplified LLM manager for basic use cases."""

    def __init__(self):
        """Initialize simple LLM manager."""
        self.llm = None
        self.current_user_id = None

    def get_llm(self, user_id: str = "default"):
        """
        Get or create LLM for user_id.

        Args:
            user_id: User ID

        Returns:
            LLM instance
        """
        if self.llm is None or self.current_user_id != user_id:
            self.llm = LLMFactory.create_llm(user_id=user_id)
            self.current_user_id = user_id
        return self.llm

