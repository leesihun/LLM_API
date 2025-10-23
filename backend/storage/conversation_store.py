"""
Conversation Storage System
Saves and retrieves conversation history
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

from backend.config.settings import settings
from backend.models.schemas import Conversation, ConversationMessage


class ConversationStore:
    """Manages conversation persistence"""

    def __init__(self):
        self.base_path = Path(settings.conversations_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_conversation_file(self, session_id: str, user_id: str = None, created_at: datetime = None) -> Path:
        """Get file path for a conversation with optional user and timestamp"""
        # Try to load existing conversation to get user_id and created_at if not provided
        if user_id is None or created_at is None:
            # Check if file exists with just session_id (old format)
            old_format_path = self.base_path / f"{session_id}.json"
            if old_format_path.exists():
                return old_format_path

            # Search for file with new format
            for file_path in self.base_path.glob(f"*_{session_id}.json"):
                return file_path

            # If not found, return old format path (for new conversations)
            return old_format_path

        # New format: user_YYYYMMDD_HHMMSS_sessionid.json
        timestamp = created_at.strftime("%Y%m%d_%H%M%S")
        filename = f"{user_id}_{timestamp}_{session_id}.json"
        return self.base_path / filename

    def create_session(self, user_id: str) -> str:
        """Create a new conversation session"""
        session_id = str(uuid.uuid4())

        conversation = Conversation(
            session_id=session_id,
            user_id=user_id,
            messages=[]
        )

        self.save_conversation(conversation)
        return session_id

    def save_conversation(self, conversation: Conversation) -> None:
        """Save conversation to disk"""
        # Update timestamp
        conversation.updated_at = datetime.utcnow()

        # Use new filename format with user_id and created_at
        file_path = self._get_conversation_file(
            conversation.session_id,
            conversation.user_id,
            conversation.created_at
        )

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(conversation.model_dump(mode="json"), f, indent=2, default=str)

    def load_conversation(self, session_id: str) -> Optional[Conversation]:
        """Load conversation from disk"""
        file_path = self._get_conversation_file(session_id)

        if not file_path.exists():
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return Conversation(**data)
        except Exception as e:
            print(f"Error loading conversation {session_id}: {e}")
            return None

    def add_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a message to a conversation"""
        conversation = self.load_conversation(session_id)

        if conversation is None:
            raise ValueError(f"Conversation {session_id} not found")

        message = ConversationMessage(
            role=role,
            content=content,
            metadata=metadata
        )

        conversation.messages.append(message)
        self.save_conversation(conversation)

    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[ConversationMessage]:
        """Get messages from a conversation"""
        conversation = self.load_conversation(session_id)

        if conversation is None:
            return []

        messages = conversation.messages

        if limit:
            messages = messages[-limit:]

        return messages

    def get_user_sessions(self, user_id: str) -> List[str]:
        """Get all session IDs for a user"""
        sessions = []

        for file_path in self.base_path.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("user_id") == user_id:
                        sessions.append(data.get("session_id"))
            except Exception:
                continue

        return sessions

    def delete_conversation(self, session_id: str) -> bool:
        """Delete a conversation"""
        file_path = self._get_conversation_file(session_id)

        if file_path.exists():
            file_path.unlink()
            return True

        return False


# Global conversation store instance
conversation_store = ConversationStore()
