"""
Session management endpoints
/api/chat/sessions - List user sessions
/api/chat/history/{session_id} - Get conversation history
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional

from backend.models.schemas import SessionsListResponse, SessionInfo, ChatHistoryResponse, ChatMessage
from backend.core.database import db, conversation_store
from backend.utils.auth import get_current_user, get_optional_user

router = APIRouter(prefix="/api/chat", tags=["sessions"])


@router.get("/sessions", response_model=SessionsListResponse)
def list_sessions(current_user: Optional[dict] = Depends(get_optional_user)):
    """
    List all sessions for the current user

    Returns:
        List of session metadata
    """
    # Use authenticated user or default to "guest"
    username = current_user["username"] if current_user else "guest"

    # Get sessions from database
    sessions = db.list_user_sessions(username)

    # Convert to response format
    session_infos = [
        SessionInfo(
            session_id=session["id"],
            created_at=session["created_at"],
            message_count=session["message_count"]
        )
        for session in sessions
    ]

    return SessionsListResponse(sessions=session_infos)


@router.get("/history/{session_id}", response_model=ChatHistoryResponse)
def get_history(
    session_id: str,
    current_user: Optional[dict] = Depends(get_optional_user)
):
    """
    Get conversation history for a specific session

    Args:
        session_id: Session ID

    Returns:
        List of messages in the conversation
    """
    # Verify session exists
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Verify user owns this session (if authenticated)
    # Allow access if:
    # 1. User is not authenticated (guest can access any session for now)
    # 2. User is authenticated AND owns the session
    # 3. User is authenticated AND session belongs to "guest"
    if current_user:
        # User is authenticated - check ownership
        if session["username"] != current_user["username"] and session["username"] != "guest":
            raise HTTPException(status_code=403, detail="Access denied")

    # Load conversation history
    messages = conversation_store.load_conversation(session_id)
    if messages is None:
        messages = []

    # Convert to Pydantic models
    chat_messages = [ChatMessage(**msg) for msg in messages]

    return ChatHistoryResponse(messages=chat_messages)
