"""
Runtime Helpers
================
Stateless helpers for sessions, file uploads, artifacts, and user management.

Replacing the previous service layer keeps the architecture flat while still
providing focused functions for each responsibility.
"""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from backend.config.settings import settings
from backend.models.schemas import (
    ConversationHistoryResponse,
    SessionListResponse,
    SignupRequest,
    SignupResponse,
    User,
)
from backend.storage.conversation_store import conversation_store
from backend.utils.auth import (
    authenticate_user,
    create_access_token,
    hash_password,
)
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

UPLOAD_ROOT = Path(settings.uploads_path)
SCRATCH_ROOT = Path(settings.python_code_execution_dir)
USERS_FILE = Path(settings.users_path)


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------


def create_user_session(user_id: str) -> str:
    """Create a new chat session for the given user."""
    session_id = conversation_store.create_session(user_id)
    logger.info(f"Created session {session_id} for {user_id}")
    return session_id


def list_user_sessions(user_id: str) -> SessionListResponse:
    """List all sessions for a user."""
    sessions = conversation_store.get_user_sessions(user_id)
    return SessionListResponse(sessions=sessions)


def get_session_history(user_id: str, session_id: str) -> ConversationHistoryResponse:
    """Return conversation history after verifying session ownership."""
    conv = conversation_store.load_conversation(session_id)
    if conv is None or conv.user_id != user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationHistoryResponse(session_id=session_id, messages=conv.messages)


def save_chat_messages(
    session_id: str,
    user_message: str,
    assistant_message: str,
    file_paths: Optional[List[str]],
    agent_type: str,
    agent_metadata: Optional[Dict[str, Any]],
) -> None:
    """Persist the latest user/assistant exchange."""
    try:
        conversation_store.add_message(
            session_id,
            "user",
            user_message,
            metadata={
                "file_paths": file_paths or None,
                "agent_type": agent_type,
            },
        )
        assistant_meta = {"agent_type": agent_type}
        if agent_metadata:
            assistant_meta["agent_metadata"] = agent_metadata
        conversation_store.add_message(
            session_id,
            "assistant",
            assistant_message,
            metadata=assistant_meta,
        )
    except Exception as exc:  # pragma: no cover - logging only
        logger.error(f"Failed to save conversation for {session_id}: {exc}")


# ---------------------------------------------------------------------------
# File upload helpers
# ---------------------------------------------------------------------------


def extract_original_filename(file_path: str) -> str:
    """Return the base filename (no prefix handling)."""
    return Path(file_path).name


async def handle_file_uploads(
    user_id: str,
    files: Optional[List[UploadFile]],
    session_id: Optional[str],
) -> Tuple[List[str], List[str], bool]:
    """
    Store uploaded files, copy them into the session scratch dir, and return paths.

    Returns:
        - scratch_paths: Paths inside the per-session scratch directory
        - uploaded_paths: Paths in the user uploads directory (for cleanup)
        - new_files_uploaded: Whether new files were provided in this request
    """
    uploads_dir = UPLOAD_ROOT / user_id
    scratch_dir = SCRATCH_ROOT / (session_id or "temp_session")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    old_files = _load_previous_file_paths(session_id)
    if files:
        # Remove previous uploads before saving new ones to avoid overwriting collisions
        if old_files:
            cleanup_uploaded_files(old_files)

        scratch_paths: List[str] = []
        upload_paths: List[str] = []
        for file in files:
            safe_name = Path(file.filename).name

            content = await file.read()

            upload_target = uploads_dir / safe_name
            scratch_target = scratch_dir / safe_name

            upload_target.write_bytes(content)
            scratch_target.write_bytes(content)

            upload_paths.append(str(upload_target))
            scratch_paths.append(str(scratch_target))

            logger.info(f"[Uploads] Saved {upload_target.name} -> {scratch_target}")

        return scratch_paths, upload_paths, True

    return (old_files, [], False) if old_files else ([], [], False)


def cleanup_uploaded_files(file_paths: Optional[List[str]]) -> None:
    """Delete uploaded files, ignoring missing paths."""
    if not file_paths:
        return
    for upload_path in file_paths:
        try:
            Path(upload_path).unlink(missing_ok=True)
        except Exception as exc:  # pragma: no cover - logging only
            logger.warning(f"Failed to cleanup {upload_path}: {exc}")


def _load_previous_file_paths(session_id: Optional[str]) -> List[str]:
    """Read the most recent set of file paths from session history."""
    if not session_id:
        return []
    try:
        conversation = conversation_store.load_conversation(session_id)
        if not conversation:
            return []
        for message in reversed(conversation.messages):
            metadata = message.metadata or {}
            file_paths = metadata.get("file_paths")
            if message.role == "user" and file_paths:
                return file_paths
    except Exception as exc:  # pragma: no cover - logging only
        logger.warning(f"Failed to load previous file paths: {exc}")
    return []


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------


def list_session_artifacts(user_id: str, session_id: str) -> Dict[str, Any]:
    """Return metadata for every file generated inside the session scratch dir."""
    _ensure_session_access(user_id, session_id)
    session_dir = SCRATCH_ROOT / session_id
    artifacts: List[Dict[str, Any]] = []
    if session_dir.exists():
        for file_path in session_dir.rglob("*"):
            if file_path.is_file():
                stat = file_path.stat()
                artifacts.append(
                    {
                        "filename": file_path.name,
                        "relative_path": str(file_path.relative_to(session_dir)),
                        "full_path": str(file_path),
                        "size_kb": round(stat.st_size / 1024, 2),
                        "modified": stat.st_mtime,
                        "extension": file_path.suffix.lower(),
                    }
                )
    artifacts.sort(key=lambda x: x["modified"], reverse=True)
    return {
        "session_id": session_id,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def download_session_artifact(
    user_id: str,
    session_id: str,
    filename: str,
) -> FileResponse:
    """Return a FileResponse for a given artifact after verifying ownership."""
    _ensure_session_access(user_id, session_id)
    session_dir = SCRATCH_ROOT / session_id
    file_path = (session_dir / filename).resolve()
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if session_dir.resolve() not in file_path.parents and file_path.parent != session_dir.resolve():
        raise HTTPException(status_code=403, detail="Access denied")

    content_type, _ = mimetypes.guess_type(str(file_path))
    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type=content_type or "application/octet-stream",
    )


def _ensure_session_access(user_id: str, session_id: str) -> None:
    conv = conversation_store.load_conversation(session_id)
    if conv is None or conv.user_id != user_id:
        raise HTTPException(status_code=404, detail="Session not found")


# ---------------------------------------------------------------------------
# User management helpers
# ---------------------------------------------------------------------------


def authenticate_credentials(username: str, password: str) -> Dict[str, Any]:
    """Authenticate and return token payload."""
    user = authenticate_user(username, password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user["username"]})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user,
    }


def create_user_account(request: SignupRequest) -> SignupResponse:
    """Create a new user with hashed password."""
    if len(request.username.strip()) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters long")
    if len(request.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters long")

    data = {"users": []}
    if USERS_FILE.exists():
        with open(USERS_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)

    if any(u.get("username") == request.username for u in data.get("users", [])):
        raise HTTPException(status_code=400, detail="Username already exists")

    new_user = {
        "username": request.username,
        "password_hash": hash_password(request.password),
        "role": request.role or "guest",
    }
    data.setdefault("users", []).append(new_user)
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)

    return SignupResponse(
        success=True,
        user=User(username=new_user["username"], role=new_user["role"]),
    )


__all__ = [
    "authenticate_credentials",
    "create_user_account",
    "create_user_session",
    "list_user_sessions",
    "get_session_history",
    "save_chat_messages",
    "handle_file_uploads",
    "cleanup_uploaded_files",
    "extract_original_filename",
    "list_session_artifacts",
    "download_session_artifact",
]

