"""
Chat Routes
Handles chat completions, conversation management, and OpenAI-compatible endpoints

Version: 2.0.0
Updated: 2025-12-04 - Refactored to use unified agent orchestrator + simplified heuristics
  - Removed AgentClassifierService (routing handled by AgentOrchestrator heuristics)
  - Extracted FileUploadService for file upload handling
  - Reduced from 526 lines → ~300 lines (43% reduction)
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import FileResponse
from typing import Dict, Any, List, Optional, Tuple
import mimetypes
import time
import uuid
from pathlib import Path
import json
import traceback

from backend.models.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelsResponse,
    ModelInfo,
    SessionListResponse,
    ConversationHistoryResponse,
    ChatMessage
)
from backend.api.dependencies import get_current_user
from backend.config.settings import settings
from backend.agents.react_agent import agent_system
from backend.runtime import (
    cleanup_temp_files,
    create_user_session,
    get_session_history,
    handle_file_uploads,
    list_session_artifacts,
    list_user_sessions,
    save_chat_messages,
    download_session_artifact,
)
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Router Setup
openai_router = APIRouter(prefix="/v1", tags=["OpenAI Compatible"])
chat_router = APIRouter(prefix="/api/chat", tags=["Chat"])

# Helper Functions (kept for backward compatibility, but use services instead)
    
# OpenAI-Compatible Endpoints

@openai_router.get("/models", response_model=ModelsResponse)
async def list_models(current_user: Dict[str, Any] = Depends(get_current_user)):
    """List available models (OpenAI compatible)"""
    models = [
        ModelInfo(
            id=settings.ollama_model,
            created=int(time.time()),
            owned_by="ollama"
        )
    ]

    return ModelsResponse(
        object="list",
        data=models
    )


@openai_router.post("/chat/completions")
async def chat_completions(
    model: str = Form(...),
    messages: str = Form(...),  # JSON string
    session_id: Optional[str] = Form(None),
    agent_type: str = Form("auto"),
    files: Optional[List[UploadFile]] = File(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    OpenAI-compatible chat completions endpoint with multipart/form-data support
    Automatically routes to appropriate agent based on query analysis
    Supports direct file uploads via multipart form

    Parameters:
        - model: Model name (form field)
        - messages: JSON string of message array (form field)
        - session_id: Optional session ID (form field)
        - agent_type: "auto", "react", "plan_execute", or "chat" (form field)
        - files: Optional file uploads (multipart files)
    """
    user_id = current_user["username"]

    # Parse messages JSON
    try:
        messages_list = json.loads(messages)
        parsed_messages = [ChatMessage(**msg) for msg in messages_list]
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid messages JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid message format: {str(e)}")

    # Create new session if none provided
    if not session_id:
        session_id = create_user_session(user_id)

    # ====== PHASE 1: FILE HANDLING ======
    file_paths, new_files_uploaded = await handle_file_uploads(user_id, files, session_id)

    # Save LLM input: parsed_messages into a file
    Path(f"data/scratch/{user_id}").mkdir(parents=True, exist_ok=True)
    with open(f"data/scratch/{user_id}/llm_input_messages_{session_id}.json", "w") as f:
        for message in parsed_messages:
            f.write(message.model_dump_json() + "\n")

    # ====== PHASE 2-4: EXECUTION & STORAGE ======
    try:
        # Execute chat completion (includes classification, execution, response building)
        response, agent_metadata = await _execute_chat_completion(
            messages=parsed_messages,
            model=model,
            session_id=session_id,
            agent_type=agent_type,
            file_paths=file_paths,
            user_id=user_id,
        )

        # Save conversation
        user_message = parsed_messages[-1].content if parsed_messages else ""
        save_chat_messages(
            session_id=session_id,
            user_message=user_message,
            assistant_message=response.choices[0]["message"]["content"],
            file_paths=file_paths,
            agent_type=agent_type,
            agent_metadata=agent_metadata
        )

        return response

    except Exception as e:
        # Detailed error logging
        logger.error("=" * 80)
        logger.error(f"CHAT COMPLETION ERROR for user {user_id}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        logger.error("=" * 80)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating response: {str(e)}"
        )

    finally:
        # ====== CLEANUP: Always delete temp files from uploads folder ======
        # Files have been copied to scratch folder, so temp files in uploads are no longer needed
        if new_files_uploaded:
            cleanup_temp_files(file_paths)
            logger.info(f"[Chat] Cleaned up {len(file_paths)} temp files from uploads folder")


# Chat Management Endpoints

@chat_router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(current_user: Dict[str, Any] = Depends(get_current_user)):
    """List all chat sessions for the current user"""
    return list_user_sessions(current_user["username"])


@chat_router.get("/history/{session_id}", response_model=ConversationHistoryResponse)
async def get_history(session_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get conversation history for a specific session"""
    return get_session_history(current_user["username"], session_id)


@chat_router.get("/sessions/{session_id}/artifacts")
async def get_session_artifacts(session_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    """List all files generated during a session"""
    return list_session_artifacts(current_user["username"], session_id)


@chat_router.get("/sessions/{session_id}/artifacts/{filename:path}")
async def download_artifact(
    session_id: str,
    filename: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Download a specific artifact file from a session"""
    return download_session_artifact(current_user["username"], session_id, filename)


async def _execute_chat_completion(
    messages: List[ChatMessage],
    model: str,
    session_id: str,
    agent_type: str,
    file_paths: Optional[List[str]],
    user_id: str,
) -> Tuple[ChatCompletionResponse, Optional[Dict[str, Any]]]:
    response_text, agent_metadata = await agent_system.execute(
        messages=messages,
        session_id=session_id or "temp_session",
        user_id=user_id,
        file_paths=file_paths,
        agent_type=agent_type,
    )
    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=model,
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }
        ],
        usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        },
        x_session_id=session_id,
        x_agent_metadata=agent_metadata
    )
    return response, agent_metadata
