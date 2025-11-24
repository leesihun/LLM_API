"""
Chat Routes
Handles chat completions, conversation management, and OpenAI-compatible endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from typing import Dict, Any, List, Optional
import time
import uuid
from pathlib import Path
import json
import traceback
import asyncio

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
from backend.storage.conversation_store import conversation_store
from backend.tasks.chat_task import chat_task
from backend.tasks.smart_agent_task import smart_agent_task, AgentType
from backend.config.settings import settings
from langchain_core.messages import SystemMessage, HumanMessage
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Router Setup
openai_router = APIRouter(prefix="/v1", tags=["OpenAI Compatible"])
chat_router = APIRouter(prefix="/api/chat", tags=["Chat"])

# Helper Functions

def _save_conversation(
    session_id: str,
    user_message: str,
    response_text: str,
    file_paths: Optional[List[str]],
    agent_type: str,
    agent_metadata: Optional[Dict[str, Any]]
):
    """Save conversation to history"""
    try:
        # Save user message
        conversation_store.add_message(session_id, "user", user_message, metadata={
            "file_paths": file_paths if file_paths else None,
            "agent_type": agent_type
        })

        # Save assistant message with agent metadata if available
        assistant_metadata = {
            "agent_type": agent_type
        }
        if agent_metadata:
            assistant_metadata["agent_metadata"] = agent_metadata

        conversation_store.add_message(session_id, "assistant", response_text, metadata=assistant_metadata)

        logger.info(f"[Chat] Saved conversation to session {session_id} (agent_type: {agent_type})")
        if agent_metadata:
            logger.info(f"[Chat] Agent metadata included: {list(agent_metadata.keys())}")
    except Exception as save_error:
        logger.error(f"[Chat] Failed to save conversation: {save_error}")
        logger.error(f"[Chat] Traceback:\n{traceback.format_exc()}")
        # Don't fail the request if conversation saving fails


def _cleanup_files(file_paths: Optional[List[str]]):
    """Cleanup temporary files"""
    if file_paths:
        for temp_path in file_paths:
            try:
                Path(temp_path).unlink(missing_ok=True)
                logger.debug(f"[Chat] Cleaned up temp file: {temp_path}")
            except Exception as e:
                logger.warning(f"[Chat] Failed to cleanup temp file {temp_path}: {e}")


def _handle_file_uploads(
    files: Optional[List[UploadFile]],
    session_id: Optional[str],
    user_id: str
) -> tuple[List[str], bool]:
    """
    Handle file uploads and retrieve old files from session history
    
    Args:
        files: List of uploaded files (can be None)
        session_id: Current session ID (can be None)
        user_id: User identifier for storage path
    
    Returns:
        Tuple of (file_paths, new_files_uploaded)
        - file_paths: List of file paths to use (new or old)
        - new_files_uploaded: Boolean indicating if new files were uploaded
    """
    file_paths = []
    old_file_paths = []
    new_files_uploaded = False

    # If continuing session, retrieve old files from conversation history
    if session_id:
        try:
            conversation = conversation_store.load_conversation(session_id)
            if conversation:
                # Get file_paths from the most recent user message with files
                for message in reversed(conversation.messages):
                    if message.role == "user" and message.metadata and message.metadata.get("file_paths"):
                        old_file_paths = message.metadata["file_paths"]
                        logger.info(f"[Chat] Retrieved {len(old_file_paths)} old files from session history")
                        break
        except Exception as e:
            logger.warning(f"[Chat] Failed to retrieve old files from session: {e}")

    # Handle new file uploads
    if files:
        uploads_path = Path(settings.uploads_path) / user_id
        uploads_path.mkdir(parents=True, exist_ok=True)

        for file in files:
            try:
                # Save with unique temp ID
                temp_id = uuid.uuid4().hex[:8]
                temp_filename = f"temp_{temp_id}_{file.filename}"
                temp_path = uploads_path / temp_filename

                # Read file content synchronously (FastAPI UploadFile requires this)
                import asyncio
                content = asyncio.run(file.read())
                
                with open(temp_path, "wb") as f:
                    f.write(content)

                file_paths.append(str(temp_path))
                logger.info(f"[Chat] Saved temp file: {temp_filename}")

            except Exception as e:
                logger.error(f"[Chat] Error saving file {file.filename}: {e}")
                continue

        if file_paths:
            new_files_uploaded = True
            logger.info(f"[Chat] Prepared {len(file_paths)} new files for session {session_id}")

            # Clean up old files since we have new ones (replacement strategy)
            if old_file_paths:
                logger.info(f"[Chat] Cleaning up {len(old_file_paths)} old files (replaced by new uploads)")
                _cleanup_files(old_file_paths)
                old_file_paths = []  # Clear old files list after cleanup

    # Use new files if uploaded, otherwise use old files from session
    if not file_paths and old_file_paths:
        file_paths = old_file_paths
        logger.info(f"[Chat] No new files uploaded; using {len(file_paths)} files from session history")

    return file_paths, new_files_uploaded


async def determine_agent_type(query: str, has_files: bool = False) -> str:
    """
    Determine agent type using LLM 3-way classification
    
    Args:
        query: User query to classify
        has_files: Whether files are attached (forces agentic workflow)
    
    Returns:
        One of: "chat", "react", or "plan_execute"
    """
    # If files are attached, force agentic workflow (at minimum react)
    if has_files:
        logger.info("[Agent Classifier] Files attached; forcing agentic workflow (minimum: react)")
        # For files, we still want to determine if it's react or plan_execute
        # So we continue with classification but won't return "chat"
    
    try:
        logger.info(f"[Agent Classifier] Classifying query: {query[:100]}...")
        from backend.utils.llm_factory import LLMFactory
        from backend.config.prompts.task_classification import get_agent_type_classifier_prompt
        
        classifier_llm = LLMFactory.create_classifier_llm()
        messages = [
            SystemMessage(content=get_agent_type_classifier_prompt()),
            HumanMessage(content=f"Query: {query}")
        ]
        
        try:
            response = await asyncio.wait_for(classifier_llm.ainvoke(messages), timeout=10)
            classification = response.content.strip().lower()
            
            # Validate classification
            if classification in ["chat", "react", "plan_execute"]:
                # If files attached, don't allow "chat" classification
                if has_files and classification == "chat":
                    logger.info("[Agent Classifier] Files attached, upgrading 'chat' to 'plan_execute'")
                    classification = "plan_execute"
                
                logger.info(f"[Agent Classifier] Classified as '{classification}': {query[:]}")
                return classification
            else:
                logger.warning(f"[Agent Classifier] Invalid response: '{classification}', using fallback")
        
        except asyncio.TimeoutError:
            logger.warning("[Agent Classifier] LLM classification timed out, using fallback")
        except Exception as llm_error:
            logger.warning(f"[Agent Classifier] LLM error: {str(llm_error)}, using fallback")
    
    except Exception as e:
        logger.error(f"[Agent Classifier] Failed to initialize classifier: {str(e)}, using fallback")
    
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
        session_id = conversation_store.create_session(user_id)

    # ====== PHASE 1: FILE HANDLING ======
    file_paths, new_files_uploaded = _handle_file_uploads(files, session_id, user_id)

    # ====== PHASE 2: CLASSIFICATION ======
    user_message = parsed_messages[-1].content if parsed_messages else ""

    # Save LLM input: parsed_messages into a file
    
    # First create the file
    Path(f"data/scratch/{user_id}").mkdir(parents=True, exist_ok=True)
    serialized_messages = [message.model_dump() for message in parsed_messages]
    with open(f"data/scratch/{user_id}/llm_input_messages_{session_id}.json", "w") as f:
        json.dump(serialized_messages, f, indent=4)

    # Determine agent type (chat/react/plan_execute)
    if agent_type == "auto":
        # Use LLM to classify
        classified_agent_type = await determine_agent_type(user_message, has_files=bool(file_paths))
    else:
        # Use explicitly specified agent type
        classified_agent_type = agent_type.lower()
        logger.info(f"[Agent Selection] Using explicitly specified agent type: {classified_agent_type}")
    
    # Validate agent type
    if classified_agent_type not in ["chat", "react", "plan_execute"]:
        logger.warning(f"[Agent Selection] Invalid agent type '{classified_agent_type}', defaulting to 'chat'")
        classified_agent_type = "chat"
        # # Raise an error
        # raise ValueError(f"Invalid agent type: {classified_agent_type}")

    # ====== PHASE 3: EXECUTION ======
    try:
        agent_metadata = None

        if classified_agent_type == "chat":
            # Use simple chat (no tools)
            logger.info(f"[Agent Execution] Using simple chat for query: {user_message[:]}")
            response_text = await chat_task.execute(
                messages=parsed_messages,
                session_id=session_id,
                use_memory=(session_id is not None)
            )
        elif classified_agent_type == "react":
            # Use ReAct agent
            logger.info(f"[Agent Execution] Using ReAct agent for query: {user_message[:]}")
            response_text, agent_metadata = await smart_agent_task.execute(
                messages=parsed_messages,
                session_id=session_id,
                user_id=user_id,
                agent_type=AgentType.REACT,
                file_paths=file_paths if file_paths else None
            )
        elif classified_agent_type == "plan_execute":
            # Use Plan-and-Execute agent
            logger.info(f"[Agent Execution] Using Plan-and-Execute agent for query: {user_message[:]}")
            response_text, agent_metadata = await smart_agent_task.execute(
                messages=parsed_messages,
                session_id=session_id,
                user_id=user_id,
                agent_type=AgentType.PLAN_EXECUTE,
                file_paths=file_paths if file_paths else None
            )
        else:
            raise ValueError(f"Unknown agent type: {classified_agent_type}")

        # ====== PHASE 4: STORAGE & CLEANUP ======
        # Save to conversation history
        _save_conversation(
            session_id, 
            user_message, 
            response_text, 
            file_paths, 
            classified_agent_type, 
            agent_metadata
        )

        # Cleanup temp files (only clean up newly uploaded files)
        if new_files_uploaded:
            _cleanup_files(file_paths)

        # Build OpenAI-compatible response
        return ChatCompletionResponse(
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


# Chat Management Endpoints

@chat_router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(current_user: Dict[str, Any] = Depends(get_current_user)):
    """List all chat sessions for the current user"""
    user_id = current_user["username"]
    sessions = conversation_store.get_user_sessions(user_id)
    return SessionListResponse(sessions=sessions)


@chat_router.get("/history/{session_id}", response_model=ConversationHistoryResponse)
async def get_history(session_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get conversation history for a specific session"""
    conv = conversation_store.load_conversation(session_id)
    if conv is None or conv.user_id != current_user["username"]:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationHistoryResponse(session_id=session_id, messages=conv.messages)


@chat_router.get("/sessions/{session_id}/artifacts")
async def get_session_artifacts(session_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    List all files generated during a session (charts, reports, code outputs, etc.)

    Returns:
        - session_id: The session identifier
        - artifacts: List of files with metadata (filename, path, size, modified time)
    """
    user_id = current_user["username"]

    # Verify session belongs to user
    conv = conversation_store.load_conversation(session_id)
    if conv is None or conv.user_id != user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    # Check session scratch directory for generated files
    session_dir = Path(settings.python_code_execution_dir) / session_id
    artifacts = []

    if session_dir.exists():
        for file_path in session_dir.rglob("*"):
            if file_path.is_file():
                try:
                    stat = file_path.stat()
                    artifacts.append({
                        "filename": file_path.name,
                        "relative_path": str(file_path.relative_to(session_dir)),
                        "full_path": str(file_path),
                        "size_kb": round(stat.st_size / 1024, 2),
                        "modified": stat.st_mtime,
                        "extension": file_path.suffix.lower()
                    })
                except Exception as e:
                    logger.warning(f"[Artifacts] Failed to stat file {file_path}: {e}")
                    continue

    # Sort by modification time (newest first)
    artifacts.sort(key=lambda x: x["modified"], reverse=True)

    logger.info(f"[Artifacts] Found {len(artifacts)} artifacts for session {session_id}")

    return {
        "session_id": session_id,
        "artifact_count": len(artifacts),
        "artifacts": artifacts
    }
