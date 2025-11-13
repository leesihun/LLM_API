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
from backend.utils.auth import get_current_user
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

async def determine_task_type(query: str) -> str:
    """Determine task type using LLM analysis, fallback to keyword matching"""
    try:
        logger.info(f"[Task Classifier] Determining task type for query: {query[:]}")
        from backend.utils.llm_factory import LLMFactory
        classifier_llm = LLMFactory.create_classifier_llm()
        messages = [
            SystemMessage(content=settings.agentic_classifier_prompt),
            HumanMessage(content=f"Query: {query}")
        ]
        try:
            response = await asyncio.wait_for(classifier_llm.ainvoke(messages), timeout=100)
            classification = response.content.strip().lower()
            if classification in ["agentic", "chat"]:
                logger.info(f"Task classified as '{classification.upper()}': {query[:100]}")
                return classification
            else:
                logger.warning(f"Invalid classifier response: '{classification}', using fallback")
        except asyncio.TimeoutError:
            logger.warning("[Task Classifier] LLM classification timed out, falling back to keywords")
        except Exception as llm_error:
            logger.warning(f"[Task Classifier] LLM classification error: {str(llm_error)}, falling back to keywords")
    except Exception as e:
        logger.error(f"[Task Classifier] Failed to initialize classifier: {str(e)}, falling back to keywords")

    # Fallback: Keyword-based classification
    logger.info("[Task Classifier] Using keyword-based fallback classification")
    agentic_keywords = ["search", "find", "look up", "research", "compare", "analyze",
                       "investigate", "current", "latest", "news", "document", "file",
                       "pdf", "code", "python", "calculate", "data"]
    if any(keyword in query.lower() for keyword in agentic_keywords):
        logger.info(f"[Task Classifier] Fallback classified as 'agentic': {query[:]}")
        return "agentic"
    logger.info(f"[Task Classifier] Fallback classified as 'chat': {query[:]}")
    return "chat"

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


@openai_router.post("/chat/completions", response_model=ChatCompletionResponse)
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
    Automatically routes to appropriate task based on query analysis
    Supports direct file uploads via multipart form

    Parameters:
        - model: Model name (form field)
        - messages: JSON string of message array (form field)
        - session_id: Optional session ID (form field)
        - agent_type: "auto", "react", or "plan_execute" (form field)
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

    # Handle file uploads - save to temp location
    file_paths = []
    if files:
        uploads_path = Path(settings.uploads_path) / user_id
        uploads_path.mkdir(parents=True, exist_ok=True)

        for file in files:
            try:
                # Save with unique temp ID
                temp_id = uuid.uuid4().hex[:8]
                temp_filename = f"temp_{temp_id}_{file.filename}"
                temp_path = uploads_path / temp_filename

                with open(temp_path, "wb") as f:
                    content = await file.read()
                    f.write(content)

                file_paths.append(str(temp_path))
                logger.info(f"[Chat] Saved temp file: {temp_filename}")

            except Exception as e:
                logger.error(f"[Chat] Error saving file {file.filename}: {e}")
                # Continue with other files
                continue

        if file_paths:
            logger.info(f"[Chat] Prepared {len(file_paths)} files for session {session_id}")

    # Determine task type based on query
    user_message = parsed_messages[-1].content if parsed_messages else ""
    task_type = await determine_task_type(user_message)

    # If files are attached, force agentic workflow to ensure file handling
    if file_paths:
        logger.info("[Chat] Files attached; forcing agentic workflow")
        task_type = "agentic"

    # Execute appropriate task
    try:
        agent_metadata = None

        if task_type == "agentic":
            # Use smart agent (auto-selects ReAct or Plan-and-Execute)
            logger.info(f"[Task Classifier] Using agentic worflow for query: {user_message[:]}")
            selected_agent_type = AgentType(agent_type) if agent_type else AgentType.AUTO
            response_text, agent_metadata = await smart_agent_task.execute(
                messages=parsed_messages,
                session_id=session_id,
                user_id=user_id,
                agent_type=selected_agent_type,
                file_paths=file_paths if file_paths else None
            )
        else:
            # Use simple chat
            response_text = await chat_task.execute(
                messages=parsed_messages,
                session_id=session_id,
                use_memory=(session_id is not None)
            )

        # Save to conversation history with metadata
        try:
            # Save user message
            conversation_store.add_message(session_id, "user", user_message, metadata={
                "file_paths": file_paths if file_paths else None,
                "task_type": task_type
            })

            # Save assistant message with agent metadata if available
            assistant_metadata = {
                "task_type": task_type
            }
            if agent_metadata:
                assistant_metadata["agent_metadata"] = agent_metadata

            conversation_store.add_message(session_id, "assistant", response_text, metadata=assistant_metadata)

            logger.info(f"[Chat] Saved conversation to session {session_id} (task_type: {task_type})")
            if agent_metadata:
                logger.info(f"[Chat] Agent metadata included: {list(agent_metadata.keys())}")
        except Exception as save_error:
            logger.error(f"[Chat] Failed to save conversation: {save_error}")
            logger.error(f"[Chat] Traceback:\n{traceback.format_exc()}")
            # Don't fail the request if conversation saving fails

        # Cleanup temp files after execution
        if file_paths:
            for temp_path in file_paths:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                    logger.debug(f"[Chat] Cleaned up temp file: {temp_path}")
                except Exception as e:
                    logger.warning(f"[Chat] Failed to cleanup temp file {temp_path}: {e}")

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
