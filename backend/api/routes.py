"""
API Routes
OpenAI-compatible endpoints and authentication
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from typing import Dict, Any, List, Optional
import time
import uuid
from pathlib import Path

from backend.models.schemas import (
    LoginRequest,
    LoginResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelsResponse,
    ModelInfo,
    User,
    SignupRequest,
    SignupResponse,
    ModelChangeRequest,
    ModelChangeResponse,
    SessionListResponse,
    ConversationHistoryResponse,
    ToolListResponse,
    ToolInfo,
    MathRequest,
    MathResponse,
    WebSearchRequest,
    WebSearchResponse,
    RAGSearchResponse
)
from backend.utils.auth import authenticate_user, create_access_token, get_current_user
from backend.storage.conversation_store import conversation_store
from backend.tasks.chat_task import chat_task
from backend.tasks.Plan_execute import PlanExecuteTask
from backend.tasks.smart_agent_task import smart_agent_task, AgentType
from backend.tools.rag_retriever import rag_retriever
from backend.tools.web_search import web_search_tool
from backend.config.settings import settings
import logging
import traceback
import json
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import asyncio

logger = logging.getLogger(__name__)


# ============================================================================
# Router Setup
# ============================================================================

auth_router = APIRouter(prefix="/api/auth", tags=["Authentication"])
openai_router = APIRouter(prefix="/v1", tags=["OpenAI Compatible"])
files_router = APIRouter(prefix="/api/files", tags=["File Management"])
admin_router = APIRouter(prefix="/api/admin", tags=["Admin"])
tools_router = APIRouter(prefix="/api/tools", tags=["Tools"])
chat_router = APIRouter(prefix="/api/chat", tags=["Chat"])


# ============================================================================
# Authentication Endpoints
# ============================================================================

@auth_router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Authenticate user and return access token"""
    try:
        logger.info(f"Login attempt for user: {request.username}")
        user = authenticate_user(request.username, request.password)

        if user is None:
            logger.warning(f"Failed login attempt for user: {request.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Create access token
        access_token = create_access_token(data={"sub": user["username"]})
        logger.info(f"Successful login for user: {request.username}")

        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            user=user
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login error: {str(e)}"
        )


@auth_router.get("/me", response_model=User)
async def get_me(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get current authenticated user"""
    return User(
        username=current_user["username"],
        role=current_user["role"]
    )


@auth_router.post("/signup", response_model=SignupResponse)
async def signup(request: SignupRequest):
    """Create a new user account (stores plaintext password for simplicity)"""
    users_path = Path(settings.users_path)
    users_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if users_path.exists():
            with open(users_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"users": []}

        # Check if user exists
        for u in data.get("users", []):
            if u.get("username") == request.username:
                raise HTTPException(status_code=400, detail="Username already exists")

        # Append new user (plaintext password for dev simplicity)
        new_user = {
            "username": request.username,
            "password": request.password,
            "role": request.role or "guest",
        }
        data.setdefault("users", []).append(new_user)

        with open(users_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return SignupResponse(success=True, user=User(username=new_user["username"], role=new_user["role"]))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signup error: {e}")
        raise HTTPException(status_code=500, detail="Error creating user")


# ============================================================================
# OpenAI-Compatible Endpoints
# ============================================================================

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
        from backend.models.schemas import ChatMessage
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

        # Save to conversation history
        conversation_store.add_message(session_id, "user", user_message)
        conversation_store.add_message(session_id, "assistant", response_text)

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


async def determine_task_type(query: str) -> str:
    """
    Determine which task type to use based on query using LLM analysis
    Falls back to keyword matching if LLM fails
    """
    try:

        logger.info(f"[Task Classifier] Determining task type for query: {query[:]}")
        # Initialize LLM for classification (using lower temperature for deterministic results)
        classifier_llm = ChatOllama(
            base_url=settings.ollama_host,
            model=settings.agentic_classifier_model,
            temperature=0.1,  # Low temperature for consistent classification
            num_ctx=2048,  # Small context window for fast classification
        )

        # Create classification prompt
        system_prompt = settings.agentic_classifier_prompt

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Query: {query}")
        ]

        # Get classification with timeout
        try:
            response = await asyncio.wait_for(
                classifier_llm.ainvoke(messages),
                timeout=100  # 10 second timeout for classification
            )

            classification = response.content.strip().lower()

            # Validate response
            if classification in ["agentic", "chat"]:
                logger.info(f"[Task Classifier] Query classified as '{classification}': {query[:]}")
                return classification
            else:
                logger.warning(f"[Task Classifier] Invalid LLM response: '{classification}', falling back to keywords")

        except asyncio.TimeoutError:
            logger.warning("[Task Classifier] LLM classification timed out, falling back to keywords")
        except Exception as llm_error:
            logger.warning(f"[Task Classifier] LLM classification error: {str(llm_error)}, falling back to keywords")

    except Exception as e:
        logger.error(f"[Task Classifier] Failed to initialize classifier: {str(e)}, falling back to keywords")

    # Fallback: Keyword-based classification
    logger.info("[Task Classifier] Using keyword-based fallback classification")
    query_lower = query.lower()

    # Keywords that suggest agentic processing
    agentic_keywords = [
        "search", "find", "look up", "research",
        "compare", "analyze", "investigate",
        "current", "latest", "news",
        "document", "file", "pdf",
        "code", "python", "calculate", "data"
    ]

    if any(keyword in query_lower for keyword in agentic_keywords):
        logger.info(f"[Task Classifier] Fallback classified as 'agentic': {query[:]}")
        return "agentic"

    logger.info(f"[Task Classifier] Fallback classified as 'chat': {query[:]}")
    return "chat"


# ============================================================================
# Chat Management Endpoints
# ============================================================================

@chat_router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(current_user: Dict[str, Any] = Depends(get_current_user)):
    user_id = current_user["username"]
    sessions = conversation_store.get_user_sessions(user_id)
    return SessionListResponse(sessions=sessions)


@chat_router.get("/history/{session_id}", response_model=ConversationHistoryResponse)
async def get_history(session_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    conv = conversation_store.load_conversation(session_id)
    if conv is None or conv.user_id != current_user["username"]:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationHistoryResponse(session_id=session_id, messages=conv.messages)


# ============================================================================
# Admin Endpoints
# ============================================================================

@admin_router.post("/model", response_model=ModelChangeResponse)
async def change_model(request: ModelChangeRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Change the active Ollama model (admin only)"""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")

    try:
        # Update settings in-memory
        settings.ollama_model = request.model

        # Reinitialize simple chat LLM to take effect immediately
        from langchain_ollama import ChatOllama
        chat_task.llm = ChatOllama(
            base_url=settings.ollama_host,
            model=settings.ollama_model,
            temperature=settings.ollama_temperature,
            num_ctx=settings.ollama_num_ctx,
            top_p=settings.ollama_top_p,
            top_k=settings.ollama_top_k,
        )

        return ModelChangeResponse(success=True, model=settings.ollama_model)
    except Exception as e:
        logger.error(f"Model change error: {e}")
        raise HTTPException(status_code=500, detail="Failed to change model")


# ============================================================================
# Tool Endpoints
# ============================================================================

@tools_router.get("/list", response_model=ToolListResponse)
async def list_tools(_: Dict[str, Any] = Depends(get_current_user)):
    tools = [
        ToolInfo(name="web_search", description="Search the web for current information"),
        ToolInfo(name="rag_retrieval", description="Retrieve from your uploaded documents"),
        ToolInfo(name="python_code", description="Run safe Python code snippets"),
        ToolInfo(name="python_coder", description="Generate and execute complex Python code"),
        ToolInfo(name="wikipedia", description="Search summaries from Wikipedia"),
        ToolInfo(name="weather", description="Get weather for a location"),
        ToolInfo(name="sql_query", description="Run parameterized SQL queries (configured)"),
    ]
    return ToolListResponse(tools=tools)


@tools_router.post("/websearch", response_model=WebSearchResponse)
async def tool_websearch(request: WebSearchRequest, _: Dict[str, Any] = Depends(get_current_user)):
    """
    Perform web search and generate LLM-based answer from results

    Returns:
        - results: Raw search results with title, URL, content
        - answer: LLM-generated answer synthesizing the search results
        - sources_used: List of URLs used as sources
    """
    logger.info(f"[Websearch Endpoint] Query: {request.query}")

    # Get search results
    results = await web_search_tool.search(request.query, max_results=request.max_results)

    # Generate LLM answer from results
    answer, sources_used = await web_search_tool.generate_answer(request.query, results)

    logger.info(f"[Websearch Endpoint] Found {len(results)} results, generated answer with {len(sources_used)} sources")

    return WebSearchResponse(
        results=results,
        answer=answer,
        sources_used=sources_used
    )


@tools_router.get("/rag/search", response_model=RAGSearchResponse)
async def tool_rag_search(query: str, top_k: int = 5, _: Dict[str, Any] = Depends(get_current_user)):
    results = await rag_retriever.retrieve(query=query, top_k=top_k)
    return RAGSearchResponse(results=results)
