"""
Data Models and Schemas
Defines all Pydantic models for API requests/responses
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime, timezone, timedelta


# ============================================================================
# Timezone Helper
# ============================================================================

def get_kst_now() -> datetime:
    """Get current time in KST (Korea Standard Time, UTC+9)"""
    kst = timezone(timedelta(hours=9))
    return datetime.now(kst)


# ============================================================================
# Authentication Models
# ============================================================================

class LoginRequest(BaseModel):
    """User login request"""
    username: str
    password: str


class LoginResponse(BaseModel):
    """User login response"""
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]


class User(BaseModel):
    """User model"""
    username: str
    role: str


# ============================================================================
# OpenAI-Compatible Chat Models
# ============================================================================

class ChatMessage(BaseModel):
    """Single chat message"""
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""
    model: str
    messages: List[ChatMessage]
    session_id: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    agent_type: Optional[Literal["auto", "react", "plan_execute"]] = "auto"


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None
    x_session_id: Optional[str] = None
    x_agent_metadata: Optional[Dict[str, Any]] = None  # Agentic workflow execution details


class ModelInfo(BaseModel):
    """Model information"""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "ollama"


class ModelsResponse(BaseModel):
    """List of available models"""
    object: str = "list"
    data: List[ModelInfo]


# ============================================================================
# Management / Admin Models
# ============================================================================

class SignupRequest(BaseModel):
    """User signup request"""
    username: str
    password: str
    role: Optional[str] = "guest"


class SignupResponse(BaseModel):
    """User signup response"""
    success: bool
    user: User


class ModelChangeRequest(BaseModel):
    """Change active LLM model (admin only)"""
    model: str


class ModelChangeResponse(BaseModel):
    """Response for model change"""
    success: bool
    model: str


# ============================================================================
# Agent State Models
# ============================================================================

class AgentState(BaseModel):
    """State object passed through LangGraph nodes"""
    messages: List[ChatMessage]
    session_id: Optional[str] = None
    user_id: str
    plan: Optional[str] = None
    tools_used: List[str] = Field(default_factory=list)
    search_results: Optional[List[Dict[str, Any]]] = None
    rag_context: Optional[str] = None
    final_output: Optional[str] = None
    verification_passed: bool = False
    iteration_count: int = 0
    max_iterations: int = 5


# ============================================================================
# Task Models
# ============================================================================

class TaskType(BaseModel):
    """Identifies the type of task to perform"""
    task: Literal["chat", "search", "rag", "agentic"]
    require_memory: bool = True


# ============================================================================
# Tool Models
# ============================================================================

class SearchQuery(BaseModel):
    """Web search query"""
    query: str
    max_results: int = 5


class SearchResult(BaseModel):
    """Web search result"""
    title: str
    url: str
    content: str
    score: Optional[float] = None


class RAGQuery(BaseModel):
    """RAG query for document retrieval"""
    query: str
    document_ids: Optional[List[str]] = None
    top_k: int = 5


class RAGResult(BaseModel):
    """RAG retrieval result"""
    content: str
    metadata: Dict[str, Any]
    score: float


class RAGSearchResponse(BaseModel):
    """RAG retrieval response"""
    results: List[RAGResult]


# ============================================================================
# Storage Models
# ============================================================================

class ConversationMessage(BaseModel):
    """Stored conversation message"""
    role: str
    content: str
    timestamp: datetime = Field(default_factory=get_kst_now)
    metadata: Optional[Dict[str, Any]] = None


class Conversation(BaseModel):
    """Complete conversation record"""
    session_id: str
    user_id: str
    messages: List[ConversationMessage]
    created_at: datetime = Field(default_factory=get_kst_now)
    updated_at: datetime = Field(default_factory=get_kst_now)
    metadata: Optional[Dict[str, Any]] = None


class SessionListResponse(BaseModel):
    """List of session IDs for a user"""
    sessions: List[str]


class ConversationHistoryResponse(BaseModel):
    """Conversation history response"""
    session_id: str
    messages: List[ConversationMessage]


# ============================================================================
# Tooling Models
# ============================================================================

class ToolInfo(BaseModel):
    name: str
    description: str


class ToolListResponse(BaseModel):
    tools: List[ToolInfo]


class MathRequest(BaseModel):
    expression: str
    return_latex: bool = False


class MathResponse(BaseModel):
    result: str
    latex: Optional[str] = None


class WebSearchRequest(BaseModel):
    query: str
    max_results: int = 5


class WebSearchResponse(BaseModel):
    results: List[SearchResult]
    answer: str  # LLM-generated answer from search results
    sources_used: List[str]  # URLs used in the answer
