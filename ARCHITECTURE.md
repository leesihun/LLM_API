# Architecture Documentation

## Table of Contents
- [System Overview](#system-overview)
- [Architecture Patterns](#architecture-patterns)
- [Core Components](#core-components)
- [Agent System](#agent-system)
- [Tool System](#tool-system)
- [Data Flow](#data-flow)
- [Authentication & Security](#authentication--security)
- [Storage Architecture](#storage-architecture)
- [Configuration Management](#configuration-management)
- [Deployment Architecture](#deployment-architecture)

---

## System Overview

This is an **Agentic AI Backend** built with FastAPI that provides OpenAI-compatible APIs. The system employs advanced multi-agent reasoning patterns using **LangGraph** for orchestration and **Ollama** for local LLM inference.

### Key Characteristics

- **OpenAI-Compatible**: Drop-in replacement for OpenAI API endpoints
- **Multi-Agent Architecture**: Intelligent routing between ReAct and Plan-and-Execute agents
- **Tool Ecosystem**: 8+ integrated tools (web search, RAG, math, data analysis, etc.)
- **Local-First**: Uses Ollama for complete data privacy
- **Production-Ready**: JWT authentication, user isolation, conversation persistence

### Technology Stack

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend Layer                        │
│                    (Static HTML/CSS/JS)                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Auth Router  │  OpenAI Router  │  Files Router     │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Task Routing Layer                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Chat Task  │  Smart Agent Task  │  Agentic Task   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      Agent System                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │        ReAct Agent       │   Plan-and-Execute       │  │
│  │   (Iterative Reasoning)  │   (LangGraph Workflow)   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                       Tool Layer                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Web Search │ RAG │ Math │ Python │ Wikipedia │ etc. │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Ollama LLM  │  FAISS/Chroma  │  File Storage      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Architecture Patterns

### 1. Hexagonal Architecture (Ports & Adapters)

The system follows hexagonal architecture principles:

```
Core Domain (Business Logic)
    ├── Agents (ReAct, Plan-and-Execute)
    └── Tools (Web Search, RAG, Math, etc.)
        ↓
    Ports (Interfaces)
    └── Tool Interface
    └── LLM Interface
    └── Storage Interface
        ↓
    Adapters (Implementations)
    ├── Ollama Adapter (LLM)
    ├── FAISS/Chroma Adapter (Vector DB)
    └── File System Adapter (Storage)
```

### 2. Strategy Pattern (Agent Selection)

The Smart Agent Router uses the Strategy pattern to dynamically select agent implementations:

```python
SmartAgentTask
    ├── _select_agent(query) → AgentType
    └── execute()
        ├── If REACT → react_agent.execute()
        └── If PLAN_EXECUTE → agentic_task.execute()
```

### 3. Chain of Responsibility (Task Routing)

Query processing follows a chain of responsibility:

```
User Query
    ↓
determine_task_type() → "chat" or "agentic"
    ↓
If "chat" → ChatTask.execute()
If "agentic" → SmartAgentTask.execute()
                    ↓
            AgentType selection
                    ↓
            ReAct or Plan-and-Execute
```

---

## Core Components

### 1. FastAPI Application

**Location**: [backend/api/app.py](backend/api/app.py)

The main application entry point that:
- Configures CORS middleware
- Registers routers (auth, OpenAI, files)
- Sets up logging
- Handles global exceptions
- Tests Ollama connectivity on startup

```python
app = FastAPI(
    title="HE Team LLM Assistant API",
    description="Agentic AI backend with LangGraph, RAG, and Web Search",
    version="1.0.0"
)
```

### 2. Routes Layer

**Location**: [backend/api/routes.py](backend/api/routes.py)

Three main routers:

#### Authentication Router (`/api/auth`)
- `POST /api/auth/login` - JWT token generation
- `GET /api/auth/me` - Current user info

#### OpenAI Router (`/v1`)
- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Main chat endpoint (OpenAI-compatible)

#### Files Router (`/api/files`)
- `POST /api/files/upload` - Upload documents for RAG
- `GET /api/files/documents` - List user documents (paginated)
- `DELETE /api/files/documents/{file_id}` - Delete document

### 3. Schema Definitions

**Location**: [backend/models/schemas.py](backend/models/schemas.py)

Pydantic models for:
- **Authentication**: `LoginRequest`, `LoginResponse`, `User`
- **Chat**: `ChatMessage`, `ChatCompletionRequest`, `ChatCompletionResponse`
- **Models**: `ModelInfo`, `ModelsResponse`
- **Agent State**: `AgentState`
- **Tools**: `SearchQuery`, `SearchResult`, `RAGQuery`, `RAGResult`
- **Storage**: `ConversationMessage`, `Conversation`

---

## Agent System

The system implements a dual-agent architecture with intelligent routing.

### Agent Comparison

| Feature | ReAct Agent | Plan-and-Execute Agent |
|---------|-------------|------------------------|
| **File** | [backend/core/react_agent.py](backend/core/react_agent.py) | [backend/core/agent_graph.py](backend/core/agent_graph.py) |
| **Pattern** | Iterative Thought-Action-Observation loops | LangGraph workflow with 5 stages |
| **Best For** | Exploratory queries, sequential reasoning | Complex batch queries, parallel tools |
| **Max Iterations** | 5 (configurable) | 3 with verification loops |
| **Tool Selection** | Dynamic per iteration | Upfront batch selection |
| **Transparency** | Full trace available via `get_trace()` | Workflow state tracking |
| **Example Query** | "Find capital of France, then its population" | "Search weather AND analyze data AND retrieve docs" |

### 1. ReAct Agent

**Location**: [backend/core/react_agent.py](backend/core/react_agent.py)

Implements the **Reasoning + Acting** pattern:

```python
while iteration < max_iterations:
    # 1. Thought: Reason about what to do
    thought = await _generate_thought(query, steps)

    # 2. Action: Select tool and input
    action, action_input = await _select_action(query, thought, steps)

    if action == "finish":
        break

    # 3. Observation: Execute and observe
    observation = await _execute_action(action, action_input)

    steps.append(ReActStep(thought, action, action_input, observation))
```

**Key Methods**:
- `execute()` - Main execution loop
- `_generate_thought()` - Reasoning step
- `_select_action()` - Tool selection step
- `_execute_action()` - Tool execution step
- `_generate_final_answer()` - Final response synthesis
- `get_trace()` - Debug trace retrieval

**Available Actions**:
```python
class ToolName(str, Enum):
    WEB_SEARCH = "web_search"
    RAG_RETRIEVAL = "rag_retrieval"
    DATA_ANALYSIS = "data_analysis"
    PYTHON_CODE = "python_code"
    MATH_CALC = "math_calc"
    WIKIPEDIA = "wikipedia"
    WEATHER = "weather"
    SQL_QUERY = "sql_query"
    FINISH = "finish"
```

### 2. Plan-and-Execute Agent (LangGraph)

**Location**: [backend/core/agent_graph.py](backend/core/agent_graph.py)

Uses **LangGraph** for workflow orchestration:

```
┌─────────────┐
│   Planning  │ ← Analyze query, create plan
└──────┬──────┘
       ↓
┌─────────────┐
│Tool Selection│ ← Select tools (web_search, rag, data_analysis)
└──────┬──────┘
       ↓
┌─────────────┐
│ Web Search  │ ← Execute web search (if needed)
└──────┬──────┘
       ↓
┌─────────────┐
│RAG Retrieval│ ← Retrieve documents (if needed)
└──────┬──────┘
       ↓
┌─────────────┐
│Data Analysis│ ← Analyze JSON data (if needed)
└──────┬──────┘
       ↓
┌─────────────┐
│  Reasoning  │ ← Generate response with context
└──────┬──────┘
       ↓
┌─────────────┐
│Verification │ ← Quality check
└──────┬──────┘
       ↓
   Passed? ────Yes───→ END
       │
       No (max_iterations < 3)
       ↓
   Loop back to Planning
```

**State Schema**:
```python
class AgentState(TypedDict):
    messages: List[ChatMessage]          # Conversation history
    session_id: str                      # Session tracking
    user_id: str                         # User context
    plan: str                            # Execution plan
    tools_used: List[str]                # Active tools
    search_results: str                  # Web search output
    rag_context: str                     # Document context
    data_analysis_results: str           # Analysis output
    current_agent: str                   # Current node
    final_output: str                    # Generated response
    verification_passed: bool            # Quality check
    iteration_count: int                 # Loop counter
    max_iterations: int                  # Loop limit (3)
```

**Nodes**:
- `planning_node()` - Query analysis and plan creation
- `tool_selection_node()` - Keyword-based tool selection
- `web_search_node()` - Web search execution
- `rag_retrieval_node()` - Document retrieval
- `data_analysis_node()` - Data analysis
- `reasoning_node()` - Response generation
- `verification_node()` - Quality verification

### 3. Smart Agent Router

**Location**: [backend/tasks/smart_agent_task.py](backend/tasks/smart_agent_task.py)

Automatically selects the best agent based on query characteristics:

```python
class AgentType(str, Enum):
    REACT = "react"
    PLAN_EXECUTE = "plan_execute"
    AUTO = "auto"
```

**Selection Heuristics**:

```python
# ReAct indicators (sequential, exploratory)
react_indicators = [
    "then", "after that", "next", "followed by",
    "step by step", "first", "second", "third",
    "if", "depending on", "based on"
]

# Plan-and-Execute indicators (parallel, comprehensive)
plan_indicators = [
    " and ", " also ", " plus ",
    "both", "all", "multiple",
    "comprehensive", "complete analysis"
]
```

**Decision Logic**:
1. Count keyword matches
2. Analyze query structure (single question vs. multiple requirements)
3. Default to ReAct for flexibility and transparency

---

## Tool System

All tools are located in [backend/tools/](backend/tools/).

### Tool Implementations

#### 1. Web Search Tool
**File**: [backend/tools/web_search.py](backend/tools/web_search.py)

```python
async def search(query: str, max_results: int = 5) -> List[Dict[str, Any]]
def format_results(results: List[Dict[str, Any]]) -> str
```

- Uses **Tavily API** for web search
- Falls back to `websearch_ts` if Tavily unavailable
- Returns structured results with title, URL, content

#### 2. RAG Retriever Tool
**File**: [backend/tools/rag_retriever.py](backend/tools/rag_retriever.py)

```python
async def index_document(file_path: str, user_id: str) -> str
async def retrieve(query: str, top_k: int = 5, user_id: str = None) -> List[Dict]
def format_results(results: List[Dict]) -> str
```

- Supports PDF, DOCX, TXT, JSON
- Uses `RecursiveCharacterTextSplitter` for chunking
- Embedding model: `all-MiniLM-L6-v2`
- Vector store: FAISS or Chroma (configurable)
- User-isolated document indexing

#### 3. Data Analysis Tool
**File**: [backend/tools/data_analysis.py](backend/tools/data_analysis.py)

```python
async def analyze_json(data_input: str) -> str
```

- Parses JSON data
- Computes statistics: min, max, mean, count, sum
- Handles nested structures

#### 4. Python Executor Tool
**File**: [backend/tools/python_executor.py](backend/tools/python_executor.py)

```python
async def execute(code: str) -> Dict[str, Any]
def format_result(result: Dict[str, Any]) -> str
```

- Safe Python code execution
- Captures stdout/stderr
- Timeout protection
- Returns execution results

#### 5. Math Calculator Tool
**File**: [backend/tools/math_calculator.py](backend/tools/math_calculator.py)

```python
async def calculate(expression: str) -> str
```

- Uses **SymPy** for symbolic math
- Supports algebra, calculus, equations
- Handles derivatives, integrals, limits

#### 6. Wikipedia Tool
**File**: [backend/tools/wikipedia_tool.py](backend/tools/wikipedia_tool.py)

```python
async def search_and_summarize(query: str, sentences: int = 3) -> str
```

- Searches Wikipedia
- Returns concise summaries
- Handles disambiguation

#### 7. Weather Tool
**File**: [backend/tools/weather_tool.py](backend/tools/weather_tool.py)

```python
async def get_weather(location: str) -> str
```

- Retrieves weather information
- Returns formatted weather data

#### 8. SQL Query Tool
**File**: [backend/tools/sql_query_tool.py](backend/tools/sql_query_tool.py)

```python
async def execute_query(query: str) -> List[Dict[str, Any]]
def format_results(results: List[Dict]) -> str
```

- Executes SQL queries
- Returns structured results
- Format results as readable text

### Tool Integration Pattern

All tools follow a consistent pattern:

```python
class ToolName:
    async def primary_method(input_data: str) -> Result:
        """Execute tool operation"""
        try:
            # Process input
            # Execute operation
            # Return structured result
        except Exception as e:
            logger.error(f"Error: {e}")
            return error_response

    def format_results(results: Result) -> str:
        """Format results for LLM consumption"""
        # Convert structured data to readable text
```

---

## Data Flow

### 1. Chat Completion Request Flow

```
User Request
    ↓
POST /v1/chat/completions (with JWT token)
    ↓
get_current_user() - Validate JWT
    ↓
determine_task_type(query) → "chat" or "agentic"
    ↓
┌────────────────┬────────────────┐
│                │                │
"chat"        "agentic"          │
↓                ↓                │
ChatTask    SmartAgentTask       │
  ↓              ↓                │
Simple      Auto-select agent    │
Ollama          ↓                │
call    ┌───────┴───────┐       │
        │               │        │
    ReActAgent  Plan-and-Execute│
        │               │        │
    Execute      Execute        │
    with tools  with tools      │
        │               │        │
        └───────┬───────┘       │
                ↓                │
        Tool Execution          │
        (web_search, rag, etc.) │
                ↓                │
        Generate Response       │
                ↓                │
        Save to conversation_store
                ↓
        Return ChatCompletionResponse
```

### 2. File Upload Flow

```
User Upload
    ↓
POST /api/files/upload (with JWT token)
    ↓
get_current_user() - Extract username
    ↓
Create user-specific folder: uploads/{username}/
    ↓
Save file: {file_id}_{filename}
    ↓
rag_retriever.index_document()
    ↓
Load document (PDF/DOCX/TXT/JSON)
    ↓
Split into chunks (RecursiveCharacterTextSplitter)
    ↓
Generate embeddings (all-MiniLM-L6-v2)
    ↓
Store in vector DB (FAISS/Chroma)
    with metadata: {user_id, doc_id, filename}
    ↓
Return {file_id, doc_id, filename, size}
```

### 3. RAG Retrieval Flow

```
Query from Agent
    ↓
rag_retriever.retrieve(query, top_k=5, user_id)
    ↓
Generate query embedding
    ↓
Similarity search in vector DB
    (filtered by user_id if provided)
    ↓
Return top_k most similar chunks
    with metadata and scores
    ↓
format_results() - Convert to readable text
    ↓
Return to Agent for reasoning
```

---

## Authentication & Security

### JWT Authentication

**Location**: [backend/utils/auth.py](backend/utils/auth.py)

#### Token Generation

```python
def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=settings.jwt_expiration_hours)
    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(
        to_encode,
        settings.secret_key,
        algorithm=settings.jwt_algorithm
    )
    return encoded_jwt
```

#### Token Validation

```python
async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    payload = jwt.decode(
        token,
        settings.secret_key,
        algorithms=[settings.jwt_algorithm]
    )
    username: str = payload.get("sub")
    user = load_user(username)
    return user
```

### User Management

**Storage**: `data/users/users.json`

```json
{
  "users": [
    {
      "username": "admin",
      "password_hash": "$2b$12$...",
      "role": "admin"
    },
    {
      "username": "guest",
      "password_hash": "$2b$12$...",
      "role": "user"
    }
  ]
}
```

**Password Hashing**: Uses `bcrypt` with 12 rounds

### Multi-User Isolation

#### File Isolation
- Upload path: `uploads/{username}/{file_id}_{filename}`
- File listing scoped by username
- File deletion scoped by username

#### Conversation Isolation
- Session IDs globally unique
- Associated with `user_id`
- Conversation history per user

#### Vector DB Isolation
- Metadata includes `user_id`
- Retrieval filtered by `user_id` (optional)

---

## Storage Architecture

### File System Structure

```
data/
├── conversations/           # Conversation history
│   ├── {session_id}.json
│   └── ...
├── uploads/                 # User uploads (RAG documents)
│   ├── {username}/
│   │   ├── {file_id}_{filename}
│   │   └── ...
│   └── ...
├── vector_db/              # Vector embeddings
│   ├── {doc_id}/
│   │   ├── index.faiss     # FAISS index
│   │   └── index.pkl       # Metadata
│   └── ...
├── users/
│   └── users.json          # User database
├── sessions/
│   └── sessions.json       # Session management
└── logs/
    └── app.log             # Application logs
```

### Conversation Storage

**Location**: [backend/storage/conversation_store.py](backend/storage/conversation_store.py)

```python
class ConversationStore:
    def create_session(user_id: str) -> str
    def add_message(session_id: str, role: str, content: str)
    def get_messages(session_id: str) -> List[Dict]
    def delete_session(session_id: str)
```

**Format** (JSON):
```json
{
  "session_id": "abc123",
  "user_id": "admin",
  "messages": [
    {
      "role": "user",
      "content": "Hello",
      "timestamp": "2025-10-23T12:00:00Z"
    },
    {
      "role": "assistant",
      "content": "Hi!",
      "timestamp": "2025-10-23T12:00:05Z"
    }
  ],
  "created_at": "2025-10-23T12:00:00Z",
  "updated_at": "2025-10-23T12:00:05Z"
}
```

### Vector Database

#### FAISS Mode
- In-memory index
- Serialized to disk per document
- Fast similarity search
- Low persistence overhead

#### Chroma Mode
- Persistent storage
- Client-server architecture
- Better for large datasets
- Supports filtering

---

## Configuration Management

**Location**: [backend/config/settings.py](backend/config/settings.py)

### Design Philosophy

**No Fallbacks**: All settings must be explicitly configured. Missing values raise descriptive errors.

### Configuration Priority

1. `.env` file (optional overrides)
2. `settings.py` defaults

### Key Settings

```python
class Settings(BaseSettings):
    # Server
    server_host: str = '0.0.0.0'
    server_port: int = 8000
    secret_key: str = 'dev-secret-key-change-in-production-please'

    # Ollama
    ollama_host: str = 'http://127.0.0.1:11434'
    ollama_model: str = 'gpt-oss:20b'
    ollama_timeout: int = 3000000  # 50 minutes
    ollama_num_ctx: int = 4096
    ollama_temperature: float = 0.7
    ollama_top_p: float = 0.9
    ollama_top_k: int = 40

    # API Keys
    tavily_api_key: str = 'tvly-...'

    # Vector DB
    vector_db_type: str = 'faiss'
    vector_db_path: str = './data/vector_db'
    embedding_model: str = 'bge-m3:latest'

    # Storage
    users_path: str = './data/users/users.json'
    sessions_path: str = './data/sessions/sessions.json'
    conversations_path: str = './data/conversations'
    uploads_path: str = './data/uploads'

    # Auth
    jwt_algorithm: str = 'HS256'
    jwt_expiration_hours: int = 24

    # Logging
    log_level: str = 'INFO'
    log_file: str = './data/logs/app.log'
```

### Environment Variables

Create `.env` file for overrides:

```bash
# Server
SERVER_HOST=localhost
SERVER_PORT=8000
SECRET_KEY=your-secure-key-here

# Ollama
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=gpt-oss:20b

# API Keys
TAVILY_API_KEY=your-tavily-key

# Vector DB
VECTOR_DB_TYPE=faiss
EMBEDDING_MODEL=bge-m3:latest

# Logging
LOG_LEVEL=INFO
```

---

## Deployment Architecture

### Development Mode

```bash
# Start Ollama
ollama serve

# Start Backend (port 8000)
python run_backend.py

# Start Frontend (port 3000)
python run_frontend.py
```

### Production Deployment

#### Option 1: Single Server

```
┌────────────────────────────────────────┐
│         Nginx Reverse Proxy            │
│  (:80/:443)                            │
│  ├── /api → Backend (localhost:8000)  │
│  └── /    → Static Frontend           │
└────────────────────────────────────────┘
         ↓                    ↓
┌─────────────────┐  ┌─────────────────┐
│ FastAPI Backend │  │ Static Frontend │
│ (localhost:8000)│  │     (port 80)   │
└─────────────────┘  └─────────────────┘
         ↓
┌─────────────────┐
│  Ollama Server  │
│ (localhost:11434)│
└─────────────────┘
```

#### Option 2: Distributed

```
┌────────────────────────────────────────┐
│         Load Balancer                  │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│    Backend Cluster (N instances)       │
│  ┌──────────┐  ┌──────────┐           │
│  │Backend #1│  │Backend #2│  ...      │
│  └──────────┘  └──────────┘           │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│    Ollama Cluster (Load Balanced)      │
│  ┌──────────┐  ┌──────────┐           │
│  │ Ollama#1 │  │ Ollama#2 │  ...      │
│  └──────────┘  └──────────┘           │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│    Shared Storage (NFS/S3)             │
│  - Conversations                       │
│  - Uploads                             │
│  - Vector DB                           │
└────────────────────────────────────────┘
```

### Docker Deployment

```yaml
# docker-compose.yml
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - SERVER_HOST=0.0.0.0
    depends_on:
      - ollama
    volumes:
      - ./data:/app/data

  frontend:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./frontend:/usr/share/nginx/html
      - ./nginx.conf:/etc/nginx/nginx.conf

volumes:
  ollama_data:
```

### Scaling Considerations

#### Bottlenecks
1. **Ollama Inference**: CPU/GPU bound
2. **Vector Search**: Memory bound (FAISS in-memory)
3. **File Storage**: Disk I/O

#### Solutions
1. **Horizontal Scaling**: Multiple Ollama instances behind load balancer
2. **Vector DB**: Switch to Chroma with persistent storage
3. **Caching**: Add Redis for conversation caching
4. **Object Storage**: S3/MinIO for file uploads
5. **Database**: PostgreSQL for user/session management

---

## Performance Optimization

### LLM Inference

```python
# Ollama timeout configuration
ollama_timeout: int = 3000000  # 50 minutes

# Connection pooling
async_client = httpx.AsyncClient(
    timeout=httpx.Timeout(settings.ollama_timeout / 1000, connect=60.0),
    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
)
```

### Vector Search

```python
# FAISS: In-memory, fast similarity search
# Chroma: Persistent, better for large datasets

# Chunk size optimization
chunk_size = 1000
chunk_overlap = 200

# Top-k retrieval
top_k = 5  # Balance between context and speed
```

### Conversation Storage

- JSON files per session
- Lazy loading (only when needed)
- No database overhead for small deployments

---

## Error Handling

### Global Exception Handler

```python
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )
```

### Tool Error Handling

```python
async def execute_action(action: str, action_input: str) -> str:
    try:
        # Execute tool
        result = await tool.execute(action_input)
        return result
    except Exception as e:
        logger.error(f"Error executing action {action}: {e}")
        return f"Error executing action: {str(e)}"
```

---

## Monitoring & Logging

### Logging Strategy

```python
# Module-level loggers
logger = logging.getLogger(__name__)

# Structured logging
logger.info(f"[Component Name] Operation: {details}")
logger.error(f"[Component Name] Error: {error}", exc_info=True)
```

### Log Levels

- **DEBUG**: Detailed execution traces, full request/response
- **INFO**: High-level operation flow (default)
- **WARNING**: Production-friendly, minimal output
- **ERROR**: Errors and exceptions

### Key Logging Points

1. **Request Processing**: User, session, query
2. **Agent Selection**: Which agent selected, why
3. **Tool Execution**: Tool name, input, result
4. **Error Handling**: Full stack traces
5. **Performance**: Operation timing

---

## Security Considerations

### Authentication
- JWT with HS256 algorithm
- Bcrypt password hashing (12 rounds)
- Token expiration (24 hours default)

### Authorization
- Role-based access control (admin/user)
- User-isolated file storage
- User-scoped conversation history

### Input Validation
- Pydantic models for all requests
- File type validation on upload
- SQL injection prevention (parameterized queries)

### Data Privacy
- Local LLM (Ollama) - no data sent to external APIs
- User data isolation
- Secure secret key management

### Production Checklist
- [ ] Change `SECRET_KEY` (generate with `secrets.token_urlsafe(32)`)
- [ ] Set `SERVER_HOST=localhost` (not `0.0.0.0`)
- [ ] Use HTTPS (nginx with SSL certificates)
- [ ] Configure CORS (restrict `allow_origins`)
- [ ] Set `LOG_LEVEL=WARNING` (reduce log verbosity)
- [ ] Secure Tavily API key
- [ ] Enable rate limiting (optional)
- [ ] Set up monitoring and alerts

---

## Extension Points

### Adding New Tools

1. Create tool file in `backend/tools/`
2. Implement async methods
3. Register in both agents:
   - ReAct: Add to `ToolName` enum and `_execute_action()`
   - Plan-and-Execute: Add tool node, update workflow edges
4. Update tool selection logic

### Adding New Agents

1. Create agent implementation in `backend/core/`
2. Register in `smart_agent_task.py` - Add to `AgentType` enum
3. Update routing logic in `_select_agent()`
4. Add detection heuristics

### Custom Storage Backends

Implement storage interface:
```python
class StorageBackend:
    async def save(key: str, data: Any) -> None
    async def load(key: str) -> Any
    async def delete(key: str) -> None
    async def list(prefix: str) -> List[str]
```

---

## References

- **FastAPI**: https://fastapi.tiangolo.com/
- **LangGraph**: https://langchain-ai.github.io/langgraph/
- **Ollama**: https://ollama.com/
- **FAISS**: https://github.com/facebookresearch/faiss
- **Chroma**: https://www.trychroma.com/
- **Pydantic**: https://docs.pydantic.dev/
