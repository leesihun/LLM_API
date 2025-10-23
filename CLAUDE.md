# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Agentic AI Backend** built with FastAPI that provides OpenAI-compatible APIs. The system uses **LangGraph** for orchestrating multi-step reasoning workflows and implements two distinct agent architectures: **ReAct** (Reasoning + Acting) and **Plan-and-Execute**. A smart router automatically selects the optimal agent based on query characteristics.

**Core Stack:**
- FastAPI with OpenAI-compatible endpoints
- LangGraph for agent workflow orchestration
- Ollama (local LLM: gpt-oss:20b)
- FAISS/Chroma for vector storage
- JWT authentication with bcrypt
- Static frontend (HTML/CSS/JS)

## Common Commands

### Running the Application

```bash
# Backend (FastAPI server on port 8000)
python run_backend.py              # Cross-platform launcher
python -m uvicorn backend.api.app:app --reload  # Direct uvicorn

# Frontend (Static server on port 3000)
python run_frontend.py             # Serve static files
python run_frontend.py --no-browser  # Don't auto-open browser

# Dependencies
pip install -r requirements.txt
```

### Ollama Setup

```bash
ollama serve                       # Start Ollama service
ollama list                        # List installed models
ollama pull gpt-oss:20b           # Download the default model
```

### Configuration

The system uses a **no fallbacks** design - all settings in [backend/config/settings.py](backend/config/settings.py:14-120) must be explicitly configured. Missing values will raise descriptive errors on startup.

**Critical variables:**
- `SECRET_KEY` - JWT signing key (generate with: `python -c "import secrets; print(secrets.token_urlsafe(32))"`)
- `TAVILY_API_KEY` - Required for web search functionality
- `OLLAMA_HOST` - Ollama endpoint (default: http://127.0.0.1:11434)
- `OLLAMA_MODEL` - Model name (default: gpt-oss:20b)
- `SERVER_HOST` - Use `0.0.0.0` for development, `localhost` for production

## Architecture

### Dual Agent System

The system implements a **smart agent router** ([backend/tasks/smart_agent_task.py](backend/tasks/smart_agent_task.py:71-124)) that analyzes queries and routes to the appropriate agent:

**1. ReAct Agent** ([backend/core/react_agent.py](backend/core/react_agent.py))
- **Pattern:** Iterative Thought → Action → Observation loops
- **Best for:** Exploratory queries, sequential reasoning, dynamic tool selection
- **Example:** "Find the capital of France, then search for its population, then calculate if it's larger than London"
- **Max iterations:** 5 (configurable in line 361)
- **How it works:**
  1. `_generate_thought()` - Reason about what to do next
  2. `_select_action()` - Choose tool and input (9 available actions)
  3. `_execute_action()` - Run tool and observe result
  4. Repeat until `finish` action or max iterations reached

**2. Plan-and-Execute Agent** ([backend/core/agent_graph.py](backend/core/agent_graph.py))
- **Pattern:** LangGraph state machine with 7 nodes (Planning → Tool Selection → 3 Tool Nodes → Reasoning → Verification)
- **Best for:** Complex batch queries, parallel tool usage, comprehensive tasks
- **Example:** "Search weather AND analyze data AND retrieve documents"
- **Max iterations:** 3 with verification loops
- **State flow:** Nodes pass `AgentState` TypedDict containing messages, plan, search_results, rag_context, etc.

**Smart Router Logic** ([backend/tasks/smart_agent_task.py](backend/tasks/smart_agent_task.py:71-124)):
- Scores queries based on indicator keywords
- ReAct indicators: "then", "after that", "step by step", "if", "depending on"
- Plan-and-Execute indicators: " and ", "both", "comprehensive", "full report"
- Default fallback: ReAct (for flexibility and transparency)

### Task Routing

```
User Query → determine_task_type() → "chat" or "agentic"
                                           ↓
                        "agentic" → SmartAgentTask → ReAct or Plan-and-Execute
                        "chat" → ChatTask (direct Ollama call)
```

**Agentic trigger keywords** ([backend/api/routes.py](backend/api/routes.py:193-210)):
- search, find, look up, research
- compare, analyze, investigate
- current, latest, news
- document, file, pdf

### Available Tools

All tools are in [backend/tools/](backend/tools/). Each tool has async methods and follows a standard interface:

1. **web_search.py** - Tavily API for web search
2. **rag_retriever.py** - Document Q&A with FAISS/Chroma vector DB
3. **data_analysis.py** - JSON data statistics (min/max/mean/count)
4. **python_executor.py** - Safe Python code execution
5. **math_calculator.py** - SymPy-based symbolic math
6. **wikipedia_tool.py** - Wikipedia search and summarization
7. **weather_tool.py** - Weather information retrieval
8. **sql_query_tool.py** - SQL database queries

### API Endpoints

**Authentication** (`/api/auth/`):
- `POST /api/auth/login` - Returns JWT token + user data
- `GET /api/auth/me` - Get current user info

**OpenAI-Compatible** (`/v1/`):
- `POST /v1/chat/completions` - Main chat endpoint
  - Optional param: `agent_type` ("auto", "react", "plan_execute")
  - Auto-creates `session_id` if not provided
  - Returns OpenAI-compatible response with `x_session_id`
- `GET /v1/models` - List available models

**File Management** (`/api/files/`):
- `POST /api/files/upload` - Upload document (user-isolated to `uploads/<username>/`)
- `GET /api/files/documents?page=1&page_size=20` - List documents (paginated)
- `DELETE /api/files/documents/{file_id}` - Delete document

**Health**:
- `GET /` - API info
- `GET /health` - Health check with Ollama status

**Default credentials:**
- Guest: `guest` / `guest_test1`
- Admin: `admin` / `administrator`

### Data Storage

All data organized under `./data/`:
- `data/vector_db/` - FAISS/Chroma embeddings
- `data/conversations/` - Chat history (JSON files per session)
- `data/uploads/<username>/` - User-isolated uploaded documents
- `data/users/users.json` - User authentication database
- `data/sessions/sessions.json` - Session management
- `data/logs/app.log` - Application logs

## Key Implementation Details

### Adding New Tools

1. Create tool file in [backend/tools/](backend/tools/) with async methods
2. Register tool in **both** agent implementations:
   - **ReAct Agent** ([backend/core/react_agent.py](backend/core/react_agent.py)):
     - Add to `ToolName` enum (line 27)
     - Add execution logic in `_execute_action()` (line 256)
     - Update tool list in `_select_action()` prompt (line 203)
   - **Plan-and-Execute Agent** ([backend/core/agent_graph.py](backend/core/agent_graph.py)):
     - Create tool node function (e.g., `async def new_tool_node(state: AgentState)`)
     - Add node to workflow in `create_agent_graph()` (line 300)
     - Update `tool_selection_node()` keyword detection (line 105)

### Adding New Agent Types

1. Create agent implementation in [backend/core/](backend/core/)
2. Add to `AgentType` enum in [backend/tasks/smart_agent_task.py](backend/tasks/smart_agent_task.py:18-22)
3. Update `execute()` method routing logic (line 38)
4. Update `_select_agent()` method with detection heuristics (line 71)

### LangGraph State Management

The Plan-and-Execute agent uses a TypedDict state ([agent_graph.py](backend/core/agent_graph.py:27-42)):

```python
class AgentState(TypedDict):
    messages: Annotated[List[ChatMessage], operator.add]  # Accumulates
    plan: str                        # Execution plan
    tools_used: List[str]            # Active tools
    search_results: str              # Web search output
    rag_context: str                 # Document context
    final_output: str                # Generated response
    verification_passed: bool        # Quality check
    iteration_count: int             # Loop counter
    # ... other fields
```

- State flows through nodes sequentially
- Use `Annotated[List, operator.add]` for accumulating lists
- Each node returns `Dict[str, Any]` with partial state updates

### ReAct Execution Tracing

The ReAct agent stores full execution history:

```python
react_agent.get_trace()  # Returns formatted Thought-Action-Observation log
```

Useful for debugging agent reasoning. Each step is a `ReActStep` object (line 40) containing:
- `thought` - Reasoning about next action
- `action` - Selected tool name
- `action_input` - Tool input
- `observation` - Tool execution result

### Multi-User Isolation

**Files:** User-specific folders prevent cross-user access
- Upload path: `uploads/<username>/<file_id>_<filename>`
- File listing/deletion scoped by JWT token in [backend/api/routes.py](backend/api/routes.py:217-356)

**Conversations:** Session IDs globally unique but associated with user_id via [backend/storage/conversation_store.py](backend/storage/conversation_store.py)

**Vector DB:** Currently shared across users (consider adding user_id metadata filter for production)

### Authentication Flow

All protected endpoints require:
```python
Authorization: Bearer <jwt_token>
```

JWT tokens ([backend/utils/auth.py](backend/utils/auth.py)):
- Signed with `SECRET_KEY` (HS256 algorithm)
- Expire after `JWT_EXPIRATION_HOURS` (default: 24)
- Include `{"sub": username}` payload
- Validated by `get_current_user()` dependency

### Conversation History

Managed by [backend/storage/conversation_store.py](backend/storage/conversation_store.py):
- JSON-based file storage (one file per session_id)
- User-isolated (keyed by user_id)
- Auto-loaded when `session_id` provided in requests
- Auto-created for new conversations

### Document Processing (RAG)

Pipeline in [backend/tools/rag_retriever.py](backend/tools/rag_retriever.py):
1. Upload → User-specific folder `uploads/<username>/`
2. Load document (supports PDF, DOCX, TXT, JSON)
3. Text chunking with `RecursiveCharacterTextSplitter`
4. Embedding with model specified in `EMBEDDING_MODEL` (default: bge-m3:latest)
5. Vector storage (FAISS or Chroma based on `VECTOR_DB_TYPE`)
6. Retrieval with similarity search (top_k=5)

**Note:** JSON files loaded as text documents using custom loader

### Prompt Engineering

Key prompts that control agent behavior:

**ReAct Agent** ([react_agent.py](backend/core/react_agent.py)):
- `_generate_thought()` (line 160) - Reasoning prompt
- `_select_action()` (line 181) - Tool selection prompt with available actions
- `_generate_final_answer()` (line 310) - Response generation

**Plan-and-Execute Agent** ([agent_graph.py](backend/core/agent_graph.py)):
- `planning_node()` (line 72) - Query analysis and planning
- `reasoning_node()` (line 200) - Response generation with context
- `verification_node()` (line 252) - Quality check criteria

### Logging

Logger instances per module:
```python
logger = logging.getLogger(__name__)
logger.info(f"[Component Name] Message")
```

Log levels (set via `LOG_LEVEL` in settings):
- DEBUG: Detailed execution traces
- INFO: High-level operation flow (default)
- WARNING: Production-friendly (minimal output)

Logs written to both console and `./data/logs/app.log`

## Important Notes

### Configuration System

Settings priority:
1. [backend/config/settings.py](backend/config/settings.py) (hardcoded defaults)
2. Optional .env file (can override defaults)

**The system will run without .env file** using settings.py defaults. This is intentional - settings.py has forced environment variable cleanup (line 128) to prevent conflicts.

### Error Handling

- All async tool calls use try/except with logging
- Return user-friendly messages (don't expose internals)
- FastAPI automatically converts `HTTPException` to proper responses
- Global exception handler in [backend/api/app.py](backend/api/app.py:102-113)

### Frontend Structure

Static files served from [frontend/static/](frontend/static/):
- `index.html` - Main chat interface (v2.0.0+ with file management UI)
- `login.html` - Login page
- `index_legacy.html` - Legacy chat interface

Frontend communicates with backend via:
- `/api/auth/login` - Authentication
- `/v1/chat/completions` - Chat interactions
- `/api/files/*` - File management

### Known Issues

1. **JSON file upload in RAG** - Custom JSON loader treats files as text documents, may not parse structured data correctly
2. **Vector DB sharing** - No user_id filtering in vector search (all users see same documents in vector DB)
3. **Timeout issues** - Large models or complex queries may timeout. Increase `OLLAMA_TIMEOUT` in settings.py

### Testing Considerations

When testing:
- ReAct agent: Test with sequential queries ("first do X, then Y")
- Plan-and-Execute: Test with multi-tool queries ("search AND analyze AND retrieve")
- Smart router: Verify correct agent selection via logs
- Test both `agent_type="auto"` and explicit agent selection
