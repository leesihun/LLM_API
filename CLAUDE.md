# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Agentic AI Backend** built with FastAPI that provides OpenAI-compatible APIs. The system uses **LangGraph** for orchestrating multi-step reasoning workflows and implements two distinct agent architectures: **ReAct** (Reasoning + Acting) and **Plan-and-Execute**. A smart router automatically selects the optimal agent based on query characteristics.

**Core Stack:**
- FastAPI with OpenAI-compatible endpoints
- LangGraph for agent workflow orchestration
- Ollama (local LLM: gemma3:12b default)
- FAISS/Chroma for vector storage
- JWT authentication with plaintext passwords (dev mode)
- Static frontend (HTML/CSS/JS)

## Common Commands

### Running the Application

```bash
# Backend (FastAPI server on port 1007)
python run_backend.py              # Launcher with checks
python server.py                   # Direct server start
python -m uvicorn backend.api.app:app --reload  # Uvicorn directly

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
ollama pull gemma3:12b            # Download the default model
```

### Configuration

The system uses a **forced defaults** design - all settings in [backend/config/settings.py](backend/config/settings.py:14-180) have hardcoded defaults. The settings system **actively removes** environment variables that might interfere (lines 188-192).

**Critical variables:**
- `secret_key` - JWT signing key (default: 'dev-secret-key-change-in-production-please')
- `tavily_api_key` - Required for web search functionality (has dev default)
- `ollama_host` - Ollama endpoint (default: http://127.0.0.1:11434)
- `ollama_model` - Model name (default: gemma3:12b)
- `server_host` - Host binding (default: 0.0.0.0)
- `server_port` - Server port (default: 1007)

**Python Code Execution Settings:**
- `python_code_enabled` - Enable/disable code execution (default: True)
- `python_code_timeout` - Max execution time in seconds (default: 300)
- `python_code_max_memory` - Memory limit in MB (default: 5120)
- `python_code_execution_dir` - Temp directory (default: ./data/code_execution)
- `python_code_max_iterations` - Max verification-modification loops (default: 10)
- `python_code_allow_partial_execution` - Execute with minor issues (default: False)
- `python_code_max_file_size` - Max input file size in MB (default: 500)

## Architecture

### Request Flow

```
User Query → determine_task_type() → "chat" or "agentic"
                                           ↓
                        "agentic" → SmartAgentTask → Plan-and-Execute (auto)
                                                   → ReAct (manual selection)
                        "chat" → ChatTask (direct Ollama call)
```

**Task classification** ([routes.py](backend/api/routes.py:256-323)):
- Uses LLM classifier (gemma3:12b) with configurable prompt from settings
- Falls back to keyword matching if LLM fails/timeouts (10s timeout)
- Agentic triggers: search, find, research, analyze, current, latest, news, document, file, code, python, calculate, data

### Agent System Architecture

**Smart Agent Router** ([smart_agent_task.py](backend/tasks/smart_agent_task.py)):
- Currently **defaults to Plan-and-Execute** for AUTO mode (line 61)
- Can manually select agents via `agent_type` parameter in API request
- Routes to either ReAct or Plan-and-Execute based on selection

**1. Plan-and-Execute Agent** ([Plan_execute.py](backend/tasks/Plan_execute.py)) - **Default**
- **Pattern:** Hybrid architecture - Planning phase + ReAct execution
- **Phase 1 (Planning):** LLM analyzes query and creates detailed execution plan (lines 65-108)
- **Phase 2 (Execution):** Injects plan into ReAct agent for guided execution (lines 150-162)
- **Phase 3 (Monitoring):** Verifies execution and builds comprehensive metadata (lines 164-181)
- **Best for:** Complex queries, batch operations, multi-tool tasks
- **Example:** "Search weather AND analyze data AND retrieve documents"
- **Max iterations:** Controlled by embedded ReAct agent (10 iterations)

**2. ReAct Agent** ([React.py](backend/tasks/React.py))
- **Pattern:** Iterative Thought → Action → Observation loops
- **Best for:** Exploratory queries, sequential reasoning, dynamic tool selection, Python code generation
- **Example:** "Find the capital of France, then search for its population, then calculate if it's larger than London"
- **Max iterations:** 10 (configurable via constructor line 72)
- **How it works:**
  1. `_generate_thought()` - Reason about what to do next (line 186)
  2. `_select_action()` - Choose tool and input (4 available actions, line 221)
  3. `_execute_action()` - Run tool and observe result (line 408)
  4. Repeat until `finish` action or max iterations reached
- **Available tools:** web_search, rag_retrieval, python_code, python_coder, finish (ToolName enum line 22)

### Available Tools

All tools are in [backend/tools/](backend/tools/). Each tool has async methods:

1. **web_search.py** - Tavily API for web search with LLM answer generation
2. **rag_retriever.py** - Document Q&A with FAISS/Chroma vector DB
3. **python_executor.py** - Safe Python code execution (simple scripts)
4. **python_coder_tool.py** - AI-driven Python code generator with iterative verification/modification
5. **python_executor_engine.py** - Subprocess execution engine for isolated code execution

#### Python Code Generator Tool

The **python_coder_tool** is an advanced code generation system that:
- Generates Python code based on natural language descriptions
- Performs static analysis and LLM-based verification
- Iteratively modifies code to fix issues (up to 10 iterations by default)
- Executes code in isolated subprocess sandboxes
- Supports file processing (CSV, Excel, PDF, JSON, etc.)

**Security features:**
- Whitelisted packages only (40+ safe packages including numpy, pandas, matplotlib)
- Blocked imports (socket, subprocess, eval, exec, pickle)
- Timeout enforcement (default: 300s)
- Filesystem isolation (temporary execution directories)
- Automatic cleanup after execution

**Workflow:**
1. Generate code using LLM
2. Verify code (static + semantic checks)
3. Modify if issues found (max 10 iterations)
4. Execute in isolated subprocess
5. Return results with full audit trail

### API Endpoints

**Authentication** (`/api/auth/`):
- `POST /api/auth/login` - Returns JWT token + user data (plaintext password check)
- `POST /api/auth/signup` - Create new user (stores plaintext password)
- `GET /api/auth/me` - Get current user info

**OpenAI-Compatible** (`/v1/`):
- `POST /v1/chat/completions` - Main chat endpoint
  - Optional param: `agent_type` ("auto", "react", "plan_execute")
  - Auto-creates `session_id` if not provided
  - Returns OpenAI-compatible response with `x_session_id` and `x_agent_metadata`
- `GET /v1/models` - List available models

**File Management** (`/api/files/`):
- `POST /api/files/upload` - Upload document (user-isolated to `uploads/<username>/`)
- `GET /api/files/documents?page=1&page_size=20` - List documents (paginated)
- `DELETE /api/files/documents/{file_id}` - Delete document

**Chat Management** (`/api/chat/`):
- `GET /api/chat/sessions` - List user sessions
- `GET /api/chat/history/{session_id}` - Get conversation history

**Admin** (`/api/admin/`):
- `POST /api/admin/model` - Change active Ollama model (admin only)

**Tools** (`/api/tools/`):
- `GET /api/tools/list` - List available tools
- `POST /api/tools/websearch` - Direct web search endpoint
- `GET /api/tools/rag/search` - Direct RAG search endpoint

**Health:**
- `GET /` - API info
- `GET /health` - Health check with Ollama status

**Default credentials:**
- Stored in `data/users/users.json` (created on first run)
- Default users depend on initial setup

### Data Storage

All data organized under `./data/`:
- `data/vector_db/` - FAISS/Chroma embeddings
- `data/conversations/` - Chat history (JSON files per session)
- `data/uploads/<username>/` - User-isolated uploaded documents
- `data/users/users.json` - User authentication database (plaintext passwords)
- `data/sessions/sessions.json` - Session management
- `data/logs/app.log` - Application logs
- `data/code_execution/` - Temporary Python code execution sandbox

## Key Implementation Details

### Adding New Tools

1. Create tool file in [backend/tools/](backend/tools/) with async methods
2. Register tool in **ReAct Agent** ([backend/tasks/React.py](backend/tasks/React.py)):
   - Add to `ToolName` enum (line 22)
   - Add execution logic in `_execute_action()` (line 408)
   - Update tool list in `_select_action()` prompt (line 243)
3. Optionally add to [backend/config/settings.py](backend/config/settings.py) `available_tools` list (line 139)

### Modifying Agent Selection Logic

**Current behavior:** Smart agent always selects Plan-and-Execute for AUTO mode ([smart_agent_task.py](backend/tasks/smart_agent_task.py:61))

**To change default agent:**
- Modify line 61 in smart_agent_task.py from `AgentType.PLAN_EXECUTE` to `AgentType.REACT`

**To implement intelligent routing:**
- Add scoring logic in `execute()` method before line 61
- Analyze query characteristics (sequential vs parallel, exploratory vs comprehensive)
- Example indicators:
  - ReAct: "then", "after that", "step by step", "if", "depending on"
  - Plan-Execute: " and ", "both", "comprehensive", "full report"

### LangGraph State Management (Plan-and-Execute)

The Plan-and-Execute agent embeds the ReAct agent, so it doesn't use a traditional LangGraph StateGraph. Instead, it:
1. Creates execution plan via LLM call
2. Injects plan into ReAct agent via enhanced messages
3. Monitors ReAct execution and builds combined metadata

The old LangGraph implementation is in [agent_graph.py](backend/core/agent_graph.py) but is **not currently used** by the system.

### ReAct Execution Tracing

The ReAct agent stores full execution history in `self.steps`:

```python
react_agent.get_trace()  # Returns formatted Thought-Action-Observation log
```

Each step is a `ReActStep` object ([React.py](backend/tasks/React.py:31-58)) containing:
- `thought` - Reasoning about next action
- `action` - Selected tool name
- `action_input` - Tool input
- `observation` - Tool execution result

### Multi-User Isolation

**Files:** User-specific folders prevent cross-user access
- Upload path: `uploads/<username>/<file_id>_<filename>`
- File listing/deletion scoped by JWT token

**Conversations:** Session IDs globally unique, associated with user_id via conversation_store

**Vector DB:** Currently shared across users (no user_id filtering - all users see same documents)

### Authentication Flow

All protected endpoints require:
```python
Authorization: Bearer <jwt_token>
```

JWT tokens ([backend/utils/auth.py](backend/utils/auth.py)):
- Signed with `secret_key` (HS256 algorithm)
- Expire after `jwt_expiration_hours` (default: 24)
- Include `{"sub": username}` payload
- Validated by `get_current_user()` dependency
- **Note:** Passwords stored in plaintext in users.json (development mode)

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
4. Embedding with model specified in `embedding_model` (default: bge-m3:latest)
5. Vector storage (FAISS or Chroma based on `vector_db_type`)
6. Retrieval with similarity search (top_k=5)

### Logging

Logger instances per module:
```python
logger = logging.getLogger(__name__)
logger.info(f"[Component Name] Message")
```

Log levels (set via `log_level` in settings):
- DEBUG: Detailed execution traces
- INFO: High-level operation flow (default)
- WARNING: Production-friendly (minimal output)

Logs written to both console and `./data/logs/app.log`

## Important Notes

### Configuration System

Settings priority:
1. [backend/config/settings.py](backend/config/settings.py) (hardcoded defaults)
2. .env file is **NOT used** - system actively removes conflicting env vars (lines 188-192)

**The system will ALWAYS use settings.py defaults.** This is intentional - settings.py has forced environment variable cleanup to prevent conflicts.

### Error Handling

- All async tool calls use try/except with logging
- Return user-friendly messages (don't expose internals)
- FastAPI automatically converts `HTTPException` to proper responses
- Global exception handler in [backend/api/app.py](backend/api/app.py)

### Frontend Structure

Static files served from [frontend/static/](frontend/static/):
- `index.html` - Main chat interface
- `login.html` - Login page
- `index_legacy.html` - Legacy chat interface
- `config.js` - Frontend configuration

Frontend communicates with backend via:
- `/api/auth/login` - Authentication
- `/v1/chat/completions` - Chat interactions
- `/api/files/*` - File management
- `/api/chat/*` - Session/history management

### Known Limitations

1. **Plaintext passwords** - Authentication uses plaintext passwords in users.json (dev mode only)
2. **Vector DB sharing** - No user_id filtering in vector search (all users see same documents in vector DB)
3. **No agent selection heuristics** - Smart agent always defaults to Plan-and-Execute in AUTO mode (manual override required for ReAct)
4. **Environment variable conflicts** - System forcefully removes OLLAMA_HOST, OLLAMA_MODEL, SERVER_HOST, SERVER_PORT env vars to use settings.py defaults
5. **Timeout issues** - Large models or complex queries may timeout. Increase `ollama_timeout` in settings.py

### Testing Agent Selection

When testing:
- **Plan-and-Execute (default):** Test with multi-tool queries ("search AND analyze AND retrieve")
- **ReAct (manual):** Test with sequential queries ("first do X, then Y") by setting `agent_type="react"` in request
- **Python code generation:** Test with various complexity levels (simple calculations, data analysis, file processing)
  - Verify iterative verification-modification loops in logs
  - Check execution isolation and cleanup in `data/code_execution/`
- **Agent metadata:** Check response `x_agent_metadata` field for execution details, tool usage, iteration count
