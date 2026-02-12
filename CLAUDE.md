# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **LLM API server** that provides OpenAI-compatible endpoints with support for multiple LLM backends (Ollama, llama.cpp). The system features a sophisticated agent-based architecture with tool calling capabilities, including web search, Python code execution, and RAG (Retrieval Augmented Generation).

**Key Architecture Pattern**: Dual-server architecture to prevent deadlock:
- **Main API Server** (port 10007): Handles chat, authentication, sessions
- **Tools API Server** (port 10006): Handles tool execution (websearch, python_coder, rag)

The separation is critical because agents running on the main server make HTTP calls to tools on the tools server. Running tools on the same server would cause deadlock.

## Development Commands

### Starting the Servers

**Start servers individually** (must start tools server first):
```bash
# Terminal 1 - Tools API (must start first)
python tools_server.py

# Terminal 2 - Main API
python run_backend.py
```

### Dependencies

```bash
pip install -r requirements.txt
```

**Important**: The project uses both `bcrypt==4.0.1` and `passlib==1.7.4` for password hashing. Bcrypt has a 72-byte password limit (not 72 characters).

### Testing

Tests are in `tests/` and require both servers running (for API tests) or can run direct tests standalone:
```bash
# Direct RAG tool tests (no servers needed)
python tests/test_rag.py

# RAG upload performance tests
python tests/test_rag_upload_performance.py
```
Tests use `requests` against `http://localhost:10007`, not pytest. They create test users/collections and clean up after themselves.

### Utility Scripts

```bash
python create_users.py          # Batch create users
python create_user_direct.py    # Create a single user directly in DB
python clear_data.py            # Clear all data (sessions, uploads, scratch)
python clear_rag_data.py        # Clear RAG indices, documents, metadata
```

### Configuration

All configuration is centralized in `config.py`. Key settings:
- `LLM_BACKEND`: Choose "ollama", "llamacpp", or "auto"
- `TOOLS_HOST`/`TOOLS_PORT`: Tools server location
- `PYTHON_EXECUTOR_MODE`: "native" or "opencode" for code execution
- `SERVER_WORKERS`/`TOOLS_SERVER_WORKERS`: Multi-worker uvicorn (default 4)
- `PRELOAD_MODEL_ON_STARTUP`: Preload LLM model at server start
- `PYTHON_CODER_SMART_EDIT`: LLM-based code merging with existing .py files

## High-Level Architecture

### 1. Agent System

All agents inherit from `backend/agents/base_agent.py`:

| Agent | Purpose | Prompts |
|-------|---------|---------|
| **ChatAgent** | Simple conversational (no tools) | `chat_system.txt` |
| **ReActAgent** | Reasoning + Acting with tools | `react_*.txt` |
| **PlanExecuteAgent** | Multi-step planning | `plan_*.txt` |
| **UltraworkAgent** | Iterative refinement via OpenCode | `ultrawork_*.txt` |
| **AutoAgent** | Auto-selects best agent | `auto_router.txt` |

**Note**: When `PYTHON_EXECUTOR_MODE="opencode"`, the `plan_execute` agent is automatically replaced with `ultrawork`.

**Agent Base Class** provides:
- `call_tool()` - HTTP requests to tools API
- `load_prompt()` - Load from `prompts/` directory
- Conversation history formatting and file attachment handling

### 2. Tool System

Tools run as separate services on port 10006:

| Tool | Location | Description |
|------|----------|-------------|
| **websearch** | `tools/web_search/tool.py` | Tavily API web search |
| **python_coder** | `tools/python_coder/` | Code execution (native or opencode mode) |
| **rag** | `tools/rag/` | FAISS-based document retrieval (basic or enhanced) |

**RAG Tool Selection** (`tools/rag/__init__.py`): At import time, `EnhancedRAGTool` is auto-selected if any of `RAG_USE_HYBRID_SEARCH`, `RAG_USE_RERANKING`, or `RAG_CHUNKING_STRATEGY != "fixed"` are set. Otherwise `BaseRAGTool` is used. EnhancedRAGTool adds hybrid BM25+FAISS search with RRF fusion, cross-encoder reranking, semantic chunking, multi-query expansion, and context window retrieval.

**Tool Call Flow**:
1. Agent calls `self.call_tool(tool_name, parameters, context)`
2. `base_agent.py` makes HTTP POST to `http://{TOOLS_HOST}:{TOOLS_PORT}/api/tools/{tool_name}`
3. `backend/api/routes/tools.py` routes to appropriate tool
4. Tool returns `{"success": bool, "answer": str, "data": dict, "metadata": dict}`

### 3. ReAct Agent Flow

Strict 2-step loop:

**Step 1** - LLM generates:
```
Thought: [reasoning]
Action: [tool_name]
Action Input: [string input]
```

**Step 2** - After tool execution, LLM generates:
```
final_answer: true/false
Observation: [analysis of tool result]
```

Loop continues until `final_answer: true`. Parsing uses regex in `_parse_action()` and `_step2_generate_observation()`.

### 4. LLM Backend Abstraction

`backend/core/llm_backend.py` provides unified interface:
- **Auto-fallback**: Tries Ollama first, falls back to llama.cpp
- **SSL handling**: Three-tier fallback: corporate cert (`C:/DigitalCity.crt`) → default SSL → disabled SSL (with warning)
- All LLM calls logged via `llm_interceptor.py` to `data/logs/prompts.log`

### 5. Streaming vs Agent Mode

**Streaming** (`stream=true` in `/v1/chat/completions`) bypasses the entire agent system and calls `llm_backend.chat_stream()` directly. This means streaming mode has **no tool calling** — no websearch, python_coder, or RAG. Only non-streaming mode uses agents with tool capabilities.

### 6. File Attachments

Files attached to messages get **automatic** rich metadata extraction (no tool call needed):
- **JSON**: Structure, keys, sample data
- **CSV/Excel**: Headers, row count, sample rows
- **Python**: Imports, function/class definitions

Handled by `extract_file_metadata()` in `backend/utils/file_handler.py`, formatted via `format_attached_files()` in `base_agent.py`.

### 7. Database & Storage

**SQLite** (`data/app.db`):
- `users`: Authentication with bcrypt hashed passwords
- `sessions`: Lightweight metadata only (no conversation data)

**Conversations**: Stored as JSON in `data/sessions/{session_id}.json` for easy debugging. Uses `filelock.FileLock` for concurrent read/write safety.

**File Storage**:
- User uploads: `data/uploads/{username}/`
- Session scratch: `data/scratch/{session_id}/`
- RAG: `data/rag_documents/`, `data/rag_indices/`, `data/rag_metadata/`

### 8. API Routes

| File | Endpoints |
|------|-----------|
| `auth.py` | `/api/auth/signup`, `/api/auth/login`, `/api/auth/me` |
| `chat.py` | `/v1/chat/completions` (OpenAI-compatible) |
| `sessions.py` | `/api/chat/sessions`, `/api/chat/history` |
| `models.py` | `/v1/models` (OpenAI-compatible) |
| `admin.py` | `/api/admin/*` (user management) |
| `tools.py` | `/api/tools/*` (tool execution) |

## Adding New Tools

1. **Create implementation** in `tools/{tool_name}/tool.py`
   - Return: `{"success": bool, "answer": str, "data": dict, "metadata": dict}`

2. **Add schema** to `tools_config.py` `TOOL_SCHEMAS` dict

3. **Add API endpoint** in `backend/api/routes/tools.py`

4. **Add parameter parsing** in `backend/agents/react_agent.py` `_convert_string_to_params()`

5. **Update config** in `config.py`:
   - Add to `AVAILABLE_TOOLS`
   - Add to `TOOL_MODELS` and `TOOL_PARAMETERS`

## Key Configuration Reference

| Category | Variables |
|----------|-----------|
| Agent behavior | `REACT_MAX_ITERATIONS`, `REACT_RETRY_ON_ERROR`, `PLAN_*`, `ULTRAWORK_*` |
| Tool settings | `TOOL_MODELS`, `TOOL_PARAMETERS`, `DEFAULT_TOOL_TIMEOUT` |
| Code execution | `PYTHON_EXECUTOR_MODE`, `OPENCODE_*` |
| Web search | `TAVILY_API_KEY`, `TAVILY_SEARCH_DEPTH` |
| RAG | `RAG_EMBEDDING_MODEL`, `RAG_CHUNK_SIZE`, `RAG_INDEX_TYPE` |
| LLM | `LLM_BACKEND`, `OLLAMA_MODEL`, `DEFAULT_TEMPERATURE` |

## Common Gotchas

1. **Always start tools server before main server** - Main server will fail health checks otherwise
2. **Password byte length != character length** - Bcrypt limit is 72 BYTES (emoji/CJK use multiple bytes)
3. **Session IDs must be unique** - Used for workspace isolation in python_coder
4. **Tool timeouts** - Default is 10 days (864000s) for long-running operations
5. **Conversation history** - Limited by `MAX_CONVERSATION_HISTORY` (default 50)
6. **Windows UTF-8** - Both servers set explicit UTF-8 encoding for console emoji support
7. **Streaming loses tools** - `stream=true` bypasses agents entirely; only direct LLM chat
8. **RAG score semantics** - FAISS `IndexFlatIP` returns cosine similarity (0-1, higher=better), NOT L2 distance. Use `dist` directly as score. Hybrid RRF scores must be normalized to 0-1 range before threshold filtering
9. **AutoAgent eager init** - All agent types are instantiated in `__init__`, not lazily
10. **Multi-worker string import** - uvicorn uses `"tools_server:app"` string format (not app object) when `workers > 1`
11. **`data/` is gitignored** - The entire `data/` directory (DB, sessions, uploads, RAG indices, logs) is excluded from version control

## File Organization

```
backend/
  agents/          - Agent implementations (base, chat, react, plan_execute, ultrawork, auto)
  api/routes/      - FastAPI route handlers
  core/            - Core services (llm_backend, database)
  models/          - Pydantic schemas
  utils/           - Utilities (auth, file_handler, conversation_store)
tools/
  web_search/      - Tavily integration
  python_coder/    - Code execution (native_tool, opencode_tool)
  rag/             - FAISS document retrieval
prompts/
  agents/          - Agent system prompts
  tools/           - Tool-specific prompts
data/
  sessions/        - Conversation JSON files
  uploads/         - User persistent uploads
  scratch/         - Session temporary files
  rag_*/           - RAG storage
  logs/            - LLM interaction logs
```

## Default Credentials

- **Admin**: `admin` / `administrator` (change in production via `config.py`)
- See `use_cases/USER_MANAGEMENT.md` for user management API reference
