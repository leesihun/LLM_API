# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **LLM API server** that provides OpenAI-compatible endpoints with support for multiple LLM backends (Ollama, llama.cpp). The system features a sophisticated agent-based architecture with tool calling capabilities, including web search, Python code execution, and RAG (Retrieval Augmented Generation).

**Key Architecture Pattern**: Dual-server architecture to prevent deadlock:
- **Main API Server** (port 10007): Handles chat, authentication, sessions
- **Tools API Server** (port 10006): Handles tool execution (websearch, python_coder, rag, ppt_maker)

The separation is critical because agents running on the main server make HTTP calls to tools on the tools server. Running tools on the same server would cause deadlock.

## Development Commands

### Starting the Servers

**Start both servers** (recommended):
```bash
bash start_servers.sh
```

**Start servers individually** (for debugging):
```bash
# Terminal 1 - Tools API (must start first)
python tools_server.py

# Terminal 2 - Main API
python server.py
```

### Testing

No formal test suite currently exists. The repository includes API examples in `API_examples.ipynb` for manual testing.

### Dependencies

Install dependencies:
```bash
pip install -r requirements.txt
```

**Important**: The project uses both `bcrypt==4.0.1` and `passlib==1.7.4` for password hashing with bcrypt's 72-byte password limit.

### Configuration

All configuration is centralized in `config.py`. Key settings:
- `LLM_BACKEND`: Choose "ollama", "llamacpp", or "auto" (tries Ollama first)
- `TOOLS_HOST`/`TOOLS_PORT`: Tools server location (change if on different machine)
- `PYTHON_EXECUTOR_MODE`: "native" or "nanocoder" for code execution

## High-Level Architecture

### 1. Agent System

The system implements multiple agent types, all inheriting from `backend/agents/base_agent.py`:

**Agent Types**:
- **ChatAgent**: Simple conversational agent (no tools)
- **ReActAgent**: Reasoning + Acting agent with tool calling
  - 2-step loop: (1) Generate Thought/Action/Input, (2) Execute tool → Observation → Decide if done
  - System prompts in `prompts/agents/react_*.txt`
  - Configurable via `REACT_MAX_ITERATIONS`, `REACT_RETRY_ON_ERROR` in config
- **PlanExecuteAgent**: Multi-step planning agent
  - Creates plan, executes steps, can re-plan on failure
  - Configurable via `PLAN_*` settings in config
- **AutoAgent**: Automatically selects best agent based on user input

**Agent Base Class** (`base_agent.py`):
- Provides `call_tool()` method that makes HTTP requests to tools API
- Handles tool execution logging and error handling
- Provides `load_prompt()` for loading system prompts from `prompts/`
- Manages conversation history formatting and file attachment handling

### 2. Tool System

Tools are **completely separate services** running on port 1006:

**Available Tools**:
1. **websearch**: Uses Tavily API for web search
   - Located: `tools/web_search/tool.py`
   - Returns LLM-generated answers from search results

2. **python_coder**: Python code execution with two modes:
   - **Native** (`tools/python_coder/native_tool.py`): Direct subprocess execution of Python code
   - **Nanocoder** (`tools/python_coder/nanocoder_tool.py`): Uses nanocoder CLI for natural language to code
   - Workspace isolation: Each session gets `data/scratch/{session_id}/`

3. **rag**: Document retrieval with FAISS
   - Located: `tools/rag/tool.py`
   - Uses sentence transformers for embeddings
   - Collections stored in `data/rag_*` directories

4. **ppt_maker**: Presentation generation with Marp
   - Located: `tools/ppt_maker/tool.py`
   - Creates PDF and PPTX from natural language instructions
   - Uses Marp CLI for markdown-to-slides conversion

**Tool Configuration** (`tools_config.py`):
- `TOOL_SCHEMAS`: Defines all tool schemas with parameters
- `format_tools_for_llm()`: Formats tool descriptions for agent prompts
- Each tool has custom timeout, temperature, and model settings

**Tool Call Flow**:
1. Agent calls `self.call_tool(tool_name, parameters, context)`
2. `base_agent.py` makes HTTP POST to `http://{TOOLS_HOST}:{TOOLS_PORT}/api/tools/{tool_name}`
3. `backend/api/routes/tools.py` receives request and routes to appropriate tool
4. Tool executes and returns structured response with `success`, `answer`, `data`, `metadata`

### 3. LLM Backend Abstraction

**LLM Backend** (`backend/core/llm_backend.py`):
- Unified interface supporting both Ollama and llama.cpp
- Auto-fallback: Tries Ollama first, falls back to llama.cpp if unavailable
- Handles streaming and non-streaming responses
- All LLM calls logged via `llm_interceptor.py` to `data/logs/prompts.log`

**Why Two Backends?**
- **Ollama**: Popular, easy to use, supports more models
- **llama.cpp**: Lightweight, good for resource-constrained environments

### 4. Database & Storage

**SQLite Database** (`backend/core/database.py`):
- Users table: Authentication with bcrypt hashed passwords
- Sessions table: Lightweight metadata only (no conversation data)

**Conversation Storage**:
- Stored as **human-readable JSON** in `data/sessions/{session_id}.json`
- Rationale: Easy debugging, version control, no database bloat
- Managed by `ConversationStore` class

**File Storage**:
- User uploads: `data/uploads/{username}/` (persistent)
- Session scratch: `data/scratch/{session_id}/` (temporary)
- RAG documents: `data/rag_documents/`
- RAG indices: `data/rag_indices/`

### 5. API Routes

All routes defined in `backend/api/routes/`:
- `auth.py`: `/api/auth/signup`, `/api/auth/login`, `/api/auth/me`
- `chat.py`: `/v1/chat/completions` (OpenAI-compatible)
- `sessions.py`: `/api/chat/sessions`, `/api/chat/history`
- `models.py`: `/v1/models` (OpenAI-compatible)
- `admin.py`: `/api/admin/*` (user management)
- `tools.py`: `/api/tools/*` (tool execution endpoints)

### 6. Authentication & Security

**JWT-based Authentication**:
- Uses `python-jose` for JWT tokens
- Token expiration: 7 days (configurable via `JWT_EXPIRATION_HOURS`)
- Protected routes use `get_current_user()` dependency

**Password Security** (Important):
- **Bcrypt has a 72-byte limit** for passwords (not 72 characters!)
- Multi-byte characters (emoji, CJK) consume multiple bytes
- Validation added in `backend/utils/auth.py` and `backend/models/schemas.py`
- See `BCRYPT_PASSWORD_FIX.md` for detailed explanation

### 7. Prompt System

System prompts stored in `prompts/`:
- `agents/`: Agent-specific prompts (ReAct, Plan-Execute)
  - `react_system.txt`: Tool descriptions and format
  - `react_thought.txt`: Reasoning step
  - `react_observation.txt`: Observation generation
  - `react_final.txt`: Final answer synthesis
- `tools/`: Tool-specific prompts (web search, RAG query optimization)

Prompts use Python `.format()` style templating with named placeholders.

## Important Implementation Details

### ReAct Agent Flow

The ReAct agent uses a strict 2-step loop:

1. **Step 1**: LLM generates structured response:
   ```
   Thought: [reasoning]
   Action: [tool_name]
   Action Input: [string input]
   ```

2. **Step 2**: Tool executes, LLM generates:
   ```
   final_answer: true/false
   Observation: [analysis of tool result]
   ```

3. If `final_answer: false`, loop continues. If `true`, generate final response.

**Parsing**: Uses regex to extract structured fields (`_parse_action()`, `_step2_generate_observation()`)

**Error Handling**: If `REACT_RETRY_ON_ERROR=True`, errors are passed back to LLM to decide next action (intelligent retry). If `False`, errors immediately fail the request.

### Python Code Execution

**Two Execution Modes**:
1. **Native**: Direct `subprocess.run()` with timeout
   - ReAct agent generates Python code directly
   - Fast execution with no LLM overhead
   - Uses `prompts/agents/react_system.txt` and `react_thought.txt`

2. **Nanocoder**: Uses nanocoder CLI for autonomous coding
   - ReAct agent generates natural language instructions
   - Nanocoder generates and executes Python code
   - Uses `prompts/agents/react_system_nanocoder.txt` and `react_thought_nanocoder.txt`
   - Requires: `npm install -g @nanocollective/nanocoder`
   - Auto-generates `.nanocoder/config.json` from `config.py` settings

**Workspace Isolation**:
- Each session gets isolated directory: `data/scratch/{session_id}/`
- Files persist across tool calls within same session
- Cleaned up when session ends

**Output Handling**:
- Always captures stdout, stderr, return code
- Tracks files created in workspace
- Nanocoder mode includes autonomous error handling

### File Attachments

Files attached to messages get **automatic** rich metadata extraction (no tool call needed):
- **JSON**: Structure, keys, sample data
- **CSV/Excel**: Headers, row count, sample rows
- **Python**: Imports, function/class definitions, preview
- **PDF/DOCX**: Basic metadata

**Flow**:
1. Files uploaded → `extract_file_metadata()` in `backend/utils/file_handler.py` extracts metadata
2. Metadata formatted via `format_attached_files()` in `base_agent.py`
3. Auto-injected into system prompt before LLM sees it

This means LLMs can "see" file contents and structure without calling any tool.

### Streaming

Chat completions support SSE (Server-Sent Events) streaming:
- Client sends `"stream": true` in request
- Server yields `data: [DONE]` when complete
- Implemented in `backend/api/routes/chat.py`

## Common Gotchas

1. **Always start tools server before main server** - Main server will fail if tools server isn't running
2. **Password byte length != character length** - Bcrypt limit is 72 BYTES, not characters
3. **Session IDs must be unique** - Used for workspace isolation in python_coder tool
4. **Tool timeouts are critical** - Long-running tools (python_coder, websearch) have extended timeouts
5. **Conversation history size** - Limited by `MAX_CONVERSATION_HISTORY`, older messages are truncated
6. **CORS configuration** - Update `CORS_ORIGINS` in config for frontend deployment

## File Organization

```
backend/
  agents/          - Agent implementations
  api/routes/      - FastAPI route handlers
  core/            - Core services (LLM, database)
  models/          - Pydantic schemas
  utils/           - Utilities (auth, file handling)
tools/
  web_search/      - Tavily web search integration
  python_coder/    - Code execution (native + nanocoder)
  rag/             - RAG document retrieval
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

## Key Configuration Settings

When modifying functionality, check these config variables:

- **Agent behavior**: `REACT_MAX_ITERATIONS`, `REACT_RETRY_ON_ERROR`, `PLAN_*`
- **Tool settings**: `TOOL_MODELS`, `TOOL_PARAMETERS`, `DEFAULT_TOOL_TIMEOUT`
- **Code execution**: `PYTHON_EXECUTOR_MODE`, `NANOCODER_PATH`, `NANOCODER_TIMEOUT`
- **Web search**: `TAVILY_API_KEY`, `TAVILY_SEARCH_DEPTH`, `WEBSEARCH_MAX_RESULTS`
- **RAG**: `RAG_EMBEDDING_MODEL`, `RAG_CHUNK_SIZE`, `RAG_INDEX_TYPE`
- **LLM**: `LLM_BACKEND`, `OLLAMA_MODEL`, `DEFAULT_TEMPERATURE`

## Database Schema

**Users Table**:
- `id`: Primary key
- `username`: Unique username
- `password_hash`: Bcrypt hash
- `role`: "admin" or "user"
- `created_at`: Timestamp

**Sessions Table**:
- `id`: Session ID (primary key)
- `username`: Owner username
- `created_at`: Timestamp
- `message_count`: Number of messages

**Conversation Storage** (JSON files):
```json
{
  "session_id": "...",
  "updated_at": "2024-...",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

## Adding New Tools

When implementing new tools, follow this pattern:

1. **Create tool implementation** in `tools/{tool_name}/tool.py`
   - Implement tool logic with proper error handling
   - Return structured response: `{"success": bool, "answer": str, "data": dict, "metadata": dict}`
   - Support session-based workspace if needed (use `config.SCRATCH_DIR / session_id`)

2. **Add tool schema** to `tools_config.py`
   - Define in `TOOL_SCHEMAS` dict with name, description, endpoint, parameters, returns
   - Tool will be auto-discovered by agents

3. **Add API endpoint** in `backend/api/routes/tools.py`
   - Create request schema (Pydantic BaseModel)
   - Add route handler: `@router.post("/api/tools/{tool_name}")`
   - Add to `/api/tools/list` endpoint

4. **Add parameter parsing** in `backend/agents/react_agent.py`
   - Add case in `_convert_string_to_params()` method to parse tool-specific input

5. **Update configuration** in `config.py`
   - Add to `AVAILABLE_TOOLS` list
   - Add tool-specific settings if needed (timeouts, models, etc.)
   - Add to `TOOL_MODELS` and `TOOL_PARAMETERS` dicts

6. **Update documentation** in this file
   - Add tool to "Available Tools" section with brief description

## Recent Changes

- **2025-12-31**: Removed `read_file` tool (redundant with automatic file metadata injection)
- See `BCRYPT_PASSWORD_FIX.md` for details on password validation improvements to handle bcrypt's 72-byte limit
