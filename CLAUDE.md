# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LLM_API is an OpenAI-compatible FastAPI backend that supports multiple LLM backends (Ollama and llama.cpp) with automatic fallback. It provides:
- **Agent System**: Chat, ReAct (reasoning+action), Plan-Execute, and Auto-routing agents
- **Tool Calling**: Extensible tool system with web search, Python execution, and RAG
- **LLM Logging**: All LLM interactions logged to prompts.log
- **Session Management**: Conversation history and file uploads
- **Authentication**: JWT-based user authentication

## Running the Application

### Start the server
```bash
python server.py
```

Server will start on `http://0.0.0.0:1007` by default.

### API Documentation
- Swagger UI: http://localhost:1007/docs
- ReDoc: http://localhost:1007/redoc
- Health Check: http://localhost:1007/health

### Testing LLM Backend Availability
```bash
# Check if Ollama is running
curl -s http://127.0.0.1:11434/api/tags

# Check if llama.cpp is running
curl -s http://127.0.0.1:8080/v1/models
```

## Agent System

### Overview

The system uses different agent types to handle requests based on complexity:

1. **ChatAgent** (`backend/agents/chat_agent.py`): Simple conversational agent for general questions
2. **ReActAgent** (`backend/agents/react_agent.py`): Reasoning and Action agent with tool calling support
3. **PlanExecuteAgent** (`backend/agents/plan_execute_agent.py`): Multi-step planning agent that uses ReAct for execution
4. **AutoAgent** (`backend/agents/auto_agent.py`): LLM-based router that selects the best agent for the task

### Agent Selection

In the chat endpoint, set `agent_type` parameter:
- `chat` - Direct conversational responses
- `react` - Single or multi-tool usage with reasoning
- `plan_execute` - Complex multi-step tasks
- `auto` - Let LLM decide which agent to use (default)

### ReAct Agent

Supports two formats (configurable via `config.REACT_FORMAT`):
- **"prompt"**: Text-based Thought-Action-Observation format
- **"native"**: Ollama native tool calling (future implementation)

Configuration:
- `REACT_MAX_ITERATIONS`: Maximum reasoning loops (default: 10)
- `REACT_RETRY_ON_ERROR`: Retry failed tool calls (default: True)

### Plan-Execute Agent

Creates a plan and executes each step using ReAct agents.

Configuration:
- `PLAN_MAX_STEPS`: Maximum plan steps (default: 10)
- `PLAN_REPLAN_ON_FAILURE`: Re-plan when steps fail (default: True)
- `PLAN_SHARE_CONTEXT`: Share context across plan steps (default: True)

### Tool System

Tools are defined in [tools_config.py](tools_config.py) with:
- Name, description, endpoint, parameters, return schema
- Currently available: websearch, python_coder, rag

Agents call tools via HTTP to `/api/tools/{tool_name}` endpoints.

## Configuration

All configuration lives in [config.py](config.py) at the project root. Key settings:

- **LLM Backend**: `LLM_BACKEND` can be "ollama", "llamacpp", or "auto" (auto-fallback)
- **Server**: `SERVER_HOST` and `SERVER_PORT` (default: 0.0.0.0:1007)
- **Database**: SQLite at `data/app.db`
- **File Storage**:
  - Persistent uploads: `data/uploads/{username}/`
  - Session scratch files: `data/scratch/{session_id}/`
- **Authentication**: JWT-based with configurable expiration
- **Default Admin**: Username: "admin", Password: "administrator" (CHANGE IN PRODUCTION)
- **Logging**: `LOG_DIR` and `PROMPTS_LOG_PATH` for LLM interaction logs
- **Prompts**: `PROMPTS_DIR` for all agent and tool prompts (configurable)

## Architecture

### Core Components

#### 1. LLM Backend Abstraction ([backend/core/llm_backend.py](backend/core/llm_backend.py))

Unified interface for multiple LLM backends with automatic fallback:

- `OllamaBackend`: Direct integration with Ollama API
- `LlamaCppBackend`: OpenAI-compatible llama.cpp server integration
- `AutoLLMBackend`: Tries Ollama first, falls back to llama.cpp

Key methods: `chat()`, `chat_stream()`, `list_models()`, `is_available()`

Global instance `llm_backend` is wrapped with `LLMInterceptor` for logging (see below).

#### 1a. LLM Interceptor ([backend/core/llm_interceptor.py](backend/core/llm_interceptor.py))

Wraps all LLM backend calls and logs to `data/logs/prompts.log`:

**Log Format**: JSON lines with:
- `timestamp`: ISO format timestamp
- `type`: "chat"
- `streaming`: true/false
- `model`: Model name
- `temperature`: Temperature used
- `messages`: Full message list sent to LLM
- `response`: Complete LLM response
- `duration_seconds`: Time taken
- `estimated_tokens`: Rough token count (input/output/total)
- `success`: true/false
- `error`: Error message (if failed)
- `backend`: Which backend was used (OllamaBackend/LlamaCppBackend)

**Important**: All agent LLM calls are automatically logged via the interceptor.

#### 2. Database Layer ([backend/core/database.py](backend/core/database.py))

Two-part storage system:

- **SQLite Database** (`Database` class): User accounts and session metadata
  - Tables: `users`, `sessions`
  - Handles user authentication and session tracking

- **JSON Conversation Store** (`ConversationStore` class): Human-readable message history
  - Location: `data/sessions/{session_id}.json`
  - Stores full conversation history with timestamps
  - Easy to inspect and debug

Global instances: `db` and `conversation_store`

#### 3. Chat Completions ([backend/api/routes/chat.py](backend/api/routes/chat.py))

OpenAI-compatible endpoint with extensions:

- **Endpoint**: `POST /v1/chat/completions`
- **Format**: Multipart form-data (supports file uploads)
- **Extensions**:
  - `session_id`: Continue existing conversations
  - `agent_type`: Future agent selection (currently unused)
  - `files`: Upload files to include in context
  - `x_session_id`: Returned in response for tracking

**Important**: Messages with uploaded files are augmented by appending file contents to the last user message. See `_prepare_messages_with_files()`.

#### 4. File Handling ([backend/utils/file_handler.py](backend/utils/file_handler.py))

Dual storage system:

- **Persistent**: `data/uploads/{username}/` - User's uploaded files persist
- **Scratch**: `data/scratch/{session_id}/` - Temporary files per session

Supported file types: `.txt`, `.md`, `.json`, `.csv`, `.py`, `.js`, `.html`, `.xml`, `.xlsx`, `.xls`

Binary files are marked as `[Binary file: filename]` in context.

### API Routes

All routes defined in [backend/api/app.py](backend/api/app.py):

- `/api/auth/*` - Authentication (login, signup)
- `/v1/models` - List available LLM models (OpenAI-compatible)
- `/v1/chat/completions` - Chat completions with streaming support
- `/api/chat/sessions` - List user sessions
- `/api/chat/history/{session_id}` - Get conversation history
- `/api/admin/*` - Admin operations (user management)
- `/api/tools/*` - Tool endpoints (websearch, etc.)

### Authentication ([backend/utils/auth.py](backend/utils/auth.py))

JWT-based authentication:

- Token format: Bearer token in Authorization header
- Optional authentication: Most endpoints work for guests ("guest" user)
- Admin routes require authenticated admin user
- Passwords hashed with bcrypt

### Request/Response Schemas ([backend/models/schemas.py](backend/models/schemas.py))

Pydantic models for validation:

- OpenAI-compatible: `ChatMessage`, `ChatCompletionResponse`, `ModelObject`
- Extensions: `SessionInfo`, `ToolInfo`, `WebSearchRequest`
- Custom field: `x_session_id` for session tracking

## Current State: Refactoring Branch

The repository is currently on the `refactoring` branch with many files deleted/modified:

- Old agent system removed (plan_execute_agent, react_agent)
- Old tool implementations removed
- Simplified to core chat functionality
- Tools API has stub implementation for future work

**Active development**: Tools endpoints exist but are not fully implemented. Web search has basic Tavily integration.

## Tool System (Future Implementation)

Configuration in [config.py](config.py):

- `AVAILABLE_TOOLS`: ["websearch", "python_coder", "rag"]
- `TOOL_MODELS`: Specific models for each tool
- `TOOL_PARAMETERS`: Custom inference parameters per tool

Web search provider: Configurable (Tavily/Serper), currently Tavily with API key in config.

## Prompt System

All agent and tool prompts are stored in `/prompts` directory and configurable:

**Agent Prompts** (`prompts/agents/`):
- `auto_router.txt`: LLM-based agent selection logic
- `chat_system.txt`: System prompt for ChatAgent
- `react_system.txt`: ReAct agent system prompt with tool descriptions
- `react_thought.txt`: ReAct thought template for each iteration
- `plan_system.txt`: Planning prompt for Plan-Execute agent
- `plan_execute.txt`: Step execution prompt

**Modifying Prompts**:
1. Edit the `.txt` files directly
2. Use `{variable}` syntax for template variables
3. Agents load prompts via `self.load_prompt(path, **kwargs)`
4. No code changes needed - prompts are loaded at runtime

## Development Patterns

### Using Agents

```python
from backend.agents import ChatAgent, ReActAgent, AutoAgent

# Create agent
agent = ReActAgent(model="deepseek-r1:1.5b", temperature=0.7)

# Run agent
response = agent.run(
    user_input="Search for Python tutorials",
    conversation_history=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
)
```

### Adding a new API endpoint

1. Define Pydantic schemas in [backend/models/schemas.py](backend/models/schemas.py)
2. Create route in `backend/api/routes/your_route.py`
3. Include router in [backend/api/app.py](backend/api/app.py)

### Using the LLM backend

```python
from backend.core.llm_backend import llm_backend

# Non-streaming
response = llm_backend.chat(messages, model, temperature)

# Streaming
for token in llm_backend.chat_stream(messages, model, temperature):
    yield token
```

### Working with sessions

```python
from backend.core.database import db, conversation_store

# Create session
session_id = str(uuid.uuid4())
db.create_session(session_id, username)

# Save conversation
conversation_store.save_conversation(session_id, messages)

# Load conversation
history = conversation_store.load_conversation(session_id)
```

## Windows-Specific Notes

This codebase is developed on Windows (git status shows Windows-style paths). When running bash commands:

- Use forward slashes or quote paths: `python server.py`
- Python available via `python` command
- Directory commands: Use `dir` for Windows or `ls` if Git Bash is available

## Dependencies

All dependencies in [requirements.txt](requirements.txt). Major libraries:

- **FastAPI**: Web framework
- **uvicorn**: ASGI server
- **httpx**: HTTP client for LLM backends
- **pydantic**: Data validation
- **passlib/bcrypt**: Password hashing
- **python-jose**: JWT handling
- **sqlite3**: Database (built-in)
- **langchain**: For future agent/tool implementations
- **tavily-python**: Web search (optional)
- **pandas**: Excel file support

Install with: `pip install -r requirements.txt`

## Important Conventions

1. **Global instances**: `llm_backend`, `db`, `conversation_store` are globally instantiated and imported
2. **Error handling**: Global exception handler in app.py catches all unhandled exceptions
3. **Streaming**: Uses Server-Sent Events (SSE) via `sse_starlette.EventSourceResponse`
4. **File uploads**: Always use multipart/form-data, files saved to both persistent and scratch locations
5. **Session IDs**: UUIDs generated for each new conversation
6. **Authentication**: Optional for most endpoints - defaults to "guest" user

## Common Pitfalls

- **Backend availability**: Always check `llm_backend.is_available()` before making requests
- **File encoding**: Text files assume UTF-8 encoding
- **JSON in forms**: Chat messages sent as JSON string in form field, must be parsed
- **Stream format**: Streaming responses use different format than non-streaming (delta vs message)
- **Session persistence**: Conversations persist in JSON files even after server restart
