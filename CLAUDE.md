# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Agentic AI Backend** with FastAPI that provides OpenAI-compatible APIs. The system uses **LangGraph** for multi-step reasoning workflows and supports two agent architectures: **ReAct** (Reasoning + Acting) and **Plan-and-Execute**. The smart agent router automatically selects the best approach based on query characteristics.

**Core Technologies:**
- FastAPI with OpenAI-compatible endpoints
- LangGraph for agent orchestration
- Ollama (local LLM: gpt-oss:20b)
- FAISS/Chroma for vector storage
- JWT authentication with bcrypt

## Common Commands

### Development

```bash
# Backend
python run_backend.py              # Cross-platform launcher (recommended)
python -m backend.api.app          # Direct execution

# Frontend
python run_frontend.py             # Serve static frontend (port 3000)
python run_frontend.py --no-browser  # Without auto-opening browser

# Configuration
python create_env.py               # Generate .env file with defaults
```

### Testing

```bash
# Comprehensive API testing
python test_all_apis.py            # Tests all endpoints with health checks

# Feature-specific tests
python test_react_agent.py         # ReAct agent functionality
python test_new_features.py        # New features validation
python test_new_tools.py           # Tool integration tests
python test_difficult_math.py      # Math calculator tests
```

### Ollama Management

```bash
ollama list                        # List installed models
ollama serve                       # Start Ollama service
ollama pull gpt-oss:20b           # Download the default model
```

### Dependency Management

```bash
pip install -r requirements.txt    # Install all dependencies
pip freeze > requirements.txt      # Update requirements (if needed)
```

## Architecture

### Agent System - Dual Architecture

The system uses a **smart agent router** that automatically selects between two agent types:

**1. ReAct Agent** ([backend/core/react_agent.py](backend/core/react_agent.py))
- **Pattern:** Iterative Thought → Action → Observation loops
- **Best for:** Exploratory queries, sequential reasoning, dynamic tool selection
- **Example:** "Find the capital of France, then search for its population, then calculate if it's larger than London"
- **Max iterations:** 5 (configurable)

**2. Plan-and-Execute Agent** ([backend/core/agent_graph.py](backend/core/agent_graph.py))
- **Pattern:** LangGraph workflow with 5 stages (Planning → Tool Selection → Tool Execution → Reasoning → Verification)
- **Best for:** Complex batch queries, parallel tool usage, well-defined comprehensive tasks
- **Example:** "Search weather AND analyze data AND retrieve documents"
- **Max iterations:** 3 with verification loops

**Smart Router** ([backend/tasks/smart_agent_task.py](backend/tasks/smart_agent_task.py))
- Analyzes query keywords and structure
- Scores queries for react_indicators vs plan_indicators
- Routes to appropriate agent automatically
- Default: ReAct for flexibility and transparency

### Task Routing Flow

```
User Query → determine_task_type() → "chat" or "agentic"
                                           ↓
                        "agentic" → SmartAgentTask → ReAct or Plan-and-Execute
                        "chat" → ChatTask (simple Ollama call)
```

**Trigger keywords for "agentic" mode:**
- search, find, look up, research
- compare, analyze, investigate
- current, latest, news
- document, file, pdf

### Available Tools

All tools are in [backend/tools/](backend/tools/):

1. **web_search.py** - Tavily API for web search (websearch_ts fallback)
2. **rag_retriever.py** - Document Q&A with FAISS/Chroma vector DB
3. **data_analysis.py** - JSON data statistics (min/max/mean/count)
4. **python_executor.py** - Safe Python code execution
5. **math_calculator.py** - SymPy-based symbolic math
6. **wikipedia_tool.py** - Wikipedia search and summarization
7. **weather_tool.py** - Weather information retrieval
8. **sql_query_tool.py** - SQL database queries

### Configuration System

**CRITICAL:** This project uses a **no fallbacks** design. All settings in `.env` must be explicitly configured. Missing values will raise descriptive errors on startup.

**Configuration priority:**
1. `.env` file (must exist)
2. [backend/config/settings.py](backend/config/settings.py) (provides defaults only for .env template)

**Key variables:**
- `SECRET_KEY` - JWT secret (generate with: `python -c "import secrets; print(secrets.token_urlsafe(32))"`)
- `TAVILY_API_KEY` - Required for web search
- `OLLAMA_HOST` - Ollama service endpoint (default: http://localhost:11434)
- `OLLAMA_MODEL` - Model name (default: gpt-oss:20b)
- `OLLAMA_TIMEOUT` - Request timeout in ms (default: 3000000 = 50 minutes)
- `SERVER_HOST` - Use `localhost` for production, `0.0.0.0` for development

### Data Storage

All data is organized under `./data/`:
- `data/vector_db/` - FAISS/Chroma vector embeddings
- `data/conversations/` - Chat history (JSON files per session)
- `data/uploads/<username>/` - User-isolated uploaded documents
- `data/users/users.json` - User authentication database
- `data/sessions/sessions.json` - Session management
- `data/logs/app.log` - Application logs

### API Endpoints

**OpenAI-Compatible** (`/v1/`):
- `POST /v1/chat/completions` - Main chat endpoint (supports `agent_type` param: "auto", "react", "plan_execute")
- `GET /v1/models` - List available models

**Authentication** (`/api/auth/`):
- `POST /api/auth/login` - Returns JWT token + user data
- `GET /api/auth/me` - Get current user info

**File Management** (`/api/files/`):
- `POST /api/files/upload` - Upload document (user-isolated, creates `uploads/<username>/`)
- `GET /api/files/documents?page=1&page_size=20` - List documents (paginated, user-isolated)
- `DELETE /api/files/documents/{file_id}` - Delete document (user-isolated)

**Health**:
- `GET /` - API info
- `GET /health` - Health check

**Default credentials:**
- Guest: `guest` / `guest_test1`
- Admin: `admin` / `administrator`

## Development Guidelines

### Adding New Tools

1. Create tool file in [backend/tools/](backend/tools/) with async methods
2. Register tool in **both** agent files:
   - [backend/core/react_agent.py](backend/core/react_agent.py) - Add to `ToolName` enum and `_execute_action()` method
   - [backend/core/agent_graph.py](backend/core/agent_graph.py) - Add tool node, update workflow edges
3. Update tool selection logic in both agents:
   - ReAct: Modify `_select_action()` prompt with new action description
   - Plan-and-Execute: Update `tool_selection_node()` keyword detection

### Adding New Agent Types

1. Create agent implementation in [backend/core/](backend/core/)
2. Register in [backend/tasks/smart_agent_task.py](backend/tasks/smart_agent_task.py) - Add to `AgentType` enum
3. Update routing logic in `_select_agent()` method
4. Add detection heuristics for when to use the new agent

### Testing New Features

1. Add test cases to [test_all_apis.py](test_all_apis.py) for comprehensive coverage
2. Create feature-specific test file if needed (see [test_react_agent.py](test_react_agent.py))
3. Test both agent types: set `agent_type` param to "react" and "plan_execute"
4. Verify agent selection logic with "auto" mode

### Authentication Flow

All protected endpoints require:
```python
Authorization: Bearer <jwt_token>
```

JWT tokens:
- Signed with `SECRET_KEY` (HS256)
- Expire after `JWT_EXPIRATION_HOURS` (default: 24)
- Include `{"sub": username}` payload
- Validated by `get_current_user()` dependency in [backend/utils/auth.py](backend/utils/auth.py)

### Conversation History

Managed by [backend/storage/conversation_store.py](backend/storage/conversation_store.py):
- JSON-based file storage (one file per session)
- User-isolated (keyed by user_id)
- Loaded automatically when `session_id` is provided
- Created automatically for new conversations

### Document Processing (RAG)

Pipeline in [backend/tools/rag_retriever.py](backend/tools/rag_retriever.py):
1. Upload → User-specific folder `uploads/<username>/`
2. Load document (PDF, DOCX, TXT, JSON)
3. Text chunking with `RecursiveCharacterTextSplitter`
4. Embedding with `all-MiniLM-L6-v2` model
5. Vector storage (FAISS or Chroma based on `VECTOR_DB_TYPE`)
6. Retrieval with similarity search (top_k=5)

**Note:** JSON files are loaded as text documents using custom loader to avoid import conflicts.

## Important Implementation Notes

### LangGraph State Management

State structure in [agent_graph.py](backend/core/agent_graph.py:27-42):
```python
class AgentState(TypedDict):
    messages: List[ChatMessage]      # Conversation
    session_id: str                  # Session tracking
    user_id: str                     # User context
    plan: str                        # Execution plan
    tools_used: List[str]            # Active tools
    search_results: str              # Web search output
    rag_context: str                 # Document context
    data_analysis_results: str       # Analysis output
    current_agent: str               # Current node
    final_output: str                # Generated response
    verification_passed: bool        # Quality check
    iteration_count: int             # Loop counter
    max_iterations: int              # Loop limit
```

State flows through nodes, accumulating information. Use `Annotated[List, operator.add]` for messages to append rather than replace.

### ReAct Step Tracing

ReAct agent stores full execution trace:
```python
react_agent.get_trace()  # Returns formatted Thought-Action-Observation history
```

Useful for debugging and understanding agent reasoning.

### Multi-User Isolation

**Files:** User-specific folders prevent cross-user access
- Upload path: `uploads/<username>/<file_id>_<filename>`
- File listing and deletion are user-scoped by JWT token

**Conversations:** Session IDs are globally unique but associated with user_id

**Vector DB:** Currently shared across users (consider adding user_id metadata for production isolation)

### Prompt Engineering

Key prompts to customize agent behavior:

**ReAct Agent** ([react_agent.py](backend/core/react_agent.py)):
- `_generate_thought()` - Line 165: Controls reasoning process
- `_select_action()` - Line 193: Tool selection prompt with available actions list

**Plan-and-Execute Agent** ([agent_graph.py](backend/core/agent_graph.py)):
- `planning_node()` - Line 74: Query analysis and plan creation
- `reasoning_node()` - Line 221: Response generation with context
- `verification_node()` - Line 255: Quality check criteria

### Error Handling

- All async tool calls should use try/except with logging
- Return user-friendly error messages (don't expose internal details)
- FastAPI automatically converts raised `HTTPException` to proper responses
- Check [backend/api/app.py](backend/api/app.py) for global exception handlers

### Logging Strategy

Logger instances in each module:
```python
logger = logging.getLogger(__name__)
logger.info(f"[Component Name] Description")
```

Log levels (set via `LOG_LEVEL` in .env):
- DEBUG: Detailed execution traces
- INFO: High-level operation flow (default)
- WARNING: Production-friendly (minimal output)

## Version History & Migration

**v2.0.0** - Frontend UI upgrade with user-isolated file management
**v1.1.4** - Comprehensive API testing suite
**v1.1.3** - Dependency fixes (LangChain, Keras compatibility)
**v1.1.2** - Cross-platform backend launcher
**v1.1.1** - API examples simplification
**v1.1.0** - Configuration management system
**v1.0.0** - Initial release with GitHub integration

See [README.md](README.md) for detailed version history.

## Troubleshooting

**"Configuration error"**
→ Copy `.env.example` to `.env` and set `SECRET_KEY` and `TAVILY_API_KEY`

**"Connection refused" to Ollama**
→ Start Ollama: `ollama serve`

**"Model not found"**
→ Pull model: `ollama pull gpt-oss:20b`

**Port 8000 in use**
→ Change `SERVER_PORT` in `.env` or kill process on port 8000

**ModuleNotFoundError**
→ Reinstall dependencies: `pip install -r requirements.txt`

**JSON file upload fails in RAG**
→ Known issue - custom JSON loader treats JSON as text documents, which may not parse structured data correctly

**Agentic queries timeout**
→ Increase `OLLAMA_TIMEOUT` in `.env` or reduce `max_iterations` in agent constructors

## Documentation Files

- [README.md](README.md) - Quick start and version history
- [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed system architecture with diagrams
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Complete feature overview
- [GETTING_STARTED.md](GETTING_STARTED.md) - 5-minute setup guide
- [API_examples.ipynb](API_examples.ipynb) - Jupyter notebook with API usage examples
