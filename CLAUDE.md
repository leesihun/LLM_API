# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **AI-powered LLM API server** with sophisticated **agentic workflow capabilities**. The system integrates Large Language Models (via Ollama) with multi-agent reasoning patterns (ReAct, Plan-Execute) to handle complex tasks including web search, document retrieval (RAG), and autonomous Python code generation/execution.

**Key Technologies:**
- Backend: FastAPI + LangChain + LangGraph
- LLM: Ollama (default model: gemma3:12b)
- Agent Patterns: ReAct (Reasoning + Acting), Plan-Execute
- Tools: Web search (Tavily), RAG (FAISS), Python code generation/execution

## Essential Commands

### Development Workflow

**Start the backend server:**
```bash
python run_backend.py
# Or use the main entry point:
python server.py
```
Server runs at: `http://0.0.0.0:1007`

**Start the frontend (separate terminal):**
```bash
python run_frontend.py
```
Frontend runs at: `http://localhost:3000`

**Prerequisites:**
- Ollama must be running: `ollama serve`
- Required models must be pulled:
  ```bash
  ollama pull gemma3:12b
  ollama pull gpt-oss:20b
  ollama pull bge-m3:latest
  ```

**Environment Setup:**
```bash
# Install dependencies
pip install -r requirements.txt

# Configuration is in backend/config/settings.py
# .env file is OPTIONAL - only override specific values if needed
```

### Testing

**Manual testing with Jupyter:**
```bash
jupyter notebook API_examples.ipynb
```

**Check Ollama connection:**
```bash
curl http://127.0.0.1:11434/api/tags
```

## Architecture Overview

### High-Level System Flow

```
User Request → Classifier → [Agentic Flow vs Simple Chat]
                ↓
         Agentic Flow:
         1. Task Detection (chat_task.py)
         2. ReAct Agent (React.py) OR Smart Agent (smart_agent_task.py)
         3. Tool Execution (web_search, rag, python_coder)
         4. Response Synthesis
```

### Critical Architecture Patterns

**1. Task Classification (backend/tasks/chat_task.py)**
- Every user message first goes through the classifier
- Classifier decides: "agentic" (requires tools) vs "chat" (simple response)
- Classifier model: `gpt-oss:20b` (configurable in settings.py)
- Triggers: keywords like "search", "analyze", "code", "calculate", "current", "latest"

**2. ReAct Agent Pattern (backend/tasks/React.py)**
- The primary agentic execution engine
- Implements Thought → Action → Observation loop
- Max iterations: 10 (configurable via `max_iterations` parameter)
- Tools available: `web_search`, `rag_retrieval`, `python_coder`, `finish`
- **Critical optimization:** Combined thought-action generation (1 LLM call instead of 2)
- **Early exit:** Auto-finish when observation contains complete answer
- **Context pruning:** If >3 steps, summarizes early steps and keeps last 2 in detail

**3. Python Code Generation (backend/tools/python_coder_tool.py)**
- **Unified architecture** (v1.2.0): Single file for generation + execution
- **Verification phase:** Max 3 iterations, focused ONLY on "Does code answer user's question?"
- **Execution retry:** Max 5 attempts with auto-fixing between retries
- **File handling:** Supports CSV, Excel, JSON, PDF, images, etc.
- **Security:** Sandboxed execution, import restrictions, timeout controls
- Important: Uses session-based execution directories (persisted if session_id provided)

**4. Dual Agent Strategy (backend/tasks/smart_agent_task.py)**
- For complex multi-step tasks
- Planner creates structured plan with steps
- Executor (ReAct agent) executes each step with tool fallback
- Each step has: goal, success_criteria, primary_tools, fallback_tools

### Key Configuration Files

**backend/config/settings.py:**
- Central configuration - ALL defaults defined here
- `.env` file is OPTIONAL (only for overrides)
- Critical settings:
  - `ollama_model`: Default 'gemma3:12b'
  - `python_code_max_iterations`: 5 (was 3, updated for better verification)
  - `python_code_timeout`: 3000 seconds
  - `python_code_allow_partial_execution`: True
  - `agentic_classifier_prompt`: Controls when to use agentic vs chat flow

**Important:** System forces settings.py defaults over environment variables to prevent conflicts.

### Directory Structure

```
backend/
├── api/              # FastAPI routes and application
│   ├── app.py       # Main FastAPI app, CORS, middleware, startup
│   └── routes.py    # API endpoints (chat, auth, files, admin, tools)
├── config/
│   └── settings.py  # Centralized configuration (all defaults here)
├── core/
│   └── agent_graph.py  # LangGraph workflow (alternative to ReAct)
├── tasks/
│   ├── chat_task.py        # Entry point: task classification
│   ├── React.py            # PRIMARY: ReAct agent implementation
│   ├── smart_agent_task.py # Plan-Execute workflow
│   └── Plan_execute.py     # Legacy plan-execute
├── tools/
│   ├── python_coder_tool.py  # Python code gen/exec (unified v1.2.0)
│   ├── web_search.py         # Tavily web search
│   └── rag_retriever.py      # Document retrieval (FAISS)
├── storage/
│   └── conversation_store.py  # Conversation persistence
├── models/
│   └── schemas.py            # Pydantic data models
└── utils/
    └── auth.py               # JWT authentication

data/
├── conversations/    # Stored chat history (JSON files)
├── uploads/          # User uploaded files
├── scratch/          # Code execution workspace (session-based)
└── logs/             # Application logs
```

## Working with Python Code Tool

### Code Generation Flow
1. **File Preparation:** Validate file types, extract metadata (columns, dtypes, preview)
2. **Code Generation:** LLM generates code with file context
3. **Verification Loop (max 3):**
   - Static analysis (import checks)
   - LLM semantic check: "Does code answer question?"
   - Modify code if issues found
4. **Execution Loop (max 5):**
   - Execute code in subprocess
   - On failure: LLM analyzes error and fixes code
   - Retry with fixed code

### When Modifying python_coder_tool.py:
- **CodeExecutor class:** Low-level execution (subprocess, file handling, import validation)
- **PythonCoderTool class:** High-level orchestration (generation, verification, retry)
- Backward compatibility: `PythonExecutor = CodeExecutor` (legacy alias)
- File metadata extraction is critical for good code generation (see `_extract_file_metadata`)

### Security Considerations:
- Blocked imports: socket, subprocess, eval, exec, pickle, etc.
- Execution timeout enforced via subprocess
- Session-based directories prevent cross-contamination
- Import validation uses AST parsing (not runtime)

## Working with ReAct Agent

### Key Methods in React.py:
- `execute()`: Main entry point, runs full ReAct loop
- `execute_with_plan()`: Guided mode for structured plan execution
- `_generate_thought_and_action()`: Combined LLM call (performance optimization)
- `_execute_action()`: Routes to appropriate tool (web_search, rag, python_coder)
- `_generate_final_answer()`: Synthesizes observations into final response

### ReAct Execution Optimizations:
1. **Pre-step file handling:** If files attached, tries python_coder before starting loop
2. **Combined thought-action:** Single LLM call instead of separate thought + action calls
3. **Context pruning:** Summarizes old steps, keeps recent 2 in full detail
4. **Early exit:** Auto-finish if observation contains complete answer (≥200 chars, answer phrases)
5. **Guard logic:** If RAG requested but files exist, tries python_coder first

### Common ReAct Issues:
- **Empty final answers:** Check `_generate_final_answer()`, uses observations from all steps
- **Infinite loops:** Check `_should_auto_finish()` logic and max_iterations
- **Tool selection:** LLM decides via `_parse_thought_and_action()`, fuzzy matching applied

## API Endpoints

**Main Chat Endpoint:**
```python
POST /api/chat
{
  "message": "user message",
  "session_id": "optional-session-id",
  "user_id": "user-id",
  "file_paths": ["optional", "file", "paths"]  # Uploaded file references
}
```

**File Upload:**
```python
POST /api/upload
# Multipart form data with files
# Returns file paths for use in /api/chat
```

**Conversation History:**
```python
GET /api/conversations?user_id=<id>
```

**Health Check:**
```python
GET /health
# Returns Ollama connection status
```

## Important Implementation Details

### Ollama Configuration
- Default timeout: 3000 seconds (50 minutes)
- Context window: 4096 tokens
- Temperature: 0.3 (conservative for code generation)
- Connection tested on startup (see app.py startup_event)

### LangGraph vs ReAct
- **agent_graph.py:** LangGraph implementation (structured workflow with verification node)
- **React.py:** Direct ReAct implementation (more flexible, currently primary)
- Both are available; ReAct is the current production choice

### Session Management
- User authentication via JWT (utils/auth.py)
- Session data stored in `data/conversations/`
- File format: `{user_id}_{timestamp}_{session_id}.json`

### Conversation Storage
- Each message saved with: role, content, timestamp, metadata
- Tools track execution history (steps, observations, code, etc.)
- Retrieval by user_id or session_id

## Common Development Tasks

### Adding a New Tool
1. Create tool class in `backend/tools/`
2. Add to `ToolName` enum in `React.py`
3. Add execution logic in `ReActAgent._execute_action()`
4. Update tool selection in `_generate_thought_and_action()` prompt
5. Add to `settings.available_tools` list

### Modifying Agentic Classifier
- Edit `settings.agentic_classifier_prompt` in settings.py
- Controls when to trigger agentic flow vs simple chat
- Be conservative: agentic flow is slower but more capable

### Changing Verification/Retry Limits
```python
# In settings.py:
python_code_max_iterations: int = 5  # Verification iterations
# In python_coder_tool.py:
self.max_execution_attempts = 5      # Execution retry attempts
# In React.py:
react_agent = ReActAgent(max_iterations=10)  # ReAct loop limit
```

### Debugging Execution Issues
- Check logs in `data/logs/app.log`
- Python code execution dirs: `data/scratch/<session_id>/`
- Execution preserves directory if session_id provided
- Review `verification_history` and `execution_attempts_history` in tool response

## Version History & Breaking Changes

### Version 1.2.0 (October 31, 2024)
**Major unification of Python code tools:**
- Merged `python_coder_tool.py` and `python_executor_engine.py` into single file
- Removed `python_executor.py` (unused legacy)
- Reduced verification iterations: 10 → 3
- Added execution retry logic: max 5 attempts with auto-fixing
- `ToolName.PYTHON_CODE` removed → use `ToolName.PYTHON_CODER`
- Import changes: `PythonExecutor` now alias for `CodeExecutor`

### Critical Breaking Changes from v1.1.0:
- Old import: `from backend.tools.python_executor_engine import PythonExecutor`
- New import: `from backend.tools.python_coder_tool import CodeExecutor` (or use `PythonExecutor` alias)

## Performance Tips

### Reducing LLM Calls
- Combined thought-action generation saves 50% calls in ReAct loop
- Early exit with auto-finish prevents unnecessary iterations
- Context pruning reduces token usage for long conversations

### Optimizing Python Code Execution
- Set realistic timeouts (default 3000s is generous)
- Use session-based directories to cache file processing
- Extract file metadata to guide code generation (reduces verification failures)

### When to Use Which Agent
- **Simple tasks:** Direct chat (no agentic flow)
- **Single tool tasks:** ReAct agent (fast, flexible)
- **Complex multi-step:** Smart agent with Plan-Execute (structured, reliable)

## Troubleshooting

**Ollama Connection Errors:**
- Verify Ollama is running: `ollama serve`
- Check settings.ollama_host (default: http://127.0.0.1:11434)
- Settings.py forces this default even if env vars exist

**Code Execution Timeouts:**
- Increase settings.python_code_timeout
- Check for infinite loops in generated code
- Review execution directory: data/scratch/<session_id>/

**Empty or Incomplete Responses:**
- Check agentic classifier - may need to adjust prompt
- Review ReAct final answer generation (`_generate_final_answer`)
- Verify observations contain data (check tool execution logs)

**Import Errors in Generated Code:**
- Review BLOCKED_IMPORTS in python_coder_tool.py
- Ensure required packages installed in environment
- Check AST validation in `CodeExecutor.validate_imports()`

---

**Last Updated:** January 2025
**Version:** 1.2.0
