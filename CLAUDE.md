# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AI-powered LLM API server** with sophisticated agentic workflow capabilities. The system integrates Large Language Models via dual backend support (Ollama + llama.cpp) with multi-agent reasoning patterns (ReAct, Plan-Execute) to handle complex tasks including web search, document retrieval (RAG), and autonomous Python code generation/execution.

**Key Technologies:**
- Backend: FastAPI + LangChain + LangGraph
- **LLM Backends:**
  - **Ollama** (default): Server-based inference with model management
  - **llama.cpp**: Direct GGUF model loading with fine-grained hardware control
- Agent Patterns: ReAct (Reasoning + Acting), Plan-Execute
- Tools: Web search (Tavily), RAG (FAISS), Python code generation/execution, Vision analysis

---

## Essential Commands

### Development Workflow

**Start the backend server:**
```bash
python run_backend.py
# Or directly:
python server.py
```
Server runs at: `http://0.0.0.0:1007`

**API Documentation:**
- Swagger UI: `http://localhost:1007/docs`
- ReDoc: `http://localhost:1007/redoc`

**Prerequisites (Ollama Backend - default):**
- Ollama must be running: `ollama serve`
- Models must be available:
  ```bash
  ollama pull gpt-oss:20b
  ollama pull llama3.2-vision:11b
  ```

**Prerequisites (llama.cpp Backend):**
- Download GGUF model files from Hugging Face
- Place models in location specified in `backend/config/settings.py` (default: `../models/`)
- **Recommended models:**
  - Qwen Coder: `https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF`
  - Llama 3: `https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF`

**Environment Setup:**
```bash
# Install dependencies
pip install -r requirements.txt

# Note: llama-cpp-python may need GPU-specific build
# For CUDA support (NVIDIA):
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# For Metal support (Apple Silicon):
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

**Configuration:**
- All configuration in `backend/config/settings.py` with defaults
- `.env` file is OPTIONAL - only override specific values if needed
- Settings are 12-factor app compliant (environment variables supported)

### Testing

**Manual testing with Jupyter:**
```bash
jupyter notebook API_examples.ipynb
```

**Check Ollama connection:**
```bash
curl http://127.0.0.1:11434/api/tags
```

**Check backend health:**
```bash
curl http://localhost:1007/health
```

---

## High-Level Architecture

### Backend Selection: Ollama vs llama.cpp

Configure backend in [backend/config/settings.py](backend/config/settings.py):

```python
llm_backend: str = 'ollama'  # or 'llamacpp'
```

| Feature | Ollama | llama.cpp |
|---------|--------|-----------|
| **Deployment** | Server-based (external service) | Direct in-process loading |
| **Model Format** | Ollama model library | GGUF files |
| **GPU Control** | Automatic | Fine-grained (layer-by-layer) |
| **Memory Mapping** | Managed by Ollama | Configurable (mmap, mlock) |
| **Context Extension** | Limited by model | RoPE scaling support |
| **Best For** | Quick setup, model management | Production, custom deployments, fine control |

### Agent Orchestration Flow

```
User Request → AgentOrchestrator → Resolve Agent Type
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
            SimpleChatAgent   ReActAgent   Plan-Execute
              (no tools)     (flexible)    (structured)
                    ↓               ↓               ↓
               Response    Tool Execution   Step-by-Step
                           ↓
                   ┌───────┼───────┐
                   ↓       ↓       ↓
            web_search  python_coder  rag_retrieval
            vision_analyzer  file_analyzer
```

**Agent Type Resolution:**
- **AUTO mode** (default): Automatically chooses based on query complexity and file attachments
- **CHAT**: Simple LLM response (no tools)
- **REACT**: Flexible tool-based reasoning (3-6 iterations)
- **PLAN_EXECUTE**: Structured multi-step execution with planning phase

### Critical Architecture Patterns

**1. Agent Routing (backend/agents/react_agent.py:146)**

The `AgentOrchestrator` automatically determines which execution mode to use:
- File attachments → Always use ReAct agent
- Query length >120 chars → Likely ReAct
- Keywords ("analyze", "search", "report") → Likely ReAct
- Otherwise → Simple chat

**2. ReAct Agent Pattern**

Main reasoning loop in `ReActAgent.execute()`:

```python
for i in range(max_iterations):
    # Generate thought and action
    thought, action, action_input = await thought_generator.generate(...)

    # Execute tool
    if action == "finish":
        break
    observation = await tool_executor.execute(action, action_input, ...)

    # Continue loop with observation
```

**Components:**
- `ThoughtActionGenerator`: LLM-based thought/action generation
- `ToolExecutor`: Routes to appropriate tool (web_search, python_coder, etc.)
- `AnswerGenerator`: Synthesizes final answer from observations
- `ContextFormatter`: Manages execution history with context pruning

**3. Python Code Generation & Execution**

The python_coder tool (`backend/tools/python_coder.py`) supports autonomous code generation with:

- **Context awareness**: Uses conversation history, plan context, and ReAct iteration history
- **File handling**: Automatically processes CSV, Excel, JSON, PDF, images
- **Execution sandbox**: Isolated execution with timeout controls
- **Session persistence**: Variables and code persist across requests via session_id
- **Retry logic**: Max 3 attempts with automatic error fixing

Key execution flow:
```python
result = await python_coder_tool.execute_code_task(
    query=query,
    file_paths=file_paths,
    session_id=session_id,
    react_context=react_context,  # Previous attempts, failures
    plan_context=plan_context,     # Plan step information
    conversation_history=history    # Full conversation context
)
```

**4. Tool System**

All tools inherit from `BaseTool` (backend/core/base_tool.py):

```python
class BaseTool(ABC):
    @abstractmethod
    async def execute(self, query: str, context: Optional[str] = None, **kwargs) -> ToolResult:
        pass

    @abstractmethod
    def validate_inputs(self, **kwargs) -> bool:
        pass
```

**Available tools:**
- `web_search_tool`: Tavily API integration for real-time web search
- `rag_retriever_tool`: FAISS-based retrieval from uploaded documents
- `python_coder_tool`: Autonomous Python code generation and execution
- `file_analyzer`: File metadata extraction and analysis
- `vision_analyzer_tool`: Image understanding with multimodal LLMs

### File Handling System

**Unified file handler registry** (`backend/core/file_handlers/`):
- Single source of truth for all file operations
- Supports: CSV, Excel, JSON, Text, PDF, DOCX, Images
- Accessed via singleton: `from backend.core.file_handlers import file_handler_registry`

**Usage:**
```python
handler = file_handler_registry.get_handler(file_path)
metadata = handler.extract_metadata(file_path)
```

### LLM Factory Pattern

Centralized LLM creation via `backend/utils/llm_factory.py`:

```python
from backend.utils.llm_factory import LLMFactory

# Standard LLM
llm = LLMFactory.create_llm(temperature=0.7, user_id="alice")

# Classifier LLM (low temperature for consistency)
classifier = LLMFactory.create_classifier_llm(user_id="alice")

# Coder LLM (optimized for code generation)
coder = LLMFactory.create_coder_llm(user_id="alice")

# Vision LLM (for image understanding)
vision = LLMFactory.create_vision_llm(user_id="alice")
```

**Features:**
- Automatic backend selection (Ollama or llama.cpp)
- Prompt logging with multiple formats (STRUCTURED, JSON, COMPACT)
- User-specific LLM instances with LLMManager
- Health checks and model listing

---

## Key Configuration Files

### backend/config/settings.py

Central configuration with all defaults defined here. `.env` file is OPTIONAL.

**Critical settings:**

```python
# Backend selection
llm_backend: str = 'ollama'  # or 'llamacpp'

# Ollama configuration
ollama_host: str = 'http://127.0.0.1:11434'
ollama_model: str = 'gpt-oss:20b'
ollama_num_ctx: int = 2048  # Context window
ollama_temperature: float = 1.0

# llama.cpp configuration
llamacpp_model_path: str = '../models/gpt-oss-120b.gguf'
llamacpp_n_gpu_layers: int = -1  # -1 = all layers to GPU
llamacpp_n_ctx: int = 2048
llamacpp_temperature: float = 1.0

# ReAct agent limits
react_max_iterations: int = 6
python_code_max_iterations: int = 3

# Python code execution
python_code_timeout: int = 3000  # seconds
python_code_execution_dir: str = './data/scratch'
python_code_use_persistent_repl: bool = True
python_code_preload_libraries: list = ['pandas as pd', 'numpy as np', 'matplotlib.pyplot as plt']

# Available tools
available_tools: list = ['web_search', 'rag', 'python_coder', 'vision_analyzer']
```

### Directory Structure

```
backend/
├── agents/                # Agent implementations
│   ├── react_agent.py     # Main ReAct agent + orchestrator
│   └── __init__.py
├── api/                   # FastAPI routes and application
│   ├── app.py             # Main app, CORS, middleware, startup
│   ├── routes/            # API endpoints
│   │   ├── chat.py        # Chat completions (OpenAI-compatible)
│   │   ├── auth.py        # Authentication (JWT)
│   │   ├── files.py       # File upload/management
│   │   ├── admin.py       # Admin endpoints
│   │   └── tools.py       # Tool-specific endpoints
│   └── middleware.py      # Security headers
├── config/
│   ├── settings.py        # All configuration with defaults
│   └── prompts/           # Prompt templates (if needed)
├── core/                  # Core infrastructure
│   ├── base_tool.py       # Abstract base class for all tools
│   ├── result_types.py    # ToolResult and other result types
│   ├── exceptions.py      # Exception hierarchy
│   ├── file_handlers/     # Unified file handling system
│   │   ├── registry.py    # FileHandlerRegistry (singleton)
│   │   ├── csv_handler.py, excel_handler.py, json_handler.py, etc.
│   └── llm_backends/      # LLM backend implementations
│       ├── llamacpp_wrapper.py  # llama.cpp integration
│       └── interceptor.py       # Prompt logging interceptor
├── tools/                 # Tool implementations
│   ├── python_coder.py    # Python code generation & execution
│   ├── web_search.py      # Tavily web search integration
│   ├── rag_retriever.py   # FAISS-based RAG retrieval
│   ├── file_analyzer.py   # File metadata extraction
│   └── vision_analyzer.py # Vision/image understanding
├── services/              # Business logic services
│   ├── conversation_store.py  # Conversation persistence
│   └── file_metadata_service.py  # File metadata caching
├── storage/               # Data persistence
│   └── conversation_store.py
├── models/
│   └── schemas.py         # Pydantic data models (ChatMessage, PlanStep, etc.)
└── utils/
    ├── llm_factory.py     # Centralized LLM instance creation
    ├── llm_manager.py     # User-specific LLM management
    ├── logging_utils.py   # Structured logging utilities
    ├── llm_response_parser.py  # LLM response parsing utilities
    ├── session_file_loader.py  # Load session code/files
    └── conversation_loader.py  # Load conversation history

data/
├── conversations/    # Stored chat history (JSON files)
├── uploads/          # User uploaded files
├── scratch/          # Code execution workspace (session-based)
│   └── {session_id}/
│       ├── script_*.py       # Generated code files
│       ├── *.png, *.csv      # Output files
│       └── variables/        # Persisted variables (Parquet, JSON, etc.)
└── logs/             # Application logs
```

---

## Common Development Tasks

### Adding a New Tool

1. **Create tool module** in `backend/tools/your_tool.py`:

```python
from backend.core.base_tool import BaseTool
from backend.core.result_types import ToolResult

class YourTool(BaseTool):
    async def execute(self, query: str, context: Optional[str] = None, **kwargs) -> ToolResult:
        self._log_execution_start(query=query)

        if not self.validate_inputs(query=query):
            return self._handle_validation_error("Invalid query")

        try:
            # Your tool logic here
            result = do_something(query)

            return ToolResult.success_result(
                output=result,
                execution_time=self._elapsed_time()
            )
        except Exception as e:
            return self._handle_error(e, "execute")

    def validate_inputs(self, **kwargs) -> bool:
        query = kwargs.get("query", "")
        return len(query.strip()) > 0
```

2. **Add to ToolName enum** in `backend/agents/react_agent.py`:

```python
class ToolName(str, Enum):
    # ... existing tools
    YOUR_TOOL = "your_tool"
```

3. **Add execution logic** in `ToolExecutor.execute()`:

```python
if action == ToolName.YOUR_TOOL:
    return await self._execute_your_tool(action_input)
```

4. **Add to available tools** in `backend/config/settings.py`:

```python
available_tools: list = ['web_search', 'rag', 'python_coder', 'your_tool']
```

5. **Update prompts** to include tool description in `_build_thought_action_prompt()`

### Switching LLM Backends

**To switch from Ollama to llama.cpp:**

1. Edit `backend/config/settings.py`:
```python
llm_backend: str = 'llamacpp'
llamacpp_model_path: str = './models/your-model.gguf'
llamacpp_n_gpu_layers: int = -1  # All layers to GPU
```

2. Download GGUF model and place at configured path
3. Restart server

**To switch back to Ollama:**

1. Edit `backend/config/settings.py`:
```python
llm_backend: str = 'ollama'
```

2. Ensure Ollama is running: `ollama serve`
3. Restart server

### Session Management & Code Persistence

Sessions are identified by `session_id` (UUID). When a session_id is provided:

**Conversation history** is loaded automatically:
- Stored in `data/conversations/{user_id}_{timestamp}_{session_id}.json`
- Retrieved via `conversation_store.get_messages(session_id, limit=500)`

**Python code execution** persists variables:
- Execution directory: `data/scratch/{session_id}/`
- Variables saved as: Parquet (DataFrames), JSON (dicts/lists), .npy (arrays)
- Code files saved as: `script_{timestamp}.py`
- Variables automatically loaded in subsequent executions

**Usage:**
```python
# First request - create session
response1, session_id = await agent_system.execute(
    messages=[...],
    session_id=None,  # New session
    user_id="alice"
)

# Follow-up request - reuse session
response2, _ = await agent_system.execute(
    messages=[...],
    session_id=session_id,  # Reuse session
    user_id="alice"
)
```

### Debugging Execution Issues

**Check logs:**
```bash
tail -f data/logs/app.log
```

**Python code execution artifacts:**
```bash
# View execution directory for session
ls -la data/scratch/{session_id}/

# View generated code
cat data/scratch/{session_id}/script_*.py

# View output files
ls data/scratch/{session_id}/*.png
ls data/scratch/{session_id}/*.csv
```

**Enable verbose logging** in settings.py:
```python
log_level: str = 'DEBUG'
llamacpp_verbose: bool = True  # For llama.cpp backend
```

**LLM prompt logging** - prompts are automatically logged:
- Location: `data/scratch/prompts.log`
- Formats: STRUCTURED (default), JSON, COMPACT
- Per-user logging when user_id provided

---

## Important Implementation Details

### ReAct Agent Execution Loop

The `ReActAgent` class orchestrates the full thought-action-observation loop:

**Key optimizations:**
1. **Context pruning**: When >3 steps, summarizes early steps and keeps last 2 in full detail
2. **Early exit**: Auto-finish detection when observation contains complete answer
3. **File-first strategy**: If files attached, attempts python_coder before other tools

**Main loop (backend/agents/react_agent.py:674):**
```python
for i in range(max_iterations):
    step = ReActStep(i + 1)

    # Generate thought and action
    thought, action, action_input = await thought_generator.generate(...)

    # Check for finish
    if action == ToolName.FINISH:
        final_answer = await answer_generator.generate(...)
        break

    # Execute tool
    observation = await tool_executor.execute(...)
    step.observation = observation
    steps.append(step)

# Build final answer if not already finished
if not final_answer:
    final_answer = await answer_generator.generate(...)
```

### Python Code Tool - Context Building

The python_coder tool builds rich context from multiple sources (backend/agents/react_agent.py:464):

```python
# Load conversation history
conversation_history = ConversationLoader.load_as_dicts(session_id, limit=10)

# Build ReAct context (previous attempts, failures)
react_context = self._build_react_history(steps, session_id)

# Build plan context (if executing a plan)
plan_context = {
    "current_step": plan_step.step_num,
    "total_steps": len(all_plan_steps),
    "previous_results": [...]
}

# Execute with full context
result = await python_coder_tool.execute_code_task(
    query=query,
    file_paths=file_paths,
    session_id=session_id,
    react_context=react_context,
    plan_context=plan_context,
    conversation_history=conversation_history
)
```

**This prevents:**
- Asking user same questions repeatedly
- Repeating failed code approaches
- Losing context between execution attempts

### Conversation Storage Format

Conversations stored as JSON in `data/conversations/`:

```json
{
  "session_id": "uuid",
  "user_id": "alice",
  "created_at": "2025-01-20T10:30:00",
  "updated_at": "2025-01-20T10:35:00",
  "messages": [
    {
      "role": "user",
      "content": "Analyze this CSV file",
      "timestamp": "2025-01-20T10:30:00",
      "metadata": {
        "file_paths": ["data.csv"]
      }
    },
    {
      "role": "assistant",
      "content": "Analysis complete...",
      "timestamp": "2025-01-20T10:35:00",
      "metadata": {
        "agent_type": "react",
        "steps": [...]
      }
    }
  ]
}
```

### OpenAI-Compatible API

The `/api/chat/completions` endpoint (backend/api/routes/chat.py) provides OpenAI-compatible interface:

**Request:**
```json
{
  "messages": [{"role": "user", "content": "Hello"}],
  "model": "gpt-oss:20b",
  "session_id": "optional-session-id",
  "agent_type": "auto"  // or "chat", "react", "plan_execute"
}
```

**Response:**
```json
{
  "id": "chatcmpl-xxx",
  "created": 1705750200,
  "model": "gpt-oss:20b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Response"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
  "x_session_id": "session-uuid",
  "x_agent_metadata": {"agent_type": "react", "steps": [...]}
}
```

---

## Troubleshooting

### Ollama Connection Errors

**Symptom:** `Request URL is missing an 'http://' or 'https://' protocol`

**Fix:** Settings automatically prepends `http://` if missing. Ensure `ollama_host` in settings.py starts with protocol.

**Verify Ollama is running:**
```bash
ollama serve
curl http://127.0.0.1:11434/api/tags
```

### llama.cpp Model Not Found

**Symptom:** `LLAMA.CPP MODEL FILE NOT FOUND!` during startup

**Fix:**
1. Download GGUF model from Hugging Face
2. Place at path specified in `llamacpp_model_path` setting
3. Check file exists: `ls -lh ./models/`

### Code Execution Timeouts

**Symptom:** Python code execution times out

**Fix:**
1. Increase `python_code_timeout` in settings.py
2. Check for infinite loops in generated code
3. Review execution logs: `data/scratch/{session_id}/`

### Empty or Incomplete Responses

**Symptom:** ReAct agent returns empty final answer

**Causes:**
- `max_iterations` reached without finishing
- Final answer generation failed

**Debug:**
1. Check agent metadata in response: `x_agent_metadata.steps`
2. Review last observation - should contain useful info
3. Increase `react_max_iterations` if needed

### Import Errors in Generated Code

**Symptom:** Code execution fails with `ImportError`

**Causes:**
- Package not in requirements.txt
- Blocked import (security sandbox)

**Fix:**
1. Check if package is installed: `pip show package_name`
2. Add to requirements.txt if needed
3. Blocked imports list in settings: Python's `ast` module validates imports before execution

---

## Version History

### v2.0.3 - 2025-12-04
- Fixed Ollama model configuration mismatch
- Updated default models to match installed Ollama models (gpt-oss:20b, llama3.2-vision:11b)

### v2.0.2 - 2025-01-13
- Fixed bcrypt compatibility (downgraded to 4.0.1)
- Fixed StructuredLogger to accept `*args` for old-style logging
- Fixed Ollama host validation with automatic protocol prefix

### v2.0.1 - 2025-12-04
- Fixed circular import in base_tool.py (lazy LLMFactory import)
- Fixed text handler newline splitting
- Added missing abstract methods to tool implementations

### v2.0.0 - 2025-12-03
- Refactored LLM backends into separate modules (backend/core/llm_backends/)
- Extracted LlamaCppWrapper and LLMInterceptor from llm_factory.py
- 61% file size reduction in llm_factory.py

### v1.3.0 - 2025-01-13
- Added LLMFactory for centralized LLM instance creation
- Support for multiple LLM configurations (standard, classifier, coder, vision)

---

**Last Updated:** 2025-12-04
**Repository:** LLM_API
**Backend Framework:** FastAPI 0.119.1 + LangChain 1.0.2 + LangGraph 1.0.1
