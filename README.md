# LLM API Project

Python-based LLM API server with agentic workflow capabilities.

## Overview

This project provides an AI-powered API service that leverages Large Language Models (LLMs) through Ollama, with sophisticated agentic capabilities including web search, document retrieval (RAG), and Python code generation/execution.

## Features

- **Multi-Agent Architecture**: ReAct and Plan-Execute patterns for complex task handling
- **Python Code Generation**: AI-driven code generation with iterative verification and execution
- **Web Search Integration**: Real-time information retrieval via Tavily API
- **RAG (Retrieval Augmented Generation)**: Document-based context retrieval
- **Session Management**: User authentication and conversation history
- **File Upload Support**: Process various file formats (CSV, Excel, JSON, PDF, etc.)
- **Security**: Sandboxed code execution with import restrictions and timeouts

## Architecture

```
LLM_API/
├── backend/
│   ├── api/              # FastAPI routes and application
│   ├── config/           # Configuration and settings
│   ├── core/             # Core agentic graph logic
│   ├── models/           # Pydantic schemas
│   ├── storage/          # Conversation persistence
│   ├── tasks/            # Task handlers (ReAct, Plan-Execute, Smart Agent)
│   ├── tools/            # Tool implementations
│   │   ├── python_coder_tool.py    # Unified Python code generation & execution
│   │   ├── rag_retriever.py        # Document retrieval
│   │   └── web_search.py           # Web search
│   └── utils/            # Authentication utilities
├── frontend/
│   └── static/           # Web interface
├── data/
│   ├── conversations/    # Conversation history
│   ├── uploads/          # User uploaded files
│   ├── scratch/          # Temporary code execution
│   └── logs/             # Application logs
└── requirements.txt
```

## Installation

### Prerequisites

- Python 3.9+
- Ollama (for LLM inference)
- Tavily API key (for web search)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd LLM_API
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure Ollama:
```bash
# Install Ollama from https://ollama.ai/
# Pull required models
ollama pull gemma3:12b
ollama pull gpt-oss:20b
ollama pull bge-m3:latest
```

5. Configure settings:
Edit `backend/config/settings.py` to customize:
- Ollama host and model
- API keys (Tavily)
- Server host and port
- Python code execution limits

## Usage

### Start the Server

```bash
python run_backend.py
```

Or for frontend:
```bash
python run_frontend.py
```

Default server runs at: `http://0.0.0.0:1007`

### API Endpoints

- `POST /api/chat` - Send chat message with agentic capabilities
- `POST /api/upload` - Upload files for processing
- `POST /api/auth/login` - User authentication
- `GET /api/conversations` - Retrieve conversation history

### Example Request

```python
import requests

response = requests.post(
    "http://localhost:1007/api/chat",
    json={
        "message": "Analyze the uploaded CSV file and show statistics",
        "session_id": "test-session",
        "user_id": "admin"
    }
)
```

## Configuration

Key settings in `backend/config/settings.py`:

```python
# Ollama Configuration
ollama_host: str = 'http://127.0.0.1:11434'
ollama_model: str = 'qwen3:8b'  # General purpose model
ollama_num_ctx: int = 16384

# Agentic Flow - Model Selection
agentic_classifier_model: str = 'qwen3:8b'  # Task classification
ollama_coder_model: str = 'qwen3:8b'  # Code generation (can use specialized model like 'deepseek-coder:6.7b')
ollama_coder_model_temperature: float = 0.1  # Dedicated sampling temp for coder LLM

# Python Code Execution
python_code_enabled: bool = True
python_code_timeout: int = 3000
python_code_max_memory: int = 5120
python_code_max_iterations: int = 5

# Security
secret_key: str = 'your-secret-key-here'
jwt_expiration_hours: int = 24
```

## Python Code Execution

The system can generate and execute Python code safely:

### Features
- **Unified Tool**: Single integrated tool for code generation, verification, and execution
- **Iterative Verification**: Up to 3 iterations to ensure code answers user's question
- **Execution Retry**: Up to 5 retry attempts on execution failure with auto-fixing
- **Sandboxed Execution**: Isolated subprocess execution with security restrictions
- **File Support**: Process multiple file formats (CSV, Excel, JSON, etc.)

### Security Measures
- Import restrictions (blocked: socket, subprocess, eval, exec, etc.)
- Execution timeout (default: 300 seconds)
- Memory limits
- Isolated temporary directories
- Static code analysis before execution

## Version History

### Version 2.0.1 (November 26, 2025)

**Bugfix: Temp File Cleanup in Uploads Folder**

Fixed an issue where temporary files in `uploads/{user_id}` were never deleted after being copied to the scratch folder for execution.

**Problem:**
When files were uploaded via `/v1/chat/completions`, they were:
1. Saved as `temp_{id}_{filename}` in `uploads/{user_id}/`
2. Copied to `data/scratch/{session_id}/` for code execution
3. **Never deleted** from the uploads folder if any exception occurred during agent execution

The cleanup code was inside the `try` block, so any exception during chat/agent execution would skip the cleanup, leaving orphaned temp files.

**Fix:**
Moved the temp file cleanup to a `finally` block, ensuring files are always deleted regardless of success or failure.

**Modified Files:**
- `backend/api/routes/chat.py` - Moved `_cleanup_files()` call from `try` block to `finally` block

**Impact:**
- Temp files in `uploads/` folder are now properly cleaned up after processing
- Reduces disk usage from accumulated orphan files
- No functional change to file handling - files are still copied (not moved) to scratch

---

### Version 2.0.0 (November 26, 2025)

**Enhancement: Intelligent Retry System for Python Coder**

Major overhaul of the Python coder's error recovery mechanism to prevent the model from getting "stuck" on repeated errors. The system now learns from previous failed attempts and adapts its approach.

**Problem Solved:**
When code failed with errors like `IndexError: list index out of range`, the model would often repeat the same mistake because:
- It didn't see its previous failed code
- It didn't know what variables actually contained at failure time
- Retry prompts were identical, producing identical wrong outputs

**Key Changes:**

1. **Full Attempt History Tracking** (`orchestrator.py`)
   - `attempt_history` now stores: full code, error message, error type, and variable namespace
   - Error classification with specific guidance for each error type
   - New `_classify_error()` method categorizes 15+ error types with actionable guidance

2. **Runtime Variable Capture on Error** (`repl_manager.py`)
   - Namespace is now captured even when execution fails
   - New `<<<ERROR_NAMESPACE_*>>>` markers for error-state variables
   - LLM can see: "data: list (EMPTY - length=0)" - explaining WHY IndexError occurred

3. **Enhanced Retry Prompts** (`fixing.py`)
   - New `get_retry_prompt_with_history()` shows all previous failed attempts
   - Escalating strategy: Attempt 2 suggests different approach, Attempt 3+ forces complete rethink
   - `get_execution_fix_prompt()` now includes debug context section with variable states

4. **Forced Different Approach on Repeated Errors** (`orchestrator.py`)
   - Detects when same error type repeats (e.g., KeyError twice in a row)
   - Skips incremental patching, forces full regeneration with "COMPLETELY RETHINK" prompt
   - Prevents stuck loops where model keeps trying the same broken approach

5. **Improved Error Context Building** (`orchestrator.py`)
   - `_build_retry_context()` enhanced with attempt_history, force_different_approach flags
   - `_format_namespace_for_prompt()` formats variable states for LLM consumption
   - Clearly shows empty lists, dict keys, DataFrame shapes at failure point

**Error Type Classifications:**
- IndexError, KeyError, TypeError, NoneType, FileNotFound, JSONDecode
- ValueError, AttributeError, NameError, ImportError, ZeroDivision
- EncodingError, PermissionError, MemoryError, Timeout, RuntimeError

**Example Improvement:**

Before (stuck loop):
```
Attempt 1: data[0] → IndexError
Attempt 2: data[0] → IndexError (same mistake)
Attempt 3: data[0] → IndexError (still same)
```

After (learns from failures):
```
Attempt 1: data[0] → IndexError
Attempt 2: (sees "data: list (EMPTY - length=0)") → adds len() check → still fails
Attempt 3: (forced different approach) → uses data.get() pattern → SUCCESS
```

**Modified Files:**
- `backend/tools/python_coder/orchestrator.py` - Major retry logic overhaul
- `backend/tools/python_coder/executor/repl_manager.py` - Error namespace capture
- `backend/config/prompts/python_coder/fixing.py` - New retry prompts with history
- `backend/config/prompts/python_coder/__init__.py` - New exports
- `backend/config/prompts/__init__.py` - Registry updates
- `backend/tools/python_coder/code_fixer.py` - Error namespace parameter

**Benefits:**
- Dramatically reduced "stuck" scenarios where model repeats same error
- Better error understanding through runtime variable inspection
- Escalating retry strategies prevent repetitive failures
- 15+ classified error types with specific fix guidance

---

### Version 1.9.0 (November 25, 2025)

**Enhancement: Prompt System Overhaul + Output File Handling**

Major restructuring of the prompt system to improve output quality, reduce token usage, and establish maintainable architecture. Also added file-based output handling to prevent CMD window truncation issues.

**Key Changes:**

1. **Output File Handling for Python Coder** (NEW)
   - Prompts now instruct LLM to save results to files (`result.csv`, `result.txt`) instead of printing large data
   - System automatically loads result files after execution for adequacy checking
   - Solves CMD window truncation issues with large pandas DataFrames
   - New setting: `python_code_output_max_llm_chars` (default: 8000)
   - New rule block: `OUTPUT_FILE_RULES` in `base.py`

2. **New Base Utilities** (`backend/config/prompts/base.py`)
   - Standardized ASCII markers: `[OK]`, `[X]`, `[!!!]`, `[WARNING]`
   - `get_current_time_context()` - Temporal awareness for all prompts
   - `section_border()` - Consistent visual separators
   - Reusable rule blocks: `FILENAME_RULES`, `NO_ARGS_RULES`, `JSON_SAFETY_RULES`, `OUTPUT_FILE_RULES`

2. **Token Optimization**
   - `plan_execute.py`: Reduced from 242 to ~130 lines (~45% reduction)
   - `task_classification.py`: Removed 60 lines of commented examples
   - `python_coder/*.py`: Consolidated duplicate rules across files

3. **Temporal Awareness**
   - All relevant prompts now include current time context
   - Format: `Current Time: {date} ({day_of_week}) {time} {timezone}`
   - Helps LLM understand time-sensitive queries

4. **Expanded File Analyzer** (`backend/config/prompts/file_analyzer.py`)
   - New specialized prompts:
     - `get_json_analysis_prompt()` - JSON structure analysis
     - `get_csv_analysis_prompt()` - CSV column/type analysis
     - `get_excel_analysis_prompt()` - Multi-sheet Excel analysis
     - `get_structure_comparison_prompt()` - Compare two files
     - `get_anomaly_detection_prompt()` - Data quality issues

5. **Improved Plan Context Passing** (`backend/tasks/react/planning.py`)
   - Changed generic "Instructions" to structured "Execution Guidance"
   - Now includes both approach and success criteria
   - Better context for tool execution

6. **Enhanced Plan Section** (`backend/config/prompts/python_coder/templates.py`)
   - Clearer step relationships with status markers
   - Data flow visualization from previous steps
   - Truncated summaries for token efficiency

**Modified Files:**
- `backend/config/prompts/base.py` - Added `OUTPUT_FILE_RULES`
- `backend/config/prompts/__init__.py`
- `backend/config/prompts/task_classification.py`
- `backend/config/prompts/plan_execute.py`
- `backend/config/prompts/file_analyzer.py`
- `backend/config/prompts/python_coder/generation.py` - Added output file instructions
- `backend/config/prompts/python_coder/templates.py` - Added output file rules
- `backend/tools/python_coder/orchestrator.py` - Added `_load_result_files()` method
- `backend/config/settings.py` - Added `python_code_output_max_llm_chars`
- `backend/tasks/react/planning.py`

**Benefits:**
- ~25-35% average token reduction across prompts
- Consistent structure and formatting
- Better temporal awareness for time-sensitive queries
- Improved code generation context
- More specialized file analysis capabilities
- Reliable output handling for large data (no CMD truncation)

---

### Version 1.8.0 (November 25, 2025)

**Enhancement: Structured LLM Prompt Logging**

Completely redesigned the `LLMInterceptor` class for much more readable and structured prompt/response logging.

**Key Changes:**

1. **Three Log Formats**
   - `STRUCTURED` (default): Human-readable with clear visual sections, box-drawing characters, and hierarchical layout
   - `JSON`: JSON Lines format for programmatic parsing and analysis
   - `COMPACT`: Minimal single-line format for quick scanning

2. **Request/Response Pairing**
   - Each request now gets a unique `call_id` (UUID)
   - Responses are linked to their originating requests via the same `call_id`
   - Makes it easy to trace conversation flow

3. **Enhanced Metadata**
   - Token estimation for tracking approximate usage
   - Duration tracking (in ms) for performance analysis
   - Clear role labels (SYSTEM, HUMAN, ASSISTANT, etc.)

4. **Improved Visual Structure**
   - Box-drawing header with log metadata
   - Clear visual separation between requests (📤) and responses (📥)
   - Content properly indented and wrapped for readability
   - No more repetitive `====` lines cluttering the log

5. **Streaming Response Logging**
   - Stream methods now aggregate chunks and log the complete response after streaming completes
   - Previously, streaming responses were not logged at all

**New Classes:**
- `LogFormat` - Enum for output format selection
- `LogMessage` - Dataclass for structured message representation
- `LogEntry` - Dataclass for complete log entries with metadata

**Factory Method Updates:**
- `create_llm()`, `create_classifier_llm()`, `create_coder_llm()` now accept:
  - `log_format: LogFormat` - Select output format
  - `log_file: Path` - Custom log file path

**Example Structured Output:**
```
════════════════════════════════════════════════════════════════════════════════
  📤 REQUEST   │  ID: a1b2c3d4  │  2025-11-25 14:30:45.123
────────────────────────────────────────────────────────────────────────────────
  Model: qwen3:8b                        User: admin
  Tokens: ~125                           Duration: 
────────────────────────────────────────────────────────────────────────────────

  [SYSTEM]
  ········································
    You are a helpful assistant.

  [HUMAN]
  ········································
    What is the capital of France?

════════════════════════════════════════════════════════════════════════════════
```

**Modified Files:**
- `backend/utils/llm_factory.py` - Complete rewrite of `LLMInterceptor` class (v1.0.0 → v1.1.0)

---

### Version 1.7.5 (November 24, 2025)

**Refactor: Prompt registry cleanup**

- Removed unused prompt modules (`backend/config/prompts/agent_graph.py`, `context_formatting.py`, `phase_manager.py`, `python_coder_legacy.py`, `rag.py`, `templates.py`, `validators.py`) to keep the prompt directory aligned with real runtime usage.
- `backend/config/prompts/__init__.py` - pruned imports, registry entries, and `__all__` exports so the centralized registry only exposes prompts that are actually queried.
- `backend/config/prompts/react_agent.py` - deleted the unused finish-step and action-input prompts that previously caused parameter mismatches in historical logs.
- `backend/config/prompts/python_coder/fixing.py` & `backend/config/prompts/python_coder/__init__.py` - removed the orphaned `get_smart_fix_prompt` helper and its export to reduce dead code.
- Documentation updated to reflect the leaner prompt surface area.

### Version 1.7.2 (November 24, 2025)

**Bugfix: Resilient plan JSON parsing**

- `backend/tasks/Plan_execute.py` - added `_parse_plan_text()` plus helpers that strip markdown fences, pull out JSON blocks, and fall back to `ast.literal_eval`, eliminating `JSONDecodeError: Expecting ':' at line 24 column 7` when the planner LLM emits narration or slightly malformed arrays.

### Version 1.7.3 (November 24, 2025)

**Bugfix: Default plan fallback**

- `backend/tasks/Plan_execute.py` - `_parse_plan_text()` now emits a single-step fallback plan (web_search or python_coder) instead of raising when the LLM output can’t be parsed, preventing “Unable to parse execution plan as JSON array” from aborting Plan-and-Execute runs.

### Version 1.7.4 (November 24, 2025)

**Feature: Coder temperature setting**

- `backend/utils/llm_factory.py` - `create_coder_llm()` now defaults to `settings.ollama_coder_model_temperature`, decoupling code-generation sampling from the general-purpose LLM temperature.
- `backend/config/settings.py` / `README` - documented `ollama_coder_model_temperature` so deployments can tune deterministic code output separately.

### Version 1.7.1 (November 24, 2025)

**Bugfix: Async file upload handling**

- `backend/api/routes/chat.py` - made `_handle_file_uploads()` async and now awaits `UploadFile.read()` directly, eliminating the `asyncio.run()` call that crashed under FastAPI’s running event loop when saving uploaded files.

### Version 1.7.0 (November 24, 2025)

**Refactor: File analyzer clarity & metadata**

- `backend/tools/file_analyzer/analyzer.py` - rewrote the orchestration layer around a `FileAnalysisPayload` helper, added normalized input validation, richer error/warning reporting, per-run metadata, and clearer handler execution utilities so multi-file analyses are easier to reason about and debug.
- `backend/tools/file_analyzer/__init__.py` - bumped the module version to `1.1.0` to reflect the refactor.

Users now receive structured warnings for unsupported formats or missing files, plus an `analysis_id` and success counters in the returned payload.

### Version 1.6.5 (November 24, 2024)

**Enhancement: Multi-line info logging**

- `backend/utils/logging_utils.py` - `StructuredLogger.info()` now detects newline characters and logs each line separately (using the same formatting as `multiline`), so `logger.info()` can handle multi-line commands without manual `logger.multiline()` calls.

### Version 1.6.4 (November 24, 2024)

**Bugfix: Restored ReAct variable loading**

- `backend/tasks/react/agent.py` - reintroduced `_load_variables()` and invoke it before guided plan execution so sessions with saved python_coder variables no longer crash with `'ReActAgent' object has no attribute '_load_variables'`.

### Version 1.6.3 (November 24, 2024)

**Documentation: Plan-and-Execute workflow**

- Added `PLAN_EXECUTE_WORKFLOW.md`, a comprehensive description of every phase (planning, guided execution, monitoring, adaptation) in the hybrid Plan-and-Execute agent.

### Version 1.6.2 (November 24, 2024)

**Change: Removed ReAct fuzzy action matching**

- `backend/tasks/react/thought_action_generator.py` - deleted fuzzy action mapping logic and now default to `finish` whenever the LLM emits an unknown action label.

### Version 1.6.1 (November 24, 2024)

**Bugfix: Chat Message Serialization**

- `backend/api/routes/chat.py` - serialize `ChatMessage` instances with `model_dump()` before writing them to `data/scratch/{user}/llm_input_messages_{session}.json`, preventing `TypeError: Object of type ChatMessage is not JSON serializable` when persisting request payloads.

### Version 1.6.0 (November 24, 2024)

**Simplification: Session Notepad Removal**

Removed the automatic session notepad feature to simplify the architecture and reduce overhead.

**Rationale:**
- The session notepad added redundancy with existing conversation history and variable storage
- Extra LLM call after every execution added latency and cost without sufficient benefit
- Simpler architecture is easier to maintain and reason about
- Variable persistence remains intact and provides sufficient session continuity

**Changes:**
1. **Deleted Files:**
   - `backend/tools/notepad.py` - Removed SessionNotepad class

2. **Modified Files:**
   - `backend/tasks/react/agent.py` - Removed notepad loading and generation hooks
   - `backend/tasks/react/context_manager.py` - Simplified to only inject variable context
   - `backend/config/prompts/__init__.py` - Removed notepad prompt imports and registrations
   - `backend/config/prompts/react_agent.py` - Removed notepad entry generation prompt
   - `backend/config/prompts/context_formatting.py` - Removed notepad context formatter

3. **What Remains:**
   - ✅ Variable persistence (DataFrames, arrays, etc.) - still works
   - ✅ Conversation history - still tracked
   - ✅ Session management - still functional
   - ✅ Variable metadata injection - agent still sees available variables

**Benefits:**
- Reduced execution overhead (no post-execution LLM call)
- Simpler codebase with less coupling
- Lower latency and cost per execution
- Cleaner separation of concerns

**Migration Guide:**
- No action needed - variable persistence works automatically
- Variables are still saved and loaded across executions within the same session
- Conversation history provides sufficient context for most use cases

---

### Version 1.2.0 (2024-10-31)

**Major Changes: Python Code Tool Unification**

1. **Unified Python Code Tool**
   - Merged `python_coder_tool.py` and `python_executor_engine.py` into a single module
   - Removed redundant `python_executor.py` (unused legacy code)
   - Simplified architecture with `CodeExecutor` class for low-level execution
   - Single `PythonCoderTool` class for high-level code generation and orchestration

2. **Verification System Redesign**
   - **Reduced scope**: Verifier now focuses ONLY on "Does the code answer the user's question?"
   - **Reduced iterations**: Maximum verification iterations reduced from 10 to 3
   - Removed overly detailed checks (performance, code quality, file handling details)
   - Simplified verification prompt for faster and more focused validation

3. **Execution Retry Logic**
   - Added automatic retry mechanism for failed code execution
   - Maximum 5 execution attempts with auto-fixing between retries
   - New `_fix_execution_error()` method uses LLM to analyze and fix runtime errors
   - Execution history tracking for debugging

4. **Code Quality Improvements**
   - Consolidated constants (`BLOCKED_IMPORTS`, `SUPPORTED_FILE_TYPES`)
   - Better separation of concerns (CodeExecutor vs PythonCoderTool)
   - Improved logging and error messages
   - Backward compatibility exports for existing imports

5. **Configuration Updates**
   - `settings.py`: `python_code_max_iterations` default changed from 10 to 3
   - Removed obsolete `python_executor` references from `React.py`
   - Consolidated `PYTHON_CODE` and `PYTHON_CODER` tool types into single `PYTHON_CODER`

6. **File Cleanup**
   - Deleted: `backend/tools/python_executor_engine.py`
   - Deleted: `backend/tools/python_executor.py` (unused)
   - Updated: `backend/tasks/React.py` (removed python_executor imports)
   - Updated: `backend/tools/python_coder_tool.py` (unified implementation)

**Benefits:**
- Faster verification (3 iterations vs 10)
- More reliable execution (5 retry attempts with auto-fixing)
- Simpler codebase (1 file instead of 3)
- Focused verification on core goal (answering user's question)
- Better error recovery through retry logic

**Breaking Changes:**
- `python_executor_engine.PythonExecutor` moved to `python_coder_tool.CodeExecutor`
- `ToolName.PYTHON_CODE` removed, use `ToolName.PYTHON_CODER` instead
- Import changes required in custom code using these tools

---

### Version 1.3.0 (November 19, 2024)

**Feature: Variable Persistence System** *(Note: Session Notepad portion removed in v1.6.0)*

Implemented variable persistence system for maintaining data continuity across executions within the same session.

**New Components:**
- `backend/tools/python_coder/variable_storage.py` - Type-specific variable serialization

**Key Features:**
1. **Variable Persistence**: Variables from successful code executions are automatically saved with type-specific serialization:
   - DataFrames → Parquet format
   - NumPy arrays → .npy files
   - Simple types → JSON
   - Matplotlib figures → PNG images
2. **Context Auto-Injection**: Variable metadata is automatically injected into subsequent executions within the same session
3. **Namespace Capture**: Enhanced REPL mode captures execution namespace and returns variable metadata

**Modified Files:**
- `backend/tools/python_coder/executor.py` - Added namespace capture in REPL bootstrap
- `backend/tools/python_coder/orchestrator.py` - Added namespace to success/final results

**Storage Structure:**
```
./data/scratch/{session_id}/
└── variables/                   # Persisted variables
    ├── variables_metadata.json  # Variable catalog with load instructions
    ├── df_*.parquet            # DataFrames
    ├── *.json                  # Simple types
    └── *.npy                   # NumPy arrays
```

**Benefits:**
- Seamless session continuity - variables persist across executions
- Efficient data reuse - no need to recompute variables
- Explicit variable loading - agent sees what's available and writes code to load when needed
- Safe serialization - no pickle, uses native formats

**Example Flow:**
1. User: "Analyze the warpage data"
   - Agent executes code, creates `df_warpage` and `stats_summary`
   - Variables saved: `df_warpage.parquet`, `stats_summary.json`

2. User: "Create a heatmap visualization"
   - System injects variable context showing available variables
   - Agent sees `df_warpage` is available and writes code to load it
   - New visualization code builds on previous work

---

### Version 1.5.0 (November 24, 2024)

**Major Refactor: 3-Way Agent Classification & Code Simplification**

Completely redesigned the agent routing system to use direct LLM-powered 3-way classification, removed streaming support, and significantly simplified the codebase for better maintainability.

**Key Changes:**

1. **3-Way LLM Agent Classification**
   - Replaced 2-stage classification (agentic vs chat → react vs plan_execute)
   - New single-stage LLM classification directly returns: "chat", "react", or "plan_execute"
   - More accurate classification with comprehensive examples for each agent type
   - Eliminated keyword-based heuristics in favor of intelligent LLM decision-making

2. **Simplified chat.py** (`backend/api/routes/chat.py`)
   - Reduced from 510 lines to ~400 lines (22% reduction)
   - Removed all streaming-related code (StreamingResponse, SSE chunks, stream parameter)
   - Extracted file handling to `_handle_file_uploads()` helper function
   - Restructured endpoint into 4 clear phases:
     - Phase 1: File Handling
     - Phase 2: Classification (LLM-powered)
     - Phase 3: Execution (route to appropriate agent)
     - Phase 4: Storage & Cleanup
   - Much easier to read, understand, and maintain

3. **Updated Task Classification** (`backend/config/prompts/task_classification.py`)
   - New `get_agent_type_classifier_prompt()` function for 3-way classification
   - Comprehensive examples for each agent type:
     - **chat**: Pure knowledge questions (10+ examples)
     - **react**: Single-goal tool tasks (10+ examples)
     - **plan_execute**: Multi-step complex workflows (10+ examples)
   - Clear decision rules and edge case handling
   - Backward-compatible with deprecated `get_agentic_classifier_prompt()`

4. **Simplified Smart Agent** (`backend/tasks/smart_agent_task.py`)
   - Removed `_select_agent()` heuristic method (50+ lines removed)
   - Agent now simply routes to specified agent type
   - Classification happens in chat.py before calling smart_agent_task
   - Cleaner separation of concerns

**Agent Type Definitions:**

- **chat**: Simple questions answerable from LLM knowledge base (no tools)
  - Example: "What is Python?", "Explain machine learning"
  
- **react**: Single-goal tasks requiring tool usage (web search, code execution, file analysis)
  - Example: "Search for current weather", "Analyze this CSV file"
  
- **plan_execute**: Multi-step complex tasks requiring planning and structured execution
  - Example: "Analyze 3 files, create visualizations, and generate a report"

**Benefits:**

- **Clearer Logic**: Single LLM decision instead of fragmented 2-stage classification
- **Better Accuracy**: LLM understands nuanced differences between agent types
- **Simpler Codebase**: Removed ~160 lines of complex logic and streaming code
- **Easier Maintenance**: Clear separation of concerns, well-defined phases
- **Better Logging**: Comprehensive logging at each phase for debugging

**Breaking Changes:**

- `stream` parameter removed from `/v1/chat/completions` endpoint
- Streaming responses no longer supported
- `AgentType.AUTO` deprecated in smart_agent_task (classification moved to chat.py)
- `determine_task_type()` renamed to `determine_agent_type()` and returns 3 values

**Modified Files:**

- `backend/api/routes/chat.py` - Major refactor (510 → 400 lines)
- `backend/config/prompts/task_classification.py` - 3-way classification prompt
- `backend/tasks/smart_agent_task.py` - Removed heuristic selection
- `backend/config/settings.py` - Added `ollama_coder_model` setting
- `backend/utils/llm_factory.py` - Coder LLM now uses configurable model

**Migration Guide:**

If you have custom code calling these endpoints:

```python
# Before (streaming - NO LONGER SUPPORTED)
response = requests.post("/v1/chat/completions", data={
    "stream": "true",  # ❌ Parameter removed
    ...
})

# After (non-streaming only)
response = requests.post("/v1/chat/completions", data={
    "model": "qwen3:8b",
    "messages": json.dumps([...]),
    "agent_type": "auto",  # or "chat", "react", "plan_execute"
    ...
})
```

**Testing:**

Tested with various query types to ensure correct agent routing:
- Simple questions → chat
- Single tool tasks → react  
- Multi-step complex tasks → plan_execute
- Explicit agent_type parameter correctly overrides classification

---

### Version 1.4.0 (November 20, 2024)

**Enhancement: Context-Aware Python Code Generation**

Significantly improved Python code generation by integrating comprehensive context from different agent workflows and conversation history.

**Key Changes:**

1. **Enhanced Orchestrator Context Passing** (`backend/tools/python_coder/orchestrator.py`)
   - Added `conversation_history`, `plan_context`, and `react_context` parameters to `execute_code_task()` method
   - Modified `_generate_code_with_self_verification()` to accept and pass context to prompt generation
   - Context now flows through the entire code generation pipeline

2. **Improved Python Coder Prompts** (`backend/config/prompts/python_coder.py`)
   - Updated `get_code_generation_with_self_verification_prompt()` to handle new context parameters
   - Enhanced `get_python_code_generation_prompt()` with three new context sections:
     - **PAST HISTORIES**: Shows previous conversation turns with timestamps
     - **PLANS**: Displays Plan-Execute workflow context with step status and previous results
     - **REACTS**: Shows ReAct iteration history including failed code attempts and errors
   - Structured prompt now has 8 sections: HISTORIES → INPUT → PLANS → REACTS → TASK → METADATA → RULES → CHECKLISTS

3. **ReAct Agent Integration** (`backend/tasks/react/tool_executor.py`)
   - Added `_build_react_context()` method to extract failed code attempts and errors
   - Added `_load_conversation_history()` method to load conversation from store
   - Modified `_execute_python_coder()` to automatically gather and pass all contexts
   - react_context structure includes iteration history with failed code, observations, and error reasons

4. **Plan-Execute Agent Integration** (`backend/tasks/react/plan_executor.py`)
   - Added `_build_plan_context()` method to create structured plan context
   - Modified `execute_step()` to accept and track all plan steps and results
   - plan_context includes current step, total steps, full plan with status, and previous results
   - Automatic context injection when python_coder tool is invoked during plan execution

5. **Conversation History Integration** (`backend/storage/conversation_store.py`)
   - Leveraged existing `get_messages()` method for conversation history retrieval
   - History automatically loaded when session_id is available
   - Recent conversation context (last 10 messages) passed to code generation

**Benefits:**
- **Better Context Awareness**: Python coder now sees full conversation history, reducing repeated questions
- **Learn from Failures**: ReAct context shows previous failed attempts, preventing the same mistakes
- **Plan Alignment**: Plan context ensures generated code aligns with overall execution strategy
- **Improved Code Quality**: LLM can generate more appropriate code with full context visibility
- **Reduced Iterations**: Context-aware generation reduces need for multiple retry attempts

**Example Context Flow:**

Plan-Execute scenario:
```
User: "Analyze sales data and create visualizations"
→ Plan Step 1: Load data (python_coder called)
  → Context includes: conversation history, plan (step 1 of 3)
→ Plan Step 2: Calculate metrics (python_coder called)
  → Context includes: conversation history, plan (step 2 of 3), previous step result
→ Plan Step 3: Create charts (python_coder called)
  → Context includes: conversation history, plan (step 3 of 3), all previous results
```

ReAct scenario with retries:
```
User: "Calculate average from data.json"
→ Iteration 1: python_coder generates code → fails (FileNotFoundError)
→ Iteration 2: python_coder called again
  → Context includes: conversation history, react_context with failed attempt #1
→ Iteration 3: python_coder called again
  → Context includes: conversation history, react_context with failed attempts #1 and #2
```

**Modified Files:**
- `backend/tools/python_coder/orchestrator.py` - Context parameter additions
- `backend/config/prompts/python_coder.py` - Prompt structure enhancement
- `backend/tasks/react/tool_executor.py` - Context gathering and passing
- `backend/tasks/react/plan_executor.py` - Plan context building

**New Test File:**
- `test_python_coder_prompt.py` - Comprehensive test for prompt generation with all context sections

**Testing:**
Run the test to verify prompt generation:
```bash
python test_python_coder_prompt.py
```

The test verifies:
- All 8 prompt sections are present
- Conversation history is properly formatted
- Plan context shows step progression
- ReAct context includes failed attempts
- File metadata and access patterns are included
- Rules and checklists are present

---

### Version 1.1.0 (Previous)
- Multi-agent architecture implementation
- ReAct pattern for reasoning and acting
- Plan-Execute workflow for complex tasks
- Web search and RAG integration

### Version 1.0.0 (Initial)
- Basic LLM API server
- Ollama integration
- Simple chat functionality

## Development

### Project Structure

- **API Layer** (`backend/api/`): FastAPI routes and HTTP handling
- **Agent Layer** (`backend/tasks/`, `backend/core/`): Agentic reasoning and task execution
- **Tool Layer** (`backend/tools/`): External integrations (web, RAG, code execution)
- **Storage Layer** (`backend/storage/`): Data persistence

### Adding New Tools

1. Create tool in `backend/tools/`
2. Register in `backend/tasks/React.py` or `backend/core/agent_graph.py`
3. Add to `ToolName` enum if needed

### Testing

```bash
# Run API tests
python -m pytest tests/

# Manual testing with Jupyter notebook
jupyter notebook API_examples.ipynb
```

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   - Ensure Ollama is running: `ollama serve`
   - Check `ollama_host` in settings

2. **Code Execution Timeout**
   - Increase `python_code_timeout` in settings
   - Check for infinite loops in generated code

3. **Import Errors in Generated Code**
   - Review `BLOCKED_IMPORTS` in `python_coder_tool.py`
   - Required packages must be installed in environment

4. **Memory Issues**
   - Reduce `ollama_num_ctx` for lower memory usage
   - Increase `python_code_max_memory` if needed

## Security Considerations

- **Never expose** the server directly to the internet without proper authentication
- **Change** `secret_key` in production
- **Review** `BLOCKED_IMPORTS` before modifying
- **Monitor** code execution logs for suspicious activity
- **Limit** file upload sizes and types

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes with clear messages
4. Submit a pull request

## License

[Your License Here]

## Contact

[Your Contact Information]

---

**Last Updated**: November 26, 2025
**Version**: 2.0.1

