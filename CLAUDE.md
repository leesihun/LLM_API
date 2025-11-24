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

### Git Workflow

**Auto-commit and push:**
```bash
bash git-auto-push.sh
```
This script automatically stages all changes, commits with timestamp, pulls remote changes (with rebase), and pushes to the remote repository.

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

**2. ReAct Agent Pattern (backend/tasks/react/)**
- The primary agentic execution engine
- Implements Thought → Action → Observation loop
- Max iterations: 10 (configurable via `max_iterations` parameter)
- Tools available: `web_search`, `rag_retrieval`, `python_coder`, `file_analyzer`, `finish`
- **Critical optimization:** Combined thought-action generation (1 LLM call instead of 2)
- **Early exit:** Auto-finish when observation contains complete answer
- **Context pruning:** If >3 steps, summarizes early steps and keeps last 2 in detail
- **Modular structure** (v2.0.0): Separated into specialized modules for maintainability

**3. Python Code Generation (backend/tools/python_coder/)**
- **Modular architecture** (v2.0.0): Separated into specialized modules
- **Context-aware generation** (v1.4.0): Uses conversation history, plan context, and ReAct iteration history
- **Verification phase:** Max 3 iterations, focused ONLY on "Does code answer user's question?"
- **Execution retry:** Max 5 attempts with auto-fixing between retries
- **Variable persistence** (v1.3.0): Automatic variable serialization and reuse across session
- **File handling:** Supports CSV, Excel, JSON, PDF, images, etc.
- **Security:** Sandboxed execution, import restrictions, timeout controls
- **Session continuity:** Uses session-based execution directories with notepad memory

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
├── api/                      # FastAPI routes and application
│   ├── app.py               # Main FastAPI app, CORS, middleware, startup
│   └── routes/              # Modular API routes (v2.0.0)
│       ├── __init__.py      # Routes registration
│       ├── chat.py          # Chat endpoints
│       ├── auth.py          # Authentication endpoints
│       ├── files.py         # File upload/management
│       ├── admin.py         # Admin endpoints
│       └── tools.py         # Tool-specific endpoints
├── config/
│   ├── settings.py          # Centralized configuration (all defaults here)
│   └── prompts.py           # Centralized prompts (PromptRegistry)
├── core/
│   └── agent_graph.py       # LangGraph workflow (alternative to ReAct)
├── tasks/
│   ├── chat_task.py         # Entry point: task classification
│   ├── react/               # Modular ReAct implementation (v2.0.0)
│   │   ├── __init__.py      # Public exports
│   │   ├── agent.py         # Main ReActAgent orchestration
│   │   ├── models.py        # ReAct data models (ReActStep, ToolName)
│   │   ├── thought_action_generator.py  # Thought/action generation
│   │   ├── tool_executor.py             # Tool execution routing
│   │   ├── answer_generator.py          # Final answer synthesis
│   │   ├── context_manager.py           # Context formatting & pruning
│   │   ├── verification.py              # Step verification logic
│   │   ├── plan_executor.py             # Plan-based execution
│   │   └── utils.py                     # Helper utilities
│   ├── smart_agent_task.py  # Plan-Execute workflow router
│   ├── Plan_execute.py      # Hybrid Plan+ReAct implementation
│   └── legacy/
│       └── React.py         # Legacy monolithic ReAct (deprecated)
├── tools/
│   ├── python_coder/        # Modular Python code tool (v2.0.0)
│   │   ├── __init__.py      # Public exports
│   │   ├── tool.py          # PythonCoderTool main class
│   │   ├── orchestrator.py  # High-level orchestration with context
│   │   ├── generator.py     # Code generation
│   │   ├── executor.py      # Code execution (CodeExecutor)
│   │   ├── verifier.py      # Code verification
│   │   ├── variable_storage.py  # Variable persistence (v1.3.0)
│   │   └── models.py        # Data models
│   ├── notepad.py           # Session memory/notepad (v1.3.0)
│   ├── file_analyzer/       # Modular file analyzer (v2.0.0)
│   │   ├── __init__.py      # Public exports
│   │   ├── tool.py          # FileAnalyzer main class
│   │   ├── analyzers.py     # Format-specific analyzers
│   │   ├── llm_analyzer.py  # LLM-powered deep analysis
│   │   └── models.py        # Data models
│   ├── web_search/          # Modular web search (v2.0.0)
│   │   ├── __init__.py      # Public exports
│   │   └── tool.py          # WebSearchTool
│   ├── rag_retriever/       # Modular RAG retriever (v2.0.0)
│   │   ├── __init__.py      # Public exports (includes backward compatibility)
│   │   ├── tool.py          # RAGRetrieverTool
│   │   ├── retriever.py     # RAGRetrieverCore
│   │   └── models.py        # Data models
│   ├── web_search.py        # Legacy monolithic web search (deprecated)
│   ├── rag_retriever.py     # Legacy monolithic RAG (deprecated, kept for fallback)
│   └── legacy/
│       ├── python_coder_tool.py      # Legacy monolithic (deprecated)
│       └── file_analyzer_tool.py     # Legacy monolithic (deprecated)
├── storage/
│   └── conversation_store.py  # Conversation persistence
├── models/
│   └── schemas.py            # Pydantic data models
└── utils/
    ├── auth.py               # JWT authentication
    ├── llm_factory.py        # Centralized LLM instance creation
    └── logging_utils.py      # Logging utilities

data/
├── conversations/    # Stored chat history (JSON files)
├── uploads/          # User uploaded files
├── scratch/          # Code execution workspace (session-based)
└── logs/             # Application logs
```

**Architecture Improvements (v2.0.0):**
- **Modular structure:** Large monolithic files split into focused modules
- **Separation of concerns:** Each module has a single, clear responsibility
- **Centralized utilities:** LLM creation, prompts, and logging consolidated
- **Legacy support:** Old monolithic files moved to legacy/ for backward compatibility
- **Improved maintainability:** Easier to test, debug, and extend individual components

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

### When Modifying python_coder/:
- **tool.py (PythonCoderTool):** High-level orchestration (generation, verification, retry)
- **orchestrator.py:** Context-aware orchestration, integrates conversation/plan/react history
- **generator.py (CodeGenerator):** LLM-based code generation with file context
- **executor.py (CodeExecutor):** Low-level execution (subprocess, file handling, import validation, namespace capture)
- **verifier.py (CodeVerifier):** Static and semantic code verification
- **variable_storage.py:** Type-specific variable serialization (DataFrames→Parquet, arrays→.npy, etc.)
- **models.py:** Data models for code generation/execution
- Backward compatibility: `PythonExecutor = CodeExecutor` (legacy alias)
- File metadata extraction is critical for good code generation (handled in generator.py)
- **Context injection:** orchestrator.py passes conversation history, plan context, and react iteration history to prompts

### Security Considerations:
- Blocked imports: socket, subprocess, eval, exec, pickle, etc.
- Execution timeout enforced via subprocess
- Session-based directories prevent cross-contamination
- Import validation uses AST parsing (not runtime)

## Working with ReAct Agent

### Key Modules in backend/tasks/react/:
- **agent.py (ReActAgent):** Main orchestration class
  - `execute()`: Main entry point, runs full ReAct loop
  - `execute_with_plan()`: Guided mode for structured plan execution
- **thought_action_generator.py:** Generates thoughts and selects actions
  - Combined thought-action generation (1 LLM call instead of 2)
- **tool_executor.py:** Routes and executes tool actions
  - Routes to appropriate tool (web_search, rag, python_coder, file_analyzer)
- **answer_generator.py:** Synthesizes observations into final response
- **context_manager.py:** Formats context with pruning optimization
- **verification.py:** Auto-finish detection and step verification
- **plan_executor.py:** Plan-based execution for structured workflows
- **models.py:** Data models (ReActStep, ToolName, ReActResult)

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
- **backend/tasks/react/:** Modular ReAct implementation (more flexible, currently primary)
- Both are available; ReAct is the current production choice

### Session Management
- User authentication via JWT (utils/auth.py)
- Session data stored in `data/conversations/`
- File format: `{user_id}_{timestamp}_{session_id}.json`

### Conversation Storage
- Each message saved with: role, content, timestamp, metadata
- Tools track execution history (steps, observations, code, etc.)
- Retrieval by user_id or session_id

### Session Notepad & Variable Persistence (v1.3.0)
- **Automatic memory generation:** After each ReAct execution, LLM creates structured notepad entry
- **Variable persistence:** Successful code execution variables auto-saved with type-specific serialization
- **Context auto-injection:** Notepad entries and variable metadata automatically injected into subsequent executions
- **Storage structure:** `data/scratch/{session_id}/notepad.json` and `variables/` directory
- **Load-on-demand:** Agent sees available variables and writes code to load when needed
- **Safe serialization:** No pickle; uses Parquet, .npy, JSON, PNG for different types

## Common Development Tasks

### Adding a New Tool
1. Create tool module in `backend/tools/` (follow modular pattern)
2. Add to `ToolName` enum in `backend/tasks/react/models.py`
3. Add execution logic in `backend/tasks/react/tool_executor.py`
4. Update tool selection in PromptRegistry (backend/config/prompts.py)
5. Add to `settings.available_tools` list in `backend/config/settings.py`

### Modifying Agentic Classifier
- Edit `settings.agentic_classifier_prompt` in settings.py
- Controls when to trigger agentic flow vs simple chat
- Be conservative: agentic flow is slower but more capable

### Changing Verification/Retry Limits
```python
# In backend/config/settings.py:
python_code_max_iterations: int = 5  # Verification iterations
# In backend/tools/python_coder/executor.py:
self.max_execution_attempts = 5      # Execution retry attempts
# In backend/tasks/react/agent.py:
react_agent = ReActAgent(max_iterations=10)  # ReAct loop limit
```

### Debugging Execution Issues
- Check logs in `data/logs/app.log`
- Python code execution dirs: `data/scratch/<session_id>/`
- Execution preserves directory if session_id provided
- Review `verification_history` and `execution_attempts_history` in tool response

## Version History & Breaking Changes

### Version 2.0.0 (January 2025)
**Major refactoring: Modular architecture for all core components**

Comprehensive code restructuring that splits large monolithic files into focused, maintainable modules while preserving backward compatibility.

**ReAct Agent Modularization:**
- Split `backend/tasks/React.py` (2000+ lines) → `backend/tasks/react/` (8 focused modules)
- Old import: `from backend.tasks.React import ReActAgent`
- New import: `from backend.tasks.react import ReActAgent`
- Legacy file moved to `backend/tasks/legacy/React.py`

**Python Coder Tool Modularization:**
- Split `backend/tools/python_coder_tool.py` → `backend/tools/python_coder/` (5 modules)
- Old import: `from backend.tools.python_coder_tool import python_coder_tool`
- New import: `from backend.tools.python_coder import python_coder_tool`
- Modules: tool.py, generator.py, executor.py, verifier.py, models.py
- Legacy file moved to `backend/tools/legacy/python_coder_tool.py`

**File Analyzer Tool Modularization:**
- Split `backend/tools/file_analyzer_tool.py` → `backend/tools/file_analyzer/` (4 modules)
- Old import: `from backend.tools.file_analyzer_tool import file_analyzer`
- New import: `from backend.tools.file_analyzer import file_analyzer`
- Modules: tool.py, analyzers.py, llm_analyzer.py, models.py
- Legacy file moved to `backend/tools/legacy/file_analyzer_tool.py`

**Web Search Tool Modularization:**
- Split `backend/tools/web_search.py` → `backend/tools/web_search/` (2 modules)
- Import unchanged: `from backend.tools.web_search import web_search_tool`
- Modules: tool.py, models.py

**RAG Retriever Tool Modularization:**
- Split `backend/tools/rag_retriever.py` → `backend/tools/rag_retriever/` (4 modules)
- Old import: `from backend.tools.rag_retriever import rag_retriever` (still works via backward compatibility)
- New import: `from backend.tools.rag_retriever import rag_retriever_tool`
- Modules: tool.py, retriever.py, models.py
- Backward compatibility: `rag_retriever = rag_retriever_tool` (legacy alias)

**API Routes Modularization:**
- Split `backend/api/routes.py` → `backend/api/routes/` (5 modules)
- Old import: `from backend.api.routes import router`
- New import: `from backend.api.routes import create_routes`
- Modules: chat.py, auth.py, files.py, admin.py, tools.py

**Infrastructure Improvements:**
- Added `backend/utils/llm_factory.py` - centralized LLM instance creation
- Added `backend/config/prompts.py` - centralized prompt management (PromptRegistry)
- All old imports updated throughout codebase
- Legacy files preserved for backward compatibility

### Version 1.4.0 (November 20, 2024)
**Enhancement: Context-Aware Python Code Generation**

Significantly improved Python code generation by integrating comprehensive context from agent workflows and conversation history.

**Key Changes:**
- Enhanced `orchestrator.py` with `conversation_history`, `plan_context`, and `react_context` parameters
- Improved prompts with 8 sections: HISTORIES → INPUT → PLANS → REACTS → TASK → METADATA → RULES → CHECKLISTS
- ReAct agent integration: `_build_react_context()` extracts failed code attempts and errors
- Plan-Execute integration: `_build_plan_context()` creates structured plan context
- Automatic conversation history loading via `ConversationStore`

**Benefits:**
- Better context awareness reduces repeated questions
- Learn from previous failures to prevent same mistakes
- Plan alignment ensures code matches overall strategy
- Improved code quality with full context visibility
- Reduced iterations through context-aware generation

### Version 1.3.0 (November 19, 2024)
**Feature: Session Notepad & Variable Persistence**

Automatic session memory system that maintains continuity across ReAct executions.

**New Components:**
- `backend/tools/notepad.py` - SessionNotepad class for persistent memory
- `backend/tools/python_coder/variable_storage.py` - Type-specific variable serialization

**Key Features:**
- Automatic notepad generation after each ReAct execution
- Variable persistence with type-specific serialization (DataFrames→Parquet, arrays→.npy)
- Context auto-injection into subsequent executions
- Namespace capture in REPL mode
- Task-based organization with descriptive names

**Storage Structure:**
```
./data/scratch/{session_id}/
├── notepad.json                 # Session memory entries
├── {task}_{timestamp}.py        # Saved code files
└── variables/                   # Persisted variables
    ├── variables_metadata.json  # Variable catalog
    ├── df_*.parquet            # DataFrames
    ├── *.json                  # Simple types
    └── *.npy                   # NumPy arrays
```

### Version 1.2.0 (October 31, 2024)
**Major unification of Python code tools**

- Merged `python_coder_tool.py` and `python_executor_engine.py` into single file
- Removed `python_executor.py` (unused legacy)
- Reduced verification iterations: 10 → 3
- Added execution retry logic: max 5 attempts with auto-fixing
- `ToolName.PYTHON_CODE` removed → use `ToolName.PYTHON_CODER`
- Import changes: `PythonExecutor` now alias for `CodeExecutor`

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

### Multi-Phase Workflows: Files as Fallback Pattern
**Key insight:** For multi-step workflows (analyze → visualize → report), process data files ONCE and reuse conversation memory.

**Pattern:**
```
Phase 1: Process files → findings stored in conversation context
           ↓
Phase 2: Reuse Phase 1 findings from memory (files only for verification)
           ↓
Phase 3: Reuse Phase 1 & 2 findings from memory
```

**Benefits:**
- 90% fewer LLM calls - no redundant file parsing
- Faster execution - reuse existing calculations
- Better consistency - all phases use same base analysis
- Lower costs - reduced token usage

**Implementation:**
```python
# Phase 1: Analysis (process files ONCE)
phase1_prompt = """
Analyze the attached CSV file.
Calculate and store in memory: [list of calculations]
I'll ask follow-up questions in subsequent messages.
"""
result1, session_id = client.chat_new(MODEL, phase1_prompt, files=[csv_path])

# Phase 2: Visualization (reuse Phase 1 findings)
phase2_prompt = """
**PRIORITY: Use your Phase 1 analysis from conversation memory.**

You already calculated: [summary of Phase 1 findings]

**DO NOT re-analyze the raw files.** Use your Phase 1 findings.
The attached files are ONLY for verification if needed.

Current Task: Create visualizations based on Phase 1 results
"""
result2, _ = client.chat_continue(MODEL, session_id, phase2_prompt, files=[csv_path])

# Phase 3: Reporting (reuse Phase 1 & 2)
phase3_prompt = """
**PRIORITY: Use Phase 1 & 2 findings from conversation memory.**

Based on your previous analysis and visualizations:
[specific task using prior work]
"""
result3, _ = client.chat_continue(MODEL, session_id, phase3_prompt)
```

**Key phrases for phase handoff:**
- "**PRIORITY: Use your Phase X findings from conversation memory**"
- "You already calculated/analyzed..."
- "**DO NOT re-analyze the raw files**"
- "The attached files are ONLY for verification if needed"

**See examples:**
- [PPTX_Report_Generator_Agent_v2.ipynb](PPTX_Report_Generator_Agent_v2.ipynb) - PowerPoint generation with 90% shorter prompts
- [Multi_Phase_Workflow_Example.ipynb](Multi_Phase_Workflow_Example.ipynb) - Detailed pattern demonstration
- Backend utilities: `backend/utils/phase_manager.py`, `backend/tasks/react/context_manager.py`

**Anti-pattern (avoid):**
```python
# BAD: Re-processing file in every phase
result1, sid = client.chat_new(MODEL, "Analyze file X", files=[path])
result2, _ = client.chat_continue(MODEL, sid, "Analyze file X again for Y", files=[path])  # Wasteful!
```

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
- Review BLOCKED_IMPORTS in backend/tools/python_coder/executor.py
- Ensure required packages installed in environment
- Check AST validation in `CodeExecutor.validate_imports()`

## Additional Resources

**Example Notebooks:**
- [API_examples.ipynb](API_examples.ipynb) - Basic API usage examples
- [PPTX_Report_Generator_Agent_v2.ipynb](PPTX_Report_Generator_Agent_v2.ipynb) - PowerPoint generation with multi-phase workflow
- [Multi_Phase_Workflow_Example.ipynb](Multi_Phase_Workflow_Example.ipynb) - Detailed pattern demonstration

**Utility Scripts:**
- [git-auto-push.sh](git-auto-push.sh) - Automated git workflow (stage, commit, pull with rebase, push)
- [run_backend.py](run_backend.py) - Backend server launcher
- [run_frontend.py](run_frontend.py) - Frontend server launcher

---

**Last Updated:** January 2025
**Version:** 2.0.0 (Modular Architecture)
**Previous Versions:** 1.4.0 (Context-Aware Code Gen), 1.3.0 (Session Notepad), 1.2.0 (Tool Unification)
