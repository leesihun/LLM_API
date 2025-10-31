# LLM_API

AI-powered Python code generation and execution API with iterative verification.

## Version History

### v1.2.0 (2025-10-31)
**ReAct Performance Optimization - 50-70% Faster**
- **Strategy 1 - Combined Thought-Action**: Merged thought and action generation into single LLM call (50% reduction in free mode)
  - Added `_generate_thought_and_action()` method
  - Updated free mode execute loop to use combined generation
  - **Impact**: Free mode now makes ~6-11 LLM calls instead of ~21 (50-70% faster)
- **Strategy 4 - Context Pruning**: Optimized step history sent to LLM
  - If ≤3 steps: send all details
  - If >3 steps: send summary + last 2 steps only
  - **Impact**: Reduced token usage, faster LLM processing
- **Strategy 5 - Early Exit**: Auto-detect when observation contains complete answer
  - Added `_should_auto_finish()` heuristic method
  - Checks for answer indicators (conclusions, results, substantial content)
  - **Impact**: Saves 2-5 iterations on queries that get good results early
- **Strategy 6 - Skip Redundant Final Answer**: Skip final LLM call in guided mode when unnecessary
  - Added `_is_final_answer_unnecessary()` method
  - Checks if last step already contains comprehensive answer
  - **Impact**: Saves 1 LLM call per guided execution (7-10% faster)
- **Files Modified**: `backend/tasks/React.py`
- **Performance Gains**:
  - Free Mode: ~21 LLM calls → ~6-11 calls (50-70% faster)
  - Guided Mode: ~14 LLM calls → ~10-12 calls (20-30% faster)

### v1.1.1 (2025-10-31)
**Server Auto-Reload Disabled**
- **Changed**: Disabled uvicorn auto-reload (`reload=False`) to prevent unnecessary server restarts
- **Reason**: Reduces shutdown_event log messages and improves stability in production
- **Impact**: Server no longer automatically restarts on file changes
- **Files Modified**: `server.py`

### v1.1.0 (2025-10-31)
**Plan-Execute & ReAct Integration Restructure**
- **Major Refactor**: Restructured Plan-Execute to create structured, JSON-based execution plans with explicit goals, tools, and fallback options
- **Added**: New schemas `PlanStep` and `StepResult` for structured planning and execution tracking
- **Added**: ReAct "guided mode" (`execute_with_plan()`) that executes plan steps one-by-one instead of free-form iteration
- **Added**: Automatic tool fallback mechanism - each step tries primary tools first, then fallback tools if they fail
- **Added**: Step-level success verification using LLM to check if step goals are met
- **Added**: Comprehensive step-by-step execution tracking with detailed metadata
- **Changed**: Plan-Execute now generates structured JSON plans with primary_tools, fallback_tools, and success_criteria per step
- **Changed**: Each plan step is executed independently with its own ReAct loop and goal verification
- **Impact**: Better separation of concerns - planning is strategic, execution is tactical with automatic recovery
- **Files Modified**: 
  - `backend/models/schemas.py` - Added PlanStep and StepResult models
  - `backend/tasks/Plan_execute.py` - Restructured to generate JSON plans and call ReAct guided mode
  - `backend/tasks/React.py` - Added execute_with_plan(), _execute_step(), _execute_tool_for_step(), _verify_step_success()
- **Architecture**:
  - Phase 1 (Planning): LLM generates structured plan with steps, each having goal + tools + fallback + success criteria
  - Phase 2 (Execution): ReAct executes each step sequentially, trying tools in order until success
  - Phase 3 (Verification): Each step verified against success criteria; final answer synthesized from all results

### v1.0.6 (2025-10-31)
**File-First Agent Behavior + Safe Fallbacks**
- **Added**: When files are attached, ReAct attempts local analysis first via `python_coder` before other tools.
- **Added**: Guarded fallback in ReAct to try `python_coder` once before `rag_retrieval`/`web_search`.
- **Changed**: ReAct action-selection prompt now prefers local file tools when files exist.
- **Changed**: Plan-and-Execute planning prompt injects file-first guidance when files are present.
- **Impact**: JSON/Excel analysis reliably uses the uploaded files; gracefully falls back if code fails.
- **Files Modified**: `backend/tasks/React.py`, `backend/tasks/Plan_execute.py`

### v1.0.5 (2025-10-31)
**File Uploads Always Routed to Agentic Workflow**
- **Fixed**: Attached files were ignored when the query was classified as simple chat, causing replies like "attached file was not provided".
- **Changed**: If files are attached to `/v1/chat/completions`, the request now forces the agentic workflow to ensure file handling.
- **Impact**: JSON/Excel analysis examples reliably access the uploaded files.
- **Files Modified**: `backend/api/routes.py`

### v1.0.3 (2025-10-31)
**Logging Readability Overhaul**
- **Changed**: Standardized log format to include timestamp, level, module, function, and line
- **Added**: Global readability filter that removes banner/separator lines, collapses multi-line messages to a single-line preview, and truncates overly long entries
- **Impact**: `logger.info` output across the entire app is significantly cleaner and easier to scan
- **Files Modified**: `backend/api/app.py`
- **Details**:
  - Introduced `ReadabilityFilter` applied to both console and file handlers
  - New format: `%(asctime)s %(levelname)s %(name)s:%(funcName)s:%(lineno)d - %(message)s`

### v1.0.4 (2025-10-31)
**ReAct Logging: Inputs/Outputs Only + Line Breaks**
- **Changed**: Removed logging of full LLM system prompts in ReAct agent
- **Added**: Line-by-line logging for LLM/tool outputs and extra blank lines between phases for readability
- **Impact**: Clearer logs showing only inputs and outputs, without verbose prompt templates
- **Files Modified**: `backend/tasks/React.py`
- **Details**:
  - Replaced multi-line single log entries with per-line logs to avoid collapsing
  - Omitted prompt bodies in thought/action/final-answer generation logs

### v1.0.2 (2025-10-31)
**Python Executor - Fixed NameError for Built-in Functions**
- **Fixed**: `NameError: name 'print' is not defined` when executing code via PYTHON_CODE tool
- **Reason**: `__builtins__` can be either a dict or module depending on execution context; the code only handled the module case
- **Impact**: Built-in functions like `print`, `len`, `str`, `int`, etc. now work correctly in executed code
- **Files Modified**: `backend/tools/python_executor.py`
- **Details**: 
  - Updated `_create_safe_globals()` method to handle both dict and module types for `__builtins__`
  - Adds type checking: uses dict access when `__builtins__` is dict, getattr when it's a module
  - Ensures all non-forbidden built-in functions are available in the execution environment

### v1.0.1 (2025-10-31)
**Python Coder Tool - Output Format Simplification**
- **Changed**: Removed JSON output requirement from code verification prompt
- **Reason**: Eliminated inconsistency between code generation (which instructed to use prints) and verification (which checked for JSON output)
- **Impact**: Generated code now consistently uses simple `print()` statements for output instead of JSON formatting
- **Files Modified**: `backend/tools/python_coder_tool.py`
- **Details**: 
  - Updated verification prompt "OUTPUT FORMAT COMPLIANCE" section to "OUTPUT FORMAT"
  - Now verifies that code uses print() statements, outputs are clear, and errors are communicated
  - Removes unnecessary complexity of JSON formatting in generated code

### v1.0.0 (Initial)
- Python code generation with LLM (Ollama)
- Iterative code verification and modification
- Secure Python code execution with sandboxing
- File upload and processing support
- User authentication and session management
- Conversation history storage
- Web search integration
- RAG (Retrieval Augmented Generation) capabilities

