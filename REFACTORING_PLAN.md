# Backend Comprehensive Refactoring Plan

**Date:** 2025-11-13
**Branch:** claude/refactor-backend-comprehensive-011CV5JXQqhtcBzTtvABSemt
**Estimated Token Budget:** 150,000 tokens
**Parallelization Strategy:** 11 phases, ~6 can run in parallel

---

## Executive Summary

### Critical Issues Identified:
1. **Two MASSIVE files** (1,700+ lines each): `React.py`, `python_coder_tool.py`
2. **Prompt duplication**: 60% inline prompts, 40% centralized in config/prompts/
3. **Old and new code coexist**: Partial migration incomplete
4. **14 files exceed 300 lines** (target: all < 300 lines)
5. **Overly complex workflows**: 3-level nested loops, 8-level method call depth
6. **Code duplication**: Context building, metadata extraction, LLM initialization

### Refactoring Goals:
- ✅ All files < 300 lines
- ✅ All prompts centralized in config/prompts/
- ✅ Remove old monolithic files (React.py, python_coder_tool.py)
- ✅ Simplify workflows (reduce iterations, flatten nesting)
- ✅ Eliminate code duplication
- ✅ Improve debuggability and maintainability

---

## Phase 1: Centralize All LLM Prompts
**Priority:** CRITICAL
**Parallelizable:** Yes
**Estimated Tokens:** 8,000

### Tasks:
1. **Extract React.py inline prompts** → `config/prompts/react_agent.py`
   - Lines 623-642: Auto-finish synthesis
   - Lines 663-680: Context pruning
   - Lines 713-730: Plan step context
   - Lines 762-776: Final answer with steps
   - Lines 1490-1519: Legacy prompts

2. **Extract python_coder_tool.py inline prompts** → `config/prompts/python_coder.py`
   - Lines 1163-1337: Code generation
   - Lines 1375-1509: Code verification
   - Lines 1660-1712: Code modification
   - Lines 1714-1766: Error fixing

3. **Extract web_search.py inline prompts** → `config/prompts/web_search.py`
   - Lines 318-357: Answer generation system
   - Lines 457-478: Answer generation user

4. **Remove duplicate from settings.py**
   - Line 117: `agentic_classifier_prompt` (use task_classification.py instead)

5. **Create prompt registry**
   - `config/prompts/__init__.py` with PromptRegistry class
   - Central access point for all prompts
   - Validation and caching

### Files Modified:
- `config/prompts/react_agent.py` (add 5 new functions)
- `config/prompts/python_coder.py` (add 4 new functions)
- `config/prompts/web_search.py` (add 2 new functions)
- `config/prompts/__init__.py` (NEW)
- `config/settings.py` (remove inline prompt)

### Deliverables:
- All prompts accessible via: `from config.prompts import PromptRegistry`
- Zero inline prompts remaining in tool files

---

## Phase 2: Create Utility Modules
**Priority:** CRITICAL
**Parallelizable:** Yes (concurrent with Phase 1)
**Estimated Tokens:** 5,000

### Tasks:
1. **Create LLM Factory** (`utils/llm_factory.py`)
   - Single source of truth for LLM initialization
   - Support for different models (classifier, coder, etc.)
   - Connection pooling and retry logic
   - ~100 lines

2. **Create Prompt Builder** (`utils/prompt_builder.py`)
   - Load prompts from registry
   - Format with context variables
   - Validation and error handling
   - ~80 lines

3. **Create Metadata Models** (`models/tool_metadata.py`)
   - Standardized metadata structures
   - Dataclasses for tool responses
   - ~120 lines

4. **Create File Metadata Service** (`services/file_metadata_service.py`)
   - Unified file metadata extraction
   - Used by python_coder and file_analyzer
   - Eliminates duplication
   - ~200 lines

### Files Created:
- `utils/llm_factory.py` (NEW)
- `utils/prompt_builder.py` (NEW)
- `models/tool_metadata.py` (NEW)
- `services/file_metadata_service.py` (NEW)

### Deliverables:
- All tools use `llm_factory.create_llm()`
- All prompts loaded via `prompt_builder.get_prompt()`
- Standardized metadata across tools

---

## Phase 3: Refactor API Routes
**Priority:** HIGH
**Parallelizable:** Yes (concurrent with Phases 1-2)
**Estimated Tokens:** 6,000

### Tasks:
1. **Split routes.py (521 lines)** into:
   ```
   api/routes/
   ├── __init__.py (route aggregation)
   ├── auth.py (~70 lines)
   ├── chat.py (~150 lines)
   ├── admin.py (~50 lines)
   ├── files.py (~80 lines)
   └── tools.py (~100 lines)
   ```

2. **Create route factory**
   - `api/routes/__init__.py` aggregates all routers
   - Single import point: `from api.routes import create_routes`

3. **Update app.py**
   - Replace individual router imports
   - Use aggregated routes from factory

### Files Created:
- `api/routes/__init__.py` (NEW)
- `api/routes/auth.py` (NEW)
- `api/routes/chat.py` (NEW)
- `api/routes/admin.py` (NEW)
- `api/routes/files.py` (NEW)
- `api/routes/tools.py` (NEW)

### Files Modified:
- `api/app.py` (simplified imports)

### Files Removed:
- `api/routes.py` (521 lines → DELETED)

---

## Phase 4: Migrate React.py to Modular Structure
**Priority:** CRITICAL
**Parallelizable:** Partially (dependencies on Phases 1-2)
**Estimated Tokens:** 20,000

### Current State:
- Old: `tasks/React.py` (1,768 lines) - STILL IN USE
- New: `tasks/react/` (partially complete)

### Tasks:

#### 4.1: Create Missing Modules
1. **Create `tasks/react/agent.py`** (~250 lines)
   - Main ReActAgent class
   - Public methods: execute(), execute_with_plan()
   - Orchestration logic only
   - Use modules for sub-tasks

2. **Create `tasks/react/context_manager.py`** (~180 lines)
   - Consolidate all context building methods
   - Methods: build_tool_context(), prune_context(), build_plan_context()
   - Remove duplication

3. **Create `tasks/react/verification.py`** (~150 lines)
   - Step verification logic
   - Auto-finish detection
   - Success criteria checking

4. **Create `tasks/react/plan_executor.py`** (~200 lines)
   - Plan-based execution logic
   - Step execution with fallbacks
   - Extracted from React.py execute_with_plan()

5. **Create `tasks/react/models.py`** (~80 lines)
   - ReActStep dataclass
   - ReActMetadata dataclass
   - All data structures

#### 4.2: Refactor Existing Modules
1. **Simplify `thought_action_generator.py`** (453 → 250 lines)
   - Remove inline prompts
   - Use PromptRegistry
   - Use llm_factory

2. **Simplify `tool_executor.py`** (386 → 250 lines)
   - Remove guard logic (python_coder before RAG)
   - Simplify tool routing
   - Use standardized metadata

3. **Simplify `answer_generator.py`** (320 → 200 lines)
   - Remove inline prompts
   - Use PromptRegistry
   - Cleaner synthesis logic

#### 4.3: Update Imports
- Update all files importing from `tasks.React`
- Point to `tasks.react.agent` instead

### Files Created:
- `tasks/react/agent.py` (NEW)
- `tasks/react/context_manager.py` (NEW)
- `tasks/react/verification.py` (NEW)
- `tasks/react/plan_executor.py` (NEW)
- `tasks/react/models.py` (NEW)

### Files Modified:
- `tasks/react/thought_action_generator.py` (453 → 250 lines)
- `tasks/react/tool_executor.py` (386 → 250 lines)
- `tasks/react/answer_generator.py` (320 → 200 lines)

### Files Moved to Legacy:
- `tasks/React.py` (1,768 lines) → `legacy/React.py.bak`

---

## Phase 5: Consolidate Python Coder Modules
**Priority:** CRITICAL
**Parallelizable:** Partially (dependencies on Phases 1-2)
**Estimated Tokens:** 15,000

### Current State:
- Old: `tools/python_coder_tool.py` (1,782 lines) - STILL IN USE
- New: `tools/python_coder/` (partially complete)

### Tasks:

#### 5.1: Simplify Orchestrator
**File:** `tools/python_coder/orchestrator.py` (482 → 280 lines)

Changes:
- Remove inline prompts (use PromptRegistry)
- Use file_metadata_service instead of local extraction
- Reduce verification loop: 3 → 2 iterations
- Reduce execution loop: 5 → 3 attempts
- Simplify error handling

#### 5.2: Refactor Code Generator
**File:** `tools/python_coder/code_generator.py` (needs work)

Changes:
- Clean separation: generation vs modification
- Use prompt_builder for all prompts
- Better context building using context_builder module

#### 5.3: Simplify Code Verifier
**File:** `tools/python_coder/code_verifier.py` (needs simplification)

Changes:
- Remove auto-fix logic (move to separate pre-processor)
- Focus on verification only
- Use standardized verification models

#### 5.4: Create Auto-Fix Pre-processor
**File:** `tools/python_coder/auto_fixer.py` (NEW)

Extract auto-fix logic:
- Filename corrections
- Import additions
- Encoding fixes
- Run BEFORE verification, not during

#### 5.5: Refactor File Handlers
**Directory:** `tools/python_coder/file_handlers/`

Current issue: Some handlers too large
- `json_handler.py` (222 lines) - OK
- Others need review and simplification

#### 5.6: Update Main Tool Interface
**File:** `tools/python_coder/__init__.py`

Export single entry point:
```python
from .orchestrator import PythonCoderTool
```

### Files Created:
- `tools/python_coder/auto_fixer.py` (NEW, ~120 lines)

### Files Modified:
- `tools/python_coder/orchestrator.py` (482 → 280 lines)
- `tools/python_coder/code_generator.py` (refactor)
- `tools/python_coder/code_verifier.py` (simplify)
- `tools/python_coder/executor.py` (use file_metadata_service)
- `tools/python_coder/__init__.py` (clean exports)

### Files Moved to Legacy:
- `tools/python_coder_tool.py` (1,782 lines) → `legacy/python_coder_tool.py.bak`

---

## Phase 6: Refactor File Analyzer Tool
**Priority:** HIGH
**Parallelizable:** Yes (concurrent with Phases 4-5)
**Estimated Tokens:** 10,000

### Current State:
- Monolithic: `tools/file_analyzer_tool.py` (857 lines)
- Needs: Strategy pattern with separate handlers

### Tasks:

#### 6.1: Create Handler Structure
```
tools/file_analyzer/
├── __init__.py (main export)
├── analyzer.py (~200 lines) - main FileAnalyzer class
├── base_handler.py (~80 lines) - abstract base
└── handlers/
    ├── __init__.py
    ├── csv_handler.py (~120 lines)
    ├── excel_handler.py (~100 lines)
    ├── json_handler.py (~140 lines)
    ├── text_handler.py (~80 lines)
    ├── pdf_handler.py (~100 lines)
    ├── docx_handler.py (~100 lines)
    └── image_handler.py (~80 lines)
```

#### 6.2: Create Base Handler
**File:** `tools/file_analyzer/base_handler.py`

Abstract class with:
- `supports(file_path: str) -> bool`
- `analyze(file_path: str) -> FileAnalysisResult`
- Common utilities

#### 6.3: Extract Format-Specific Handlers
Split current monolithic file by format:
- CSV: DataFrame analysis, column types, stats
- Excel: Sheet info, data preview
- JSON: Structure analysis, schema inference
- PDF: Text extraction, page count
- DOCX: Text extraction, metadata
- Images: Dimensions, format, metadata

#### 6.4: Create Main Analyzer
**File:** `tools/file_analyzer/analyzer.py`

Responsibilities:
- Register handlers
- Route to appropriate handler based on file extension
- Aggregate results
- Error handling

#### 6.5: Integrate with file_metadata_service
- Use shared metadata service for common operations
- Avoid duplication with python_coder metadata extraction

### Files Created:
- `tools/file_analyzer/__init__.py` (NEW)
- `tools/file_analyzer/analyzer.py` (NEW, ~200 lines)
- `tools/file_analyzer/base_handler.py` (NEW, ~80 lines)
- `tools/file_analyzer/handlers/csv_handler.py` (NEW, ~120 lines)
- `tools/file_analyzer/handlers/excel_handler.py` (NEW, ~100 lines)
- `tools/file_analyzer/handlers/json_handler.py` (NEW, ~140 lines)
- `tools/file_analyzer/handlers/text_handler.py` (NEW, ~80 lines)
- `tools/file_analyzer/handlers/pdf_handler.py` (NEW, ~100 lines)
- `tools/file_analyzer/handlers/docx_handler.py` (NEW, ~100 lines)
- `tools/file_analyzer/handlers/image_handler.py` (NEW, ~80 lines)

### Files Moved to Legacy:
- `tools/file_analyzer_tool.py` (857 lines) → `legacy/file_analyzer_tool.py.bak`

---

## Phase 7: Refactor Web Search Tool
**Priority:** MEDIUM
**Parallelizable:** Yes (concurrent with Phase 6)
**Estimated Tokens:** 5,000

### Current State:
- `tools/web_search.py` (509 lines)
- Has inline prompts (duplicates config/prompts/web_search.py)

### Tasks:

#### 7.1: Remove Inline Prompts
- Delete lines 318-357 (answer generation system prompt)
- Delete lines 457-478 (answer generation user prompt)
- Import from config.prompts.web_search instead

#### 7.2: Use LLM Factory
- Replace direct ChatOllama initialization
- Use llm_factory.create_llm()

#### 7.3: Simplify Search Flow
Current flow is complex:
1. Query refinement (LLM call)
2. Tavily search (API call)
3. Answer generation (LLM call)

Simplify:
- Make query refinement optional (flag)
- Better error handling
- Cleaner result formatting

#### 7.4: Extract Search Result Processor
Create `tools/web_search/result_processor.py`:
- Parse Tavily results
- Filter and rank
- Format for LLM consumption

#### 7.5: Split into Modules
```
tools/web_search/
├── __init__.py (main export)
├── searcher.py (~150 lines) - main WebSearchTool
├── query_refiner.py (~100 lines)
├── result_processor.py (~120 lines)
└── answer_generator.py (~100 lines)
```

### Files Created:
- `tools/web_search/__init__.py` (NEW)
- `tools/web_search/searcher.py` (NEW, ~150 lines)
- `tools/web_search/query_refiner.py` (NEW, ~100 lines)
- `tools/web_search/result_processor.py` (NEW, ~120 lines)
- `tools/web_search/answer_generator.py` (NEW, ~100 lines)

### Files Moved to Legacy:
- `tools/web_search.py` (509 lines) → `legacy/web_search.py.bak`

---

## Phase 8: Simplify Workflow Complexity
**Priority:** HIGH
**Parallelizable:** No (depends on Phases 4-5)
**Estimated Tokens:** 8,000

### Tasks:

#### 8.1: Reduce Iteration Limits
**Files to modify:**
- `tasks/react/agent.py`: max_iterations: 10 → 6
- `tools/python_coder/orchestrator.py`: verification: 3 → 2, execution: 5 → 3
- `tasks/smart_agent_task.py`: plan steps limit

#### 8.2: Remove Guard Logic
**File:** `tasks/react/tool_executor.py`

Remove complex guard:
```python
# REMOVE THIS:
if tool == ToolName.RAG_RETRIEVAL and files_exist:
    # Try python_coder first
    # Then fallback to RAG
```

Reason: Adds unnecessary complexity, let LLM decide

#### 8.3: Simplify Auto-Finish Detection
**File:** `tasks/react/verification.py`

Current: 10 different heuristics
Reduce to: 3 core heuristics
- Observation length > 200 chars
- Contains "answer" keywords
- No errors in observation

#### 8.4: Flatten Plan-Execute Nesting
**File:** `tasks/react/plan_executor.py`

Remove fallback tools:
- Each step gets ONE tool (not primary + fallbacks)
- If fails, move to next step (don't retry with different tool)
- Reduces nesting from 4 levels to 2 levels

#### 8.5: Simplify Context Pruning
**File:** `tasks/react/context_manager.py`

Current: Complex logic with summarization
New: Simple sliding window
- Keep last 3 steps in full detail
- Discard older steps entirely (no summarization LLM call)

### Files Modified:
- `tasks/react/agent.py` (reduce max_iterations)
- `tasks/react/tool_executor.py` (remove guard logic)
- `tasks/react/verification.py` (simplify auto-finish)
- `tasks/react/plan_executor.py` (remove fallback tools)
- `tasks/react/context_manager.py` (simplify pruning)
- `tools/python_coder/orchestrator.py` (reduce iterations)

### Expected Impact:
- **Reduce average LLM calls:** 10-15 → 5-8 per request
- **Reduce execution time:** 30-60s → 15-30s
- **Simplify debugging:** Fewer branches to trace
- **Lower token usage:** ~40% reduction

---

## Phase 9: Update All Imports
**Priority:** CRITICAL
**Parallelizable:** No (must run after Phases 3-7)
**Estimated Tokens:** 6,000

### Tasks:

#### 9.1: Find All Old Imports
Search for:
- `from backend.tasks.React import`
- `from backend.tools.python_coder_tool import`
- `from backend.tools.file_analyzer_tool import`
- `from backend.tools.web_search import` (if moved to module)
- `from backend.api.routes import`

#### 9.2: Update to New Imports
Replace with:
- `from backend.tasks.react.agent import ReActAgent`
- `from backend.tools.python_coder import PythonCoderTool`
- `from backend.tools.file_analyzer import FileAnalyzer`
- `from backend.tools.web_search import WebSearchTool`
- `from backend.api.routes import create_routes`

#### 9.3: Update Tests (if any)
- Search for test files
- Update imports
- Verify tests still pass

#### 9.4: Update Documentation
- Update CLAUDE.md with new import paths
- Update architecture diagrams
- Update file structure documentation

### Files to Check:
- `api/app.py`
- `tasks/chat_task.py`
- `tasks/smart_agent_task.py`
- `tasks/Plan_execute.py`
- `core/agent_graph.py`
- Any other files importing refactored modules

### Verification:
```bash
# Check for old imports
grep -r "from backend.tasks.React" backend/
grep -r "from backend.tools.python_coder_tool" backend/
grep -r "from backend.tools.file_analyzer_tool" backend/
grep -r "from backend.api.routes import" backend/

# Should return NO results
```

---

## Phase 10: Create Legacy Backup and Cleanup
**Priority:** MEDIUM
**Parallelizable:** No (must run after Phase 9)
**Estimated Tokens:** 2,000

### Tasks:

#### 10.1: Create Legacy Directory
```bash
mkdir -p backend/legacy
```

#### 10.2: Move Old Files
```bash
mv backend/tasks/React.py backend/legacy/React.py.bak
mv backend/tools/python_coder_tool.py backend/legacy/python_coder_tool.py.bak
mv backend/tools/file_analyzer_tool.py backend/legacy/file_analyzer_tool.py.bak
mv backend/tools/web_search.py backend/legacy/web_search.py.bak
mv backend/api/routes.py backend/legacy/routes.py.bak
```

#### 10.3: Update .gitignore
Add:
```
backend/legacy/
```

#### 10.4: Verify No References Remain
```bash
grep -r "React.py" backend/ --exclude-dir=legacy
grep -r "python_coder_tool.py" backend/ --exclude-dir=legacy
# etc.
```

#### 10.5: Clean Up Unused Imports
Remove any imports that are no longer needed

---

## Phase 11: Final Verification and Testing
**Priority:** CRITICAL
**Parallelizable:** No (must run last)
**Estimated Tokens:** 5,000

### Tasks:

#### 11.1: Static Analysis
- Check all files < 300 lines
- Verify no circular imports
- Check for unused imports
- Run type checker (if using mypy)

#### 11.2: Test Server Startup
```bash
python run_backend.py
```
Verify:
- Server starts without errors
- Ollama connection works
- All routes registered

#### 11.3: Test Core Workflows
Manual testing:
1. Simple chat message
2. Agentic task (web search)
3. Python code generation
4. File analysis
5. RAG retrieval
6. Plan-Execute task

#### 11.4: Performance Comparison
Compare old vs new:
- Average response time
- Token usage per request
- LLM calls per request
- Memory usage

#### 11.5: Update Documentation

**Update CLAUDE.md:**
- New architecture section
- New file structure
- New import paths
- Remove references to old files

**Create REFACTORING_SUMMARY.md:**
- What changed
- Before/after metrics
- Migration guide
- Breaking changes

#### 11.6: Git Commit
```bash
git add .
git commit -m "Comprehensive backend refactoring

- Broke down 2 giant files (1,700+ lines → modules < 300 lines)
- Centralized all LLM prompts to config/prompts/
- Created utility modules (LLM factory, prompt builder)
- Refactored API routes into separate modules
- Migrated React.py to tasks/react/ module
- Consolidated python_coder_tool.py to tools/python_coder/
- Refactored file_analyzer into handler modules
- Refactored web_search into separate modules
- Simplified workflows (reduced iterations, removed guard logic)
- Eliminated code duplication
- Standardized metadata structures
- All files now < 300 lines
- Improved debuggability and maintainability"

git push -u origin claude/refactor-backend-comprehensive-011CV5JXQqhtcBzTtvABSemt
```

---

## Parallelization Strategy

### Parallel Group 1 (Can run simultaneously):
- Phase 1: Centralize prompts
- Phase 2: Create utility modules
- Phase 3: Refactor API routes

### Parallel Group 2 (After Group 1):
- Phase 4: Migrate React.py
- Phase 5: Consolidate python_coder
- Phase 6: Refactor file_analyzer
- Phase 7: Refactor web_search

### Sequential Group 3 (Must run in order):
- Phase 8: Simplify workflow complexity (after 4-5)
- Phase 9: Update all imports (after 3-7)
- Phase 10: Create legacy backup (after 9)
- Phase 11: Final verification (after 10)

### Total Parallel Agents: 7
- Agent 1: Phase 1 (Centralize prompts)
- Agent 2: Phase 2 (Utility modules)
- Agent 3: Phase 3 (API routes)
- Agent 4: Phase 4 (React migration)
- Agent 5: Phase 5 (Python coder)
- Agent 6: Phase 6 (File analyzer)
- Agent 7: Phase 7 (Web search)

---

## Success Metrics

### Before Refactoring:
- Files > 300 lines: 14 files
- Largest file: 1,782 lines
- Total lines: 13,095
- Inline prompts: ~60%
- Avg LLM calls per request: 10-15
- Code duplication: High
- Workflow nesting: 4 levels
- Max method depth: 8 levels

### After Refactoring (Target):
- Files > 300 lines: 0 files ✅
- Largest file: < 300 lines ✅
- Total lines: ~14,000 (slightly more due to module overhead) ✅
- Inline prompts: 0% ✅
- Avg LLM calls per request: 5-8 ✅
- Code duplication: Minimal ✅
- Workflow nesting: 2 levels ✅
- Max method depth: 4 levels ✅

---

## Estimated Timeline

With parallel execution:
- Group 1: 30-40 minutes
- Group 2: 60-80 minutes
- Group 3: 30-40 minutes
- **Total: 2-3 hours**

---

## Risk Mitigation

### Risks:
1. **Breaking imports:** Mitigated by thorough Phase 9 verification
2. **Lost functionality:** Mitigated by comprehensive testing in Phase 11
3. **Performance regression:** Mitigated by benchmarking in Phase 11
4. **Merge conflicts:** Using dedicated branch, clean history

### Rollback Strategy:
- Keep old files in legacy/ directory
- Git history preserves all changes
- Can revert commit if issues found
- Phase 11 testing catches issues before push

---

## Notes for Execution

1. **Use parallel agents wherever possible** (7 agents in Group 2)
2. **Each agent should:**
   - Focus on single phase
   - Report completion status
   - List files created/modified/deleted
   - Verify no syntax errors

3. **After each phase:**
   - Verify files compile (Python syntax)
   - Check line counts (must be < 300)
   - Run quick smoke test

4. **Communication between agents:**
   - Not required (phases are independent)
   - Coordinator (main agent) tracks completion

5. **Token budget allocation:**
   - Phase 1: 8,000 tokens
   - Phase 2: 5,000 tokens
   - Phase 3: 6,000 tokens
   - Phase 4: 20,000 tokens (largest)
   - Phase 5: 15,000 tokens
   - Phase 6: 10,000 tokens
   - Phase 7: 5,000 tokens
   - Phase 8: 8,000 tokens
   - Phase 9: 6,000 tokens
   - Phase 10: 2,000 tokens
   - Phase 11: 5,000 tokens
   - **Total: 90,000 tokens (well within budget)**

---

**Ready to execute!** 🚀
