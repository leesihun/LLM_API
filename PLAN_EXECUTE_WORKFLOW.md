# Plan-and-Execute Workflow

This document captures the end-to-end flow of the Plan-and-Execute agent implemented in this repository (`backend/tasks/Plan_execute.py`). The workflow blends deliberate planning with the flexibility of the ReAct agent, resulting in three tightly-coupled phases:

1. **Planning** – Build a structured, tool-aware plan with success criteria.
2. **Execution** – Drive each plan step through ReAct’s Thought→Action→Observation loop (guided mode).
3. **Monitoring** – Track outcomes, adapt the plan if needed, and synthesize a final answer plus metadata.

The following sections walk through every stage and component involved, from request entry to final response generation.

---

## 0. Inputs & Pre-Flight

When `PlanExecuteTask.execute()` is invoked via `smart_agent_task`, it receives:

- `messages`: full chat history (list of `ChatMessage` objects).
- `session_id`: optional identifier for conversation + persisted artifacts.
- `user_id`: used for logging and scratch storage.
- `file_paths`: optional local file references (uploads or previous outputs).
- `max_iterations`: per-step cap handed to the ReAct executor.

Before planning begins, the agent attempts to hydrate context via `_load_notepad_and_variables()`:

- Loads session notebook entries (`backend/tools/notepad.py`) if present.
- Loads persisted variable metadata through `backend.tools.python_coder.variable_storage.VariableStorage`.
- Failures only log warnings, ensuring the workflow is resilient even when persistence is unavailable.

The last user message becomes the **active query**, while earlier turns form the **conversation history** string handed to the planner. Attached files are noted so the planner can consider file-aware strategies.

---

## 1. Planning Phase

File: `backend/tasks/Plan_execute.py`, method `_create_execution_plan()`.

1. **Prompt Assembly**  
   - Uses `prompts.get_execution_plan_prompt(...)` with the current query, serialized conversation history, available tools (`settings.available_tools`), and `has_files` flag.
   - Prompt instructs the LLM to output a JSON array describing executable steps (`goal`, `primary_tools`, `success_criteria`, optional `context`).

2. **LLM Invocation**  
   - Runs through `LLMFactory.create_llm(temperature=0.3)` (lazy-loaded) for deterministic planning.
   - Response is logged (truncated) for auditability.

3. **Parsing & Validation**  
   - Extracts the JSON payload even when the LLM wraps it in Markdown fences.
   - Each object is converted into a `PlanStep` Pydantic model (`backend/models/schemas.py`), auto-numbering steps when the LLM omits `step_num`.
   - Errors in parsing are surfaced immediately (JSON decode, missing fields, etc.), since a valid plan is required for execution.

The output of this phase is an ordered list of `PlanStep` instances capturing:

- `goal` – human-readable objective for the step.
- `primary_tools` – ordered list (first entry drives execution).
- `success_criteria` – textual definition of “done”.
- `context` – extra hints or constraints for the executor.

---

## 2. Execution Phase (Guided ReAct)

Files: `backend/tasks/Plan_execute.py`, `backend/tasks/react/agent.py`, `backend/tasks/react/plan_executor.py`.

### 2.1 Hand-off to ReAct (Guided Mode)

```
PlanExecuteTask.execute()
└── react_agent.execute_with_plan(...)
    └── PlanExecutor.execute_plan(...)
```

- `react_agent.execute_with_plan()` mirrors the free-form ReAct loop but injects the structured plan.
- Session/file context is stored on the agent (`self.file_paths`, `self.session_id`), and `_load_variables()` is called if a session exists (variable metadata → `ContextManager`).
- The `PlanExecutor` is initialized with shared collaborators: `ToolExecutor`, `ContextManager`, `StepVerifier`, `AnswerGenerator`, and the LLM instance for any follow-up prompts.

### 2.2 Step Loop

For each `PlanStep`, `PlanExecutor.execute_plan()` performs:

1. **Logging & Context Prep**
   - Emits structured logs (goal, primary tool, step index).
   - Builds a lightweight textual context from prior observations (keeps the last 5 to avoid prompt bloat).

2. **Single-Tool Execution (`execute_step`)**
   - Picks the first entry in `primary_tools`. There is no automatic fallback—plans should explicitly encode retries or alternates.
   - Builds the action input via `_build_action_input_from_plan()`:
     - Original user query.
     - Step-specific goal.
     - Optional instructions/constraints from `PlanStep.context`.
     - Condensed results from previous steps when available.
   - For Python-heavy steps, constructs a rich `plan_context` object that includes:
     - Current vs. total steps.
     - Status of each plan step (completed/current/pending).
     - Summaries of previous observations.
     - This context flows into the python_coder prompt to keep generated code aligned with the global plan.
   - Delegates tool execution to `ToolExecutor.execute(...)`, which knows how to call `web_search`, `rag_retrieval`, `python_coder`, etc. The tool run inherits `file_paths` and `session_id`, so code execution can load uploaded files or persisted variables.

3. **Result Capture**
   - Wraps each outcome in a `StepResult` model, storing `success`, `tool_used`, `attempts`, `observation`, and any error text.
   - Appends concise observation snippets to `accumulated_observations` for later synthesis and for feeding subsequent steps.

### 2.3 Dynamic Plan Adaptation

After each step (except the last), the executor checks whether the remaining plan is still viable using `PlanAdapter.should_replan()`:

- **Trigger 1:** Step failed after ≥3 attempts.
- **Trigger 2:** Observation text contains keywords signaling missing prerequisites or new requirements.
- **Trigger 3:** LLM-based viability check that weighs current results against remaining steps (requires ≥0.7 confidence to demand re-planning).

If any trigger fires:

1. `PlanAdapter.adapt_plan()` compiles context from completed steps and the current issue.
2. An LLM prompt requests a brand-new JSON plan that **preserves completed work**, addresses the blocker, and continues toward the original goal.
3. Newly generated `PlanStep`s replace the unfinished portion of the plan. Step numbers are renumbered so downstream logging stays monotonic.

This adaptive loop allows the workflow to recover from missing files, unexpected data, or tool failures without discarding successful work.

### 2.4 Final Answer Assembly

Once all steps (original or adapted) finish, `PlanExecutor._generate_final_answer_from_steps()`:

- Builds a structured summary of each step (status icon, goal, tool, observation preview).
- Concatenates accumulated observations for richer context.
- Uses `PromptRegistry.get('react_final_answer_from_steps', ...)` to synthesize the final user-facing response via the shared LLM instance.

---

## 3. Monitoring & Metadata

Back in `PlanExecuteTask.execute()` Phase 3:

1. **Statistics** – Computes totals (steps executed, success rate, attempts, unique tools).
2. **Metadata Assembly** – `_build_metadata_from_steps()` returns a JSON-serializable dict containing:
   - `planning`: the complete plan as executed, including any adapted steps.
   - `execution`: every `StepResult` (truncated observations) plus tool usage and attempt counts.
   - `summary`: success/failure counts, success rate %, total tool attempts.
   - `agent_type` and architectural descriptor for downstream analytics.
3. **Return Payload** – The final response tuple is `(final_output, metadata)` where `final_output` is the synthesized answer and `metadata` powers UI displays or audits.

Throughout all phases, the workflow produces verbose logs (section headers, key-value dumps, observations) to ease debugging and telemetry collection.

---

## 4. Failure Handling & Resilience

- **JSON parsing failures** in the planning phase surface immediately with raw LLM output logged, encouraging prompt fixes.
- **Tool failures** are captured inside `StepResult` (observation + error string). The plan is expected to include retry steps, or re-planning can inject them automatically.
- **Context loading issues** (`_load_notepad_and_variables`) are non-fatal; they degrade gracefully with warnings.
- **Plan adaptation errors** log the failure and re-raise, making it clear when the LLM did not return valid JSON.
- **Observation pruning** prevents extremely long contexts from derailing later prompts, yet keeps recent history available.

---

## 5. Component Reference

| Component | Location | Responsibility |
| --- | --- | --- |
| `PlanExecuteTask` | `backend/tasks/Plan_execute.py` | Entry point; orchestrates planning, execution, monitoring; builds metadata |
| `react_agent.execute_with_plan` | `backend/tasks/react/agent.py` | Prepares ReAct for guided execution, loads session context, delegates to plan executor |
| `PlanExecutor` | `backend/tasks/react/plan_executor.py` | Steps through the plan, executes tools, accumulates observations, finalizes answer |
| `PlanAdapter` | `backend/tasks/react/plan_adapter.py` | Detects when to re-plan and generates adapted plan steps |
| `ToolExecutor` | `backend/tasks/react/tool_executor.py` | Unified interface for invoking web search, RAG, python_coder, etc. |
| `ContextManager` | `backend/tasks/react/context_manager.py` | Builds context strings (tools, previous observations, variable metadata) |
| `PromptRegistry` | `backend/config/prompts/__init__.py` | Houses prompt templates for planning, final answer synthesis, etc. |

Use this document when onboarding, debugging Plan-and-Execute behavior, or extending the architecture (e.g., adding new tools, altering re-planning triggers, or modifying prompts). It reflects the state of the workflow as of Version 1.6.3.

