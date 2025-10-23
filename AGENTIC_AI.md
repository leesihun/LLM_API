# Agentic AI Architecture & Capabilities

This document explains how the project implements agentic behaviour on top of FastAPI and LangChain tooling. It covers the request flow, routing logic, individual agents, shared tools, persistence, configuration, and current limitations.

## System Overview
- **Backend stack**: FastAPI app (`backend/api/app.py`) exposes OpenAI-compatible chat endpoints. Requests authenticate through JWT (`backend/utils/auth.py`) and persist conversation context via the JSON-based `ConversationStore` (`backend/storage/conversation_store.py`).
- **Agent entry point**: The OpenAI `chat/completions` route (`backend/api/routes.py`) inspects the latest user utterance with `determine_task_type`. If the query suggests multi-step reasoning (keywords such as “search”, “analyze”, “document”), it forwards the call to the agent router; otherwise it uses the lightweight `ChatTask`.
- **Agent router**: `SmartAgentTask` (`backend/tasks/smart_agent_task.py`) chooses between the LangGraph plan-and-execute workflow and the ReAct loop. The choice can be forced via the `agent_type` request field or automatically inferred from linguistic heuristics (`then`, `after that`, `comprehensive`, etc.).

## Request Flow (Agentic Path)
1. **Authentication & sessioning** – JWT validation yields the `user_id`; `ConversationStore` creates or loads a `session_id` and attaches recent context when memory is enabled.
2. **Task detection** – `determine_task_type` classifies the message as `agentic` or `chat` using keyword-based heuristics.
3. **Agent selection** – `SmartAgentTask` returns `react` or `plan_execute`:
   - ReAct is favoured for sequential or exploratory questions.
   - Plan-and-execute is favoured for multi-tool, batch-style requests containing multiple `and`/`,` clauses.
4. **Agent execution** – The chosen agent orchestrates LLM reasoning, tool calls, and verification.
5. **Persistence** – Both user prompt and assistant reply are appended to the session transcript for future turns.

## ReAct Agent (Reasoning + Acting Loop)
File: `backend/core/react_agent.py`

- **Model**: Each agent instance uses `ChatOllama` configured via `backend/config/settings.py`. Async HTTPX clients provide timeouts and connection pooling.
- **Loop structure**:
  1. `_generate_thought` prompts the model to reflect on the next action, incorporating a formatted trace of prior steps.
  2. `_select_action` presents a menu of tool descriptions and expects the strict `Action: … / Action Input: …` response format.
  3. `_execute_action` delegates to concrete tool wrappers. Observations are appended to the step trace.
  4. The loop repeats until the model emits `finish`, an error occurs, or `max_iterations` (default 5) is reached. If it stops without an explicit answer, `_generate_final_answer` asks the LLM to summarise all observations.
- **Traceability**: `ReActStep` objects record thought/action/observation pairs; `get_trace` renders a human-readable audit trail for debugging.
- **Tool coverage**: ReAct can trigger `web_search`, `rag_retrieval`, `data_analysis`, `python_code`, `math_calc`, `wikipedia`, `weather`, and `sql_query`. Each branch reports failures back to the loop so the LLM can adjust strategy.

## Plan-and-Execute Agent (LangGraph Workflow)
File: `backend/core/agent_graph.py`

- **State contract**: `AgentState` (a `TypedDict`) tracks the message list, planning notes, tool outputs, the active node, verification flag, and iteration counters.
- **Graph nodes**:
  1. `planning_node` asks the LLM to produce a high-level plan (2–4 steps) for the latest user request.
  2. `tool_selection_node` inspects the plan and user prompt to choose which tool stages should run (`web_search`, `rag`, `data_analysis`, or fallback `chat`).
  3. `web_search_node`, `rag_retrieval_node`, `data_analysis_node` execute the selected tools and record textual summaries in state.
  4. `reasoning_node` combines conversation history and collected context into a response prompt for the LLM.
  5. `verification_node` performs an automated QA loop by asking the LLM to judge adequacy (`YES`/`NO`). Failures trigger another planning iteration until `max_iterations` is reached.
- **Control flow**: Implemented with `langgraph.StateGraph`. Edges provide a linear tool pipeline with a conditional edge from verification back to planning for iterative refinement.

## Shared Tooling & Capabilities

| Tool | Location | Purpose & Notes |
| --- | --- | --- |
| Web search | `backend/tools/web_search.py` | Attempts Tavily API first; if unavailable, falls back to the bundled `websearch_ts` scraper. Results are normalised into `SearchResult` objects and formatted text. |
| Retrieval-Augmented Generation | `backend/tools/rag_retriever.py` | Supports PDF, DOCX, TXT, and JSON ingestion. Uses `RecursiveCharacterTextSplitter` and `OllamaEmbeddings` to populate FAISS (default) or Chroma vector stores under `./data/vector_db`. Retrieval returns top-k chunks with metadata. |
| Data analysis | `backend/tools/data_analysis.py` | Parses natural-language commands to perform aggregate statistics (min/max/mean/sum/count) over numeric fields in uploaded JSON files (`settings.uploads_path`). |
| Python executor | `backend/tools/python_executor.py` | Executes sandboxed Python snippets with restricted built-ins, vetted imports, output capture, and optional POSIX signal timeouts. Useful for custom calculations and transformations. |
| Math calculator | `backend/tools/math_calculator.py` | Wraps SymPy for algebra, calculus, factoring, and expression manipulation, falling back to a basic evaluator when SymPy is absent. |
| Wikipedia | `backend/tools/wikipedia_tool.py` | Uses the public Wikipedia API to search and summarise articles, supporting configurable language and sentence count. |
| Weather | `backend/tools/weather_tool.py` | Calls Open-Meteo’s geocoding and forecast APIs to return current conditions plus a short forecast. |
| SQL query | `backend/tools/sql_query_tool.py` | Executes read-only SELECT statements against an auto-initialised SQLite database with sample `users` and `products` tables. Enforces forbidden keywords and row limits before formatting tabular output. |

Each tool exposes a coroutine-based API returning either structured objects (for internal chaining) or user-ready text. Formatting helpers standardise how observations feed back into the agents.

## Memory & Persistence
- **Conversation transcripts**: `ConversationStore` serialises sessions to JSON, rolling filenames that include `user_id`, timestamp, and `session_id`. Retrieval APIs can load, limit, or delete history.
- **Session routing**: The OpenAI endpoint creates sessions on demand and stores both user and assistant turns to maintain context for follow-up questions.
- **Document storage**: Vector indexes and uploads live under `./data`, as configured in `settings.py`. The RAG retriever lazily loads FAISS indexes from disk so previously ingested documents stay queryable.

## Configuration & Observability
- **Settings**: `backend/config/settings.py` centralises tuning parameters for Ollama, vector DBs, storage paths, and API keys. Defaults target local development, and helper functions can emit an `.env` template.
- **Logging**: `backend/api/app.py` initialises rotating file and console logging; each agent and tool logs at key decision points (plan generation, tool execution, verification results) to aid debugging.
- **Health checks**: On startup the API pings the configured Ollama host and lists available models, providing clear errors when LLM connectivity fails.

## Current Capabilities & Limitations
- Supports **automatic agent selection**, **multi-step planning**, **tool orchestration**, and **LLM-based verification** for complex tasks.
- Tools cover **web intelligence**, **document retrieval**, **data analytics**, **code execution**, **math reasoning**, **encyclopaedia lookups**, **weather**, and **SQL exploration**, enabling diverse workflows without manual wiring.
- Conversation memory and document indexes persist across sessions, allowing follow-up questions to reuse previous context.
- Limitations to note:
  - Task classification and agent routing rely on rule-based keyword heuristics; nuanced intent detection may misroute queries.
  - Verification currently accepts any response containing a leading “yes”, which may allow false positives.
  - Tool selection inside the LangGraph agent is keyword-driven and runs tools sequentially, even if only one is required.
  - Python execution timeouts fall back to no timeout on Windows because `signal.SIGALRM` is unavailable.
  - Several responses include emoji characters from tool formatters that may not render uniformly across clients.

## Potential Enhancements
1. Replace keyword heuristics with embedding-based or classifier-driven intent detection for task and agent selection.
2. Extend verification to cross-check facts (e.g., compare against tool outputs) before approving a final answer.
3. Add streaming responses and telemetry on tool latency to improve UX and monitoring.
4. Introduce parallel tool execution in the LangGraph workflow when multiple stages are selected.

This architecture balances flexibility (through ReAct) and structure (through LangGraph). By centralising tool wrappers and persistence, the system can grow to support additional agents or specialised workflows without reworking the surrounding infrastructure.
