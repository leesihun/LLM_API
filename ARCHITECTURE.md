# System Architecture

## Overview

The HE Team LLM Assistant uses a modern agentic AI architecture built with **LangGraph** for multi-step reasoning and tool orchestration.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (HTML/JS)                       │
│                      http://localhost:3000                       │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP/REST
                             │ OpenAI-compatible API
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend Server                        │
│                      http://localhost:8000                       │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Auth Router  │  │ OpenAI Router│  │ Files Router │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                  │                  │                  │
│         └──────────────────┼──────────────────┘                 │
│                            ▼                                     │
│              ┌─────────────────────────────┐                    │
│              │   Task Router/Dispatcher    │                    │
│              └─────────────┬───────────────┘                    │
│                            │                                     │
│         ┌──────────────────┼──────────────────┐                │
│         ▼                  ▼                  ▼                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Chat Task   │  │ Agentic Task │  │ Search Task  │         │
│  └──────────────┘  └──────┬───────┘  └──────────────┘         │
│                            │                                     │
│                            ▼                                     │
│              ┌─────────────────────────────┐                    │
│              │  LangGraph Agent Workflow   │                    │
│              └─────────────┬───────────────┘                    │
│                            │                                     │
└────────────────────────────┼─────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Ollama LLM   │   │ Tavily Search│   │ Vector DB    │
│ (Local)      │   │ API (Web)    │   │ (FAISS)      │
└──────────────┘   └──────────────┘   └──────────────┘
```

## LangGraph Agentic Workflow

### Detailed Flow

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ PLANNING NODE                                               │
│ • Analyze user query                                        │
│ • Determine information needs                               │
│ • Create step-by-step plan                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ TOOL SELECTION NODE                                         │
│ • Parse plan and query                                      │
│ • Identify required tools:                                  │
│   - Web Search (current info)                               │
│   - RAG (document retrieval)                                │
│   - Calculator (math)                                       │
│   - Chat (conversational)                                   │
└──────┬──────────────────┬─────────────────┬────────────────┘
       │                  │                 │
       ▼                  ▼                 ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ WEB SEARCH  │   │ RAG RETRIEV │   │ OTHER TOOLS │
│   NODE      │   │    NODE     │   │             │
└──────┬──────┘   └──────┬──────┘   └──────┬──────┘
       │                  │                 │
       └──────────────────┼─────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ REASONING NODE                                              │
│ • Combine retrieved information                             │
│ • Apply LLM reasoning                                       │
│ • Generate comprehensive response                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ VERIFICATION NODE                                           │
│ • Check response quality                                    │
│ • Validate answer completeness                              │
│ • Decide: output or refine?                                 │
└──────┬────────────────────────────────────┬─────────────────┘
       │                                    │
       │ ✓ Passes                          │ ✗ Needs work
       │                                    │ (iteration < max)
       ▼                                    │
   Final Output ◄──────────────────────────┘
                                        (loop back to planning)
```

## Component Details

### 1. API Layer (`backend/api/`)

**app.py** - FastAPI application
- CORS middleware
- Error handling
- Startup/shutdown events
- Health checks

**routes.py** - API endpoints
- `/api/auth/login` - Authentication
- `/v1/chat/completions` - Main chat endpoint
- `/v1/models` - List models
- `/api/files/upload` - Document upload

### 2. Core Logic (`backend/core/`)

**agent_graph.py** - LangGraph workflow
- State management
- Node definitions
- Edge routing
- Conditional logic

### 3. Task Handlers (`backend/tasks/`)

**chat_task.py** - Simple conversations
- Direct LLM calls
- Optional memory
- Fast responses

**agentic_task.py** - Complex workflows
- Multi-step reasoning
- Tool orchestration
- Verification loops

### 4. Tools (`backend/tools/`)

**web_search.py** - Internet search
- Primary: Tavily API
- Fallback: websearch_ts
- Result formatting

**rag_retriever.py** - Document Q&A
- Document loading (PDF, DOCX, etc.)
- Text chunking
- Vector embeddings
- Similarity search

### 5. Storage (`backend/storage/`)

**conversation_store.py** - Persistence
- JSON-based storage
- Session management
- Message history
- User isolation

### 6. Authentication (`backend/utils/`)

**auth.py** - Security
- JWT tokens
- Password hashing (bcrypt)
- User validation
- FastAPI dependencies

## Data Flow Examples

### Example 1: Simple Chat

```
User: "Hello, how are you?"
    │
    ▼
[Task Router] → Detects: Simple chat
    │
    ▼
[Chat Task]
    │
    ├─ Load conversation history (if session_id)
    ├─ Build context
    └─ Call Ollama LLM
    │
    ▼
Response: "Hello! I'm doing well..."
    │
    ▼
[Save to conversation store]
    │
    ▼
Return to user
```

### Example 2: Web Search

```
User: "What's the latest news about AI?"
    │
    ▼
[Task Router] → Detects: "latest" keyword → Agentic
    │
    ▼
[LangGraph Workflow]
    │
    ├─ [Planning] "Need current web information"
    ├─ [Tool Selection] → Web Search
    ├─ [Web Search Node]
    │   └─ Tavily API call → 5 results
    ├─ [Reasoning]
    │   └─ LLM synthesizes search results
    └─ [Verification] → Quality check → Pass
    │
    ▼
Response: "Based on recent sources..."
    │
    ▼
[Save to conversation store]
    │
    ▼
Return to user
```

### Example 3: Document RAG

```
User: "What does this PDF say about X?"
    │
    ├─ Upload PDF via /api/files/upload
    │   └─ [RAG Retriever] indexes document
    │
    ▼
User: "Summarize the key points"
    │
    ▼
[Task Router] → Detects: "document" keyword → Agentic
    │
    ▼
[LangGraph Workflow]
    │
    ├─ [Planning] "Need to retrieve from document"
    ├─ [Tool Selection] → RAG
    ├─ [RAG Node]
    │   ├─ Embed query
    │   ├─ Search vector DB
    │   └─ Return top 5 chunks
    ├─ [Reasoning]
    │   └─ LLM answers using context
    └─ [Verification] → Pass
    │
    ▼
Response: "According to the document..."
    │
    ▼
Return to user
```

## State Management

### Agent State Structure

```python
AgentState = {
    "messages": List[ChatMessage],      # Conversation
    "session_id": str,                  # Session ID
    "user_id": str,                     # User identifier
    "plan": str,                        # Execution plan
    "tools_used": List[str],            # Tools activated
    "search_results": str,              # Web search output
    "rag_context": str,                 # Document context
    "final_output": str,                # Generated response
    "verification_passed": bool,        # Quality check
    "iteration_count": int,             # Loop counter
    "max_iterations": int               # Loop limit
}
```

State flows through nodes and accumulates information at each step.

## Configuration System

### No Fallbacks Design

All settings **must** be explicitly configured in `.env`:

```python
# ✓ Correct
OLLAMA_HOST=http://localhost:11434

# ✗ Wrong (will raise error)
# OLLAMA_HOST=  (empty/missing)
```

This prevents silent failures and undefined behavior.

### Settings Validation

```python
from backend.config.settings import settings

# On import, validates all required variables
# Raises descriptive error if any missing
# Creates necessary directories
```

## Security Model

### Authentication Flow

```
1. User sends credentials → /api/auth/login
2. Server verifies against users.json
3. Generate JWT token (signed with SECRET_KEY)
4. Return token to client
5. Client includes token in Authorization header
6. Server validates token on each request
7. Token expires after JWT_EXPIRATION_HOURS
```

### Password Storage

- Passwords hashed with bcrypt
- Salt automatically generated
- Never stored in plaintext
- Hash verification on login

## Scalability Considerations

### Current Design
- Single-process server
- File-based storage
- Local vector DB
- Synchronous LLM calls

### Production Improvements
1. **Database:** PostgreSQL for conversations
2. **Async:** Fully async LLM calls
3. **Queue:** Celery for background tasks
4. **Cache:** Redis for session storage
5. **Vector DB:** Pinecone or Weaviate
6. **Horizontal scaling:** Multiple workers

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Framework | FastAPI | High-performance async API |
| Workflow | LangGraph | Agent orchestration |
| LLM | Ollama | Local model inference |
| Search | Tavily | Web search API |
| Vector DB | FAISS | Document embeddings |
| Embeddings | Sentence Transformers | Text vectorization |
| Auth | JWT + bcrypt | Secure authentication |
| Docs | PyPDF, python-docx | Document loading |

## Extension Points

### Adding New Tools

1. **Create tool** in `backend/tools/new_tool.py`
2. **Add node** in `agent_graph.py`
3. **Update selection** logic in `tool_selection_node`
4. **Connect edges** in workflow

### Adding New Tasks

1. **Create handler** in `backend/tasks/new_task.py`
2. **Update router** in `routes.py`
3. **Add detection** logic in `determine_task_type`

### Customizing LLM Behavior

Edit prompts in:
- `planning_node` - Query analysis
- `reasoning_node` - Response generation
- `verification_node` - Quality check

---

**This architecture ensures:**
- ✅ Modularity (easy to extend)
- ✅ Maintainability (clear structure)
- ✅ Testability (isolated components)
- ✅ Scalability (can be distributed)
- ✅ Reliability (verification loops)
