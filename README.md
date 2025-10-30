# LLM API – Sample Client and Usage

Version: 0.2.1

This document shows a minimal Python client and examples for common tasks.

## Quickstart

Prereqs: Python 3.10+.

```bash
pip install httpx
```

Set base URL (default `http://127.0.0.1:8000`).

## Minimal Python Client

```python
import httpx

class LLMApiClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")
        self.token = None

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    # 1) Create account
    def signup(self, username: str, password: str, role: str = "guest"):
        r = httpx.post(f"{self.base_url}/api/auth/signup", json={
            "username": username, "password": password, "role": role
        })
        r.raise_for_status()
        return r.json()

    # 2) Login
    def login(self, username: str, password: str):
        r = httpx.post(f"{self.base_url}/api/auth/login", json={
            "username": username, "password": password
        })
        r.raise_for_status()
        data = r.json()
        self.token = data["access_token"]
        return data

    # 3) Change model (admin)
    def change_model(self, model: str):
        r = httpx.post(f"{self.base_url}/api/admin/model", json={"model": model}, headers=self._headers())
        r.raise_for_status()
        return r.json()

    # List models (OpenAI-compatible)
    def list_models(self):
        r = httpx.get(f"{self.base_url}/v1/models", headers=self._headers())
        r.raise_for_status()
        return r.json()

    # 4) Start new chat
    def chat_new(self, model: str, user_message: str, agent_type: str = "auto"):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": user_message}],
            "agent_type": agent_type
        }
        r = httpx.post(f"{self.base_url}/v1/chat/completions", json=payload, headers=self._headers())
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"], data["x_session_id"]

    # 5) Continue a chat
    def chat_continue(self, model: str, session_id: str, user_message: str, agent_type: str = "auto"):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": user_message}],
            "session_id": session_id,
            "agent_type": agent_type
        }
        r = httpx.post(f"{self.base_url}/v1/chat/completions", json=payload, headers=self._headers())
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"], data["x_session_id"]

    # 6) See chat history
    def chat_sessions(self):
        r = httpx.get(f"{self.base_url}/api/chat/sessions", headers=self._headers())
        r.raise_for_status()
        return r.json()["sessions"]

    def chat_history(self, session_id: str):
        r = httpx.get(f"{self.base_url}/api/chat/history/{session_id}", headers=self._headers())
        r.raise_for_status()
        return r.json()["messages"]

    # 7) Web search
    def websearch(self, query: str, max_results: int = 5):
        r = httpx.post(f"{self.base_url}/api/tools/websearch", json={
            "query": query, "max_results": max_results
        }, headers=self._headers())
        r.raise_for_status()
        return r.json()["results"]

    # 8) Tools list and math
    def tools(self):
        r = httpx.get(f"{self.base_url}/api/tools/list", headers=self._headers())
        r.raise_for_status()
        return r.json()["tools"]

    def math(self, expression: str):
        r = httpx.post(f"{self.base_url}/api/tools/math", json={"expression": expression}, headers=self._headers())
        r.raise_for_status()
        return r.json()["result"]

    # 9) Answer based on JSON (use chat with instruction)
    def answer_from_json(self, model: str, json_blob: dict, question: str):
        prompt = f"Given this JSON: {json_blob}\nAnswer: {question}"
        return self.chat_new(model, prompt)[0]
```

## Examples

Replace variables as needed.

```python
client = LLMApiClient()

# 1) Create a new account
client.signup("alice", "alice_pw")

# 2) Login
login = client.login("alice", "alice_pw")
print("Logged in as:", login["user"])  # {'username': 'alice', 'role': 'guest'}

# 3) Change models (admin only)
# client.login("admin", "administrator")
# print(client.change_model("llama3:8b"))

# List models (OpenAI-compatible)
print(client.list_models())

MODEL = client.list_models()["data"][0]["id"]

# 4) Start a new chat, with response
reply, session_id = client.chat_new(MODEL, "Hello! Give me a short haiku about autumn.")
print("Assistant:", reply)

# 5) Continue a chat
reply2, _ = client.chat_continue(MODEL, session_id, "Now do one about winter.")
print("Assistant:", reply2)

# 6) See chat history
print("Sessions:", client.chat_sessions())
print("History:", client.chat_history(session_id))

# 7) Perform a websearch
results = client.websearch("Who is the president of S. Korea as of October 2025?", max_results=3)
for r in results:
    print(r["title"], r["url"])  # cite in UI as needed

# 8) All tool calls
print("Tools:", client.tools())
print("11.951/3.751 =", client.math("11.951/3.751"))
print("Which is bigger?", client.math("max(1.951, 19.51)"))
ko_reasoning = (
    "나는 시장에 가서 사과 10개를 샀어. 사과 2개를 이웃에게 주고, 2개를 수리공에게 주었어. "
    "그리고 사과 5개를 더 사서 1개는 내가 먹었어. 나는 몇 개의 사과를 가지고 있었니?"
)
print("Korean apples:", client.math("10 - 2 - 2 + 5 - 1"))

# 9) Answer based on JSON
sample = {"users": [{"name": "Tom", "age": 30}, {"name": "Jane", "age": 28}]}
print(client.answer_from_json(MODEL, sample, "Who is older?"))
```

## Changelog

- 0.2.1: Fix import path case for Plan-and-Execute module in `backend/api/routes.py`.
- 0.2.0: Added single notebook `API_examples.ipynb` covering tasks 1–9.
- 0.1.0: Added signup, admin model change, chat sessions/history, tools list, math, web search, and sample client with examples.
# Agentic AI Backend with FastAPI

**OpenAI-Compatible API with Multi-Agent Reasoning**

A production-ready AI backend featuring intelligent agent routing, RAG (Retrieval-Augmented Generation), web search, and 8+ integrated tools powered by local LLM inference via Ollama.

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Agent System](#agent-system)
- [Tools](#tools)
- [Configuration](#configuration)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Features

### Core Capabilities

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI endpoints (`/v1/chat/completions`, `/v1/models`)
- **Dual-Agent System**: Intelligent routing between ReAct and Plan-and-Execute agents
- **8+ Integrated Tools**: Web search, RAG, math, Python execution, Wikipedia, weather, SQL, data analysis
- **Local LLM**: Complete data privacy using Ollama (no external API calls)
- **RAG with Vector DB**: Document Q&A with FAISS/Chroma
- **JWT Authentication**: Secure user authentication with bcrypt
- **Multi-User Isolation**: User-scoped file storage and conversation history
- **Conversation Memory**: Persistent chat history per session
- **Production-Ready**: Comprehensive logging, error handling, and configuration management

### Technical Highlights

- **FastAPI**: High-performance async web framework
- **LangGraph**: Workflow orchestration for Plan-and-Execute agent
- **Ollama**: Local LLM inference (default model: `gpt-oss:20b`)
- **FAISS/Chroma**: Vector similarity search for RAG
- **Tavily API**: Web search integration
- **Pydantic**: Type-safe request/response validation

---

## Quick Start

### Prerequisites

- Python 3.9+
- Ollama installed and running
- 16GB+ RAM recommended (for 20B model)

### 5-Minute Setup

```bash
# 1. Clone repository
git clone <repository-url>
cd LLM_API

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Ollama
ollama serve

# 4. Pull the model
ollama pull gpt-oss:20b

# 5. Start backend
python run_backend.py

# 6. Start frontend (optional, in new terminal)
python run_frontend.py
```

**Access the API**:
- Backend: http://localhost:8000
- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs

**Default Login Credentials**:
- **Admin**: `admin` / `administrator`
- **Guest**: `guest` / `guest_test1`

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────┐
│              User Interface                     │
│      (Web Frontend / API Clients)               │
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│         FastAPI Backend (Port 8000)             │
│  ┌──────────────────────────────────────────┐  │
│  │  /api/auth  │  /v1  │  /api/files       │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│            Smart Agent Router                   │
│  ┌──────────────────────────────────────────┐  │
│  │  Auto-Select Agent Based on Query        │  │
│  │  ├── ReAct (Iterative Reasoning)         │  │
│  │  └── Plan-and-Execute (LangGraph)        │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│              Tool Ecosystem                     │
│  Web Search │ RAG │ Math │ Python │ Wikipedia  │
│  Weather │ SQL │ Data Analysis                 │
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│         Infrastructure Layer                    │
│  Ollama LLM │ FAISS/Chroma │ File Storage      │
└─────────────────────────────────────────────────┘
```

### Agent Comparison

| Feature | ReAct Agent | Plan-and-Execute Agent |
|---------|-------------|------------------------|
| **Pattern** | Iterative Thought-Action-Observation | LangGraph workflow with 5 stages |
| **Best For** | Exploratory, sequential reasoning | Complex batch queries, parallel tools |
| **Max Iterations** | 5 | 3 with verification |
| **Transparency** | Full trace via `get_trace()` | State tracking |
| **Example** | "Find capital, then its population" | "Search weather AND analyze data AND retrieve docs" |

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed technical documentation.

---

## Installation

### System Requirements

- **OS**: Windows, macOS, Linux
- **Python**: 3.9 or higher
- **RAM**: 16GB+ (for 20B model), 8GB+ (for 7B model)
- **Storage**: 10GB+ free space

### Step-by-Step Installation

#### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies**:
```
fastapi==0.115.5
uvicorn==0.32.1
langchain==0.3.7
langgraph==0.2.45
langchain-ollama==0.2.0
faiss-cpu==1.9.0
PyJWT==2.10.1
bcrypt==4.2.1
pydantic==2.10.3
httpx==0.28.1
```

#### 2. Install Ollama

**macOS/Linux**:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows**:
Download installer from https://ollama.com/download

#### 3. Pull LLM Model

```bash
# Default model (20B parameters)
ollama pull gpt-oss:20b

# Alternative models
ollama pull llama3.2:latest    # 8B parameters (faster)
ollama pull mistral:latest     # 7B parameters (efficient)
```

#### 4. Configure Environment (Optional)

The system works with default settings, but you can customize:

```bash
# Copy example (optional)
cp .env.example .env

# Edit .env with your preferences
nano .env
```

**Important Settings**:
```bash
# Generate secure secret key
SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")

# API keys
TAVILY_API_KEY=your-tavily-key-here

# Ollama configuration
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=gpt-oss:20b

# Server
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
```

---

## Usage

### Starting the Backend

#### Option 1: Cross-Platform Launcher (Recommended)

```bash
python run_backend.py
```

This script:
- Validates Ollama connection
- Checks model availability
- Starts FastAPI server on port 8000

#### Option 2: Direct Execution

```bash
python -m backend.api.app
```

#### Option 3: Uvicorn (Development)

```bash
uvicorn backend.api.app:app --reload --host 0.0.0.0 --port 8000
```

### Starting the Frontend

```bash
python run_frontend.py              # Opens browser automatically
python run_frontend.py --no-browser # Without browser
```

Frontend serves on port 3000.

### Verifying Installation

```bash
# Check health
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "ollama_host": "http://127.0.0.1:11434",
  "model": "gpt-oss:20b"
}
```

---

## API Documentation

### Authentication

#### Login

```bash
POST /api/auth/login
```

**Request**:
```json
{
  "username": "admin",
  "password": "administrator"
}
```

**Response**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "user": {
    "username": "admin",
    "role": "admin"
  }
}
```

#### Get Current User

```bash
GET /api/auth/me
Authorization: Bearer <token>
```

**Response**:
```json
{
  "username": "admin",
  "role": "admin"
}
```

### Chat Completions (OpenAI-Compatible)

#### Basic Chat

```bash
POST /v1/chat/completions
Authorization: Bearer <token>
Content-Type: application/json
```

**Request**:
```json
{
  "model": "gpt-oss:20b",
  "messages": [
    {
      "role": "user",
      "content": "What is the capital of France?"
    }
  ],
  "agent_type": "auto"
}
```

**Parameters**:
- `model`: Model name (from `/v1/models`)
- `messages`: Array of message objects with `role` and `content`
- `session_id`: Optional, for conversation continuity
- `agent_type`: `"auto"` (default), `"react"`, or `"plan_execute"`
- `temperature`: Optional, 0.0-1.0 (default: 0.7)
- `max_tokens`: Optional

**Response**:
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1730000000,
  "model": "gpt-oss:20b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  },
  "x_session_id": "session-abc123"
}
```

#### Agentic Query with Tools

```bash
POST /v1/chat/completions
```

**Request**:
```json
{
  "model": "gpt-oss:20b",
  "messages": [
    {
      "role": "user",
      "content": "Search for the current weather in Seoul and analyze the temperature data"
    }
  ],
  "agent_type": "auto"
}
```

This automatically:
1. Detects need for web search (weather)
2. Detects need for data analysis
3. Routes to appropriate agent (likely Plan-and-Execute)
4. Executes tools in parallel
5. Generates response with context

#### Conversation with Memory

```bash
# First message
POST /v1/chat/completions
{
  "model": "gpt-oss:20b",
  "messages": [{"role": "user", "content": "My name is Alice"}]
}

# Response includes x_session_id: "session-xyz"

# Follow-up message (remembers context)
POST /v1/chat/completions
{
  "model": "gpt-oss:20b",
  "messages": [{"role": "user", "content": "What's my name?"}],
  "session_id": "session-xyz"
}
```

### File Management

#### Upload Document (RAG)

```bash
POST /api/files/upload
Authorization: Bearer <token>
Content-Type: multipart/form-data
```

**Request**:
```bash
curl -X POST http://localhost:8000/api/files/upload \
  -H "Authorization: Bearer <token>" \
  -F "file=@document.pdf"
```

**Response**:
```json
{
  "success": true,
  "file_id": "a1b2c3d4",
  "doc_id": "d8a4c890f6ee07d71386dbe1934de12e",
  "filename": "document.pdf",
  "size": 102400,
  "message": "File uploaded and indexed successfully"
}
```

**Supported Formats**: PDF, DOCX, TXT, JSON

#### List Documents

```bash
GET /api/files/documents?page=1&page_size=20
Authorization: Bearer <token>
```

**Response**:
```json
{
  "documents": [
    {
      "file_id": "a1b2c3d4",
      "filename": "document.pdf",
      "full_path": "a1b2c3d4_document.pdf",
      "size": 102400,
      "created": 1730000000
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 20,
  "total_pages": 1
}
```

#### Delete Document

```bash
DELETE /api/files/documents/{file_id}
Authorization: Bearer <token>
```

**Response**:
```json
{
  "success": true,
  "message": "File deleted successfully"
}
```

### Models

#### List Available Models

```bash
GET /v1/models
Authorization: Bearer <token>
```

**Response**:
```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt-oss:20b",
      "object": "model",
      "created": 1730000000,
      "owned_by": "ollama"
    }
  ]
}
```

---

## Agent System

### Smart Agent Router

The system automatically selects the best agent based on query characteristics:

#### ReAct Agent (Reasoning + Acting)

**Best for**:
- Exploratory queries
- Sequential reasoning ("first X, then Y")
- Dynamic tool selection
- Iterative refinement

**Example Queries**:
```
"Find the capital of France, then search for its population"
"If the weather is sunny, recommend outdoor activities"
"Step by step, analyze this data"
```

**Execution Pattern**:
```
Iteration 1:
  Thought: I need to find the capital of France
  Action: wikipedia
  Action Input: capital of France
  Observation: Paris is the capital of France

Iteration 2:
  Thought: Now I need to find its population
  Action: web_search
  Action Input: Paris population 2025
  Observation: Paris has approximately 2.2 million people

Iteration 3:
  Thought: I have all the information needed
  Action: finish
  Action Input: The capital of France is Paris, with...
```

#### Plan-and-Execute Agent (LangGraph)

**Best for**:
- Complex batch queries
- Parallel tool usage
- Comprehensive tasks with multiple requirements
- Well-defined workflows

**Example Queries**:
```
"Search weather in Seoul AND analyze uploaded data AND retrieve climate docs"
"Give me a complete report on market trends and competitor analysis"
"Summarize everything about AI and provide statistics"
```

**Execution Pattern**:
```
Planning → Tool Selection → Execute All Tools → Reasoning → Verification
   ↓            ↓                    ↓               ↓           ↓
Create plan   [web_search,      Run parallel    Generate    Quality
              rag, analysis]     execution      response     check
```

### Manual Agent Selection

Override automatic selection:

```bash
POST /v1/chat/completions
{
  "model": "gpt-oss:20b",
  "messages": [...],
  "agent_type": "react"  # or "plan_execute" or "auto"
}
```

---

## Tools

### 1. Web Search

**Purpose**: Search the internet for current information

**Implementation**: Uses Tavily API (primary) with websearch_ts fallback

**Example Queries**:
- "What's the latest news about AI?"
- "Current weather in Tokyo"
- "Recent developments in quantum computing"

**Configuration**:
```bash
TAVILY_API_KEY=your-key-here
```

### 2. RAG Retriever (Document Q&A)

**Purpose**: Retrieve and answer questions from uploaded documents

**Supported Formats**: PDF, DOCX, TXT, JSON

**Vector DB**: FAISS (default) or Chroma

**Example Queries**:
- "What does my contract say about termination?"
- "Summarize the uploaded research paper"
- "Find information about pricing in the document"

**Configuration**:
```bash
VECTOR_DB_TYPE=faiss          # or "chroma"
EMBEDDING_MODEL=bge-m3:latest
```

### 3. Data Analysis

**Purpose**: Analyze JSON data with statistics

**Capabilities**: min, max, mean, count, sum, nested structures

**Example Queries**:
- "Find the maximum value in this dataset"
- "Calculate average sales from uploaded JSON"
- "Count occurrences in data"

### 4. Python Executor

**Purpose**: Execute Python code safely

**Features**: Stdout/stderr capture, timeout protection

**Example Queries**:
- "Write Python code to calculate factorial of 10"
- "Generate Fibonacci sequence using Python"
- "Parse this JSON using Python"

### 5. Math Calculator

**Purpose**: Advanced mathematical calculations

**Engine**: SymPy (symbolic mathematics)

**Capabilities**: Algebra, calculus, derivatives, integrals, limits, equations

**Example Queries**:
- "Calculate the derivative of x^2 + 3x"
- "Solve equation: 2x + 5 = 15"
- "Integrate sin(x) from 0 to pi"

### 6. Wikipedia Tool

**Purpose**: Search Wikipedia for factual information

**Features**: Summarization, disambiguation handling

**Example Queries**:
- "What is quantum entanglement?"
- "History of the Roman Empire"
- "Biography of Albert Einstein"

### 7. Weather Tool

**Purpose**: Get current weather information

**Example Queries**:
- "What's the weather in London?"
- "Current temperature in New York"
- "Weather forecast for Paris"

### 8. SQL Query Tool

**Purpose**: Execute SQL queries on databases

**Features**: Structured results, safe execution

**Example Queries**:
- "SELECT * FROM users WHERE age > 25"
- "Query database for sales data"

---

## Configuration

### Environment Variables

All settings have sensible defaults in [backend/config/settings.py](backend/config/settings.py).

Create `.env` file to override:

```bash
# ============================================================================
# Server Configuration
# ============================================================================
SERVER_HOST=0.0.0.0              # Use 'localhost' in production
SERVER_PORT=8000
SECRET_KEY=your-secret-key-here  # Generate: python -c "import secrets; print(secrets.token_urlsafe(32))"

# ============================================================================
# Ollama Configuration
# ============================================================================
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=gpt-oss:20b
OLLAMA_TIMEOUT=3000000           # 50 minutes
OLLAMA_NUM_CTX=4096              # Context window
OLLAMA_TEMPERATURE=0.7           # 0.0=conservative, 1.0=creative
OLLAMA_TOP_P=0.9
OLLAMA_TOP_K=40

# ============================================================================
# API Keys
# ============================================================================
TAVILY_API_KEY=your-tavily-key   # Get from https://tavily.com/

# ============================================================================
# Vector Database
# ============================================================================
VECTOR_DB_TYPE=faiss             # or "chroma"
VECTOR_DB_PATH=./data/vector_db
EMBEDDING_MODEL=bge-m3:latest

# ============================================================================
# Storage Paths
# ============================================================================
USERS_PATH=./data/users/users.json
SESSIONS_PATH=./data/sessions/sessions.json
CONVERSATIONS_PATH=./data/conversations
UPLOADS_PATH=./data/uploads

# ============================================================================
# Authentication
# ============================================================================
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# ============================================================================
# Logging
# ============================================================================
LOG_LEVEL=INFO                   # DEBUG, INFO, WARNING, ERROR
LOG_FILE=./data/logs/app.log
```

### Model Selection

Choose based on hardware:

| Model | Parameters | RAM Required | Speed | Quality |
|-------|-----------|--------------|-------|---------|
| `gpt-oss:20b` | 20B | 16GB+ | Moderate | High |
| `llama3.2:latest` | 8B | 8GB+ | Fast | Good |
| `mistral:latest` | 7B | 6GB+ | Very Fast | Good |

Change model:
```bash
OLLAMA_MODEL=llama3.2:latest
```

---

## Development

### Project Structure

```
LLM_API/
├── backend/
│   ├── api/
│   │   ├── app.py              # FastAPI application
│   │   └── routes.py           # API endpoints
│   ├── core/
│   │   └── agent_graph.py      # LangGraph workflow (deprecated)
│   ├── tasks/
│   │   ├── chat_task.py        # Simple chat task
│   │   ├── React.py            # ReAct agent (Reasoning + Acting)
│   │   ├── Plan_execute.py    # Plan-and-Execute agent (hybrid)
│   │   └── smart_agent_task.py # Smart agent router
│   ├── tools/
│   │   ├── web_search.py       # Web search tool
│   │   ├── rag_retriever.py    # RAG tool
│   │   ├── math_calculator.py  # Math tool
│   │   ├── python_executor.py  # Python execution
│   │   ├── wikipedia_tool.py   # Wikipedia tool
│   │   ├── weather_tool.py     # Weather tool
│   │   ├── sql_query_tool.py   # SQL tool
│   │   └── data_analysis.py    # Data analysis tool
│   ├── models/
│   │   └── schemas.py          # Pydantic models
│   ├── storage/
│   │   └── conversation_store.py # Conversation management
│   ├── utils/
│   │   └── auth.py             # Authentication utilities
│   └── config/
│       └── settings.py         # Configuration management
├── frontend/                   # Static frontend files
├── data/                       # Data storage
│   ├── conversations/          # Chat history
│   ├── uploads/               # User documents
│   ├── vector_db/             # Vector embeddings
│   ├── users/                 # User database
│   ├── sessions/              # Session management
│   └── logs/                  # Application logs
├── run_backend.py             # Backend launcher
├── run_frontend.py            # Frontend server
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── ARCHITECTURE.md            # Technical documentation
├── API_DOCUMENTATION.md       # API reference
└── CLAUDE.md                  # Development guide
```

### Adding New Tools

1. **Create tool file** in `backend/tools/`:

```python
# backend/tools/my_tool.py
import logging

logger = logging.getLogger(__name__)

class MyTool:
    async def execute(self, input_data: str) -> str:
        """Execute tool operation"""
        try:
            # Your logic here
            result = process(input_data)
            return self.format_result(result)
        except Exception as e:
            logger.error(f"Error: {e}")
            return f"Error: {str(e)}"

    def format_result(self, result) -> str:
        """Format for LLM consumption"""
        return str(result)

# Global instance
my_tool = MyTool()
```

2. **Register in ReAct Agent** ([backend/tasks/React.py](backend/tasks/React.py)):

```python
# Add to ToolName enum
class ToolName(str, Enum):
    MY_TOOL = "my_tool"
    # ... existing tools

# Add to _execute_action() method
elif action == ToolName.MY_TOOL:
    result = await my_tool.execute(action_input)
    return result
```

3. **Register in Plan-and-Execute Agent** ([backend/core/agent_graph.py](backend/core/agent_graph.py)):

```python
# Add node function
async def my_tool_node(state: AgentState) -> Dict[str, Any]:
    if "my_tool" not in state.get("tools_used", []):
        return {"my_tool_results": "", "current_agent": "my_tool"}

    result = await my_tool.execute(state["messages"][-1].content)
    return {"my_tool_results": result, "current_agent": "my_tool"}

# Add to workflow
workflow.add_node("my_tool", my_tool_node)
workflow.add_edge("previous_node", "my_tool")
```

4. **Update tool selection logic** in both agents.

### Code Style

- **Type Hints**: Use Pydantic models for validation
- **Async/Await**: All I/O operations are async
- **Logging**: Use module-level loggers with structured messages
- **Error Handling**: Try/except with detailed logging
- **Documentation**: Docstrings for all public methods

---

## Testing

### Comprehensive API Testing

```bash
python test_all_apis.py
```

Tests all endpoints with health checks:
- Authentication (login, user info)
- Chat completions (simple, agentic, memory)
- File upload/list/delete
- Models endpoint

### Feature-Specific Tests

```bash
# ReAct agent
python test_react_agent.py

# New features
python test_new_features.py

# Tool integration
python test_new_tools.py

# Math calculator
python test_difficult_math.py
```

### Manual Testing with cURL

```bash
# Login
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"administrator"}'

# Chat
TOKEN="your-jwt-token"
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss:20b",
    "messages": [{"role":"user","content":"Hello!"}]
  }'

# Upload file
curl -X POST http://localhost:8000/api/files/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@document.pdf"
```

### Unit Testing

Create tests in `tests/` directory:

```python
import pytest
from backend.tools.math_calculator import math_calculator

@pytest.mark.asyncio
async def test_math_calculator():
    result = await math_calculator.calculate("2 + 2")
    assert "4" in result

@pytest.mark.asyncio
async def test_react_agent():
    from backend.tasks.React import react_agent
    messages = [{"role": "user", "content": "What is 5 + 3?"}]
    result = await react_agent.execute(messages, None, "test_user")
    assert "8" in result
```

Run tests:
```bash
pytest tests/
```

---

## Deployment

### Production Checklist

Before deploying to production:

- [ ] **Security**
  - [ ] Change `SECRET_KEY` (generate with: `python -c "import secrets; print(secrets.token_urlsafe(32))"`)
  - [ ] Set `SERVER_HOST=localhost` (not `0.0.0.0`)
  - [ ] Use HTTPS with SSL certificates
  - [ ] Configure CORS (`allow_origins` in [app.py](backend/api/app.py))
  - [ ] Secure Tavily API key

- [ ] **Performance**
  - [ ] Set `LOG_LEVEL=WARNING` (reduce log verbosity)
  - [ ] Adjust `OLLAMA_TIMEOUT` based on model
  - [ ] Configure `OLLAMA_NUM_CTX` for context window
  - [ ] Set up connection pooling

- [ ] **Monitoring**
  - [ ] Set up log aggregation (ELK, Datadog, etc.)
  - [ ] Configure health check monitoring
  - [ ] Set up alerts for errors

- [ ] **Backups**
  - [ ] Backup `data/users/users.json`
  - [ ] Backup `data/conversations/`
  - [ ] Backup `data/uploads/`
  - [ ] Backup `data/vector_db/`

### Docker Deployment

**Dockerfile**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY backend/ backend/
COPY frontend/ frontend/
COPY run_backend.py .
COPY run_frontend.py .

# Create data directories
RUN mkdir -p data/conversations data/uploads data/vector_db data/logs

EXPOSE 8000

CMD ["python", "run_backend.py"]
```

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - SERVER_HOST=0.0.0.0
      - SECRET_KEY=${SECRET_KEY}
      - TAVILY_API_KEY=${TAVILY_API_KEY}
    depends_on:
      - ollama
    volumes:
      - ./data:/app/data

  frontend:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./frontend:/usr/share/nginx/html
      - ./nginx.conf:/etc/nginx/nginx.conf

volumes:
  ollama_data:
```

**Run**:
```bash
docker-compose up -d
```

### Nginx Reverse Proxy

**nginx.conf**:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Frontend
    location / {
        root /var/www/frontend;
        try_files $uri $uri/ /index.html;
    }

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /v1/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Systemd Service (Linux)

**llm-api.service**:
```ini
[Unit]
Description=LLM API Backend
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/llm-api
Environment="PATH=/opt/llm-api/venv/bin"
ExecStart=/opt/llm-api/venv/bin/python run_backend.py
Restart=always

[Install]
WantedBy=multi-user.target
```

**Enable**:
```bash
sudo systemctl enable llm-api
sudo systemctl start llm-api
sudo systemctl status llm-api
```

---

## Troubleshooting

### Common Issues

#### 1. "Connection refused" to Ollama

**Symptoms**:
```
Error connecting to Ollama at http://127.0.0.1:11434
```

**Solution**:
```bash
# Check if Ollama is running
ollama list

# Start Ollama service
ollama serve

# Verify connectivity
curl http://127.0.0.1:11434/api/tags
```

#### 2. "Model not found"

**Symptoms**:
```
Model 'gpt-oss:20b' not found
```

**Solution**:
```bash
# List installed models
ollama list

# Pull required model
ollama pull gpt-oss:20b

# Verify
ollama list
```

#### 3. Port 8000 already in use

**Symptoms**:
```
Address already in use: 0.0.0.0:8000
```

**Solution**:
```bash
# Find process using port 8000
# Windows
netstat -ano | findstr :8000
taskkill /PID <pid> /F

# Linux/macOS
lsof -i :8000
kill -9 <pid>

# Or change port
SERVER_PORT=8001 python run_backend.py
```

#### 4. "Configuration error"

**Symptoms**:
```
Configuration error: Missing required settings
```

**Solution**:
All settings have defaults in `backend/config/settings.py`. You can either:

**Option 1: Edit settings.py directly**
```bash
# Edit backend/config/settings.py
nano backend/config/settings.py

# Update required values like:
# secret_key = 'your-secure-key'
# tavily_api_key = 'your-tavily-key'
```

**Option 2: Create .env file to override defaults**
```bash
# Create .env file
nano .env

# Add your custom values:
SECRET_KEY=your-secure-key
TAVILY_API_KEY=your-tavily-key
```

#### 5. "ModuleNotFoundError"

**Symptoms**:
```
ModuleNotFoundError: No module named 'fastapi'
```

**Solution**:
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Verify installation
pip list
```

#### 6. Agentic queries timeout

**Symptoms**:
```
Request timeout after 2 minutes
```

**Solution**:
```bash
# Increase timeout in .env
OLLAMA_TIMEOUT=3000000  # 50 minutes

# Or reduce max iterations in agents
# backend/tasks/React.py: max_iterations=3
# backend/core/agent_graph.py: max_iterations=2
```

#### 7. JSON file upload fails in RAG

**Known Issue**: Custom JSON loader treats JSON as text, may not parse structured data correctly.

**Workaround**: Convert JSON to TXT before upload, or process as string data.

### Debug Mode

Enable detailed logging:

```bash
LOG_LEVEL=DEBUG python run_backend.py
```

View logs:
```bash
tail -f data/logs/app.log
```

---

## Contributing

Contributions are welcome! Please follow these guidelines:

### Development Workflow

1. **Fork** the repository
2. **Create branch**: `git checkout -b feature/my-feature`
3. **Make changes** with tests
4. **Run tests**: `python test_all_apis.py`
5. **Commit**: `git commit -m "Add my feature"`
6. **Push**: `git push origin feature/my-feature`
7. **Submit Pull Request**

### Code Guidelines

- Follow PEP 8 style guide
- Add type hints to all functions
- Include docstrings for public methods
- Write unit tests for new features
- Update documentation

### Testing Requirements

- All tests must pass
- New features require tests
- Maintain >80% code coverage

---

## License

MIT License

Copyright (c) 2025 HE Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Acknowledgments

- **Ollama** - Local LLM inference
- **LangChain** - LLM framework
- **LangGraph** - Workflow orchestration
- **FastAPI** - Web framework
- **Tavily** - Web search API
- **FAISS** - Vector similarity search

---

## Support

- **Documentation**: [ARCHITECTURE.md](ARCHITECTURE.md), [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
- **Issues**: GitHub Issues
- **API Docs**: http://localhost:8000/docs

---

**Built with AI by HE Team**
