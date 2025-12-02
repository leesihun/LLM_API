# LLM API - AI-Powered Agentic Workflow System

> **Production-ready FastAPI server with advanced multi-agent reasoning, dual LLM backend support, and autonomous code generation capabilities**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 🚀 Overview

An enterprise-grade LLM API server that combines the power of Large Language Models with sophisticated agentic workflows. The system features **dual backend support** (Ollama + llama.cpp) and implements cutting-edge reasoning patterns (ReAct, Plan-Execute) to handle complex, multi-step tasks autonomously.

### Key Capabilities

- **🤖 Dual LLM Backend**: Switch between Ollama (easy setup) and llama.cpp (production-grade) without code changes
- **🧠 Multi-Agent Architecture**: ReAct and Plan-Execute patterns for complex reasoning and task decomposition
- **💻 Autonomous Code Generation**: AI-driven Python code generation with iterative verification and sandboxed execution
- **🔍 Web Intelligence**: Real-time information retrieval via Tavily API integration
- **📚 Document Understanding**: RAG (Retrieval Augmented Generation) with FAISS for context-aware responses
- **📁 Multi-Format File Processing**: CSV, Excel, JSON, PDF, images, and more
- **🔒 Enterprise Security**: Sandboxed execution, JWT authentication, import restrictions, timeout controls
- **💾 Session Continuity**: Variable persistence, conversation history, and task memory across executions

---

## 🎯 When to Use This

| Use Case | Example |
|----------|---------|
| **Complex Data Analysis** | "Analyze these 3 CSV files, calculate correlations, create visualizations, and generate a PowerPoint report" |
| **Real-time Research** | "Search for the latest AI research papers from 2025 and summarize key findings" |
| **Automated Reporting** | "Load sales data, calculate KPIs, create charts, and export to Excel with formatting" |
| **Document Intelligence** | "Analyze these PDFs, extract key metrics, and compare trends across documents" |
| **Iterative Problem Solving** | "Debug this dataset, identify anomalies, and propose fixes with statistical validation" |

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Backend Selection](#-llm-backend-selection)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Agent Types](#-agent-types)
- [Tools & Capabilities](#-tools--capabilities)
- [Security](#-security)
- [Development](#-development)
- [Troubleshooting](#-troubleshooting)
- [Version History](#-version-history)

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
# Clone and setup
git clone <repository-url>
cd LLM_API
pip install -r requirements.txt
```

### 2. Choose Your Backend

#### Option A: Ollama (Recommended for Getting Started)

```bash
# Install Ollama from https://ollama.ai/
ollama serve

# Pull required models
ollama pull gemma3:12b
ollama pull gpt-oss:20b
ollama pull bge-m3:latest
```

#### Option B: llama.cpp (Recommended for Production)

```bash
# Download GGUF models from Hugging Face
# Example: Qwen Coder 7B Q4 quantization
wget https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf

# Place in ./models/ directory
mkdir -p models
mv qwen2.5-coder-7b-instruct-q4_k_m.gguf models/

# For GPU support, install llama-cpp-python with CUDA
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### 3. Configure Backend

Edit `backend/config/settings.py`:

```python
# Line 33: Choose your backend
llm_backend: str = 'ollama'  # or 'llamacpp'
```

### 4. Start the Server

```bash
python run_backend.py
# Server runs at http://0.0.0.0:1007
```

### 5. Test the API

```python
import requests

response = requests.post(
    "http://localhost:1007/api/chat",
    json={
        "message": "Search for the latest news about AI and summarize the top 3 articles",
        "session_id": "test-session",
        "user_id": "admin"
    }
)

print(response.json()["response"])
```

---

## 🏗️ Architecture

### System Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    User Request + Files                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              LLM-Powered Task Classifier                     │
│   (Analyzes intent → routes to chat/react/plan-execute)    │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
   ┌────────┐      ┌─────────┐     ┌──────────────┐
   │  CHAT  │      │  REACT  │     │ PLAN-EXECUTE │
   │ Simple │      │ Single  │     │  Multi-Step  │
   │ Answer │      │  Tool   │     │   Complex    │
   └────────┘      └────┬────┘     └──────┬───────┘
                        │                 │
                        ▼                 ▼
              ┌──────────────────────────────────┐
              │        Tool Execution             │
              │  • Web Search (Tavily)           │
              │  • RAG Retrieval (FAISS)         │
              │  • Python Code Gen/Exec          │
              │  • File Analysis                 │
              └──────────────────┬───────────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────┐
              │     Response Synthesis            │
              │  + Session Storage                │
              │  + Variable Persistence           │
              └──────────────────────────────────┘
```

### Directory Structure

```
LLM_API/
├── backend/
│   ├── api/                    # FastAPI application
│   │   ├── app.py              # Main app, CORS, middleware
│   │   └── routes/             # Modular API endpoints (v2.0+)
│   │       ├── chat.py         # Chat & completions
│   │       ├── auth.py         # Authentication
│   │       ├── files.py        # File upload/download
│   │       ├── admin.py        # Admin endpoints
│   │       └── tools.py        # Tool-specific routes
│   │
│   ├── config/
│   │   ├── settings.py         # Central configuration (all defaults)
│   │   └── prompts/            # Centralized prompt management (v2.0+)
│   │       ├── __init__.py     # PromptRegistry
│   │       ├── task_classification.py
│   │       ├── react_agent.py
│   │       ├── plan_execute.py
│   │       ├── file_analyzer.py
│   │       └── python_coder/   # Code generation prompts
│   │
│   ├── tasks/                  # Agent implementations
│   │   ├── chat_task.py        # Entry point & classifier
│   │   ├── react/              # Modular ReAct agent (v2.0+)
│   │   │   ├── agent.py        # Main orchestration
│   │   │   ├── thought_action_generator.py
│   │   │   ├── tool_executor.py
│   │   │   ├── answer_generator.py
│   │   │   ├── context_manager.py
│   │   │   ├── verification.py
│   │   │   └── plan_executor.py
│   │   ├── Plan_execute.py     # Plan-Execute workflow
│   │   └── smart_agent_task.py # High-level router
│   │
│   ├── tools/                  # Tool implementations (v2.0+)
│   │   ├── python_coder/       # Code generation & execution
│   │   │   ├── tool.py         # Main orchestration
│   │   │   ├── orchestrator.py # Context-aware coordination (v1.4+)
│   │   │   ├── generator.py    # Code generation
│   │   │   ├── executor/       # Execution engine
│   │   │   │   ├── code_executor.py
│   │   │   │   └── repl_manager.py
│   │   │   ├── verifier.py     # Code verification
│   │   │   └── variable_storage.py  # Persistence (v1.3+)
│   │   ├── web_search/         # Tavily integration
│   │   ├── rag_retriever/      # FAISS RAG
│   │   └── file_analyzer/      # Multi-format file processing
│   │
│   ├── utils/
│   │   ├── llm_factory.py      # Centralized LLM creation (v2.0+)
│   │   ├── auth.py             # JWT authentication
│   │   └── logging_utils.py    # Structured logging (v1.8+)
│   │
│   └── storage/
│       └── conversation_store.py  # Conversation persistence
│
├── data/
│   ├── conversations/          # Chat history (JSON)
│   ├── uploads/                # User uploaded files
│   ├── scratch/                # Code execution workspace
│   │   └── {session_id}/
│   │       ├── variables/      # Persisted variables (v1.3+)
│   │       └── *.py            # Generated code
│   └── logs/                   # Application logs
│
├── frontend/                   # Web interface
├── requirements.txt
└── run_backend.py             # Server launcher
```

---

## 🔄 LLM Backend Selection

### Comparison

| Feature | **Ollama** | **llama.cpp** |
|---------|-----------|--------------|
| **Setup Complexity** | ⭐⭐⭐⭐⭐ Easy | ⭐⭐⭐ Moderate |
| **Model Management** | Built-in model library | Manual GGUF downloads |
| **GPU Control** | Automatic | Fine-grained (layer-by-layer) |
| **Memory Mapping** | Managed by Ollama | Configurable (mmap, mlock) |
| **Context Extension** | Limited by model | RoPE scaling support |
| **Production Use** | ⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Excellent |
| **Best For** | Development, quick setup | Production, embedded, fine control |

### Configuration

```python
# backend/config/settings.py (line 33)
llm_backend: str = 'ollama'  # or 'llamacpp'

# Ollama Settings (lines 40-52)
ollama_host: str = 'http://127.0.0.1:11434'
ollama_model: str = 'gemma3:12b'
ollama_num_ctx: int = 4096
ollama_timeout: int = 3000

# llama.cpp Settings (lines 54-83)
llamacpp_model_path: str = './models/qwen-coder-30b.gguf'
llamacpp_n_gpu_layers: int = -1      # -1 = all layers to GPU
llamacpp_n_ctx: int = 16384          # Context window
llamacpp_temperature: float = 0.6
llamacpp_rope_freq_scale: float = 1.0  # Context extension
llamacpp_use_mmap: bool = True       # Memory mapping
llamacpp_use_mlock: bool = False     # Lock in RAM
```

### Switching Backends at Runtime

```python
from backend.utils.llm_factory import LLMFactory

# Ollama
llm = LLMFactory.create_llm(backend='ollama', model='qwen3-coder:30b')

# llama.cpp
llm = LLMFactory.create_llm(
    backend='llamacpp',
    model_path='./models/qwen-coder-7b-q4.gguf',
    n_gpu_layers=-1  # All layers to GPU
)

# Both return ChatOllama-compatible interface
response = await llm.ainvoke("Hello, world!")
```

---

## 🛠️ Installation

### Prerequisites

- **Python 3.9+**
- **LLM Backend**:
  - **Ollama** (easy): Install from [ollama.ai](https://ollama.ai/)
  - **llama.cpp** (advanced): GGUF models from [Hugging Face](https://huggingface.co/models?library=gguf)
- **API Keys** (optional):
  - Tavily API key for web search

### Step-by-Step Installation

#### 1. Clone Repository

```bash
git clone <repository-url>
cd LLM_API
```

#### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Install llama-cpp-python (If Using llama.cpp Backend)

```bash
# NVIDIA GPU (CUDA)
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Apple Silicon (Metal)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# CPU Only
pip install llama-cpp-python
```

#### 5. Setup Ollama Models (If Using Ollama Backend)

```bash
# Start Ollama service
ollama serve

# Pull required models
ollama pull gemma3:12b       # General purpose
ollama pull gpt-oss:20b      # Task classification
ollama pull bge-m3:latest    # Embeddings for RAG
```

#### 6. Download GGUF Models (If Using llama.cpp Backend)

```bash
# Create models directory
mkdir -p models

# Download recommended models from Hugging Face
# Qwen Coder (excellent for coding tasks)
wget https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf -P models/

# Update settings.py with model path
# llamacpp_model_path: str = './models/qwen2.5-coder-7b-instruct-q4_k_m.gguf'
```

---

## ⚙️ Configuration

### Central Configuration File

All settings are in `backend/config/settings.py`. The `.env` file is **optional** and only used for overrides.

### Key Settings

```python
# Backend Selection
llm_backend: str = 'ollama'  # or 'llamacpp'

# Ollama Configuration
ollama_model: str = 'gemma3:12b'
agentic_classifier_model: str = 'gpt-oss:20b'
ollama_coder_model: str = 'gemma3:12b'  # Can use specialized model

# llama.cpp Configuration
llamacpp_model_path: str = './models/qwen-coder-30b.gguf'
llamacpp_n_gpu_layers: int = -1  # GPU acceleration

# Python Code Execution
python_code_enabled: bool = True
python_code_timeout: int = 3000  # seconds
python_code_max_iterations: int = 5  # retry attempts

# Security
secret_key: str = 'your-secret-key-here'
jwt_expiration_hours: int = 24

# API Keys (optional)
tavily_api_key: str = ''  # For web search
```

### Environment Variables (Optional Overrides)

Create `.env` file:

```bash
OLLAMA_HOST=http://127.0.0.1:11434
TAVILY_API_KEY=your-api-key
SECRET_KEY=your-secret-key
```

---

## 📖 Usage

### Starting the Server

```bash
# Backend API server
python run_backend.py
# Server runs at: http://0.0.0.0:1007

# Frontend (separate terminal)
python run_frontend.py
# Frontend runs at: http://localhost:3000
```

### Basic Chat Request

```python
import requests

response = requests.post(
    "http://localhost:1007/api/chat",
    json={
        "message": "What is Python?",
        "session_id": "my-session",
        "user_id": "admin"
    }
)

print(response.json()["response"])
```

### File Upload and Analysis

```python
import requests

# 1. Upload file
files = {'file': open('data.csv', 'rb')}
upload_response = requests.post(
    "http://localhost:1007/api/files/upload",
    files=files
)
file_path = upload_response.json()["file_path"]

# 2. Chat with file context
response = requests.post(
    "http://localhost:1007/api/chat",
    json={
        "message": "Analyze this CSV and show statistics",
        "file_paths": [file_path],
        "session_id": "analysis-session",
        "user_id": "admin"
    }
)

print(response.json()["response"])
```

### OpenAI-Compatible API

```python
import requests
import json

response = requests.post(
    "http://localhost:1007/v1/chat/completions",
    json={
        "model": "gemma3:12b",
        "messages": json.dumps([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing"}
        ]),
        "agent_type": "auto"  # or "chat", "react", "plan_execute"
    }
)

print(response.json())
```

### Multi-Phase Workflow Pattern

```python
# Phase 1: Initial analysis (process files ONCE)
phase1_prompt = """
Analyze the attached CSV file.
Calculate: mean, median, correlations, and outliers.
I'll ask follow-up questions in subsequent messages.
"""
result1, session_id = client.chat_new(MODEL, phase1_prompt, files=[csv_path])

# Phase 2: Visualization (reuse Phase 1 findings from memory)
phase2_prompt = """
**PRIORITY: Use your Phase 1 analysis from conversation memory.**

You already calculated: mean, median, correlations, and outliers.

**DO NOT re-analyze the raw files.** Use your Phase 1 findings.

Current Task: Create 3 visualizations based on Phase 1 results:
1. Distribution plot
2. Correlation heatmap
3. Outlier detection chart
"""
result2, _ = client.chat_continue(MODEL, session_id, phase2_prompt)

# Phase 3: Report generation (reuse all previous work)
phase3_prompt = """
**PRIORITY: Use Phase 1 & 2 findings from conversation memory.**

Create a PowerPoint report with:
- Phase 1 statistics
- Phase 2 visualizations
- Executive summary
"""
result3, _ = client.chat_continue(MODEL, session_id, phase3_prompt)
```

**Benefits of Multi-Phase Pattern:**
- 90% fewer LLM calls - no redundant processing
- Faster execution - reuse existing calculations
- Better consistency - all phases use same base analysis

See: `PPTX_Report_Generator_Agent_v2.ipynb` for detailed example.

---

## 🔌 API Reference

### Core Endpoints

#### `POST /api/chat`

Main chat interface with agentic capabilities.

**Request:**
```json
{
  "message": "string",
  "session_id": "string (optional)",
  "user_id": "string",
  "file_paths": ["string (optional)"]
}
```

**Response:**
```json
{
  "response": "string",
  "session_id": "string",
  "agent_type": "chat|react|plan_execute",
  "tool_calls": ["array (optional)"]
}
```

#### `POST /v1/chat/completions`

OpenAI-compatible chat completions endpoint.

**Request:**
```json
{
  "model": "string",
  "messages": "JSON string",
  "agent_type": "auto|chat|react|plan_execute (optional)"
}
```

#### `POST /api/files/upload`

Upload files for processing.

**Request:** `multipart/form-data` with file

**Response:**
```json
{
  "file_path": "string",
  "filename": "string",
  "size": "number"
}
```

#### `GET /api/chat/sessions/{session_id}/artifacts`

List files generated during a session.

**Response:**
```json
{
  "artifacts": [
    {
      "filename": "string",
      "size": "number",
      "modified": "timestamp"
    }
  ]
}
```

#### `GET /api/chat/sessions/{session_id}/artifacts/{filename}`

Download a generated artifact file.

**Response:** File download (FileResponse)

---

## 🤖 Agent Types

### 1. Chat Agent (Simple Q&A)

**When Used:** Questions answerable from LLM knowledge base (no tools required)

**Examples:**
- "What is Python?"
- "Explain machine learning"
- "How does photosynthesis work?"

**Characteristics:**
- Direct LLM response
- No tool invocation
- Fastest response time

---

### 2. ReAct Agent (Single-Goal Tool Tasks)

**When Used:** Single-objective tasks requiring tool usage

**Examples:**
- "Search for the latest AI news"
- "Analyze this CSV file"
- "Calculate the Fibonacci sequence"

**How It Works:**
```
Loop (max 10 iterations):
  1. Thought: "I need to search for AI news"
  2. Action: web_search("latest AI news")
  3. Observation: [search results]
  4. Thought: "Now I have enough information"
  5. Action: finish([final answer])
```

**Tools Available:**
- `web_search`: Tavily API for real-time search
- `rag_retrieval`: Document retrieval from FAISS
- `python_coder`: Code generation and execution
- `file_analyzer`: Multi-format file analysis
- `finish`: Return final answer

**Optimizations:**
- Combined thought-action generation (1 LLM call instead of 2)
- Context pruning for long conversations
- Early exit detection when answer is complete
- Variable persistence across executions

---

### 3. Plan-Execute Agent (Multi-Step Complex Tasks)

**When Used:** Complex tasks requiring planning and structured execution

**Examples:**
- "Analyze 3 CSV files, create visualizations, and generate a PowerPoint report"
- "Research latest AI papers, summarize findings, and create comparison table"
- "Load data, calculate KPIs, detect anomalies, and export results"

**How It Works:**
```
Phase 1: Planning
  - Planner creates structured plan with steps
  - Each step has: goal, success_criteria, primary_tools, fallback_tools

Phase 2: Guided Execution
  - Executor (ReAct agent) executes each step
  - Tool selection based on step requirements
  - Previous results passed to next step

Phase 3: Monitoring & Adaptation
  - Verify each step completion
  - Auto-retry on failure
  - Final answer synthesis
```

**Example Plan:**
```json
[
  {
    "step": 1,
    "goal": "Load and validate data files",
    "success_criteria": "All files loaded without errors",
    "primary_tools": ["file_analyzer", "python_coder"],
    "fallback_tools": ["rag_retrieval"]
  },
  {
    "step": 2,
    "goal": "Calculate statistics and correlations",
    "success_criteria": "Statistics calculated and validated",
    "primary_tools": ["python_coder"],
    "fallback_tools": []
  },
  {
    "step": 3,
    "goal": "Create visualizations",
    "success_criteria": "3 charts generated and saved",
    "primary_tools": ["python_coder"],
    "fallback_tools": []
  }
]
```

---

## 🛠️ Tools & Capabilities

### 1. Python Code Generator & Executor

**What It Does:** Generates and executes Python code in a sandboxed environment

**Features:**
- Context-aware generation (v1.4+): Uses conversation history, plan context, react iteration history
- Iterative verification: Max 3 verification loops
- Execution retry: Max 5 attempts with auto-fixing
- Variable persistence (v1.3+): DataFrames, arrays, objects saved across executions
- File support: CSV, Excel, JSON, PDF, images, etc.

**Security Measures:**
- Blocked imports: `socket`, `subprocess`, `eval`, `exec`, `pickle`, etc.
- Execution timeout: 3000 seconds (configurable)
- Memory limits: 5120 MB (configurable)
- Isolated session directories
- AST-based import validation (no runtime eval)

**Example Flow:**
```
User: "Analyze warpage data and calculate statistics"
  → Code generated with file metadata
  → Static analysis (import checks)
  → LLM verification: "Does code answer question?"
  → Execute code in subprocess
  → On error: LLM analyzes and fixes code
  → Retry with fixed code (max 5 attempts)
  → Variables saved: df_warpage.parquet, stats.json
```

**Variable Persistence:**
```
./data/scratch/{session_id}/variables/
├── variables_metadata.json    # Variable catalog
├── df_warpage.parquet         # DataFrames
├── stats_summary.json         # Simple types
└── correlation_matrix.npy     # NumPy arrays
```

---

### 2. Web Search (Tavily API)

**What It Does:** Real-time web search for current information

**Features:**
- Tavily API integration
- Result ranking and filtering
- Source attribution
- Configurable search depth

**Example:**
```
Query: "Latest AI breakthroughs in 2025"
→ Tavily API search
→ Top 5 results with summaries
→ Source URLs and timestamps
```

---

### 3. RAG Retriever (FAISS)

**What It Does:** Document-based context retrieval

**Features:**
- FAISS vector store
- BGE-M3 embeddings
- Semantic similarity search
- Multi-document support

**Example:**
```
Query: "What is the pricing model?"
→ Embed query with BGE-M3
→ FAISS similarity search
→ Retrieve top 3 relevant chunks
→ Context passed to LLM
```

---

### 4. File Analyzer

**What It Does:** Multi-format file analysis and extraction

**Supported Formats:**
- **Tabular:** CSV, Excel (.xlsx, .xls)
- **Documents:** PDF, TXT, Markdown
- **Data:** JSON, XML
- **Images:** PNG, JPG (with vision model)

**Features:**
- Format-specific analyzers
- LLM-powered deep analysis
- Metadata extraction (columns, dtypes, preview)
- Structure comparison
- Anomaly detection

**Example:**
```
File: sales_data.csv
→ Extract columns, dtypes, shape
→ Preview first 5 rows
→ Statistical summary
→ LLM analysis: "Sales data with 12 columns, 1000 rows..."
```

---

## 🔒 Security

### Code Execution Sandbox

- **Blocked Imports:** 30+ dangerous modules (socket, subprocess, eval, etc.)
- **Timeout Enforcement:** 3000 seconds (configurable)
- **Memory Limits:** 5120 MB (configurable)
- **Isolated Directories:** Session-based workspace prevents cross-contamination
- **AST Validation:** Static analysis before execution
- **No Runtime Eval:** Import checks use AST parsing, not `eval()`

### Authentication

- **JWT Tokens:** Secure user authentication
- **Session Management:** User-scoped sessions
- **Path Traversal Protection:** Blocks `../` attacks in file downloads
- **Session Ownership Validation:** Users can only access their own data

### Best Practices

- ✅ Change `secret_key` in production
- ✅ Never expose server directly to internet without authentication
- ✅ Review `BLOCKED_IMPORTS` before modifying
- ✅ Monitor logs for suspicious activity
- ✅ Limit file upload sizes and types
- ✅ Use HTTPS in production
- ✅ Implement rate limiting for public deployments

---

## 🧪 Development

### Project Structure Philosophy

- **API Layer** (`backend/api/`): HTTP handling, routing, middleware
- **Agent Layer** (`backend/tasks/`): Reasoning, planning, orchestration
- **Tool Layer** (`backend/tools/`): External integrations and execution
- **Storage Layer** (`backend/storage/`): Persistence and data management
- **Utils Layer** (`backend/utils/`): Shared utilities and factories

### Testing

```bash
# Run API tests
python -m pytest tests/

# Manual testing with Jupyter
jupyter notebook API_examples.ipynb

# Check Ollama connection
curl http://127.0.0.1:11434/api/tags
```

### Adding a New Tool

1. **Create Tool Module** in `backend/tools/`
   ```python
   # backend/tools/my_tool/tool.py
   class MyTool:
       def run(self, query: str) -> dict:
           # Implementation
           return {"result": "..."}

   my_tool = MyTool()
   ```

2. **Add to ToolName Enum** in `backend/tasks/react/models.py`
   ```python
   class ToolName(str, Enum):
       MY_TOOL = "my_tool"
   ```

3. **Add Execution Logic** in `backend/tasks/react/tool_executor.py`
   ```python
   elif action.tool_name == ToolName.MY_TOOL:
       from backend.tools.my_tool import my_tool
       observation = my_tool.run(action.tool_input)
   ```

4. **Update Prompts** in `backend/config/prompts/react_agent.py`
   ```python
   AVAILABLE_TOOLS = [
       "my_tool: Description of what this tool does"
   ]
   ```

5. **Register in Settings** in `backend/config/settings.py`
   ```python
   available_tools: list = ["my_tool", ...]
   ```

---

## 🔧 Troubleshooting

### Ollama Backend Issues

**Problem:** "Connection refused" when starting server

**Solution:**
```bash
# 1. Check Ollama is running
ollama serve

# 2. Test connection
curl http://127.0.0.1:11434/api/tags

# 3. Verify settings.py
# ollama_host: str = 'http://127.0.0.1:11434'
```

**Problem:** "Model not found" error

**Solution:**
```bash
# Pull required models
ollama pull gemma3:12b
ollama pull gpt-oss:20b
ollama pull bge-m3:latest

# List available models
ollama list
```

---

### llama.cpp Backend Issues

**Problem:** "CUDA error" when using GPU

**Solution:**
```bash
# 1. Verify NVIDIA drivers
nvidia-smi

# 2. Rebuild llama-cpp-python with CUDA
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# 3. Test in Python
python -c "from llama_cpp import Llama; print('CUDA available')"

# 4. Reduce GPU layers if out of VRAM
# settings.py: llamacpp_n_gpu_layers = 20  (instead of -1)
```

**Problem:** "Out of memory" error

**Solution:**
```python
# settings.py
llamacpp_n_ctx = 4096          # Reduce context size
llamacpp_low_vram = True       # Enable low VRAM mode
llamacpp_n_gpu_layers = 0      # CPU only
llamacpp_use_mmap = False      # Disable memory mapping
```

**Problem:** Slow inference

**Solution:**
```python
# settings.py
llamacpp_n_gpu_layers = -1     # All layers to GPU
llamacpp_n_batch = 2048        # Increase batch size
llamacpp_use_mmap = True       # Enable memory mapping

# Use smaller quantization (Q4_K_M instead of Q8_0)
# Download from Hugging Face
```

---

### Code Execution Issues

**Problem:** "Code execution timeout"

**Solution:**
```python
# settings.py
python_code_timeout: int = 5000  # Increase timeout (seconds)
```

**Problem:** "Import not allowed" error

**Solution:**
```python
# Review blocked imports in:
# backend/tools/python_coder/executor/code_executor.py

# BLOCKED_IMPORTS list includes:
# socket, subprocess, eval, exec, pickle, etc.

# If you need to allow an import, remove it from the list
# (only if you trust the code source!)
```

**Problem:** Empty or incomplete responses

**Solution:**
```bash
# 1. Check logs
cat data/logs/app.log

# 2. Verify classifier is working
# Edit settings.py: agentic_classifier_model

# 3. Check max_tokens setting
# settings.py: llamacpp_max_tokens = 4096
```

---

### General Issues

**Problem:** Variables not persisting across sessions

**Solution:**
```python
# Ensure you're using the same session_id
# Variables are session-scoped, not user-scoped

# Check variable storage:
ls data/scratch/{session_id}/variables/
cat data/scratch/{session_id}/variables/variables_metadata.json
```

**Problem:** File upload fails

**Solution:**
```bash
# 1. Check upload directory exists
mkdir -p data/uploads

# 2. Verify file size limits
# FastAPI default: 2MB
# Increase in backend/api/app.py if needed

# 3. Check file permissions
chmod 755 data/uploads
```

---

## 📚 Version History

### Version 2.1.0 (December 2, 2025)
**Feature: Dual Backend Support - llama.cpp Integration**

- Added native llama.cpp support alongside Ollama
- `LlamaCppWrapper` in `llm_factory.py` for ChatOllama-compatible interface
- Backend selection via `llm_backend` setting
- Comprehensive llama.cpp configuration (GPU layers, context extension, memory optimization)
- GGUF model support with fine-grained hardware control
- RoPE scaling for extended context windows
- Lazy loading for faster startup
- Backward compatible with existing Ollama deployments

**Configuration:**
```python
llm_backend: str = 'ollama'  # or 'llamacpp'
llamacpp_model_path: str = './models/qwen-coder-30b.gguf'
llamacpp_n_gpu_layers: int = -1  # GPU offloading
llamacpp_n_ctx: int = 16384      # Context window
```

---

### Version 2.0.0 (January 2025)
**Major Refactoring: Modular Architecture**

Comprehensive code restructuring that splits large monolithic files into focused, maintainable modules.

**ReAct Agent Modularization:**
- Split 2000+ line `React.py` → 8 focused modules in `backend/tasks/react/`
- Modules: agent.py, thought_action_generator.py, tool_executor.py, answer_generator.py, context_manager.py, verification.py, plan_executor.py, models.py

**Python Coder Tool Modularization:**
- Split `python_coder_tool.py` → 5 modules in `backend/tools/python_coder/`
- Modules: tool.py, generator.py, executor.py, verifier.py, models.py

**Other Modularizations:**
- File Analyzer: `file_analyzer_tool.py` → 4 modules
- Web Search: `web_search.py` → 2 modules
- RAG Retriever: `rag_retriever.py` → 4 modules
- API Routes: `routes.py` → 5 route modules

**Infrastructure Improvements:**
- Added `llm_factory.py` - centralized LLM creation
- Added `prompts.py` - centralized prompt management (PromptRegistry)
- Legacy files preserved for backward compatibility

---

### Version 1.9.0 (November 25, 2025)
**Enhancement: Prompt System Overhaul + Output File Handling**

- Output file handling for Python coder (solves CMD truncation)
- New base utilities: ASCII markers, temporal awareness, reusable rule blocks
- Token optimization: 25-35% average reduction across prompts
- Expanded file analyzer with specialized prompts
- Improved plan context passing

---

### Version 1.8.0 (November 25, 2025)
**Enhancement: Structured LLM Prompt Logging**

- Redesigned `LLMInterceptor` class for readable logging
- Three log formats: STRUCTURED, JSON, COMPACT
- Request/response pairing with unique call IDs
- Enhanced metadata: token estimation, duration tracking
- Streaming response logging

---

### Version 1.7.0 (November 24, 2025)
**Enhancement: Intelligent Retry System for Python Coder**

- Full attempt history tracking with error classification
- Runtime variable capture on error
- Enhanced retry prompts with escalating strategies
- Forced different approach on repeated errors
- 15+ classified error types with specific guidance

**Problem Solved:** Model no longer gets "stuck" on repeated errors (e.g., IndexError loops)

---

### Version 1.6.0 (November 24, 2024)
**Simplification: Session Notepad Removal**

- Removed automatic session notepad feature
- Simpler architecture with less overhead
- Variable persistence remains intact
- Reduced latency and cost per execution

---

### Version 1.5.0 (November 24, 2024)
**Major Refactor: 3-Way Agent Classification**

- LLM-powered 3-way classification: chat, react, plan_execute
- Simplified `chat.py`: 510 → 400 lines (22% reduction)
- Removed streaming support
- Clearer logic and better accuracy

---

### Version 1.4.0 (November 20, 2024)
**Enhancement: Context-Aware Python Code Generation**

- Conversation history integration
- Plan context for Plan-Execute workflows
- ReAct context with failed attempts
- 8-section prompt structure: HISTORIES → INPUT → PLANS → REACTS → TASK → METADATA → RULES → CHECKLISTS

---

### Version 1.3.0 (November 19, 2024)
**Feature: Variable Persistence System**

- Automatic variable saving with type-specific serialization
- DataFrames → Parquet, arrays → .npy, simple types → JSON
- Context auto-injection into subsequent executions
- Namespace capture in REPL mode

---

### Version 1.2.0 (October 31, 2024)
**Major Changes: Python Code Tool Unification**

- Merged multiple files into single `python_coder_tool.py`
- Reduced verification iterations: 10 → 3
- Added execution retry logic: max 5 attempts with auto-fixing
- Simplified architecture

---

### Earlier Versions
- **v1.1.0:** Multi-agent architecture implementation
- **v1.0.0:** Initial LLM API server with Ollama integration

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📧 Support

- **Issues:** [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions:** [GitHub Discussions](https://github.com/your-repo/discussions)
- **Documentation:** See [CLAUDE.md](CLAUDE.md) for detailed developer docs

---

## 🙏 Acknowledgments

- **LangChain** - Agent framework
- **LangGraph** - Workflow orchestration
- **Ollama** - LLM inference
- **llama.cpp** - High-performance LLM runtime
- **Tavily** - Web search API
- **FAISS** - Vector similarity search

---

**Built with ❤️ for AI-powered automation**

*Last Updated: December 2, 2025 | Version 2.1.0*
