# LLM API

> **AI-Powered Agentic Backend** with ReAct reasoning, autonomous code execution, and multi-tool orchestration

A comprehensive FastAPI-based LLM backend featuring sophisticated agentic workflows, dual LLM backend support (Ollama + llama.cpp), and intelligent tool orchestration for complex tasks.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.119.1-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-1.0.2-orange.svg)](https://www.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ Features

### 🤖 **Intelligent Agent System**
- **ReAct Agent**: Reasoning + Acting pattern with thought-action-observation loops
- **Plan-Execute Mode**: Structured multi-step task decomposition and execution
- **Auto Mode**: Automatically chooses optimal execution strategy
- **Context-Aware**: Maintains conversation history and session state

### 🛠️ **Powerful Tool Ecosystem**
- **🐍 Python Code Generation**: Autonomous code writing, execution, and debugging
  - Session-based variable persistence
  - Automatic file handling (CSV, Excel, JSON, PDF, images)
  - Sandboxed execution with retry logic
- **🔍 Web Search**: Real-time information retrieval via Tavily API
- **📚 RAG Retrieval**: FAISS-based document search and retrieval
- **📊 File Analysis**: Smart metadata extraction and analysis
- **👁️ Vision Analysis**: Image understanding with multimodal LLMs

### 🚀 **Dual LLM Backend Support**

<table>
<tr>
<th>Ollama</th>
<th>llama.cpp</th>
</tr>
<tr>
<td>

```python
# Quick setup, managed models
llm_backend: str = 'ollama'
ollama_host: str = 'http://127.0.0.1:11434'
ollama_model: str = 'gpt-oss:20b'
```

</td>
<td>

```python
# Production-ready, fine-grained control
llm_backend: str = 'llamacpp'
llamacpp_model_path: str = './models/model.gguf'
llamacpp_n_gpu_layers: int = -1  # Full GPU
```

</td>
</tr>
</table>

### 📡 **OpenAI-Compatible API**
- Drop-in replacement for OpenAI Chat Completions API
- Extended metadata for agent execution details
- JWT authentication with role-based access control
- File upload support with multipart/form-data

### 💾 **Session Management**
- Persistent conversation history
- Automatic code and variable persistence across sessions
- Session-based execution directories
- User-specific file management

---

## 🚀 Quick Start

### Prerequisites

**For Ollama Backend (Recommended for Development):**
```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Or download from: https://ollama.com/download

# Start Ollama server
ollama serve

# Pull models
ollama pull gpt-oss:20b
ollama pull llama3.2-vision:11b
```

**For llama.cpp Backend (Recommended for Production):**
```bash
# Download GGUF models from Hugging Face
# Example: Qwen Coder
wget https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf

# Place in models directory
mkdir -p models
mv qwen2.5-coder-7b-instruct-q4_k_m.gguf models/
```

### Installation

```bash
# Clone repository
git clone <repository-url>
cd LLM_API

# Install dependencies
pip install -r requirements.txt

# Optional: Install with GPU support (CUDA)
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Optional: Install with GPU support (Metal for Apple Silicon)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Configuration

Edit `backend/config/settings.py` or create `.env` file:

```python
# LLM Backend Selection
llm_backend = 'ollama'  # or 'llamacpp'

# Ollama Configuration
ollama_host = 'http://127.0.0.1:11434'
ollama_model = 'gpt-oss:20b'
ollama_num_ctx = 2048

# Server Configuration
server_host = '0.0.0.0'
server_port = 1007

# Agent Configuration
react_max_iterations = 6
python_code_max_iterations = 3
```

### Run the Server

```bash
# Start backend server
python run_backend.py

# Or directly
python server.py
```

Server will be available at:
- **API**: `http://localhost:1007`
- **Swagger UI**: `http://localhost:1007/docs`
- **ReDoc**: `http://localhost:1007/redoc`

---

## 📖 Usage

### Basic Chat Request

```bash
curl -X POST http://localhost:1007/api/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "model": "gpt-oss:20b"
  }'
```

### Python Code Generation with File Analysis

```bash
curl -X POST http://localhost:1007/api/chat/completions \
  -H "Content-Type: multipart/form-data" \
  -F 'messages=[{"role": "user", "content": "Analyze this CSV and create a bar chart"}]' \
  -F 'files=@data.csv' \
  -F 'agent_type=react'
```

### Web Search Request

```bash
curl -X POST http://localhost:1007/api/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What are the latest developments in AI?"}],
    "agent_type": "react"
  }'
```

### Using Python Client

```python
import httpx

async def chat_completion(message: str, session_id: str = None):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:1007/api/chat/completions",
            json={
                "messages": [{"role": "user", "content": message}],
                "model": "gpt-oss:20b",
                "session_id": session_id,
                "agent_type": "auto"
            }
        )
        return response.json()

# First request - creates new session
result1 = await chat_completion("Analyze this sales data")
session_id = result1["x_session_id"]

# Follow-up request - reuses session context
result2 = await chat_completion("Create a report based on that analysis", session_id)
```

---

## 🏗️ Architecture

### High-Level Flow

```
User Request
    ↓
AgentOrchestrator (Auto-routing)
    ↓
    ├── SimpleChatAgent (No tools needed)
    │       ↓
    │   Direct LLM Response
    │
    ├── ReActAgent (Tool-based reasoning)
    │       ↓
    │   Thought-Action-Observation Loop
    │       ↓
    │   ├── web_search_tool
    │   ├── python_coder_tool
    │   ├── rag_retriever_tool
    │   ├── file_analyzer
    │   └── vision_analyzer_tool
    │
    └── Plan-Execute Mode (Structured tasks)
            ↓
        1. Create Plan (multiple steps)
        2. Execute each step with ReAct
        3. Synthesize final result
```

### Core Components

**Agent System** (`backend/agents/`)
- `AgentOrchestrator`: Main entry point, handles routing
- `SimpleChatAgent`: Direct LLM interaction without tools
- `ReActAgent`: Implements ReAct reasoning pattern
- `ThoughtActionGenerator`: LLM-based thought and action generation
- `ToolExecutor`: Routes and executes tool calls
- `AnswerGenerator`: Synthesizes final responses

**Tool System** (`backend/tools/`)
- `BaseTool`: Abstract base class for all tools
- `python_coder_tool`: Autonomous code generation and execution
- `web_search_tool`: Tavily API integration
- `rag_retriever_tool`: FAISS-based retrieval
- `file_analyzer`: File metadata extraction
- `vision_analyzer_tool`: Image understanding

**Infrastructure** (`backend/core/`)
- `base_tool.py`: Tool interface and utilities
- `file_handlers/`: Unified file handling system
- `llm_backends/`: LLM backend implementations
- `result_types.py`: Standardized result formats
- `exceptions.py`: Custom exception hierarchy

**API Layer** (`backend/api/`)
- `app.py`: FastAPI application setup
- `routes/chat.py`: Chat completions endpoint
- `routes/auth.py`: JWT authentication
- `routes/files.py`: File upload/management
- `middleware.py`: Security headers

### Key Design Patterns

**1. Factory Pattern** - `LLMFactory` for centralized LLM creation
```python
from backend.utils.llm_factory import LLMFactory

llm = LLMFactory.create_llm(temperature=0.7, user_id="alice")
coder = LLMFactory.create_coder_llm(user_id="alice")
```

**2. Singleton Pattern** - File handler registry
```python
from backend.core.file_handlers import file_handler_registry

handler = file_handler_registry.get_handler("data.csv")
```

**3. Strategy Pattern** - Agent type selection
```python
# Auto-selects optimal strategy based on query
agent_type = orchestrator._resolve_agent_type(
    agent_type="auto",
    messages=messages,
    file_paths=file_paths
)
```

**4. Template Method** - BaseTool defines execution flow
```python
class BaseTool(ABC):
    @abstractmethod
    async def execute(self, query: str, **kwargs) -> ToolResult:
        pass

    @abstractmethod
    def validate_inputs(self, **kwargs) -> bool:
        pass
```

---

## 🔧 Configuration

### Environment Variables

Create `.env` file (optional - all settings have defaults):

```bash
# LLM Backend
LLM_BACKEND=ollama

# Ollama
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=gpt-oss:20b
OLLAMA_NUM_CTX=2048
OLLAMA_TEMPERATURE=1.0

# llama.cpp
LLAMACPP_MODEL_PATH=./models/model.gguf
LLAMACPP_N_GPU_LAYERS=-1
LLAMACPP_N_CTX=2048
LLAMACPP_TEMPERATURE=1.0

# Server
SERVER_HOST=0.0.0.0
SERVER_PORT=1007
SECRET_KEY=your-secret-key-change-in-production

# Agent
REACT_MAX_ITERATIONS=6
PYTHON_CODE_MAX_ITERATIONS=3
PYTHON_CODE_TIMEOUT=3000

# Logging
LOG_LEVEL=INFO
LOG_FILE=./data/logs/app.log

# API Keys
TAVILY_API_KEY=your-tavily-api-key
```

### Backend-Specific Settings

**Ollama Backend:**
```python
ollama_host: str = 'http://127.0.0.1:11434'
ollama_model: str = 'gpt-oss:20b'
ollama_coder_model: str = 'gpt-oss:20b'
ollama_vision_model: str = 'llama3.2-vision:11b'
ollama_num_ctx: int = 2048
ollama_temperature: float = 1.0
ollama_timeout: int = 3000000
```

**llama.cpp Backend:**
```python
llamacpp_model_path: str = './models/gpt-oss-120b.gguf'
llamacpp_coder_model_path: str = './models/GLM-4.6-REAP.gguf'
llamacpp_vision_model_path: str = './models/gemma3-12b-it-q8_0.gguf'
llamacpp_n_gpu_layers: int = -1  # -1 = all, 0 = CPU only
llamacpp_n_ctx: int = 2048
llamacpp_temperature: float = 1.0
llamacpp_use_mmap: bool = True
llamacpp_use_mlock: bool = False
```

---

## 🧪 Testing

### Manual Testing with Jupyter

```bash
jupyter notebook API_examples.ipynb
```

### Health Check

```bash
curl http://localhost:1007/health
```

Expected response:
```json
{
  "status": "healthy",
  "ollama_connection": "connected",
  "models_available": ["gpt-oss:20b", "llama3.2-vision:11b"],
  "backend": "ollama"
}
```

### Test Individual Tools

```python
from backend.tools.python_coder import python_coder_tool

result = await python_coder_tool.execute_code_task(
    query="Calculate the sum of numbers 1 to 100",
    file_paths=[],
    session_id="test-session"
)

print(result["output"])
```

---

## 📁 Project Structure

```
LLM_API/
├── backend/
│   ├── agents/              # Agent implementations
│   │   └── react_agent.py   # Main ReAct agent + orchestrator
│   ├── api/                 # FastAPI routes
│   │   ├── app.py           # Main application
│   │   └── routes/          # API endpoints
│   ├── config/              # Configuration
│   │   └── settings.py      # All settings with defaults
│   ├── core/                # Core infrastructure
│   │   ├── base_tool.py     # Tool base class
│   │   ├── file_handlers/   # Unified file handling
│   │   └── llm_backends/    # LLM backend implementations
│   ├── tools/               # Tool implementations
│   │   ├── python_coder.py
│   │   ├── web_search.py
│   │   ├── rag_retriever.py
│   │   ├── file_analyzer.py
│   │   └── vision_analyzer.py
│   ├── services/            # Business logic
│   ├── storage/             # Data persistence
│   ├── models/              # Pydantic schemas
│   └── utils/               # Utilities
│       ├── llm_factory.py   # LLM creation
│       ├── llm_manager.py   # User-specific LLMs
│       └── logging_utils.py # Logging
├── data/
│   ├── conversations/       # Chat history
│   ├── uploads/             # User files
│   ├── scratch/             # Code execution workspace
│   └── logs/                # Application logs
├── models/                  # GGUF model files (for llama.cpp)
├── requirements.txt
├── run_backend.py           # Server launcher
├── server.py                # Main entry point
├── README.md                # This file
└── CLAUDE.md                # Developer guide for Claude Code
```

---

## 🔒 Security

### Authentication

JWT-based authentication with role-based access control:

```bash
# Register user
curl -X POST http://localhost:1007/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "password": "secure-password"}'

# Login
curl -X POST http://localhost:1007/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "password": "secure-password"}'

# Returns JWT token
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer"
}

# Use token in requests
curl -X GET http://localhost:1007/api/conversations \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
```

### Code Execution Sandbox

Python code execution is sandboxed with:
- **Import restrictions**: Blocked modules (subprocess, eval, exec, pickle, etc.)
- **Timeout controls**: Configurable execution timeout (default: 3000 seconds)
- **Session isolation**: Each session has separate execution directory
- **AST validation**: Static analysis before execution

### Security Headers

Automatic security headers via middleware:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000; includeSubDomains`

---

## 🎯 Use Cases

### 1. Data Analysis & Visualization

```python
# Upload CSV, analyze, and generate charts
messages = [{
    "role": "user",
    "content": "Analyze sales_data.csv and create a bar chart of revenue by region"
}]

response = await client.post(
    "/api/chat/completions",
    files={"files": open("sales_data.csv", "rb")},
    data={"messages": json.dumps(messages)}
)
```

**Agent will:**
- Automatically read CSV using pandas
- Calculate revenue by region
- Generate matplotlib chart
- Save as PNG in session directory
- Return analysis summary

### 2. Multi-Step Research Tasks

```python
messages = [{
    "role": "user",
    "content": """Research the latest developments in quantum computing,
    summarize key breakthroughs, and create a comparison table of
    major quantum computing companies"""
}]

response = await client.post(
    "/api/chat/completions",
    json={"messages": messages, "agent_type": "plan_execute"}
)
```

**Agent will:**
- Create structured plan (research → summarize → compare)
- Execute web searches for each step
- Synthesize information across multiple sources
- Generate structured comparison table

### 3. RAG-Based Document Q&A

```python
# First, upload documents
files = [open("report1.pdf", "rb"), open("report2.pdf", "rb")]
upload_response = await client.post("/api/upload", files=files)

# Then query
messages = [{
    "role": "user",
    "content": "What are the key findings from these reports?"
}]

response = await client.post(
    "/api/chat/completions",
    json={"messages": messages, "agent_type": "react"}
)
```

**Agent will:**
- Use RAG retriever to search documents
- Extract relevant passages
- Synthesize comprehensive answer

### 4. Vision + Code Generation

```python
messages = [{
    "role": "user",
    "content": "Analyze this chart image and recreate it as a matplotlib plot"
}]

response = await client.post(
    "/api/chat/completions",
    files={"files": open("chart.png", "rb")},
    data={"messages": json.dumps(messages)}
)
```

**Agent will:**
- Use vision analyzer to understand chart
- Extract data and styling
- Generate Python code to recreate chart
- Execute code and save new plot

---

## 🐛 Troubleshooting

### Ollama Connection Issues

**Error:** `Request URL is missing an 'http://' or 'https://' protocol`

**Solution:**
```bash
# Ensure Ollama is running
ollama serve

# Check connection
curl http://127.0.0.1:11434/api/tags

# Verify settings.py has correct host
ollama_host: str = 'http://127.0.0.1:11434'
```

### llama.cpp Model Not Loading

**Error:** `LLAMA.CPP MODEL FILE NOT FOUND!`

**Solution:**
```bash
# Download GGUF model
wget https://huggingface.co/model/path/model.gguf

# Place in correct location
mkdir -p models
mv model.gguf models/

# Update settings.py
llamacpp_model_path: str = './models/model.gguf'

# Check file exists
ls -lh models/
```

### Python Code Execution Timeout

**Error:** Code execution exceeds timeout

**Solution:**
```python
# Increase timeout in settings.py
python_code_timeout: int = 6000  # 6000 seconds

# Or check for infinite loops in generated code
# Review execution logs: data/scratch/{session_id}/
```

### Import Errors

**Error:** `ImportError: No module named 'xxx'`

**Solution:**
```bash
# Check if module is installed
pip show xxx

# Install missing package
pip install xxx

# Add to requirements.txt for persistence
echo "xxx==1.0.0" >> requirements.txt
```

### GPU Out of Memory (llama.cpp)

**Error:** CUDA out of memory

**Solution:**
```python
# Reduce GPU layers in settings.py
llamacpp_n_gpu_layers: int = 20  # Instead of -1 (all)

# Or enable low VRAM mode
llamacpp_low_vram: bool = True

# Or reduce context window
llamacpp_n_ctx: int = 1024  # Instead of 2048
```

---

## 📊 Performance Tips

### Optimizing LLM Calls

**1. Use Classifier for Routing**
```python
# Let classifier decide if tools are needed
agent_type = "auto"  # Instead of always using "react"
```

**2. Adjust Max Iterations**
```python
# Reduce iterations for simple tasks
react_max_iterations: int = 3  # Instead of 6

# Increase for complex tasks
react_max_iterations: int = 10
```

### Optimizing Code Execution

**1. Session Reuse**
```python
# Reuse session_id to persist variables
# First call: df = pd.read_csv('data.csv')
# Second call: df.describe()  # df already loaded
```

**2. Preload Common Libraries**
```python
# In settings.py
python_code_preload_libraries = [
    'pandas as pd',
    'numpy as np',
    'matplotlib.pyplot as plt'
]
```

### Scaling for Production

**1. Use llama.cpp Backend**
- Faster inference
- Better GPU control
- Lower memory overhead

**2. Connection Pooling**
- LLMFactory uses connection pooling by default
- Reuse LLM instances when possible

**3. Enable Caching**
```python
# File metadata caching
enable_file_metadata_cache: bool = True
file_metadata_cache_ttl_hours: int = 24
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Test thoroughly**: Ensure all existing functionality works
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open Pull Request**

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt  # If available

# Enable debug logging
LOG_LEVEL=DEBUG python server.py

# Run with auto-reload (development)
uvicorn backend.api.app:app --reload --host 0.0.0.0 --port 1007
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **LangChain** - Framework for LLM applications
- **LangGraph** - Graph-based agent orchestration
- **FastAPI** - Modern web framework
- **Ollama** - Easy LLM deployment
- **llama.cpp** - Efficient LLM inference
- **Tavily** - Web search API

---

## 📞 Support

- **Documentation**: See [CLAUDE.md](CLAUDE.md) for developer guide
- **Examples**: Check `API_examples.ipynb` for usage examples
- **Issues**: Report bugs via GitHub Issues
- **API Docs**: `http://localhost:1007/docs` when server is running

---

## 🗺️ Roadmap

### Current Version: 2.0.4

**Completed:**
- ✅ Dual backend support (Ollama + llama.cpp)
- ✅ ReAct agent with tool orchestration
- ✅ Python code generation with session persistence
- ✅ File handling for multiple formats
- ✅ Vision analysis capabilities
- ✅ OpenAI-compatible API
- ✅ JWT authentication

**v2.0.4 Changes:**
- 🔧 Completely rewritten ReAct response parser (`_parse_response`)
  - Multi-strategy parsing: structured regex + line-by-line fallback
  - Handles all case variations: `THOUGHT:`, `Thought:`, `thought:`
  - Handles all action input formats: `ACTION INPUT:`, `Action Input:`, `ACTION_INPUT:`, `Input:`
  - Better multi-line content handling
- 🔧 Enhanced LLM response handling in ThoughtActionGenerator
  - Added retry logic (3 attempts) for empty responses
  - Added comprehensive debug logging for LLM invocations
  - Added empty response detection with detailed error logging
  - Proper extraction of content from various AIMessage formats
  - Logs LLM type, model, prompt length for debugging

**Planned:**
- 🔄 Streaming responses (Server-Sent Events)
- 🔄 WebSocket support for real-time updates
- 🔄 Multi-user conversation support
- 🔄 Advanced RAG with re-ranking
- 🔄 Tool usage analytics and monitoring
- 🔄 Docker deployment with docker-compose
- 🔄 Kubernetes deployment configs
- 🔄 Integration tests and CI/CD

---

**Built with ❤️ by HE Team**

**Version:** 2.0.4
**Last Updated:** 2025-12-04
