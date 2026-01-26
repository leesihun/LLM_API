# LLM API Server

A powerful, production-ready API server that provides OpenAI-compatible endpoints with support for multiple LLM backends and intelligent agent-based tool calling capabilities.

## Features

### Core Capabilities
- **OpenAI-Compatible API** - Drop-in replacement for OpenAI's chat completions endpoint
- **Multi-Backend Support** - Switch between Ollama and llama.cpp with automatic fallback
- **Intelligent Agents** - Multiple agent types (Chat, ReAct, Plan-Execute, Auto) for different use cases
- **Tool Calling** - Built-in tools for web search, Python code execution, RAG, and presentation generation
- **Streaming Support** - Server-Sent Events (SSE) for real-time response streaming
- **Session Management** - Persistent conversation history with JSON-based storage
- **Authentication & Security** - JWT-based authentication with bcrypt password hashing
- **File Attachments** - Automatic metadata extraction from JSON, CSV, PDF, Python, and more

### Built-in Tools
1. **Web Search** - Powered by Tavily API for up-to-date information retrieval
2. **Python Code Execution** - Sandboxed Python execution with workspace isolation
3. **RAG (Retrieval Augmented Generation)** - FAISS-based document retrieval with embeddings
4. **Presentation Maker** - Generate PDF/PPTX presentations from natural language using Marp

### Agent Types
- **ChatAgent** - Simple conversational agent without tools
- **ReActAgent** - Reasoning + Acting agent with iterative tool calling
- **PlanExecuteAgent** - Multi-step planning and execution with re-planning capabilities
- **AutoAgent** - Automatically selects the best agent based on user input

## Architecture

### Dual-Server Design
The system uses a dual-server architecture to prevent deadlock:

- **Main API Server** (port 10007): Handles chat, authentication, sessions
- **Tools API Server** (port 10006): Handles tool execution

This separation is critical because agents running on the main server make HTTP calls to tools on the tools server. Running tools on the same server would cause deadlock.

```
┌─────────────────┐         ┌──────────────────┐
│   Main API      │────────>│   Tools API      │
│   (port 10007)  │  HTTP   │   (port 10006)   │
│                 │<────────│                  │
│ • Chat          │         │ • websearch      │
│ • Auth          │         │ • python_coder   │
│ • Sessions      │         │ • rag            │
│ • Agents        │         │ • ppt_maker      │
└─────────────────┘         └──────────────────┘
```

### Technology Stack
- **Framework**: FastAPI
- **LLM Backends**: Ollama, llama.cpp
- **Database**: SQLite (user auth & sessions)
- **Storage**: JSON files for conversations
- **Vector Store**: FAISS
- **Embeddings**: sentence-transformers
- **Authentication**: JWT (python-jose)
- **Password Hashing**: bcrypt (with passlib)

## Installation

### Prerequisites
- Python 3.8+
- Node.js (for nanocoder mode, optional)
- Ollama or llama.cpp server running
- Tavily API key (for web search)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd LLM_API
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: (Optional) Install Nanocoder
For autonomous code generation mode:
```bash
npm install -g @nanocollective/nanocoder
```

### Step 4: Configure Environment
Create or edit [config.py](config.py) with your settings:

```python
# LLM Backend
LLM_BACKEND = "ollama"  # "ollama", "llamacpp", or "auto"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:14b"

# API Keys
TAVILY_API_KEY = "your-tavily-api-key-here"

# Server Ports
MAIN_API_PORT = 10007
TOOLS_PORT = 10006
```

## Quick Start

### Starting the Servers

**Option 1: Start both servers at once (recommended)**
```bash
bash start_servers.sh
```

**Option 2: Start servers individually (for debugging)**
```bash
# Terminal 1 - Tools API (must start FIRST)
python tools_server.py

# Terminal 2 - Main API
python server.py
```

### Create Your First User

```bash
python create_user_direct.py
```

Follow the prompts to create an admin or regular user.

### Make Your First API Call

```bash
curl -X POST http://localhost:10007/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "model": "qwen2.5:14b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "agent_type": "chat"
  }'
```

## Configuration

All configuration is centralized in [config.py](config.py). Key settings:

### LLM Backend
```python
LLM_BACKEND = "ollama"           # "ollama", "llamacpp", or "auto"
OLLAMA_MODEL = "qwen2.5:14b"     # Default model for Ollama
DEFAULT_TEMPERATURE = 0.7        # Default temperature
MAX_CONVERSATION_HISTORY = 20    # Max messages to keep
```

### Agent Configuration
```python
# ReAct Agent
REACT_MAX_ITERATIONS = 10        # Max tool calling iterations
REACT_RETRY_ON_ERROR = True      # Retry on tool errors

# Plan-Execute Agent
PLAN_MAX_STEPS = 10              # Max planning steps
PLAN_MAX_RETRIES = 2             # Retries on step failure
```

### Tool Configuration
```python
# Python Code Execution
PYTHON_EXECUTOR_MODE = "native"  # "native" or "nanocoder"
NANOCODER_TIMEOUT = 300          # Timeout in seconds

# Web Search
TAVILY_SEARCH_DEPTH = "advanced" # "basic" or "advanced"
WEBSEARCH_MAX_RESULTS = 5        # Max search results

# RAG
RAG_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RAG_CHUNK_SIZE = 512
RAG_INDEX_TYPE = "Flat"          # FAISS index type
```

### Security
```python
JWT_SECRET_KEY = "your-secret-key-here"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 168       # 7 days
```

## API Documentation

### Authentication

**Sign Up**
```http
POST /api/auth/signup
Content-Type: application/json

{
  "username": "user1",
  "password": "password123",
  "role": "user"
}
```

**Login**
```http
POST /api/auth/login
Content-Type: application/json

{
  "username": "user1",
  "password": "password123"
}

Response:
{
  "access_token": "eyJ0eXAi...",
  "token_type": "bearer"
}
```

### Chat Completions

**Basic Chat**
```http
POST /v1/chat/completions
Authorization: Bearer YOUR_TOKEN
Content-Type: application/json

{
  "model": "qwen2.5:14b",
  "messages": [
    {"role": "user", "content": "What is the weather like?"}
  ],
  "agent_type": "chat",
  "stream": false
}
```

**ReAct Agent with Tools**
```http
POST /v1/chat/completions
Authorization: Bearer YOUR_TOKEN
Content-Type: application/json

{
  "model": "qwen2.5:14b",
  "messages": [
    {"role": "user", "content": "Search for recent AI news and summarize"}
  ],
  "agent_type": "react",
  "available_tools": ["websearch"],
  "stream": false
}
```

**Streaming Response**
```http
POST /v1/chat/completions
Authorization: Bearer YOUR_TOKEN
Content-Type: application/json

{
  "model": "qwen2.5:14b",
  "messages": [{"role": "user", "content": "Tell me a story"}],
  "agent_type": "chat",
  "stream": true
}
```

### Session Management

**List Sessions**
```http
GET /api/chat/sessions
Authorization: Bearer YOUR_TOKEN
```

**Get Conversation History**
```http
GET /api/chat/history/{session_id}
Authorization: Bearer YOUR_TOKEN
```

**Delete Session**
```http
DELETE /api/chat/sessions/{session_id}
Authorization: Bearer YOUR_TOKEN
```

### Available Models

```http
GET /v1/models
Authorization: Bearer YOUR_TOKEN

Response:
{
  "object": "list",
  "data": [
    {
      "id": "qwen2.5:14b",
      "object": "model",
      "created": 1704067200,
      "owned_by": "ollama"
    }
  ]
}
```

## Agent Usage Examples

### Chat Agent (No Tools)
Simple conversational agent without tool calling capabilities.

```python
{
  "agent_type": "chat",
  "messages": [{"role": "user", "content": "Explain quantum computing"}]
}
```

### ReAct Agent (Reasoning + Acting)
Iteratively reasons and uses tools to solve problems.

```python
{
  "agent_type": "react",
  "available_tools": ["websearch", "python_coder"],
  "messages": [{
    "role": "user",
    "content": "Search for the current Bitcoin price and calculate its value in EUR"
  }]
}
```

### Plan-Execute Agent
Creates a plan, executes steps, and can re-plan on failures.

```python
{
  "agent_type": "plan_execute",
  "available_tools": ["websearch", "python_coder", "rag"],
  "messages": [{
    "role": "user",
    "content": "Research recent AI developments and create a summary report"
  }]
}
```

### Auto Agent
Automatically selects the best agent based on input complexity.

```python
{
  "agent_type": "auto",
  "available_tools": ["websearch", "python_coder"],
  "messages": [{"role": "user", "content": "Your question here"}]
}
```

## Tools

### 1. Web Search (websearch)
Searches the web using Tavily API and returns LLM-generated answers.

**Example:**
```json
{
  "agent_type": "react",
  "available_tools": ["websearch"],
  "messages": [{"role": "user", "content": "What are the latest developments in quantum computing?"}]
}
```

### 2. Python Code Execution (python_coder)
Executes Python code in an isolated workspace with two modes:

**Native Mode** - Direct subprocess execution:
```json
{
  "agent_type": "react",
  "available_tools": ["python_coder"],
  "messages": [{"role": "user", "content": "Calculate the first 10 Fibonacci numbers"}]
}
```

**Nanocoder Mode** - Autonomous code generation (requires nanocoder CLI):
```python
# In config.py
PYTHON_EXECUTOR_MODE = "nanocoder"
```

### 3. RAG (rag)
Retrieves relevant documents from indexed collections.

**Example:**
```json
{
  "agent_type": "react",
  "available_tools": ["rag"],
  "messages": [{
    "role": "user",
    "content": "Find information about user authentication in the docs"
  }]
}
```

### 4. Presentation Maker (ppt_maker)
Generates PDF and PPTX presentations from natural language instructions.

**Example:**
```json
{
  "agent_type": "react",
  "available_tools": ["ppt_maker"],
  "messages": [{
    "role": "user",
    "content": "Create a 5-slide presentation about machine learning basics"
  }]
}
```

## File Attachments

The system automatically extracts metadata from attached files without requiring tool calls:

**Supported Formats:**
- JSON: Structure, keys, sample data
- CSV/Excel: Headers, row count, sample rows
- Python: Imports, function/class definitions
- PDF/DOCX: Basic metadata
- Images: Dimensions, format

**Usage:**
```python
# File metadata is automatically injected into the system prompt
# The LLM can "see" file contents without calling any tool
```

## Development

### Project Structure
```
LLM_API/
├── backend/
│   ├── agents/          # Agent implementations
│   ├── api/routes/      # FastAPI route handlers
│   ├── core/            # Core services (LLM, database)
│   ├── models/          # Pydantic schemas
│   └── utils/           # Utilities (auth, file handling)
├── tools/
│   ├── web_search/      # Tavily web search
│   ├── python_coder/    # Code execution
│   ├── rag/             # Document retrieval
│   └── ppt_maker/       # Presentation generation
├── prompts/
│   ├── agents/          # Agent system prompts
│   └── tools/           # Tool-specific prompts
├── data/
│   ├── sessions/        # Conversation JSON files
│   ├── uploads/         # User persistent uploads
│   ├── scratch/         # Session temporary files
│   └── logs/            # LLM interaction logs
├── config.py            # Configuration settings
├── server.py            # Main API server
├── tools_server.py      # Tools API server
└── requirements.txt     # Python dependencies
```

### Adding New Tools

See [CLAUDE.md](CLAUDE.md) for detailed instructions on adding custom tools.

Quick overview:
1. Create tool implementation in `tools/{tool_name}/tool.py`
2. Add tool schema to `tools_config.py`
3. Add API endpoint in `backend/api/routes/tools.py`
4. Update configuration in `config.py`

### Database Schema

**Users Table:**
- `id`: Primary key
- `username`: Unique username
- `password_hash`: Bcrypt hash
- `role`: "admin" or "user"
- `created_at`: Timestamp

**Sessions Table:**
- `id`: Session ID (primary key)
- `username`: Owner username
- `created_at`: Timestamp
- `message_count`: Number of messages

**Conversations:** Stored as JSON files in `data/sessions/{session_id}.json`

### Logging

All LLM interactions are logged to `data/logs/prompts.log` via the LLM interceptor, including:
- Full prompts sent to the LLM
- LLM responses
- Timestamps
- Model information

## Troubleshooting

### Common Issues

**1. "Connection refused" when calling tools**
- Ensure the tools server is running on port 10006
- Always start `tools_server.py` BEFORE `server.py`

**2. Password validation errors**
- Bcrypt has a 72-byte limit (not 72 characters!)
- Multi-byte characters (emoji, CJK) consume multiple bytes
- See [USER_MANAGEMENT.md](USER_MANAGEMENT.md) for details

**3. "Model not found" errors**
- Check that Ollama/llama.cpp is running
- Verify the model name in `config.py` matches your installed model
- Try `ollama list` to see available models

**4. Web search not working**
- Verify `TAVILY_API_KEY` is set in `config.py`
- Check your Tavily API quota at https://tavily.com

**5. Python code execution timeout**
- Increase `NANOCODER_TIMEOUT` or `DEFAULT_TOOL_TIMEOUT` in `config.py`
- Long-running computations may need higher timeout values

**6. CORS errors from frontend**
- Update `CORS_ORIGINS` in `config.py` to include your frontend URL
- Restart the server after config changes

### Debug Mode

Enable detailed logging:
```python
# In config.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check LLM interaction logs:
```bash
tail -f data/logs/prompts.log
```

## Performance Optimization

### Memory Management
- Set `MAX_CONVERSATION_HISTORY` to limit context size
- Use smaller embedding models for RAG if memory is constrained
- Consider using llama.cpp for lower memory usage

### Speed
- Use `stream: true` for faster perceived response times
- Set `TAVILY_SEARCH_DEPTH: "basic"` for faster web searches
- Use haiku/smaller models for tool calling (configured in `TOOL_MODELS`)

### Scaling
- Deploy tools server on a separate machine (update `TOOLS_HOST` in config)
- Use a reverse proxy (nginx) for load balancing
- Consider using Redis for session storage instead of JSON files

## Security Considerations

1. **Change the JWT Secret** - Update `JWT_SECRET_KEY` in production
2. **Use HTTPS** - Always use TLS in production
3. **Rate Limiting** - Implement rate limiting for production deployments
4. **API Key Rotation** - Rotate Tavily API keys regularly
5. **File Upload Validation** - Validate and sanitize all uploaded files
6. **SQL Injection** - System uses parameterized queries (SQLAlchemy)
7. **Password Security** - Bcrypt hashing with 72-byte limit validation

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Support

For issues and questions:
- Check [CLAUDE.md](CLAUDE.md) for detailed architecture documentation
- Check [USER_MANAGEMENT.md](USER_MANAGEMENT.md) for authentication details
- Open an issue on GitHub

## Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/)
- [Ollama](https://ollama.ai/)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Tavily](https://tavily.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [sentence-transformers](https://www.sbert.net/)
