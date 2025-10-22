# HE Team LLM Assistant - Setup Guide

Complete setup instructions for the agentic AI backend with LangGraph.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the Application](#running-the-application)
5. [Architecture Overview](#architecture-overview)
6. [API Documentation](#api-documentation)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

1. **Python 3.10+**
   ```bash
   python --version  # Should be 3.10 or higher
   ```

2. **Ollama**
   - Download from: https://ollama.ai
   - Start Ollama service:
     ```bash
     ollama serve
     ```
   - Pull the model:
     ```bash
     ollama pull gpt-oss:20b
     ```

3. **Tavily API Key**
   - Sign up at: https://tavily.com
   - Get your API key from the dashboard

---

## Installation

### 1. Clone or Navigate to Project

```bash
cd /path/to/LLM_API
```

### 2. Create Virtual Environment

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

### 1. Create Environment File

Copy the example environment file:

```bash
cp .env.example .env
```

### 2. Edit Configuration

Open `.env` and configure all required variables:

```env
# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
SECRET_KEY=your-strong-secret-key-here

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gpt-oss:20b
OLLAMA_TIMEOUT=600000
OLLAMA_NUM_CTX=8192
OLLAMA_TEMPERATURE=0.7
OLLAMA_TOP_P=0.9
OLLAMA_TOP_K=40

# Tavily Search API
TAVILY_API_KEY=your-tavily-api-key-here

# Vector Database
VECTOR_DB_TYPE=faiss
VECTOR_DB_PATH=./backend/storage/vectordb
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Storage Paths
USERS_PATH=./backend/config/users.json
SESSIONS_PATH=./backend/config/sessions.json
CONVERSATIONS_PATH=./conversations
UPLOADS_PATH=./backend/storage/uploads

# Authentication
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# Logging
LOG_LEVEL=INFO
LOG_FILE=./backend/logs/app.log
```

### 3. Important Configuration Notes

- **SECRET_KEY**: Generate a secure random key for production
- **TAVILY_API_KEY**: Required for web search functionality
- **OLLAMA_HOST**: Must match your Ollama installation
- **OLLAMA_MODEL**: Must be pulled via `ollama pull` first

---

## Running the Application

### Option 1: Run Backend Only

**Linux/Mac:**
```bash
chmod +x run_backend.sh
./run_backend.sh
```

**Windows:**
```cmd
run_backend.bat
```

**Or directly with Python:**
```bash
python server.py
```

### Option 2: Run Backend + Frontend

**Linux/Mac:**
```bash
chmod +x start_all.sh
./start_all.sh
```

**Windows:**
```cmd
start_all.bat
```

### Access Points

- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Frontend UI**: http://localhost:3000 (if started)

---

## Architecture Overview

### Project Structure

```
LLM_API/
├── backend/
│   ├── api/              # FastAPI routes and app
│   │   ├── app.py       # Main application
│   │   └── routes.py    # API endpoints
│   ├── config/          # Configuration files
│   │   ├── settings.py  # Settings manager
│   │   └── users.json   # User database
│   ├── core/            # Core logic
│   │   └── agent_graph.py  # LangGraph workflow
│   ├── models/          # Data models
│   │   └── schemas.py   # Pydantic schemas
│   ├── storage/         # Data persistence
│   │   └── conversation_store.py
│   ├── tasks/           # Task handlers
│   │   ├── chat_task.py      # Simple chat
│   │   └── agentic_task.py   # Agentic workflow
│   ├── tools/           # External tools
│   │   ├── web_search.py     # Tavily + fallback
│   │   └── rag_retriever.py  # Document RAG
│   └── utils/           # Utilities
│       └── auth.py      # Authentication
├── frontend/            # Frontend static files
├── conversations/       # Saved conversations
├── .env                 # Configuration (create from .env.example)
├── requirements.txt     # Python dependencies
├── server.py            # Server launcher
└── SETUP.md             # This file
```

### LangGraph Workflow

The agentic workflow follows this pipeline:

```
User Prompt
    ↓
[1] Planning Node
    ├─ Analyze query
    └─ Create execution plan
    ↓
[2] Tool Selection Node
    ├─ Determine required tools
    └─ Choose: chat, search, RAG, or agentic
    ↓
[3] Tool Execution Nodes
    ├─ Web Search (Tavily API)
    ├─ RAG Retriever (FAISS/Chroma)
    └─ Execute in parallel when possible
    ↓
[4] Reasoning Node
    ├─ Combine retrieved information
    └─ Generate response with LLM
    ↓
[5] Verification Node
    ├─ Check response quality
    └─ Loop back if needed (max 3 iterations)
    ↓
Final Output
```

### Task Types

1. **Normal Chat** (`chat_task.py`)
   - Simple conversational AI
   - Optional conversation memory
   - Direct LLM invocation

2. **Agentic Workflow** (`agentic_task.py`)
   - Multi-step reasoning
   - Tool integration (search, RAG)
   - Self-verification loops
   - Automatically triggered by keywords

---

## API Documentation

### Authentication

#### Login
```http
POST /api/auth/login
Content-Type: application/json

{
  "username": "guest",
  "password": "guest_test1"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "user": {
    "username": "guest",
    "role": "guest"
  }
}
```

#### Get Current User
```http
GET /api/auth/me
Authorization: Bearer <token>
```

### Chat Completions (OpenAI Compatible)

```http
POST /v1/chat/completions
Authorization: Bearer <token>
Content-Type: application/json

{
  "model": "gpt-oss:20b",
  "messages": [
    {
      "role": "user",
      "content": "What is the weather like today?"
    }
  ],
  "session_id": "optional-session-id"
}
```

**Response:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "gpt-oss:20b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I'll search for current weather information..."
      },
      "finish_reason": "stop"
    }
  ],
  "x_session_id": "generated-session-id"
}
```

### File Upload (RAG)

```http
POST /api/files/upload
Authorization: Bearer <token>
Content-Type: multipart/form-data

file: <binary-file-data>
```

**Supported formats:** PDF, DOCX, TXT, JSON

### List Models

```http
GET /v1/models
Authorization: Bearer <token>
```

---

## Default Users

The system includes two default users:

| Username | Password      | Role  |
|----------|---------------|-------|
| guest    | guest_test1   | guest |
| admin    | administrator | admin |

**Important:** Change these credentials in production!

Edit `backend/config/users.json` or use password hashing:

```python
from backend.utils.auth import get_password_hash
print(get_password_hash("your-new-password"))
```

---

## Troubleshooting

### Backend Won't Start

1. **Missing .env file**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Ollama not running**
   ```bash
   ollama serve
   ollama pull gpt-oss:20b
   ```

3. **Port already in use**
   - Change `SERVER_PORT` in `.env`
   - Or kill process using port 8000

### Tavily Search Fails

- Verify API key in `.env`
- Check internet connection
- System will fall back to websearch_ts

### RAG Not Working

1. **Check embedding model download**
   - First run downloads sentence-transformers model
   - Requires internet connection

2. **Upload file via API**
   ```bash
   curl -X POST http://localhost:8000/api/files/upload \
     -H "Authorization: Bearer <token>" \
     -F "file=@document.pdf"
   ```

### Authentication Issues

- Check `.env` SECRET_KEY is set
- Verify user credentials in `backend/config/users.json`
- Token expires after JWT_EXPIRATION_HOURS

### Out of Memory

- Reduce `OLLAMA_NUM_CTX` in `.env`
- Use smaller embedding model
- Switch to Chroma instead of FAISS

---

## Development Tips

### Enable Debug Logging

```env
LOG_LEVEL=DEBUG
```

### Test API with curl

```bash
# Login
TOKEN=$(curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"guest","password":"guest_test1"}' \
  | jq -r '.access_token')

# Chat
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss:20b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Add New Tools

1. Create tool in `backend/tools/your_tool.py`
2. Import in `backend/core/agent_graph.py`
3. Add node to LangGraph workflow
4. Update tool selection logic

### Customize LLM Parameters

Edit `.env`:
```env
OLLAMA_TEMPERATURE=0.8  # Higher = more creative
OLLAMA_TOP_P=0.95       # Nucleus sampling
OLLAMA_TOP_K=50         # Top-K sampling
```

---

## Production Deployment

### Security Checklist

- [ ] Change SECRET_KEY to strong random value
- [ ] Update default user passwords
- [ ] Configure CORS allowed origins
- [ ] Use HTTPS (reverse proxy)
- [ ] Set secure JWT expiration
- [ ] Enable rate limiting
- [ ] Configure firewall rules

### Recommended Setup

1. **Reverse Proxy (Nginx)**
   ```nginx
   location /api {
       proxy_pass http://localhost:8000;
   }
   ```

2. **Process Manager (systemd/supervisor)**
   ```ini
   [program:llm_api]
   command=/path/to/venv/bin/python server.py
   directory=/path/to/LLM_API
   autostart=true
   autorestart=true
   ```

3. **Environment Variables**
   - Use secrets management (Vault, AWS Secrets)
   - Never commit `.env` to version control

---

## Support

For issues and questions:
- Check logs: `backend/logs/app.log`
- Review API docs: http://localhost:8000/docs
- Verify configuration in `.env`

---

**Built with LangGraph, FastAPI, and Ollama**
