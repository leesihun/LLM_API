# LLM API - Dual Server Setup

## Overview

The system now runs **two separate servers** to avoid deadlock issues:

1. **Main API Server** (Port 1007) - Handles chat, authentication, and routing
2. **Tools API Server** (Port 1006) - Handles tool execution (websearch, python_coder, rag)

This architecture prevents deadlock when the chat agent needs to call tools internally.

## Quick Start

### Windows

Double-click `start_servers.bat` or run:
```batch
start_servers.bat
```

This will open two command windows:
- **Tools API** running on port 1006
- **Main API** running on port 1007

### Linux/Mac

```bash
chmod +x start_servers.sh
./start_servers.sh
```

Or manually start both servers in separate terminals:

**Terminal 1 - Tools API:**
```bash
python tools_server.py
```

**Terminal 2 - Main API:**
```bash
python server.py
```

## Manual Startup

If you prefer to run servers separately:

### 1. Start Tools Server (FIRST)
```bash
python tools_server.py
```

You should see:
```
======================================================================
LLM Tools API Server
======================================================================
Host: 0.0.0.0
Port: 1006
======================================================================

Available Tools:
  - Web Search (Tavily)
  - Python Code Executor
  - RAG (Retrieval Augmented Generation)

Health Check: http://localhost:1006/health

Starting server...
======================================================================
```

### 2. Start Main Server (SECOND)
```bash
python server.py
```

You should see:
```
======================================================================
LLM API - Backend Server
======================================================================
Host: 0.0.0.0
Port: 1007
LLM Backend: auto
Ollama: http://localhost:11434
Llama.cpp: http://localhost:8080
Default Model: gemma3:1b
======================================================================

API Documentation:
  - Swagger UI: http://localhost:1007/docs
  - ReDoc: http://localhost:1007/redoc
  - Health Check: http://localhost:1007/health

Starting server...
======================================================================
```

## Health Checks

Verify both servers are running:

```bash
# Check Tools API
curl http://localhost:1006/health

# Check Main API
curl http://localhost:1007/health
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│          Client (API Request)                   │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
         ┌─────────────────────┐
         │   Main API Server   │
         │     Port 1007       │
         │  - Chat             │
         │  - Auth             │
         │  - Sessions         │
         └──────────┬──────────┘
                    │
                    │ (Tool Call via HTTP)
                    ▼
         ┌─────────────────────┐
         │  Tools API Server   │
         │     Port 1006       │
         │  - websearch        │
         │  - python_coder     │
         │  - rag              │
         └─────────────────────┘
```

## Why Two Servers?

**Problem:** When the main server receives a chat request, the ReAct agent needs to call tools. If tools are on the same server, making an HTTP request back to itself causes a **deadlock** - the server is waiting for itself to respond.

**Solution:** Run tools on a **separate port (1006)**. The main server can make HTTP requests to port 1006 without blocking itself.

## Configuration

Edit `config.py` to change ports:

```python
SERVER_PORT = 1007  # Main API server
TOOLS_PORT = 1006   # Tools API server
```

## Troubleshooting

### Port Already in Use

If you see "Address already in use" errors:

**Windows:**
```batch
netstat -ano | findstr :1006
netstat -ano | findstr :1007
taskkill /PID <process_id> /F
```

**Linux/Mac:**
```bash
lsof -i :1006
lsof -i :1007
kill -9 <process_id>
```

### Tools Server Not Responding

Make sure the tools server is started **before** the main server. The main server needs the tools server to be available for agent operations.

### Connection Refused

Check that both servers are running:
```bash
curl http://localhost:1006/health
curl http://localhost:1007/health
```

## Logs

All LLM interactions and tool executions are logged to:
```
data/logs/prompts.log
```

Console output shows real-time execution:
- `[LLM]` - Language model calls
- `[TOOL CALL]` - Tool invocations
- `[TAVILY API]` - Web search API calls
- `[PYTHON]` - Python code execution
- `[RESULT]` - Tool results
- `[ERROR]` - Errors

## API Documentation

Once both servers are running:

- **Main API Docs:** http://localhost:1007/docs
- **Tools API Docs:** http://localhost:1006/docs

## Stopping Servers

### Windows
Close both command windows or press `Ctrl+C` in each window.

### Linux/Mac
If using `start_servers.sh`, press `Ctrl+C` once to stop both servers.

If running manually, press `Ctrl+C` in each terminal.
