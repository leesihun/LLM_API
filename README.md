# LLM API Project

Python-based LLM API server with agentic workflow capabilities.

## Overview

This project provides an AI-powered API service that leverages Large Language Models (LLMs) through Ollama, with sophisticated agentic capabilities including web search, document retrieval (RAG), and Python code generation/execution.

## Features

- **Multi-Agent Architecture**: ReAct and Plan-Execute patterns for complex task handling
- **Python Code Generation**: AI-driven code generation with iterative verification and execution
- **Web Search Integration**: Real-time information retrieval via Tavily API
- **RAG (Retrieval Augmented Generation)**: Document-based context retrieval
- **Session Management**: User authentication and conversation history
- **File Upload Support**: Process various file formats (CSV, Excel, JSON, PDF, etc.)
- **Security**: Sandboxed code execution with import restrictions and timeouts

## Architecture

```
LLM_API/
├── backend/
│   ├── api/              # FastAPI routes and application
│   ├── config/           # Configuration and settings
│   ├── core/             # Core agentic graph logic
│   ├── models/           # Pydantic schemas
│   ├── storage/          # Conversation persistence
│   ├── tasks/            # Task handlers (ReAct, Plan-Execute, Smart Agent)
│   ├── tools/            # Tool implementations
│   │   ├── python_coder_tool.py    # Unified Python code generation & execution
│   │   ├── rag_retriever.py        # Document retrieval
│   │   └── web_search.py           # Web search
│   └── utils/            # Authentication utilities
├── frontend/
│   └── static/           # Web interface
├── data/
│   ├── conversations/    # Conversation history
│   ├── uploads/          # User uploaded files
│   ├── scratch/          # Temporary code execution
│   └── logs/             # Application logs
└── requirements.txt
```

## Installation

### Prerequisites

- Python 3.9+
- Ollama (for LLM inference)
- Tavily API key (for web search)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd LLM_API
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure Ollama:
```bash
# Install Ollama from https://ollama.ai/
# Pull required models
ollama pull gemma3:12b
ollama pull gpt-oss:20b
ollama pull bge-m3:latest
```

5. Configure settings:
Edit `backend/config/settings.py` to customize:
- Ollama host and model
- API keys (Tavily)
- Server host and port
- Python code execution limits

## Usage

### Start the Server

```bash
python run_backend.py
```

Or for frontend:
```bash
python run_frontend.py
```

Default server runs at: `http://0.0.0.0:1007`

### API Endpoints

- `POST /api/chat` - Send chat message with agentic capabilities
- `POST /api/upload` - Upload files for processing
- `POST /api/auth/login` - User authentication
- `GET /api/conversations` - Retrieve conversation history

### Example Request

```python
import requests

response = requests.post(
    "http://localhost:1007/api/chat",
    json={
        "message": "Analyze the uploaded CSV file and show statistics",
        "session_id": "test-session",
        "user_id": "admin"
    }
)
```

## Configuration

Key settings in `backend/config/settings.py`:

```python
# Ollama Configuration
ollama_host: str = 'http://127.0.0.1:11434'
ollama_model: str = 'gemma3:12b'
ollama_num_ctx: int = 4096

# Python Code Execution
python_code_enabled: bool = True
python_code_timeout: int = 300
python_code_max_memory: int = 5120
python_code_max_iterations: int = 3

# Security
secret_key: str = 'your-secret-key-here'
jwt_expiration_hours: int = 24
```

## Python Code Execution

The system can generate and execute Python code safely:

### Features
- **Unified Tool**: Single integrated tool for code generation, verification, and execution
- **Iterative Verification**: Up to 3 iterations to ensure code answers user's question
- **Execution Retry**: Up to 5 retry attempts on execution failure with auto-fixing
- **Sandboxed Execution**: Isolated subprocess execution with security restrictions
- **File Support**: Process multiple file formats (CSV, Excel, JSON, etc.)

### Security Measures
- Import restrictions (blocked: socket, subprocess, eval, exec, etc.)
- Execution timeout (default: 300 seconds)
- Memory limits
- Isolated temporary directories
- Static code analysis before execution

## Version History

### Version 1.2.0 (2024-10-31)

**Major Changes: Python Code Tool Unification**

1. **Unified Python Code Tool**
   - Merged `python_coder_tool.py` and `python_executor_engine.py` into a single module
   - Removed redundant `python_executor.py` (unused legacy code)
   - Simplified architecture with `CodeExecutor` class for low-level execution
   - Single `PythonCoderTool` class for high-level code generation and orchestration

2. **Verification System Redesign**
   - **Reduced scope**: Verifier now focuses ONLY on "Does the code answer the user's question?"
   - **Reduced iterations**: Maximum verification iterations reduced from 10 to 3
   - Removed overly detailed checks (performance, code quality, file handling details)
   - Simplified verification prompt for faster and more focused validation

3. **Execution Retry Logic**
   - Added automatic retry mechanism for failed code execution
   - Maximum 5 execution attempts with auto-fixing between retries
   - New `_fix_execution_error()` method uses LLM to analyze and fix runtime errors
   - Execution history tracking for debugging

4. **Code Quality Improvements**
   - Consolidated constants (`BLOCKED_IMPORTS`, `SUPPORTED_FILE_TYPES`)
   - Better separation of concerns (CodeExecutor vs PythonCoderTool)
   - Improved logging and error messages
   - Backward compatibility exports for existing imports

5. **Configuration Updates**
   - `settings.py`: `python_code_max_iterations` default changed from 10 to 3
   - Removed obsolete `python_executor` references from `React.py`
   - Consolidated `PYTHON_CODE` and `PYTHON_CODER` tool types into single `PYTHON_CODER`

6. **File Cleanup**
   - Deleted: `backend/tools/python_executor_engine.py`
   - Deleted: `backend/tools/python_executor.py` (unused)
   - Updated: `backend/tasks/React.py` (removed python_executor imports)
   - Updated: `backend/tools/python_coder_tool.py` (unified implementation)

**Benefits:**
- Faster verification (3 iterations vs 10)
- More reliable execution (5 retry attempts with auto-fixing)
- Simpler codebase (1 file instead of 3)
- Focused verification on core goal (answering user's question)
- Better error recovery through retry logic

**Breaking Changes:**
- `python_executor_engine.PythonExecutor` moved to `python_coder_tool.CodeExecutor`
- `ToolName.PYTHON_CODE` removed, use `ToolName.PYTHON_CODER` instead
- Import changes required in custom code using these tools

---

### Version 1.1.0 (Previous)
- Multi-agent architecture implementation
- ReAct pattern for reasoning and acting
- Plan-Execute workflow for complex tasks
- Web search and RAG integration

### Version 1.0.0 (Initial)
- Basic LLM API server
- Ollama integration
- Simple chat functionality

## Development

### Project Structure

- **API Layer** (`backend/api/`): FastAPI routes and HTTP handling
- **Agent Layer** (`backend/tasks/`, `backend/core/`): Agentic reasoning and task execution
- **Tool Layer** (`backend/tools/`): External integrations (web, RAG, code execution)
- **Storage Layer** (`backend/storage/`): Data persistence

### Adding New Tools

1. Create tool in `backend/tools/`
2. Register in `backend/tasks/React.py` or `backend/core/agent_graph.py`
3. Add to `ToolName` enum if needed

### Testing

```bash
# Run API tests
python -m pytest tests/

# Manual testing with Jupyter notebook
jupyter notebook API_examples.ipynb
```

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   - Ensure Ollama is running: `ollama serve`
   - Check `ollama_host` in settings

2. **Code Execution Timeout**
   - Increase `python_code_timeout` in settings
   - Check for infinite loops in generated code

3. **Import Errors in Generated Code**
   - Review `BLOCKED_IMPORTS` in `python_coder_tool.py`
   - Required packages must be installed in environment

4. **Memory Issues**
   - Reduce `ollama_num_ctx` for lower memory usage
   - Increase `python_code_max_memory` if needed

## Security Considerations

- **Never expose** the server directly to the internet without proper authentication
- **Change** `secret_key` in production
- **Review** `BLOCKED_IMPORTS` before modifying
- **Monitor** code execution logs for suspicious activity
- **Limit** file upload sizes and types

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes with clear messages
4. Submit a pull request

## License

[Your License Here]

## Contact

[Your Contact Information]

---

**Last Updated**: October 31, 2024
**Version**: 1.2.0

