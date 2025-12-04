# LLM API

A comprehensive LLM-powered API backend with ReAct agent, tool execution, and multiple LLM backend support.

## Features

- **Multi-Backend Support**: Ollama and llama.cpp backends
- **ReAct Agent**: Intelligent agent with reasoning and tool use
- **Tools**: Web search, RAG retrieval, Python code execution, vision analysis
- **File Handling**: Support for CSV, Excel, PDF, DOCX, JSON, and images
- **Authentication**: JWT-based auth with role-based access control
- **Conversation Management**: Persistent conversation storage

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the backend server
python run_backend.py
```

## Configuration

Edit `backend/config/settings.py` or use environment variables to configure:
- LLM backend (Ollama/llama.cpp)
- Model paths and parameters
- API keys
- Server settings

---

## Changelog

### v2.0.1 - 2025-12-04
**Bug Fixes:**

1. **Circular Import Error**
   - **Issue**: `ImportError: cannot import name 'LLMFactory' from partially initialized module 'backend.utils.llm_factory'`
   - **Root Cause**: `backend/core/base_tool.py` imported `LLMFactory` at module load time, causing circular dependency when `backend.core` was initialized
   - **Fix**: Changed `LLMFactory` import in `base_tool.py` from top-level to lazy import (inside `_get_llm()` and `_get_coder_llm()` methods)
   - **Files Changed**: `backend/core/base_tool.py`

2. **Text Handler Syntax Error**
   - **Issue**: `SyntaxError` in `text_handler.py` due to literal newline character inside string
   - **Fix**: Changed `content.split('↵')` to `content.split('\n')` for proper newline splitting
   - **Files Changed**: `backend/core/file_handlers/text_handler.py`

3. **WebSearchTool Abstract Methods Missing**
   - **Issue**: `TypeError: Can't instantiate abstract class WebSearchTool without an implementation for abstract methods 'execute', 'validate_inputs'`
   - **Fix**: Added `execute()` and `validate_inputs()` methods to `WebSearchTool` class
   - **Files Changed**: `backend/tools/web_search.py`

4. **RAGRetrieverTool Missing validate_inputs**
   - **Issue**: `TypeError: Can't instantiate abstract class RAGRetrieverTool without an implementation for abstract method 'validate_inputs'`
   - **Fix**: Added `validate_inputs()` method to `RAGRetrieverTool` class
   - **Files Changed**: `backend/tools/rag_retriever.py`

5. **PythonCoderTool Abstract Methods Missing**
   - **Issue**: `TypeError: Can't instantiate abstract class PythonCoderTool without an implementation for abstract methods 'execute', 'validate_inputs'`
   - **Fix**: Added `execute()` and `validate_inputs()` methods to `PythonCoderTool` class
   - **Files Changed**: `backend/tools/python_coder.py`

6. **ReActAgentFactory Forward Reference Error**
   - **Issue**: `NameError: name 'ReActAgentFactory' is not defined` - class used before definition
   - **Fix**: Moved `agent_system = AgentOrchestrator()` instantiation after `ReActAgentFactory` class definition
   - **Files Changed**: `backend/agents/react_agent.py`

### v2.0.0 - 2025-12-03
- Refactored LLM backends into separate modules (`backend/core/llm_backends/`)
- Extracted `LlamaCppWrapper` and `LLMInterceptor` from `llm_factory.py`
- 61% file size reduction in `llm_factory.py`

### v1.3.0 - 2025-01-13
- Added `LLMFactory` for centralized LLM instance creation
- Support for multiple LLM configurations (standard, classifier, coder, vision)

