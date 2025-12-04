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

### v2.0.3 - 2025-12-04
**Bug Fixes:**

1. **Ollama Model Configuration Mismatch**
   - **Issue**: `OLLAMA CONNECTION FAILED` during startup even when Ollama is running
   - **Root Cause**: Default model settings (`gemma3:12b-it-q8_0`) did not match installed Ollama models
   - **Fix**: Updated default model settings to use available models (`gpt-oss:20b`, `llama3.2-vision:11b`)
   - **Files Changed**: `backend/config/settings.py`

### v2.0.2 - 2025-01-13
**Bug Fixes:**

1. **Bcrypt AttributeError**
   - **Issue**: `AttributeError: module 'bcrypt' has no attribute '__about__'`
   - **Root Cause**: bcrypt 5.0.0 removed the `__about__` module that passlib 1.7.4 tries to access during initialization
   - **Fix**: Downgraded bcrypt from 5.0.0 to 4.0.1 for compatibility with passlib 1.7.4
   - **Files Changed**: `requirements.txt`

2. **StructuredLogger TypeError**
   - **Issue**: `TypeError: StructuredLogger.info() takes 2 positional arguments but 4 were given`
   - **Root Cause**: `StructuredLogger` methods in `logging_utils.py` didn't accept `*args` for old-style Python logging format strings (e.g., `logger.info("Message %s", value)`)
   - **Fix**: Updated `debug()`, `info()`, `warning()`, `error()`, and `critical()` methods to accept `*args` and `**kwargs` and pass them through to the underlying logger
   - **Files Changed**: `backend/utils/logging_utils.py`, `backend/tools/file_analyzer/__init__.py`

3. **Ollama Connection Protocol Error**
   - **Issue**: `Request URL is missing an 'http://' or 'https://' protocol`
   - **Root Cause**: `ollama_host` setting could be overridden by environment variable without protocol prefix (e.g., `OLLAMA_HOST=127.0.0.1:11434`)
   - **Fix**: Added `field_validator` to automatically prepend `http://` if protocol is missing from `ollama_host` setting
   - **Files Changed**: `backend/config/settings.py`

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

