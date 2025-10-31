# LLM_API

AI-powered Python code generation and execution API with iterative verification.

## Version History

### v1.0.2 (2025-10-31)
**Python Executor - Fixed NameError for Built-in Functions**
- **Fixed**: `NameError: name 'print' is not defined` when executing code via PYTHON_CODE tool
- **Reason**: `__builtins__` can be either a dict or module depending on execution context; the code only handled the module case
- **Impact**: Built-in functions like `print`, `len`, `str`, `int`, etc. now work correctly in executed code
- **Files Modified**: `backend/tools/python_executor.py`
- **Details**: 
  - Updated `_create_safe_globals()` method to handle both dict and module types for `__builtins__`
  - Adds type checking: uses dict access when `__builtins__` is dict, getattr when it's a module
  - Ensures all non-forbidden built-in functions are available in the execution environment

### v1.0.1 (2025-10-31)
**Python Coder Tool - Output Format Simplification**
- **Changed**: Removed JSON output requirement from code verification prompt
- **Reason**: Eliminated inconsistency between code generation (which instructed to use prints) and verification (which checked for JSON output)
- **Impact**: Generated code now consistently uses simple `print()` statements for output instead of JSON formatting
- **Files Modified**: `backend/tools/python_coder_tool.py`
- **Details**: 
  - Updated verification prompt "OUTPUT FORMAT COMPLIANCE" section to "OUTPUT FORMAT"
  - Now verifies that code uses print() statements, outputs are clear, and errors are communicated
  - Removes unnecessary complexity of JSON formatting in generated code

### v1.0.0 (Initial)
- Python code generation with LLM (Ollama)
- Iterative code verification and modification
- Secure Python code execution with sandboxing
- File upload and processing support
- User authentication and session management
- Conversation history storage
- Web search integration
- RAG (Retrieval Augmented Generation) capabilities

