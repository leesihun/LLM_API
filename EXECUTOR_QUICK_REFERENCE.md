# Executor Module - Quick Reference Guide

## Import Reference

### Main Executor
```python
from backend.tools.python_coder.executor import CodeExecutor

# Create executor
executor = CodeExecutor(
    timeout=30,
    max_memory_mb=512,
    execution_base_dir="./data/scratch",
    use_persistent_repl=True
)

# Execute code
result = executor.execute(
    code="print('Hello World')",
    input_files={"path/to/file.csv": "file.csv"},
    session_id="my-session",
    stage_name="exec1"
)
```

### Import Validator
```python
from backend.tools.python_coder.executor import ImportValidator

validator = ImportValidator()
is_valid, issues = validator.validate(code)
```

### Sandbox Configuration
```python
from backend.tools.python_coder.executor import SandboxConfig

config = SandboxConfig(
    timeout=30,
    max_memory_mb=512,
    execution_base_dir="./data/scratch",
    use_persistent_repl=True
)

# Check file type
if config.is_file_type_supported("data.csv"):
    print("CSV files supported")
```

### REPL Manager
```python
from backend.tools.python_coder.executor import REPLManager

manager = REPLManager(timeout=30)
repl = manager.get_or_create("session-123", execution_dir)
result = repl.execute(code)
```

### Utility Functions
```python
from backend.tools.python_coder.executor import utils

# Format file size
size_str = utils.format_file_size(1536)  # "1.5 KB"

# Get execution environment
env = utils.get_execution_env()  # {"PYTHONIOENCODING": "utf-8", ...}

# Validate file path
if utils.validate_file_path("/path/to/file.csv"):
    print("File exists")
```

### Constants
```python
from backend.tools.python_coder.executor import (
    BLOCKED_IMPORTS,
    SUPPORTED_FILE_TYPES
)

print(f"Blocked: {BLOCKED_IMPORTS}")
print(f"Supported: {SUPPORTED_FILE_TYPES}")
```

## Module Structure

```
executor/
├── __init__.py          # Public API exports
├── core.py              # CodeExecutor main class
├── sandbox.py           # Security configuration
├── import_validator.py  # Import validation
├── repl_manager.py      # Persistent REPL
└── utils.py             # Helper functions
```

## Key Classes

### CodeExecutor
**Location:** `executor/core.py`  
**Purpose:** Main code execution orchestrator

**Key Methods:**
- `execute(code, input_files, session_id, stage_name)` - Execute Python code
- `validate_imports(code)` - Validate code imports
- `validate_file_type(file_path)` - Check file type support
- `cleanup_session(session_id)` - Clean up session data
- `get_stats()` - Get executor statistics

### SandboxConfig
**Location:** `executor/sandbox.py`  
**Purpose:** Security and execution configuration

**Key Methods:**
- `is_file_type_supported(file_path)` - Check file type
- `get_execution_dir(session_id)` - Get execution directory
- `to_dict()` - Export configuration

### ImportValidator
**Location:** `executor/import_validator.py`  
**Purpose:** Validate code imports for security

**Key Methods:**
- `validate(code)` - Validate code imports (returns tuple)
- `add_blocked_import(module)` - Add blocked module
- `remove_blocked_import(module)` - Remove blocked module
- `is_blocked(module)` - Check if module is blocked

### PersistentREPL
**Location:** `executor/repl_manager.py`  
**Purpose:** Single persistent REPL instance

**Key Methods:**
- `start()` - Start REPL process
- `execute(code)` - Execute code in REPL
- `stop()` - Stop REPL process
- `is_alive()` - Check REPL health

### REPLManager
**Location:** `executor/repl_manager.py`  
**Purpose:** Manage multiple REPL instances

**Key Methods:**
- `get_or_create(session_id, execution_dir)` - Get/create REPL
- `cleanup_session(session_id)` - Clean up session REPL
- `cleanup_all()` - Clean up all REPLs
- `get_stats()` - Get manager statistics

## Backward Compatibility

All existing code continues to work:

```python
# Old import (still works)
from backend.tools.python_coder.executor import CodeExecutor

# Old alias (still works)
from backend.tools.python_coder.executor import PythonExecutor

# Old constants (still accessible)
from backend.tools.python_coder.executor import SUPPORTED_FILE_TYPES
```

## Migration Notes

No migration required! All existing imports work without changes.

The original monolithic file is preserved at:
`backend/tools/python_coder/legacy/executor_monolithic.py`

## Testing

Run comprehensive tests:
```bash
python -c "
from backend.tools.python_coder.executor import CodeExecutor
executor = CodeExecutor(timeout=5, use_persistent_repl=False)
result = executor.execute('print(\"Test\")')
assert result['success']
print('✓ Tests passed')
"
```

## Common Patterns

### Basic Execution
```python
from backend.tools.python_coder.executor import CodeExecutor

executor = CodeExecutor()
result = executor.execute("print('Hello')")

if result['success']:
    print(result['output'])
else:
    print(f"Error: {result['error']}")
```

### With File Input
```python
result = executor.execute(
    code="import pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.head())",
    input_files={"/full/path/to/data.csv": "data.csv"},
    session_id="my-session"
)
```

### With Import Validation
```python
from backend.tools.python_coder.executor import CodeExecutor, ImportValidator

validator = ImportValidator()
is_valid, issues = validator.validate(code)

if is_valid:
    executor = CodeExecutor()
    result = executor.execute(code)
else:
    print(f"Validation failed: {issues}")
```

### Using REPL Mode
```python
executor = CodeExecutor(use_persistent_repl=True)

# First execution (starts REPL)
result1 = executor.execute(code1, session_id="sess1")

# Second execution (reuses REPL - faster!)
result2 = executor.execute(code2, session_id="sess1")

# Cleanup
executor.cleanup_session("sess1")
```

## Performance Tips

1. **Use REPL mode for multiple executions**: Set `use_persistent_repl=True`
2. **Reuse session IDs**: File caching and REPL reuse speed up retries
3. **Set appropriate timeouts**: Default 30s, adjust based on workload
4. **Clean up sessions**: Call `cleanup_session()` when done

## Security Features

- **Blocked imports**: socket, subprocess, eval, exec, pickle, etc.
- **AST-based validation**: No code execution during validation
- **Sandboxed execution**: Isolated subprocess/REPL environment
- **Timeout enforcement**: Prevents infinite loops
- **Error detection**: Enhanced pattern matching for error detection

---

**Version:** 2.0.0  
**Last Updated:** 2025-11-20  
**Status:** Production Ready
