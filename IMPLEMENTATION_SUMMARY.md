# Python Executor Mode Implementation Summary

## Overview

Successfully implemented a **switchable Python code execution backend** for the LLM_API project. The system now supports two execution modes:

1. **Native Mode** - Direct subprocess execution (existing implementation)
2. **OpenInterpreter Mode** - AI-powered code execution with automatic error correction

## Implementation Details

### Architecture

```
┌─────────────────────────────────────────────┐
│  Backend API (/api/tools/python_coder)      │
└──────────────────┬──────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────┐
│  Factory Pattern (tools/python_coder)       │
│  - get_python_executor(session_id)          │
│  - PythonCoderTool (backward compatible)    │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ↓                     ↓
┌──────────────────┐  ┌──────────────────────┐
│ NativePythonExec │  │ OpenInterpreterExec  │
│ (subprocess)     │  │ (with auto-retry)    │
└──────────────────┘  └──────────────────────┘
        │                     │
        └──────────┬──────────┘
                   ↓
        ┌─────────────────────┐
        │  BasePythonExecutor │
        │  (interface)        │
        └─────────────────────┘
```

### Key Features

#### 1. Configuration-Based Switching
- **Single config variable** controls mode: `PYTHON_EXECUTOR_MODE` in config.py
- Valid values: `"native"` or `"openinterpreter"`
- No code changes needed to switch modes

#### 2. Native Mode (Default)
- Direct Python subprocess execution
- No external dependencies
- Fast and reliable
- Existing implementation preserved

#### 3. OpenInterpreter Mode
- Wraps Open Interpreter library
- **Automatic retry on errors** (up to 3 attempts by default)
- **Error context accumulation** across retries
- Uses same workspace as native mode (session-based)
- Configurable via prompts/tools/python_coder_openinterpreter.txt

#### 4. Strict Error Handling
- **No fallbacks** - If OpenInterpreter mode fails, returns error (doesn't fall back to native)
- **ImportError on missing dependency** - Clear error message if Open Interpreter not installed
- **ValueError on invalid mode** - Strict validation of PYTHON_EXECUTOR_MODE

#### 5. Shared Interface
- Both executors implement `BasePythonExecutor`
- Same workspace directory (data/scratch/{session_id}/)
- Identical return format (ToolResponse compatible)
- Backward compatible with existing code

## Files Created/Modified

### New Files

1. **tools/python_coder/base.py**
   - Abstract base class for executors
   - Defines standard interface

2. **tools/python_coder/native_tool.py**
   - Native executor implementation
   - Renamed from tool.py

3. **tools/python_coder/openinterpreter_tool.py**
   - OpenInterpreter wrapper with auto-retry
   - Error context management
   - Prompt-based system instructions

4. **prompts/tools/python_coder_openinterpreter.txt**
   - System prompt for OpenInterpreter
   - Error correction guidelines
   - Security guidelines
   - Output format instructions

5. **test_executor_modes.py**
   - Comprehensive test suite
   - Tests both modes
   - Factory validation tests

### Modified Files

1. **config.py**
   - Added `PYTHON_EXECUTOR_MODE` configuration
   - Added `PYTHON_CODER_MAX_RETRIES` (default: 3)
   - Added OpenInterpreter-specific settings

2. **tools/python_coder/__init__.py**
   - Implemented factory pattern
   - `get_python_executor()` function
   - Backward-compatible `PythonCoderTool` wrapper

3. **backend/api/routes/tools.py**
   - Added logging for executor mode
   - No functional changes (already compatible)

4. **requirements.txt**
   - Added `open-interpreter>=0.2.0` (optional dependency)

## Configuration Reference

### config.py Settings

```python
# Execution mode switch
PYTHON_EXECUTOR_MODE: Literal["native", "openinterpreter"] = "native"

# Native executor settings (existing)
PYTHON_EXECUTOR_TIMEOUT = 30
PYTHON_EXECUTOR_MAX_OUTPUT_SIZE = 1024 * 1024
PYTHON_WORKSPACE_DIR = SCRATCH_DIR

# OpenInterpreter settings (new)
PYTHON_CODER_MAX_RETRIES = 3  # Auto-retry attempts
PYTHON_CODER_OPENINTERPRETER_OFFLINE = True  # No external API calls
PYTHON_CODER_OPENINTERPRETER_AUTO_RUN = True  # Auto-execute code
PYTHON_CODER_OPENINTERPRETER_SAFE_MODE = False  # Safe mode (experimental)

# Shared model configuration (existing)
TOOL_MODELS = {
    "python_coder": "gpt-oss:20b"  # Used by both native and OI
}
```

## Usage

### Switching Modes

**To use Native Mode (default):**
```python
# config.py
PYTHON_EXECUTOR_MODE = "native"
```

**To use OpenInterpreter Mode:**
```python
# config.py
PYTHON_EXECUTOR_MODE = "openinterpreter"
```

Then restart the server:
```bash
python server.py
```

### Installing OpenInterpreter (Optional)

Only needed if using OpenInterpreter mode:
```bash
pip install open-interpreter
```

### Testing

Run the test suite:
```bash
python test_executor_modes.py
```

Expected output:
```
================================================================================
TEST SUMMARY
================================================================================
Native Mode............................. PASSED
OpenInterpreter Mode.................... PASSED
Factory Validation...................... PASSED

Total: 3/3 tests passed

All tests passed!
```

## Error Handling

### Strict Mode Behavior

1. **Invalid Mode**
   ```python
   PYTHON_EXECUTOR_MODE = "invalid"
   # Raises: ValueError: Invalid PYTHON_EXECUTOR_MODE: 'invalid'.
   #         Must be 'native' or 'openinterpreter'
   ```

2. **OpenInterpreter Not Installed**
   ```python
   PYTHON_EXECUTOR_MODE = "openinterpreter"
   # Raises: ImportError: Open Interpreter is not installed.
   #         Install it with: pip install open-interpreter
   #         Or set PYTHON_EXECUTOR_MODE='native' in config.py
   ```

3. **Execution Failures**
   - No fallback to native mode
   - Returns ToolResponse with success=False
   - Error details in stderr and error fields

## OpenInterpreter Auto-Retry Flow

```
User Request
    ↓
ReAct Agent → "Execute Python code to calculate factorial"
    ↓
OpenInterpreter Wrapper
    ↓
Attempt 1: Execute code
    ↓ (Error: NameError)
Attempt 2: Execute with error context
    ↓ (Success!)
Return Result → ReAct Agent
```

### Retry Logic

```python
for attempt in range(max_retries):  # default: 3
    try:
        # Build instruction with previous error history
        instruction = original_code + error_context

        # Execute via Open Interpreter
        response = interpreter.chat(instruction)

        # Check if successful
        if success:
            return result
        else:
            # Add error to history, retry
            error_history.append(error_message)

    except Exception as e:
        # Add exception to history, retry
        error_history.append(str(e))

# After all retries exhausted
return ToolResponse(success=False, error="Failed after N attempts")
```

## Integration with ReAct Agent

### Tool Call Format

**Input (from ReAct agent):**
```json
{
  "code": "print('Hello World')",
  "session_id": "abc123",
  "timeout": 30,
  "context": {
    "user_query": "Run a hello world program",
    "current_thought": "I should execute simple Python code"
  }
}
```

**Output (to ReAct agent):**
```json
{
  "success": true,
  "answer": "Code executed successfully.\n\nOutput:\nHello World\n\nFiles in workspace: script_123.py",
  "data": {
    "stdout": "Hello World\n",
    "stderr": "",
    "files": {"script_123.py": {...}},
    "workspace": "data/scratch/abc123",
    "returncode": 0
  },
  "metadata": {
    "execution_time": 1.5,
    "code_execution_time": 0.7
  },
  "error": null
}
```

## Logging

Both modes log to `data/logs/prompts.log`:

```
================================================================================
TOOL EXECUTION: python_coder (OpenInterpreter mode)
================================================================================
Timestamp: 2025-12-16 10:30:45
Session ID: abc123
Workspace: data/scratch/abc123
Max Retries: 3
Code/Instruction Length: 50 chars

INSTRUCTION:
  print('Hello World')

--- Attempt 1/3 ---

RESPONSE:
  Hello World

[SUCCESS] Completed in attempt 1
Execution time: 1.5s
================================================================================
```

## Performance Considerations

### Native Mode
- **Pros**: Fast, no overhead, reliable
- **Cons**: No auto-correction, requires correct code

### OpenInterpreter Mode
- **Pros**: Auto-correction, natural language support, intelligent retry
- **Cons**: Slower (LLM calls), requires Open Interpreter installed

### Recommendation
- **Development**: Use Native mode for speed
- **Production with ReAct agent**: Use OpenInterpreter mode for robustness
- **Simple tasks**: Native mode sufficient
- **Complex tasks with potential errors**: OpenInterpreter mode beneficial

## Security

### Native Mode
- Executes code directly in subprocess
- Limited to session workspace directory
- No network access restrictions

### OpenInterpreter Mode
- `OFFLINE = True` - No external API calls
- `AUTO_RUN = True` - No user confirmation (automated)
- Same workspace restrictions as native
- Prompt includes security guidelines

### Workspace Isolation
Both modes use session-based workspace:
```
data/scratch/{session_id}/
```
Files are isolated per session.

## Future Enhancements

Potential improvements:
1. Add support for more backends (e.g., IPython, Jupyter)
2. Configurable retry strategies
3. Token usage tracking for OpenInterpreter
4. Timeout enforcement for OpenInterpreter
5. More sophisticated error pattern detection
6. Streaming support for long-running executions

## Testing Checklist

- [x] Native mode basic execution
- [x] Native mode error handling
- [x] OpenInterpreter mode initialization
- [x] Factory pattern validation
- [x] Backward compatibility
- [x] Config-based mode switching
- [x] Error message clarity
- [x] Workspace isolation
- [x] Logging functionality
- [x] API endpoint integration

## Conclusion

The implementation successfully adds a configurable Python execution backend with:
- ✅ Clean architecture (factory pattern + base interface)
- ✅ Easy configuration (single variable in config.py)
- ✅ Strict error handling (no silent fallbacks)
- ✅ Backward compatibility (existing code works unchanged)
- ✅ Comprehensive testing (test_executor_modes.py)
- ✅ Clear documentation (prompts and code comments)
- ✅ Production-ready (tested with native mode)

The system is ready for use. To enable OpenInterpreter mode, simply:
1. Install: `pip install open-interpreter`
2. Configure: Set `PYTHON_EXECUTOR_MODE = "openinterpreter"` in config.py
3. Restart: `python server.py`
