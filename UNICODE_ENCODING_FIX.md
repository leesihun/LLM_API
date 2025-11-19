# Unicode Encoding Fix for CodeExecutor

## Problem
The `backend.tools.python_coder.executor` module was unable to encode Unicode characters like `\u2011` (non-breaking hyphen) when executing code on Windows, causing `UnicodeEncodeError`.

## Root Cause
- Windows subprocess defaults to console encoding (cp949, cp1252, etc.) instead of UTF-8
- The `text=True` parameter in `subprocess.Popen()` and `subprocess.run()` uses system default encoding
- Characters like `\u2011`, `\u2013`, and others cannot be encoded in these legacy encodings

## Solution Implemented

### Changes Made to `backend/tools/python_coder/executor.py`:

**1. Fixed PersistentREPL (lines 69-78):**
```python
# BEFORE:
self.process = subprocess.Popen(
    [sys.executable, "-u", "-c", self._get_repl_bootstrap()],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=str(self.execution_dir),
    text=True,  # ← Problem: uses system encoding
    bufsize=0
)

# AFTER:
self.process = subprocess.Popen(
    [sys.executable, "-u", "-c", self._get_repl_bootstrap()],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=str(self.execution_dir),
    encoding='utf-8',      # ← Explicit UTF-8
    errors='replace',      # ← Fallback for invalid bytes
    bufsize=0
)
```

**2. Fixed traditional subprocess execution (lines 595-602):**
```python
# BEFORE:
result = subprocess.run(
    [sys.executable, str(script_path)],
    capture_output=True,
    timeout=self.timeout,
    cwd=str(execution_dir),
    text=True  # ← Problem: uses system encoding
)

# AFTER:
result = subprocess.run(
    [sys.executable, str(script_path)],
    capture_output=True,
    timeout=self.timeout,
    cwd=str(execution_dir),
    encoding='utf-8',      # ← Explicit UTF-8
    errors='replace'       # ← Fallback for invalid bytes
)
```

## What `errors='replace'` Does

- **Purpose**: Handles characters that cannot be encoded/decoded
- **Behavior**: Replaces problematic bytes with `?` (for encoding) or `�` (for decoding)
- **Benefit**: Prevents crashes while maintaining data flow
- **Note**: With UTF-8, this rarely triggers since UTF-8 can encode all Unicode characters

## Test Results

### Subprocess Mode: ✓ PASSED
- Successfully handles `\u2011` (non-breaking hyphen)
- Successfully handles `\u2014` (em dash)
- Successfully handles `\u2022` (bullet point)
- Successfully handles `\u00a9` (copyright symbol)

### REPL Mode: Partially Working
- Encoding fix applied successfully
- REPL can receive Unicode over stdin
- Note: REPL mode has additional limitation with Windows console output (not related to this fix)

## Usage

The fix is automatic and requires no code changes. Simply use CodeExecutor as before:

```python
from backend.tools.python_coder.executor import CodeExecutor

executor = CodeExecutor()

# Code with Unicode characters now works
code_with_unicode = '''
text = "Hello\u2011World"  # Non-breaking hyphen
print(len(text))
'''

result = executor.execute(code_with_unicode)
# SUCCESS: Works on both Windows and Unix
```

## Important Notes

### This Fix Addresses:
✅ Unicode characters in code being sent to subprocess
✅ Unicode characters in code file content
✅ Unicode in stdin/stdout/stderr communication
✅ Cross-platform compatibility (Windows, Linux, macOS)

### This Fix Does NOT Address:
❌ Windows console's inability to **display** certain Unicode characters
❌ User code that prints Unicode to console on legacy Windows terminals

### Example of Console Output Limitation:
```python
# This code will EXECUTE successfully (fix applied)
# But PRINTING might fail on Windows console (OS limitation)
code = '''
text = "\u2011"  # ← Stored successfully
print(text)      # ← May fail if console is cp949/cp1252
'''
```

**Workaround for console output**: Use `PYTHONIOENCODING=utf-8` environment variable or redirect output to file.

## Technical Details

### Why UTF-8?
- UTF-8 can encode all 1,112,064 valid Unicode code points
- Industry standard for text encoding
- Consistent behavior across all platforms

### Why `errors='replace'`?
- Defensive programming - prevents crashes from corrupted data
- Minimal data loss (only affects truly invalid byte sequences)
- Better user experience than crashing

### Platform-Specific Behavior

**Before Fix:**
- Windows: Uses cp949 (Korean), cp1252 (Western European), or cp437 (legacy DOS)
- Linux/macOS: Often uses UTF-8 by default
- Result: Inconsistent behavior across platforms

**After Fix:**
- All platforms: Explicitly UTF-8
- Result: Consistent, predictable behavior

## Version Information

- **Fixed in**: January 2025
- **Affects**: backend/tools/python_coder/executor.py
- **Related**: Part of Python Code Tool modularization (v2.0.0)
- **Backward Compatible**: Yes, existing code works without changes

## Testing

Run the test suite to verify:
```bash
python test_unicode_fix.py
```

Expected output:
```
Subprocess Mode: PASSED
```

## Related Issues

- Windows console encoding limitations
- Python subprocess text mode encoding
- UTF-8 standardization across tools

---

**Status**: ✅ FIXED - Unicode characters can now be encoded in code execution
**Tested on**: Windows (cp949 console)
**Compatibility**: Python 3.7+
