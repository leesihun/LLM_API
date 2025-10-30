# Math Calculator Unicode Symbol Fix

## Problem
**User reported:** ReAct Agent returning **22.75** instead of **20** for the calculation:
```
Action: calculator
Action Input: (100 × 0.15 + 25) / 2
Observation: 22.75  ❌ WRONG!
```

Correct answer should be: **20**
- Step 1: 100 × 0.15 = 15
- Step 2: 15 + 25 = 40
- Step 3: 40 / 2 = **20**

## Root Cause
1. **LLMs generate unicode math symbols**:
   - `×` (multiplication sign U+00D7) instead of `*`
   - `÷` (division sign U+00F7) instead of `/`
   - `−` (minus sign U+2212) instead of `-`

2. **SymPy doesn't recognize unicode symbols** → parsing errors or unexpected behavior

3. **No input normalization** before calculation

## Solution

### 1. Added Input Normalization (`math_calculator.py`)

Created `_normalize_expression()` method that converts unicode to ASCII:

```python
def _normalize_expression(self, expression: str) -> str:
    """
    Normalize mathematical expression by converting unicode symbols to ASCII

    Handles:
    - Unicode multiplication signs (×) → *
    - Unicode division signs (÷) → /
    - Unicode minus signs (−) → -
    - Full-width digits → half-width digits
    - Extra whitespace cleanup
    """
    replacements = {
        '×': '*',  # Multiplication sign (U+00D7)
        '÷': '/',  # Division sign (U+00F7)
        '−': '-',  # Minus sign (U+2212)
        '・': '*',  # Katakana middle dot
        '·': '*',  # Middle dot (U+00B7)
        '━': '-',  # Heavy horizontal line
        '—': '-',  # Em dash
        '–': '-',  # En dash
    }

    # Apply replacements...
    # Convert full-width digits...
    # Clean whitespace...
```

**Location:** [backend/tools/math_calculator.py:36-74](backend/tools/math_calculator.py#L36-L74)

### 2. Enhanced Logging (`React.py`)

Added detailed logging for debugging:

```python
elif action == ToolName.MATH_CALC:
    logger.info(f"[ReAct Agent] Calculating: {action_input}")
    logger.debug(f"[ReAct Agent] Math input (raw): {repr(action_input)}")
    result = await math_calculator.calculate(action_input)
    logger.info(f"[ReAct Agent] Math result: {result}")
    return result
```

**Location:** [backend/tasks/React.py:376-382](backend/tasks/React.py#L376-L382)

## Test Results

```bash
$ python test_user_problem.py

================================================================================
USER-REPORTED PROBLEM TEST
================================================================================

Test 1: Unicode multiplication symbol
Expression: (100 × 0.15 + 25) / 2
Result: Result: 20  ✓

Test 2: ASCII multiplication symbol
Expression: (100 * 0.15 + 25) / 2
Result: Result: 20  ✓

================================================================================
VERIFICATION
================================================================================
Step 1: 100 * 0.15 = 15
Step 2: 15 + 25 = 40
Step 3: 40 / 2 = 20

Correct answer: 20
Unicode result: 20.0 ✓
ASCII result: 20.0 ✓

[SUCCESS] Both calculations returned correct result: 20
The 22.75 issue has been FIXED!
```

## Supported Unicode Symbols

The calculator now automatically handles:

| Unicode | ASCII | Description |
|---------|-------|-------------|
| × | * | Multiplication sign |
| ÷ | / | Division sign |
| − | - | Minus sign |
| · | * | Middle dot |
| ・ | * | Katakana middle dot |
| ０-９ | 0-9 | Full-width digits |

## Files Modified

1. **backend/tools/math_calculator.py**
   - Lines 36-74: Added `_normalize_expression()` method
   - Line 96: Call normalization in `calculate()` method

2. **backend/tasks/React.py**
   - Lines 376-382: Enhanced logging for math calculations

## Impact

- ✅ **Fixes 22.75 bug** - Now returns correct result (20)
- ✅ **LLM compatibility** - Handles unicode symbols that LLMs naturally generate
- ✅ **Better debugging** - Detailed logs show input/output
- ✅ **Backward compatible** - ASCII operators still work perfectly
- ✅ **International support** - Handles full-width digits

## How to Test

```bash
# Test the specific user problem
python test_user_problem.py

# Test with your own expressions
python -c "
import asyncio
from backend.tools.math_calculator import math_calculator

async def test():
    # Unicode symbols
    result = await math_calculator.calculate('(100 × 0.15 + 25) / 2')
    print(result)

asyncio.run(test())
"
```

## Next Steps

1. ✅ **Input normalization** - DONE
2. ✅ **Unicode symbol support** - DONE
3. ✅ **Enhanced logging** - DONE
4. ⏳ **Integration test with ReAct Agent** - Recommended to test with real LLM
5. ⏳ **Add to CI/CD** - Create unit tests for unicode handling

## Notes

- This fix ensures math calculator works correctly regardless of how LLM formats the expression
- The normalization is transparent - users don't need to know about it
- Logging at DEBUG level captures unicode symbols for troubleshooting
- No performance impact - normalization is very fast (string replacements)
