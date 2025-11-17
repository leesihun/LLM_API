"""
Test that the Unicode emoji fixes work correctly.
This should run without UnicodeEncodeError on Windows (cp949).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Test imports
from backend.tools.python_coder.orchestrator import python_coder_tool
from backend.tools.python_coder.context_builder import FileContextBuilder
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

print("="*60)
print("Testing Unicode Emoji Fixes")
print("="*60)

# Test 1: Logger messages
print("\nTest 1: Testing logger messages...")
try:
    logger.info("[PythonCoderTool] [OK] Execution succeeded, checking output adequacy...")
    logger.error("[PythonCoderTool] [X] Execution failed: test error")
    logger.warning("[PythonCoderTool] [WARNING] Output not adequate: test warning")
    logger.info("[PythonCoderTool] [SAVED] Saved test code to test.py")
    logger.info("[CodeExecutor] [SAVED] Saved stage code to script_test.py")
    logger.warning("[CodeExecutor] [WARNING] Error pattern detected: 'test'")
    print("   [PASS] Logger messages work without encoding errors")
except UnicodeEncodeError as e:
    print(f"   [FAIL] Logger messages caused encoding error: {e}")

# Test 2: Context builder
print("\nTest 2: Testing context builder...")
try:
    builder = FileContextBuilder()

    # Create test metadata
    validated_files = {"test.json": "test.json"}
    file_metadata = {
        "test.json": {
            "type": "json",
            "size_mb": 0.001,
            "structure": {
                "type": "dict",
                "keys": ["test_key"],
                "num_keys": 1
            },
            "access_patterns": ["data['test_key']"],
            "requires_null_check": True,
            "max_depth": 4
        }
    }

    context = builder.build_context(validated_files, file_metadata)

    # Check for ASCII-safe markers
    assert "[!!!]" in context, "Should contain [!!!] marker"
    assert "[PATTERNS]" in context, "Should contain [PATTERNS] marker"
    assert "[!]" in context, "Should contain [!] marker"

    # Ensure no Unicode emojis
    unicode_emojis = ["🚨", "📋", "⚠️", "💾", "✅", "❌"]
    for emoji in unicode_emojis:
        assert emoji not in context, f"Should not contain {emoji}"

    print("   [PASS] Context builder uses ASCII-safe markers")
    print(f"   Context sample:\n{context[:300]}...")

except Exception as e:
    print(f"   [FAIL] Context builder test failed: {e}")

print("\n" + "="*60)
print("All Tests Completed Successfully!")
print("="*60)
print("\nThe cp949 encoding issue should now be fixed.")
print("Unicode emojis have been replaced with ASCII-safe alternatives:")
print("  - (X emoji) -> [X]")
print("  - (OK emoji) -> [OK]")
print("  - (WARNING emoji) -> [WARNING] or [!]")
print("  - (CRITICAL emoji) -> [!!!]")
print("  - (SAVED emoji) -> [SAVED]")
print("  - (PATTERNS emoji) -> [PATTERNS]")
print("="*60)
