"""
Test Unicode encoding fix for CodeExecutor.
Tests that the executor can handle special Unicode characters like \u2011.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.tools.python_coder.executor import CodeExecutor


def test_unicode_in_code():
    """Test that Unicode characters in code execute properly."""

    # Code containing various Unicode characters
    # The key test: can we SEND code with Unicode to the subprocess?
    test_code = """
import sys

# Test various Unicode characters in the CODE ITSELF (not in output)
# This tests if the subprocess can RECEIVE Unicode in stdin
text = "Hello\u2011World"  # Non-breaking hyphen (the problematic character)
text2 = "Em\u2014dash"      # Em dash
text3 = "Bullet\u2022point" # Bullet point
text4 = "Copyright\u00a9"   # Copyright symbol

# Don't print Unicode (that fails on cp949 console)
# Instead, verify the strings are correctly stored
assert "\u2011" in text, "Non-breaking hyphen not found"
assert "\u2014" in text2, "Em dash not found"
assert "\u2022" in text3, "Bullet not found"
assert "\u00a9" in text4, "Copyright symbol not found"

# Print safe ASCII confirmation
print("SUCCESS: All Unicode characters processed correctly in code!")
print(f"String lengths: {len(text)}, {len(text2)}, {len(text3)}, {len(text4)}")
"""

    print("=" * 60)
    print("Testing Unicode Character Support")
    print("=" * 60)

    # Test with REPL mode
    print("\n[1] Testing with Persistent REPL mode...")
    executor_repl = CodeExecutor(
        timeout=10,
        use_persistent_repl=True,
        execution_base_dir="./data/scratch"
    )

    result_repl = executor_repl.execute(
        code=test_code,
        session_id="test_unicode_repl"
    )

    print(f"   Success: {result_repl['success']}")
    print(f"   Output:\n{result_repl['output']}")
    if result_repl['error']:
        print(f"   Error: {result_repl['error']}")

    # Cleanup REPL
    executor_repl.cleanup_session("test_unicode_repl")

    # Test with subprocess mode
    print("\n[2] Testing with Subprocess mode...")
    executor_subprocess = CodeExecutor(
        timeout=10,
        use_persistent_repl=False,
        execution_base_dir="./data/scratch"
    )

    result_subprocess = executor_subprocess.execute(
        code=test_code,
        session_id=None  # Temporary execution
    )

    print(f"   Success: {result_subprocess['success']}")
    print(f"   Output:\n{result_subprocess['output']}")
    if result_subprocess['error']:
        print(f"   Error: {result_subprocess['error']}")

    # Summary
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    print(f"REPL Mode:       {'PASSED' if result_repl['success'] else 'FAILED'}")
    print(f"Subprocess Mode: {'PASSED' if result_subprocess['success'] else 'FAILED'}")

    if result_repl['success'] and result_subprocess['success']:
        print("\nAll tests passed! Unicode encoding is fixed.")
        return True
    else:
        print("\nSome tests failed. Check errors above.")
        return False


if __name__ == "__main__":
    success = test_unicode_in_code()
    sys.exit(0 if success else 1)
