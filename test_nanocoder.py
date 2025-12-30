"""
Test script for nanocoder executor
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from tools.python_coder import get_python_executor

def test_native_mode():
    """Test native Python executor"""
    print("\n" + "="*80)
    print("TESTING NATIVE MODE")
    print("="*80)

    # Temporarily set to native mode
    original_mode = config.PYTHON_EXECUTOR_MODE
    config.PYTHON_EXECUTOR_MODE = "native"

    try:
        executor = get_python_executor("test_session_native")

        # Test with Python code
        python_code = """
import math

# Calculate factorial
def factorial(n):
    return math.factorial(n)

result = factorial(5)
print(f"Factorial of 5 is: {result}")
"""

        print("\nExecuting Python code directly...")
        result = executor.execute(python_code, timeout=30)

        print(f"\nResult:")
        print(f"  Success: {result['success']}")
        print(f"  Return code: {result['returncode']}")
        print(f"  Stdout: {result['stdout']}")
        print(f"  Stderr: {result['stderr']}")
        print(f"  Execution time: {result['execution_time']:.2f}s")

        # Cleanup
        executor.clear_workspace()

        return result['success']

    finally:
        config.PYTHON_EXECUTOR_MODE = original_mode


def test_nanocoder_mode():
    """Test nanocoder executor"""
    print("\n" + "="*80)
    print("TESTING NANOCODER MODE")
    print("="*80)

    # Temporarily set to nanocoder mode
    original_mode = config.PYTHON_EXECUTOR_MODE
    config.PYTHON_EXECUTOR_MODE = "nanocoder"

    try:
        executor = get_python_executor("test_session_nanocoder")

        # Test with natural language instruction
        instruction = "Create a Python script that calculates the factorial of 5 and prints the result"

        print(f"\nExecuting instruction: {instruction}")
        result = executor.execute(instruction, timeout=60)

        print(f"\nResult:")
        print(f"  Success: {result['success']}")
        print(f"  Return code: {result['returncode']}")
        print(f"  Stdout: {result['stdout'][:500]}...")
        print(f"  Stderr: {result['stderr'][:500] if result['stderr'] else 'None'}...")
        print(f"  Execution time: {result['execution_time']:.2f}s")

        # Cleanup
        executor.clear_workspace()

        return result['success']

    finally:
        config.PYTHON_EXECUTOR_MODE = original_mode


if __name__ == "__main__":
    print("Starting Nanocoder Integration Tests")
    print("="*80)

    # Test 1: Native mode
    try:
        native_success = test_native_mode()
        print(f"\n[OK] Native mode test: {'PASSED' if native_success else 'FAILED'}")
    except Exception as e:
        print(f"\n[FAIL] Native mode test FAILED: {e}")
        import traceback
        traceback.print_exc()
        native_success = False

    # Test 2: Nanocoder mode
    try:
        nanocoder_success = test_nanocoder_mode()
        print(f"\n[OK] Nanocoder mode test: {'PASSED' if nanocoder_success else 'FAILED'}")
    except Exception as e:
        print(f"\n[FAIL] Nanocoder mode test FAILED: {e}")
        import traceback
        traceback.print_exc()
        nanocoder_success = False

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Native mode:    {'[PASS]' if native_success else '[FAIL]'}")
    print(f"Nanocoder mode: {'[PASS]' if nanocoder_success else '[FAIL]'}")

    if native_success and nanocoder_success:
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed")
        sys.exit(1)
