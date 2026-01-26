"""
Simple test script for OpenCode integration
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from tools.python_coder import get_python_executor

def test_opencode():
    """Test OpenCode executor with a simple instruction"""
    print("\n" + "=" * 80)
    print("Testing OpenCode Integration")
    print("=" * 80)
    print(f"Python Executor Mode: {config.PYTHON_EXECUTOR_MODE}")
    print(f"OpenCode Provider: {config.OPENCODE_PROVIDER}")
    print(f"OpenCode Model: {config.OPENCODE_MODEL}")
    print("=" * 80)
    print()

    # Create executor
    session_id = "test_session_123"
    executor = get_python_executor(session_id)

    print(f"Executor Type: {type(executor).__name__}")
    print(f"Session ID: {session_id}")
    print()

    # Test instruction
    instruction = "Create a Python function that calculates the factorial of 5, test it, and print the result"

    print(f"Instruction: {instruction}")
    print()
    print("Executing...")
    print("-" * 80)

    # Execute
    result = executor.execute(instruction)

    print("-" * 80)
    print()
    print("Result:")
    print(f"  Success: {result['success']}")
    print(f"  Return Code: {result['returncode']}")
    print(f"  Execution Time: {result['execution_time']:.2f}s")
    print()

    if result['stdout']:
        print("Output:")
        print(result['stdout'])

    if result['stderr']:
        print("\nStderr:")
        print(result['stderr'])

    if result['error']:
        print(f"\nError: {result['error']}")

    if result['files']:
        print(f"\nFiles created: {list(result['files'].keys())}")

    print()
    print("=" * 80)
    print("Test complete!")
    print("=" * 80)

    return result['success']


if __name__ == "__main__":
    success = test_opencode()
    sys.exit(0 if success else 1)
