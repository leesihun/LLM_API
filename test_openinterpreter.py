#!/usr/bin/env python3
"""
Test script for OpenInterpreter tool with simple division
"""
import sys
sys.path.insert(0, '.')

from tools.python_coder.openinterpreter_tool import OpenInterpreterExecutor

def test_simple_division():
    """Test simple division: 11.951 / 3.751"""
    print("=" * 80)
    print("Testing OpenInterpreter with simple division")
    print("=" * 80)

    # Create executor
    session_id = "test_session_division"
    executor = OpenInterpreterExecutor(session_id)

    # Simple code to execute
    code = "print(11.951 / 3.751)"

    print(f"\nCode to execute:")
    print(f"  {code}")
    print()

    # Execute
    result = executor.execute(code)

    # Display results
    print("\n" + "=" * 80)
    print("EXECUTION RESULT")
    print("=" * 80)
    print(f"Success: {result['success']}")
    print(f"Return Code: {result['returncode']}")
    print(f"Execution Time: {result['execution_time']:.2f}s")
    print()

    if result['stdout']:
        print("STDOUT:")
        print(result['stdout'])
        print()

    if result['stderr']:
        print("STDERR:")
        print(result['stderr'])
        print()

    if result['error']:
        print("ERROR:")
        print(result['error'])
        print()

    print(f"Workspace: {result['workspace']}")
    print(f"Files created: {len(result['files'])}")

    # Cleanup
    executor.clear_workspace()

    return result['success']

if __name__ == "__main__":
    success = test_simple_division()
    sys.exit(0 if success else 1)
