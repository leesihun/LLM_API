"""
Quick test for OpenInterpreter executor
"""
import config

# Temporarily set mode to openinterpreter
original_mode = config.PYTHON_EXECUTOR_MODE
config.PYTHON_EXECUTOR_MODE = "openinterpreter"

try:
    from tools.python_coder import PythonCoderTool

    print("=" * 80)
    print("Testing OpenInterpreter Executor")
    print("=" * 80)

    # Create executor instance
    print("\n1. Creating executor...")
    executor = PythonCoderTool(session_id="test_session")
    print(f"   Created: {type(executor).__name__}")

    # Test execution with natural language
    print("\n2. Testing execution with natural language instruction...")
    instruction = "Calculate the factorial of 5 and print the result"

    result = executor.execute(
        code=instruction,  # Natural language instruction
        timeout=60
    )

    print("\n3. Results:")
    print(f"   Success: {result['success']}")
    print(f"   Return code: {result['returncode']}")
    print(f"   Execution time: {result['execution_time']:.2f}s")

    if result['stdout']:
        print(f"\n   STDOUT:")
        print(f"   {result['stdout']}")

    if result['stderr']:
        print(f"\n   STDERR:")
        print(f"   {result['stderr']}")

    if result['files']:
        print(f"\n   Files created: {list(result['files'].keys())}")

    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)

except ImportError as e:
    print(f"\nImportError: {e}")
    print("\nNote: Open Interpreter may not be installed.")
    print("Install with: pip install open-interpreter")

except Exception as e:
    print(f"\nError: {type(e).__name__}: {e}")

finally:
    # Restore original mode
    config.PYTHON_EXECUTOR_MODE = original_mode
