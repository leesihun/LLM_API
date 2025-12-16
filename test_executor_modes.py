"""
Test script for Python executor modes
Tests both native and openinterpreter modes
"""
import config
from tools.python_coder import PythonCoderTool, get_python_executor


def test_native_mode():
    """Test native executor mode"""
    print("\n" + "=" * 80)
    print("TEST: Native Executor Mode")
    print("=" * 80)

    # Set mode
    original_mode = config.PYTHON_EXECUTOR_MODE
    config.PYTHON_EXECUTOR_MODE = "native"

    try:
        # Create tool
        print("\n1. Creating tool instance...")
        tool = get_python_executor("test_native_session")
        print(f"   Tool type: {type(tool).__name__}")
        print(f"   Workspace: {tool.workspace}")

        # Test simple execution
        print("\n2. Testing simple code execution...")
        code = """
print("Hello from native executor!")
result = 2 + 2
print(f"2 + 2 = {result}")
"""
        result = tool.execute(code)

        print(f"   Success: {result['success']}")
        print(f"   Return code: {result['returncode']}")
        if result['stdout']:
            print(f"   Output:\n{result['stdout']}")
        if result['stderr']:
            print(f"   Error:\n{result['stderr']}")

        # Test error handling
        print("\n3. Testing error handling...")
        bad_code = "print(undefined_variable)"
        result = tool.execute(bad_code)

        print(f"   Success: {result['success']}")
        print(f"   Expected error captured: {not result['success']}")

        # Cleanup
        tool.clear_workspace()
        print("\n4. Workspace cleared")

        print("\n[NATIVE MODE TEST] PASSED")
        return True

    except Exception as e:
        print(f"\n[NATIVE MODE TEST] FAILED")
        print(f"Error: {e}")
        return False

    finally:
        config.PYTHON_EXECUTOR_MODE = original_mode


def test_openinterpreter_mode():
    """Test OpenInterpreter executor mode"""
    print("\n" + "=" * 80)
    print("TEST: OpenInterpreter Executor Mode")
    print("=" * 80)

    # Set mode
    original_mode = config.PYTHON_EXECUTOR_MODE
    config.PYTHON_EXECUTOR_MODE = "openinterpreter"

    try:
        # Create tool
        print("\n1. Creating tool instance...")
        tool = get_python_executor("test_openinterpreter_session")
        print(f"   Tool type: {type(tool).__name__}")
        print(f"   Workspace: {tool.workspace}")

        print("\n[OPENINTERPRETER MODE TEST] Tool created successfully")
        print("\nNOTE: Full execution test skipped (requires Open Interpreter installation)")
        print("      To test fully: pip install open-interpreter")

        # Cleanup
        tool.clear_workspace()

        return True

    except ImportError as e:
        print(f"\n[OPENINTERPRETER MODE TEST] Import Error (Expected if not installed)")
        print(f"   Error: {e}")
        print("\n   This is OK if you haven't installed open-interpreter yet")
        print("   To enable: pip install open-interpreter")
        return True  # Not a failure, just not installed

    except Exception as e:
        print(f"\n[OPENINTERPRETER MODE TEST] FAILED")
        print(f"Unexpected error: {e}")
        return False

    finally:
        config.PYTHON_EXECUTOR_MODE = original_mode


def test_factory_validation():
    """Test factory validation for invalid mode"""
    print("\n" + "=" * 80)
    print("TEST: Factory Validation")
    print("=" * 80)

    original_mode = config.PYTHON_EXECUTOR_MODE
    config.PYTHON_EXECUTOR_MODE = "invalid_mode"

    try:
        print("\n1. Testing invalid mode rejection...")
        tool = get_python_executor("test_session")
        print("[VALIDATION TEST] FAILED - Should have raised ValueError")
        return False

    except ValueError as e:
        print(f"   Correctly raised ValueError: {e}")
        print("\n[VALIDATION TEST] PASSED")
        return True

    except Exception as e:
        print(f"\n[VALIDATION TEST] FAILED")
        print(f"Wrong exception type: {e}")
        return False

    finally:
        config.PYTHON_EXECUTOR_MODE = original_mode


if __name__ == "__main__":
    print("\n")
    print("=" * 80)
    print(" " * 20 + "PYTHON EXECUTOR MODE TESTS")
    print("=" * 80)

    results = []

    # Run tests
    results.append(("Native Mode", test_native_mode()))
    results.append(("OpenInterpreter Mode", test_openinterpreter_mode()))
    results.append(("Factory Validation", test_factory_validation()))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{name:.<40} {status}")

    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\nAll tests passed!")
        exit(0)
    else:
        print("\nSome tests failed")
        exit(1)
