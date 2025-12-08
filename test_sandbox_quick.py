"""Quick test of code sandbox functionality"""
import asyncio
from backend.tools.code_sandbox import CodeSandbox, SandboxManager

def test_sandbox_basic():
    """Test basic code execution"""
    print("=" * 60)
    print("TEST 1: Basic Code Execution")
    print("=" * 60)

    sandbox = SandboxManager.get_sandbox("test_basic")

    # Test 1: Simple calculation
    code = """
x = 10
y = 20
result = x + y
print(f"Result: {result}")
"""
    result = sandbox.execute(code)
    print(f"Success: {result.success}")
    print(f"Output: {result.output}")
    print(f"Variables: {result.variables}")
    print()

    # Test 2: Variable persistence
    print("=" * 60)
    print("TEST 2: Variable Persistence")
    print("=" * 60)
    code2 = """
print(f"Previous result was: {result}")
result2 = result * 2
print(f"New result: {result2}")
"""
    result2 = sandbox.execute(code2)
    print(f"Success: {result2.success}")
    print(f"Output: {result2.output}")
    print(f"Variables: {result2.variables}")
    print()

def test_sandbox_error_handling():
    """Test error handling"""
    print("=" * 60)
    print("TEST 3: Error Handling")
    print("=" * 60)

    sandbox = SandboxManager.get_sandbox("test_error")

    # Test error
    code = """
x = 10
y = 0
result = x / y  # Division by zero
print(result)
"""
    result = sandbox.execute(code)
    print(f"Success: {result.success}")
    print(f"Error: {result.error}")
    print(f"Error Type: {result.error_type}")
    print()

def test_sandbox_validation():
    """Test code validation"""
    print("=" * 60)
    print("TEST 4: Code Validation (Blocked Imports)")
    print("=" * 60)

    sandbox = SandboxManager.get_sandbox("test_validation")

    # Test blocked import
    code = """
import subprocess
subprocess.run(['ls'])
"""
    result = sandbox.execute(code, validate=True)
    print(f"Success: {result.success}")
    print(f"Error: {result.error}")
    print(f"Error Type: {result.error_type}")
    print()

def test_pandas_operations():
    """Test pandas operations"""
    print("=" * 60)
    print("TEST 5: Pandas Operations")
    print("=" * 60)

    sandbox = SandboxManager.get_sandbox("test_pandas")

    code = """
# Create a simple dataframe
data = {'name': ['Alice', 'Bob', 'Charlie'], 'age': [25, 30, 35]}
df = pd.DataFrame(data)
print("DataFrame created:")
print(df)
print(f"\\nShape: {df.shape}")
"""
    result = sandbox.execute(code)
    print(f"Success: {result.success}")
    print(f"Output: {result.output}")
    if not result.success:
        print(f"Error: {result.error}")
    print()

if __name__ == "__main__":
    test_sandbox_basic()
    test_sandbox_error_handling()
    test_sandbox_validation()
    test_pandas_operations()

    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
