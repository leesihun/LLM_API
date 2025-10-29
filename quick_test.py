"""
Quick integration test - Verify Python coder tool is accessible
"""
import asyncio
from backend.tools.python_coder_tool import python_coder_tool

async def test_simple_code():
    """Test simple code generation"""
    print("="*60)
    print("Quick Integration Test: Python Coder Tool")
    print("="*60)

    query = "Calculate the sum of numbers from 1 to 10 and print as JSON"

    print(f"\nQuery: {query}")
    print("\nGenerating and executing code...")

    result = await python_coder_tool.execute_code_task(query)

    print(f"\nResult:")
    print(f"  Success: {result['success']}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Execution Time: {result['execution_time']:.2f}s")

    if result['success']:
        print(f"\nOutput:\n{result['output']}")
        print("\n[SUCCESS] Python coder tool is working correctly!")
    else:
        print(f"\nError: {result.get('error', 'Unknown error')}")
        print("\n[FAILED] Tool encountered an error")

    return result['success']

if __name__ == "__main__":
    success = asyncio.run(test_simple_code())
    exit(0 if success else 1)
