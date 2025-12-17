"""
Test script for Open Interpreter integration
Tests the python_coder tool endpoint with Open Interpreter backend
"""
import requests
import json
import time

# Configuration
TOOLS_API_URL = "http://localhost:10006"

def test_python_coder(code: str, session_id: str, description: str):
    """Test the python_coder endpoint"""
    print("\n" + "=" * 80)
    print(f"TEST: {description}")
    print("=" * 80)
    print(f"Code:\n{code}")
    print("-" * 80)

    # Make request
    response = requests.post(
        f"{TOOLS_API_URL}/api/tools/python_coder",
        json={
            "code": code,
            "session_id": session_id,
            "timeout": 60
        }
    )

    # Parse response
    result = response.json()

    print(f"\nResponse Status: {response.status_code}")
    print(f"Success: {result['success']}")
    print(f"\nAnswer:\n{result['answer']}")

    if result['data']['stdout']:
        print(f"\nStdout:\n{result['data']['stdout']}")

    if result['data']['stderr']:
        print(f"\nStderr:\n{result['data']['stderr']}")

    print(f"\nExecution Time: {result['metadata']['execution_time']:.2f}s")
    print(f"Workspace: {result['data']['workspace']}")

    if result['data']['files']:
        print(f"\nFiles Created: {list(result['data']['files'].keys())}")

    print("=" * 80)

    return result


def main():
    """Run tests"""
    session_id = f"test_session_{int(time.time())}"

    print("\n" + "=" * 80)
    print("OPEN INTERPRETER TESTING")
    print("=" * 80)
    print(f"Session ID: {session_id}")
    print(f"Tools API: {TOOLS_API_URL}")

    # Test 1: Simple print statement
    print("\n\n*** TEST 1: Simple Hello World ***")
    test_python_coder(
        code='print("Hello from Open Interpreter!")',
        session_id=session_id,
        description="Simple print statement"
    )

    # Test 2: Math calculation
    print("\n\n*** TEST 2: Math Calculation ***")
    test_python_coder(
        code="""
result = 42 * 2 + 10
print(f"The answer is: {result}")
""",
        session_id=session_id,
        description="Basic math calculation"
    )

    # Test 3: Create a file
    print("\n\n*** TEST 3: File Creation ***")
    test_python_coder(
        code="""
with open('hello.txt', 'w') as f:
    f.write('Hello from Open Interpreter file!')
print('File created successfully')
""",
        session_id=session_id,
        description="Create a text file"
    )

    # Test 4: Read the file
    print("\n\n*** TEST 4: Read File ***")
    test_python_coder(
        code="""
with open('hello.txt', 'r') as f:
    content = f.read()
print(f'File content: {content}')
""",
        session_id=session_id,
        description="Read the created file"
    )

    # Test 5: Code with intentional error (to test retry logic)
    print("\n\n*** TEST 5: Error Handling (Intentional Error) ***")
    test_python_coder(
        code="""
# This will cause an error on first try
import os
file_path = 'nonexistent.txt'
with open(file_path, 'r') as f:
    print(f.read())
""",
        session_id=session_id,
        description="Code with intentional error"
    )

    print("\n\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
