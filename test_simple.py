"""
Simple test for Python executor (native mode)
"""
import requests
import json

# Configuration
TOOLS_API_URL = "http://localhost:10006"

# Test code: print hello
code = 'print("Hello from Python Executor!")'
session_id = "test_simple"

print("Testing Python Executor...")
print(f"Code: {code}")
print(f"API: {TOOLS_API_URL}")
print()

# Make request
response = requests.post(
    f"{TOOLS_API_URL}/api/tools/python_coder",
    json={
        "code": code,
        "session_id": session_id,
        "timeout": 30
    }
)

# Parse response
result = response.json()

print(f"Status Code: {response.status_code}")
print(f"Success: {result['success']}")
print()
print(f"Answer:\n{result['answer']}")
print()

if result['data']['stdout']:
    print(f"Output:\n{result['data']['stdout']}")

if result['data']['stderr']:
    print(f"Errors:\n{result['data']['stderr']}")

print(f"\nExecution Time: {result['metadata']['execution_time']:.3f}s")
