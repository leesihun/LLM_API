"""
Test script for new multipart file upload to /v1/chat/completions
"""
import httpx
import json
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:1007"
USERNAME = "leesihun"
PASSWORD = "s.hun.lee"

def test_file_upload_chat():
    """Test the new multipart file upload with chat"""

    # Step 1: Login
    print("1. Logging in...")
    login_response = httpx.post(
        f"{API_BASE_URL}/api/auth/login",
        json={"username": USERNAME, "password": PASSWORD},
        timeout=10.0
    )
    login_response.raise_for_status()
    token = login_response.json()["access_token"]
    print(f"   ✓ Logged in successfully")

    # Step 2: Prepare test file
    print("\n2. Preparing test file...")
    test_file_path = Path("test_data.csv")
    test_file_path.write_text("name,age,city\nAlice,30,Seoul\nBob,25,Busan\nCharlie,35,Incheon")
    print(f"   ✓ Created test file: {test_file_path}")

    # Step 3: Send chat request with file
    print("\n3. Sending chat request with file attachment...")

    messages = [
        {"role": "user", "content": "Analyze the data in the attached CSV file and tell me the average age"}
    ]

    with open(test_file_path, "rb") as f:
        files = {"files": (test_file_path.name, f, "text/csv")}
        data = {
            "model": "gemma3:12b",
            "messages": json.dumps(messages),
            "agent_type": "auto"
        }
        headers = {"Authorization": f"Bearer {token}"}

        response = httpx.post(
            f"{API_BASE_URL}/v1/chat/completions",
            files=files,
            data=data,
            headers=headers,
            timeout=120.0
        )

    response.raise_for_status()
    result = response.json()

    print(f"   ✓ Request successful!")
    print(f"\n4. Response:")
    print(f"   Session ID: {result.get('x_session_id')}")
    print(f"   Agent Metadata: {result.get('x_agent_metadata', {}).get('agent_type')}")
    print(f"\n   AI Response:")
    print(f"   {result['choices'][0]['message']['content']}")

    # Step 4: Cleanup
    print("\n5. Cleaning up...")
    test_file_path.unlink()
    print(f"   ✓ Removed test file")

    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    try:
        test_file_upload_chat()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
