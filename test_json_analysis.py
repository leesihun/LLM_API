"""
Test script for JSON file analysis issue
Reproduces the complex JSON analysis test from API_examples.ipynb
"""
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Use the client from API_examples.ipynb
import httpx

class LLMApiClient:
    def __init__(self, base_url: str, timeout: float = 1200.0):
        self.base_url = base_url.rstrip("/")
        self.token = None
        self.timeout = httpx.Timeout(50.0, read=timeout, write=timeout, pool=timeout)

    def _headers(self):
        h = {}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def login(self, username: str, password: str):
        r = httpx.post(f"{self.base_url}/api/auth/login", json={
            "username": username, "password": password
        }, timeout=10.0)
        r.raise_for_status()
        data = r.json()
        self.token = data["access_token"]
        return data

    def chat_new(self, model: str, user_message: str, agent_type: str = "auto", files: list = None):
        messages = [{"role": "user", "content": user_message}]

        data = {
            "model": model,
            "messages": json.dumps(messages),
            "agent_type": agent_type
        }

        files_to_upload = []
        if files:
            for file_path in files:
                f = open(file_path, "rb")
                files_to_upload.append(("files", (Path(file_path).name, f)))

        try:
            r = httpx.post(
                f"{self.base_url}/v1/chat/completions",
                data=data,
                files=files_to_upload if files_to_upload else None,
                headers=self._headers(),
                timeout=self.timeout
            )
            r.raise_for_status()
            result = r.json()
            return result["choices"][0]["message"]["content"], result["x_session_id"]

        finally:
            for _, (_, f) in files_to_upload:
                f.close()


def main():
    # Setup
    API_BASE_URL = 'http://localhost:1007'
    client = LLMApiClient(API_BASE_URL, timeout=1200.0)

    # Login
    print("Logging in...")
    client.login("leesihun", "s.hun.lee")
    print("✓ Logged in successfully\n")

    # Create test JSON
    complex_json = {
        "company": "TechMart Inc",
        "quarter": "Q3 2025",
        "departments": [
            {
                "name": "Electronics",
                "employees": 45,
                "sales": [
                    {"product": "Laptop", "units_sold": 320, "price": 1200, "revenue": 384000},
                    {"product": "Smartphone", "units_sold": 856, "price": 800, "revenue": 684800},
                    {"product": "Tablet", "units_sold": 142, "price": 500, "revenue": 71000}
                ]
            },
            {
                "name": "Home Appliances",
                "employees": 32,
                "sales": [
                    {"product": "Refrigerator", "units_sold": 89, "price": 1500, "revenue": 133500},
                    {"product": "Washing Machine", "units_sold": 124, "price": 900, "revenue": 111600},
                    {"product": "Microwave", "units_sold": 267, "price": 200, "revenue": 53400}
                ]
            },
            {
                "name": "Furniture",
                "employees": 28,
                "sales": [
                    {"product": "Desk", "units_sold": 178, "price": 450, "revenue": 80100},
                    {"product": "Chair", "units_sold": 432, "price": 150, "revenue": 64800},
                    {"product": "Bookshelf", "units_sold": 95, "price": 300, "revenue": 28500}
                ]
            }
        ]
    }

    json_name = './complex_json.json'
    with open(json_name, 'w') as f:
        json.dump(complex_json, f, indent=2)
    print(f"✓ Created test JSON file: {json_name}\n")

    # Test query
    analysis_query = """
Analyze the attached company data JSON file and tell me:
1. Which department has the highest total revenue?
2. What is the average revenue per employee across all departments?
3. Which single product generated the most revenue?
4. Calculate the total units sold across all departments.

Please provide exact numbers and show your calculations.
"""

    print("Expected Answers:")
    print("1. Electronics: 1,139,800")
    print("2. 15,356.19 (1,615,300 total revenue / 105 total employees)")
    print("3. Smartphone: 684,800")
    print("4. 2,503 units total")
    print("\n" + "="*80 + "\n")

    # Send request with file
    print("Sending request with file attachment...")
    print(f"Model: deepseek-r1:1.5b")
    print(f"Agent type: auto\n")

    json_reply, session_id = client.chat_new(
        "deepseek-r1:1.5b",
        analysis_query,
        files=[json_name]
    )

    print("="*80)
    print("RESPONSE FROM LLM:")
    print("="*80)
    print(json_reply)
    print("="*80)
    print(f"\nSession ID: {session_id}")

    # Cleanup
    Path(json_name).unlink()
    print(f"\n✓ Cleaned up {json_name}")


if __name__ == "__main__":
    main()
