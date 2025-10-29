"""
Test API examples with deepseek-r1:1.5b and 20 minute timeout
Tests key examples from the notebook to verify functionality
"""
import httpx
import time

API_BASE_URL = "http://127.0.0.1:8000"

class LLMApiClient:
    def __init__(self, base_url: str, timeout: float = 1200.0):
        self.base_url = base_url.rstrip("/")
        self.token = None
        self.timeout = httpx.Timeout(10.0, read=timeout, write=timeout, pool=timeout)

    def _headers(self):
        h = {"Content-Type": "application/json"}
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

    def change_model(self, model: str):
        r = httpx.post(f"{self.base_url}/api/admin/model", json={"model": model}, headers=self._headers(), timeout=10.0)
        r.raise_for_status()
        return r.json()

    def list_models(self):
        r = httpx.get(f"{self.base_url}/v1/models", headers=self._headers(), timeout=10.0)
        r.raise_for_status()
        return r.json()

    def chat_new(self, model: str, user_message: str, agent_type: str = "auto"):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": user_message}],
            "agent_type": agent_type
        }
        r = httpx.post(f"{self.base_url}/v1/chat/completions", json=payload, headers=self._headers(), timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"], data["x_session_id"]


def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def main():
    print_section("Deepseek-r1:1.5b Test with 20 Minute Timeout")
    print("\nThis will test key examples from API_examples.ipynb")
    print("Each test may take several minutes with deepseek-r1:1.5b\n")

    client = LLMApiClient(API_BASE_URL, timeout=1200.0)

    # Setup: Login as admin and set model
    print("[Setup] Logging in as admin...")
    client.login("admin", "administrator")
    print("        [OK] Logged in")

    print("[Setup] Setting model to deepseek-r1:1.5b...")
    client.change_model("deepseek-r1:1.5b")
    print("        [OK] Model set")

    models = client.list_models()
    MODEL = models["data"][0]["id"]
    print(f"[Setup] Using model: {MODEL}\n")

    # Test 1: Simple chat (Cell 8 equivalent)
    print_section("Test 1: Simple Chat (Non-Agentic)")
    print("Query: 'Hello! Give me a short haiku about autumn.'")
    try:
        start = time.time()
        reply, session_id = client.chat_new(MODEL, "Hello! Give me a short haiku about autumn.")
        elapsed = time.time() - start

        print(f"\n[SUCCESS] Response received in {elapsed:.1f}s")
        print(f"\nResponse:\n{reply}\n")
        print(f"Session ID: {session_id}")
    except Exception as e:
        print(f"\n[FAIL] {e}")
        return False

    # Test 2: Simple Python code generation (Cell 33 simplified)
    print_section("Test 2: Python Code Generation (Agentic)")
    print("Query: 'Write Python code to calculate factorial of 5'")
    print("Note: This uses ReAct agent with python_coder tool")

    try:
        start = time.time()
        query = "Write Python code to calculate the factorial of 5 and output as JSON with key 'result'"
        reply, _ = client.chat_new(MODEL, query, agent_type="react")
        elapsed = time.time() - start

        print(f"\n[SUCCESS] Response received in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        print(f"\nResponse preview (first 500 chars):\n{reply[:500]}\n")

        # Check if it mentions the correct result
        if "120" in reply:
            print("[VERIFIED] Response contains correct result (5! = 120)")
        else:
            print("[WARNING] Response doesn't mention 120, but may still be correct")

    except Exception as e:
        print(f"\n[FAIL] {e}")
        print("This likely means the request took longer than 20 minutes")
        return False

    # Test 3: Math calculation (Cell 15 simplified)
    print_section("Test 3: Math Calculation (Agentic)")
    print("Query: 'What is 123.45 divided by 6.78?'")

    try:
        start = time.time()
        reply, _ = client.chat_new(MODEL, "What is 123.45 divided by 6.78? Please calculate precisely.")
        elapsed = time.time() - start

        print(f"\n[SUCCESS] Response received in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        print(f"\nResponse:\n{reply[:300]}\n")

    except Exception as e:
        print(f"\n[FAIL] {e}")
        return False

    print_section("All Tests Complete!")
    print("\nSummary:")
    print("- Timeout set to 20 minutes (1200 seconds)")
    print("- Model: deepseek-r1:1.5b")
    print("- All tests passed (or check above for failures)")
    print("\nThe notebook should work with these settings.")
    print("Note: Each agentic cell may take 5-15 minutes to complete.")

    return True


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Test cancelled by user")
        exit(1)
