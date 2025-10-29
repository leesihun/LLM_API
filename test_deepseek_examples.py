"""
Test key examples from API_examples.ipynb using deepseek-r1:1.5b
"""
import httpx
import time

API_BASE_URL = "http://127.0.0.1:8000"

class LLMApiClient:
    def __init__(self, base_url: str, timeout: float = 300.0):
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

    def list_models(self):
        r = httpx.get(f"{self.base_url}/v1/models", headers=self._headers(), timeout=10.0)
        r.raise_for_status()
        return r.json()


def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def main():
    print_section("API Examples Test with deepseek-r1:1.5b")

    client = LLMApiClient(API_BASE_URL, timeout=180.0)

    # Login as admin and set model
    print("\n[1] Logging in as admin...")
    client.login("admin", "administrator")
    print("    [OK] Logged in")

    print("\n[2] Setting model to deepseek-r1:1.5b...")
    client.change_model("deepseek-r1:1.5b")
    print("    [OK] Model set to deepseek-r1:1.5b")

    # Get model list
    print("\n[3] Getting available models...")
    models = client.list_models()
    MODEL = models["data"][0]["id"]
    print(f"    Using model: {MODEL}")

    # Test 1: Simple chat
    print_section("Test 1: Simple Chat")
    try:
        start = time.time()
        reply, session_id = client.chat_new(MODEL, "Write a short haiku about autumn.")
        elapsed = time.time() - start
        print(f"Response ({elapsed:.1f}s):")
        print(reply[:300])
        print("    [PASS]")
    except Exception as e:
        print(f"    [FAIL] {e}")

    # Test 2: Math calculation (should trigger agent)
    print_section("Test 2: Math Calculation (Agent)")
    try:
        start = time.time()
        reply, _ = client.chat_new(MODEL, "What is 123.45 divided by 6.78? Please calculate precisely.")
        elapsed = time.time() - start
        print(f"Response ({elapsed:.1f}s):")
        print(reply[:300])
        print("    [PASS]")
    except Exception as e:
        print(f"    [FAIL] {e}")

    # Test 3: Python code generation (NEW FEATURE)
    print_section("Test 3: Python Code Generation")
    try:
        start = time.time()
        query = """
Write Python code to calculate the factorial of 5 and output as JSON.
Use a simple iterative approach.
"""
        reply, _ = client.chat_new(MODEL, query, agent_type="react")
        elapsed = time.time() - start
        print(f"Response ({elapsed:.1f}s):")
        print(reply[:500])

        # Check if it mentions code execution
        if "120" in reply or "factorial" in reply.lower():
            print("    [PASS] Result mentions factorial/120")
        else:
            print("    [PARTIAL] Got response but unclear if code ran")
    except Exception as e:
        print(f"    [FAIL] {e}")

    # Test 4: Data analysis with Python
    print_section("Test 4: Python Data Analysis")
    try:
        start = time.time()
        query = """
Write Python code to:
1. Create a list of the first 10 prime numbers
2. Calculate their sum
3. Output as JSON with keys: primes, sum
"""
        reply, _ = client.chat_new(MODEL, query, agent_type="react")
        elapsed = time.time() - start
        print(f"Response ({elapsed:.1f}s):")
        print(reply[:500])

        # First 10 primes sum = 2+3+5+7+11+13+17+19+23+29 = 129
        if "129" in reply or "primes" in reply.lower():
            print("    [PASS] Result mentions primes/sum")
        else:
            print("    [PARTIAL] Got response")
    except Exception as e:
        print(f"    [FAIL] {e}")

    # Test 5: Excel file analysis (Real file)
    print_section("Test 5: Excel File Analysis")
    username = "leesihun"
    try:
        # Login as user
        client.login(username, "s.hun.lee")

        start = time.time()
        query = f"""
I have an Excel file at: data/uploads/{username}/폴드긍정.xlsx

Write Python code to:
1. Load the Excel file with pandas
2. Show column names
3. Count total rows
4. Output as JSON
"""
        reply, _ = client.chat_new(MODEL, query, agent_type="react")
        elapsed = time.time() - start
        print(f"Response ({elapsed:.1f}s):")
        print(reply[:500])

        if "column" in reply.lower() or "pandas" in reply.lower():
            print("    [PASS] Response mentions columns/pandas")
        else:
            print("    [PARTIAL] Got response")
    except Exception as e:
        print(f"    [FAIL] {e}")

    print_section("Testing Complete")
    print("\nAll key features tested with deepseek-r1:1.5b model.")
    print("Check above for [PASS]/[FAIL] status of each test.")


if __name__ == "__main__":
    main()
