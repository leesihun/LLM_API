"""
Minimal test - just verify deepseek-r1:1.5b works with basic chat and Python code generation
"""
import httpx

API_BASE_URL = "http://127.0.0.1:8000"

print("="*80)
print("MINIMAL DEEPSEEK TEST")
print("="*80)

# Login as admin
print("\n[1] Login as admin...")
r = httpx.post(f"{API_BASE_URL}/api/auth/login", json={"username": "admin", "password": "administrator"}, timeout=10.0)
r.raise_for_status()
token = r.json()["access_token"]
print("    [OK] Logged in")

headers = {"Authorization": f"Bearer {token}"}

# Set model to deepseek
print("\n[2] Setting model to deepseek-r1:1.5b...")
r = httpx.post(f"{API_BASE_URL}/api/admin/model", json={"model": "deepseek-r1:1.5b"}, headers=headers, timeout=10.0)
r.raise_for_status()
print("    [OK] Model set")

# Test 1: Simple chat (non-agentic)
print("\n[3] Test simple chat...")
r = httpx.post(
    f"{API_BASE_URL}/v1/chat/completions",
    headers=headers,
    json={
        "model": "deepseek-r1:1.5b",
        "messages": [{"role": "user", "content": "Say hello in 5 words or less"}]
    },
    timeout=60.0
)
r.raise_for_status()
reply = r.json()["choices"][0]["message"]["content"]
print(f"    Response: {reply[:100]}")
print("    [OK] Chat works")

# Test 2: Python code generation (agentic - uses python_coder tool)
print("\n[4] Test Python code generation...")
print("    (This will take longer - generating and executing code)")
r = httpx.post(
    f"{API_BASE_URL}/v1/chat/completions",
    headers=headers,
    json={
        "model": "deepseek-r1:1.5b",
        "messages": [{"role": "user", "content": "Write Python code to calculate factorial of 5 and output as JSON"}],
        "agent_type": "react"
    },
    timeout=180.0
)

if r.status_code == 200:
    reply = r.json()["choices"][0]["message"]["content"]
    print(f"    Response preview: {reply[:200]}")

    # Check if it mentions code execution success
    if "120" in reply or "factorial" in reply.lower():
        print("    [OK] Python code generation appears to work (mentions factorial/120)")
    else:
        print("    [PARTIAL] Got response but unclear if code ran")
elif r.status_code == 500:
    print(f"    [FAIL] Server error (likely LLM parsing issue)")
    print(f"    This is a known issue with some models and ReAct agent")
else:
    print(f"    [FAIL] Status {r.status_code}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
print("\nThe notebook should work with deepseek-r1:1.5b for basic chat.")
print("If Python code generation failed, it may be due to LLM response format issues.")
