"""
Test script for Python Code Generator Tool
"""
import asyncio
import httpx
import json

BASE_URL = "http://localhost:8000"

# Default credentials from the system
USERNAME = "guest"
PASSWORD = "guest_test1"


async def login():
    """Login and get token"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/auth/login",
            json={"username": USERNAME, "password": PASSWORD}
        )
        if response.status_code == 200:
            data = response.json()
            print(f"[OK] Logged in as {USERNAME}")
            return data["access_token"]
        else:
            print(f"[ERROR] Login failed: {response.text}")
            return None


async def test_simple_calculation(token):
    """Test 1: Simple calculation (Fibonacci)"""
    print("\n" + "="*60)
    print("TEST 1: Simple Calculation - Fibonacci Sequence")
    print("="*60)

    query = """
Write Python code to calculate the Fibonacci sequence up to 100.
Use an efficient iterative approach and print the result as a JSON list.
"""

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{BASE_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "model": "gpt-oss:20b",
                "messages": [{"role": "user", "content": query}],
                "agent_type": "react"
            }
        )

        if response.status_code == 200:
            data = response.json()
            print(f"\n[OK] Status: Success")
            print(f"Agent: {data.get('agent_selected', 'N/A')}")
            print(f"\nResponse:\n{data['choices'][0]['message']['content'][:500]}...")
            return True
        else:
            print(f"[ERROR] Request failed: {response.status_code}")
            print(response.text)
            return False


async def test_data_analysis(token):
    """Test 2: Data analysis with pandas"""
    print("\n" + "="*60)
    print("TEST 2: Data Analysis - Pandas Operations")
    print("="*60)

    query = """
Write Python code to:
1. Create a pandas DataFrame with 50 rows of random sales data (date, product, quantity, price)
2. Calculate total revenue per product
3. Find the top 3 products by revenue
4. Output results as JSON

Use numpy for random data generation.
"""

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{BASE_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "model": "gpt-oss:20b",
                "messages": [{"role": "user", "content": query}],
                "agent_type": "react"
            }
        )

        if response.status_code == 200:
            data = response.json()
            print(f"\n[OK] Status: Success")
            print(f"Agent: {data.get('agent_selected', 'N/A')}")
            print(f"\nResponse:\n{data['choices'][0]['message']['content'][:500]}...")
            return True
        else:
            print(f"[ERROR] Request failed: {response.status_code}")
            print(response.text)
            return False


async def test_math_computation(token):
    """Test 3: Mathematical computation"""
    print("\n" + "="*60)
    print("TEST 3: Mathematical Computation - Prime Numbers")
    print("="*60)

    query = """
Write Python code to:
1. Calculate the first 15 prime numbers
2. Compute their sum and average
3. Find the largest prime in the list
4. Output results as JSON with keys: primes, sum, average, largest
"""

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{BASE_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "model": "gpt-oss:20b",
                "messages": [{"role": "user", "content": query}],
                "agent_type": "react"
            }
        )

        if response.status_code == 200:
            data = response.json()
            print(f"\n[OK] Status: Success")
            print(f"Agent: {data.get('agent_selected', 'N/A')}")
            print(f"\nResponse:\n{data['choices'][0]['message']['content'][:500]}...")
            return True
        else:
            print(f"[ERROR] Request failed: {response.status_code}")
            print(response.text)
            return False


async def main():
    """Run all tests"""
    print("="*60)
    print("Python Code Generator Tool - Test Suite")
    print("="*60)

    # Login
    token = await login()
    if not token:
        print("\n[ERROR] Cannot proceed without authentication")
        return

    # Run tests
    results = []

    # Test 1: Simple calculation
    result1 = await test_simple_calculation(token)
    results.append(("Simple Calculation", result1))

    # Test 2: Data analysis
    result2 = await test_data_analysis(token)
    results.append(("Data Analysis", result2))

    # Test 3: Math computation
    result3 = await test_math_computation(token)
    results.append(("Math Computation", result3))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results:
        status = "[PASSED]" if passed else "[FAILED]"
        print(f"{test_name:30s} {status}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All tests passed! Python Code Generator Tool is working correctly.")
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed. Please check the errors above.")


if __name__ == "__main__":
    asyncio.run(main())
