"""
Comprehensive Test Suite for New Tools
Tests: Python Executor, Math Calculator, Wikipedia, Weather, SQL Query, Pagination
"""

import requests
import time

# Configuration
BASE_URL = "http://localhost:8000"
USERNAME = "admin"
PASSWORD = "admin123"


def test_login():
    """Authenticate and get token"""
    print("\n=== Authentication Test ===")
    response = requests.post(
        f"{BASE_URL}/api/auth/login",
        json={"username": USERNAME, "password": PASSWORD}
    )

    if response.status_code == 200:
        token = response.json()["access_token"]
        print("[OK] Login successful")
        return token
    else:
        print(f"[X] Login failed: {response.status_code}")
        return None


def chat_request(token, message, agent_type="react"):
    """Send chat request and measure response time"""
    headers = {"Authorization": f"Bearer {token}"}

    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers=headers,
        json={
            "model": "llama3.2:latest",
            "messages": [{"role": "user", "content": message}],
            "agent_type": agent_type
        },
        timeout=180
    )
    elapsed = time.time() - start_time

    if response.status_code == 200:
        answer = response.json()["choices"][0]["message"]["content"]
        return True, answer, elapsed
    else:
        return False, f"Error {response.status_code}: {response.text}", elapsed


def test_python_executor(token):
    """Test Python code execution tool"""
    print("\n=== Python Executor Test ===")

    query = """
    Execute this Python code:
    result = sum([1, 2, 3, 4, 5])
    squares = [x**2 for x in range(1, 6)]
    print(f"Sum: {result}, Squares: {squares}")
    """

    success, answer, elapsed = chat_request(token, query)

    print(f"Query: {query.strip()}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Status: {'[OK]' if success else '[X]'}")
    print(f"Response:\n{answer}\n")

    return success


def test_math_calculator(token):
    """Test math calculator with symbolic computation"""
    print("\n=== Math Calculator Test ===")

    test_cases = [
        "Calculate the derivative of x^3 + 2x^2 - 5x + 7",
        "Solve the equation 2x + 5 = 15",
        "Calculate the integral of sin(x) from 0 to pi"
    ]

    results = []
    for query in test_cases:
        success, answer, elapsed = chat_request(token, query)
        print(f"\nQuery: {query}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Status: {'[OK]' if success else '[X]'}")
        print(f"Response: {answer[:200]}...")
        results.append(success)

    return all(results)


def test_wikipedia(token):
    """Test Wikipedia search and summarization"""
    print("\n=== Wikipedia Test ===")

    query = "Search Wikipedia for information about Artificial Intelligence"

    success, answer, elapsed = chat_request(token, query)

    print(f"Query: {query}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Status: {'[OK]' if success else '[X]'}")
    print(f"Response:\n{answer[:300]}...\n")

    return success


def test_weather(token):
    """Test weather API"""
    print("\n=== Weather API Test ===")

    query = "What is the current weather in Seoul?"

    success, answer, elapsed = chat_request(token, query)

    print(f"Query: {query}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Status: {'[OK]' if success else '[X]'}")
    print(f"Response:\n{answer[:300]}...\n")

    return success


def test_sql_query(token):
    """Test SQL query tool"""
    print("\n=== SQL Query Test ===")

    test_cases = [
        "Query the database: SELECT * FROM users",
        "Query the database: SELECT name, price FROM products WHERE category = 'Electronics'",
        "Query the database: SELECT COUNT(*) as total_products FROM products"
    ]

    results = []
    for query in test_cases:
        success, answer, elapsed = chat_request(token, query)
        print(f"\nQuery: {query}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Status: {'[OK]' if success else '[X]'}")
        print(f"Response: {answer[:200]}...")
        results.append(success)

    return all(results)


def test_pagination(token):
    """Test document listing pagination"""
    print("\n=== Pagination Test ===")

    headers = {"Authorization": f"Bearer {token}"}

    # Test page 1 with default page_size
    response1 = requests.get(
        f"{BASE_URL}/api/files/documents",
        headers=headers,
        params={"page": 1, "page_size": 5}
    )

    # Test page 2
    response2 = requests.get(
        f"{BASE_URL}/api/files/documents",
        headers=headers,
        params={"page": 2, "page_size": 5}
    )

    if response1.status_code == 200 and response2.status_code == 200:
        data1 = response1.json()
        data2 = response2.json()

        print(f"Page 1: {len(data1['documents'])} documents")
        print(f"  Total: {data1['total']}, Page: {data1['page']}, Page Size: {data1['page_size']}, Total Pages: {data1['total_pages']}")

        print(f"\nPage 2: {len(data2['documents'])} documents")
        print(f"  Total: {data2['total']}, Page: {data2['page']}, Page Size: {data2['page_size']}, Total Pages: {data2['total_pages']}")

        # Verify pagination metadata
        success = (
            data1['page'] == 1 and
            data2['page'] == 2 and
            data1['page_size'] == 5 and
            data2['page_size'] == 5 and
            data1['total'] == data2['total']
        )

        print(f"\nStatus: {'[OK]' if success else '[X]'}")
        return success
    else:
        print(f"[X] Pagination failed: {response1.status_code}, {response2.status_code}")
        return False


def test_react_agent(token):
    """Test ReAct agent with multi-step task"""
    print("\n=== ReAct Agent Multi-Step Test ===")

    query = """
    First, search Wikipedia for Python programming language.
    Then, execute Python code to calculate the sum of numbers from 1 to 10.
    Finally, provide a summary combining both results.
    """

    success, answer, elapsed = chat_request(token, query, agent_type="react")

    print(f"Query: {query.strip()}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Status: {'[OK]' if success else '[X]'}")
    print(f"Response:\n{answer[:400]}...\n")

    return success


def main():
    """Run all tests"""
    print("=" * 80)
    print("NEW TOOLS COMPREHENSIVE TEST SUITE")
    print("=" * 80)

    # Authenticate
    token = test_login()
    if not token:
        print("\n[X] Authentication failed, cannot proceed with tests")
        return

    results = {}

    # Run all tests
    print("\n" + "=" * 80)
    print("TESTING ALL NEW TOOLS")
    print("=" * 80)

    results['python_executor'] = test_python_executor(token)
    results['math_calculator'] = test_math_calculator(token)
    results['wikipedia'] = test_wikipedia(token)
    results['weather'] = test_weather(token)
    results['sql_query'] = test_sql_query(token)
    results['pagination'] = test_pagination(token)
    results['react_agent'] = test_react_agent(token)

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    total_tests = len(results)
    passed_tests = sum(results.values())

    for test_name, passed in results.items():
        status = "[OK]" if passed else "[X]"
        print(f"{status} {test_name}")

    print("\n" + "=" * 80)
    print(f"TOTAL: {passed_tests}/{total_tests} tests passed")
    print("=" * 80)

    if passed_tests == total_tests:
        print("\n[OK] All tests passed successfully!")
    else:
        print(f"\n[X] {total_tests - passed_tests} test(s) failed")


if __name__ == "__main__":
    main()
