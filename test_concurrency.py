"""
Test script to verify concurrent request handling
Sends multiple requests simultaneously and measures response times
"""
import asyncio
import httpx
import time
from datetime import datetime

# Configuration
API_URL = "http://localhost:10007/v1/chat/completions"
USERNAME = "admin"
PASSWORD = "administrator"
NUM_CONCURRENT_REQUESTS = 4

async def login():
    """Get JWT token"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:10007/api/auth/login",
            data={"username": USERNAME, "password": PASSWORD}
        )
        if response.status_code == 200:
            return response.json()["access_token"]
        else:
            raise Exception(f"Login failed: {response.status_code}")

async def send_request(request_id: int, token: str):
    """Send a single chat request"""
    start_time = time.time()
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Request {request_id}: Starting")

    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "model": "default",
        "messages": [
            {"role": "user", "content": f"Count from 1 to 5. This is request {request_id}."}
        ],
        "stream": False,
        "agent_type": "chat"
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(API_URL, json=payload, headers=headers)
            elapsed = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Request {request_id}: ✓ Completed in {elapsed:.2f}s")
                print(f"    Response preview: {content[:100]}...")
                return {"id": request_id, "elapsed": elapsed, "status": "success"}
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Request {request_id}: ✗ Failed ({response.status_code})")
                return {"id": request_id, "elapsed": elapsed, "status": "failed", "error": response.status_code}
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Request {request_id}: ✗ Error: {e}")
        return {"id": request_id, "elapsed": elapsed, "status": "error", "error": str(e)}

async def test_concurrent_requests():
    """Test multiple concurrent requests"""
    print("=" * 70)
    print("LLM API Concurrency Test")
    print("=" * 70)
    print()

    # Login
    print("Logging in...")
    token = await login()
    print("✓ Login successful")
    print()

    # Send concurrent requests
    print(f"Sending {NUM_CONCURRENT_REQUESTS} concurrent requests...")
    print()

    start_time = time.time()

    # Create tasks for concurrent execution
    tasks = [send_request(i+1, token) for i in range(NUM_CONCURRENT_REQUESTS)]

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)

    total_time = time.time() - start_time

    # Print results
    print()
    print("=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"Total requests: {NUM_CONCURRENT_REQUESTS}")
    print(f"Total time: {total_time:.2f}s")
    print()

    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] != "success")

    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print()

    if successful > 0:
        avg_time = sum(r["elapsed"] for r in results if r["status"] == "success") / successful
        max_time = max(r["elapsed"] for r in results if r["status"] == "success")
        min_time = min(r["elapsed"] for r in results if r["status"] == "success")

        print(f"Response times:")
        print(f"  Average: {avg_time:.2f}s")
        print(f"  Min: {min_time:.2f}s")
        print(f"  Max: {max_time:.2f}s")
        print()

        # Check if truly concurrent
        if max_time < total_time * 1.2:  # Allow 20% margin
            print("✓ Requests appear to be processed CONCURRENTLY")
            print(f"  (max single request time {max_time:.2f}s ≈ total time {total_time:.2f}s)")
        else:
            print("⚠ Requests appear to be processed SEQUENTIALLY")
            print(f"  (max single request time {max_time:.2f}s << total time {total_time:.2f}s)")

    print()
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(test_concurrent_requests())
