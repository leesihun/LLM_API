"""
Test script to verify real-time logging behavior
"""
import time
from backend.core.llm_backend import llm_backend

def test_realtime_logging():
    """Test that logging happens in real-time (request then response)"""

    print("Testing real-time logging...")
    print("Check data/logs/prompts.log for real-time updates")
    print("-" * 80)

    # Test non-streaming
    print("\n1. Testing NON-STREAMING chat...")
    print("   Request should be logged immediately")
    print("   Response should be logged after LLM responds")

    messages = [
        {"role": "user", "content": "Say 'Hello World' and nothing else."}
    ]

    try:
        response = llm_backend.chat(
            messages=messages,
            model="qwen2.5:0.5b",
            temperature=0.7,
            session_id="test-session-123",
            agent_type="test"
        )
        print(f"\n   Got response: {response[:50]}...")
    except Exception as e:
        print(f"\n   Error: {e}")

    print("\n   Check prompts.log - you should see:")
    print("   1. REQUEST log with '>>> REQUEST' header")
    print("   2. RESPONSE log with '<<< RESPONSE' header")

    # Wait a bit
    time.sleep(2)

    # Test streaming
    print("\n2. Testing STREAMING chat...")
    print("   Request should be logged immediately")
    print("   Response should be logged after streaming completes")

    try:
        collected = ""
        for token in llm_backend.chat_stream(
            messages=messages,
            model="qwen2.5:0.5b",
            temperature=0.7,
            session_id="test-session-456",
            agent_type="test"
        ):
            collected += token
            print(token, end="", flush=True)
        print(f"\n\n   Streaming complete: {len(collected)} characters")
    except Exception as e:
        print(f"\n   Error: {e}")

    print("\n   Check prompts.log again - you should see:")
    print("   1. REQUEST log with '>>> REQUEST' and '[STREAMING...]'")
    print("   2. RESPONSE log with '<<< RESPONSE' and full response")

    print("\n" + "=" * 80)
    print("Test complete! Check data/logs/prompts.log to verify real-time logging.")
    print("=" * 80)

if __name__ == "__main__":
    test_realtime_logging()
