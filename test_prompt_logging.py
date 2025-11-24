"""
Test script to verify prompt logging functionality
"""

import asyncio
from backend.utils.llm_factory import LLMFactory


async def test_prompt_logging():
    """Test that prompts are being logged to user-specific files"""

    print("=" * 80)
    print("TESTING PROMPT LOGGING FUNCTIONALITY")
    print("=" * 80)

    # Test 1: Create LLM with user_id
    print("\n1. Creating LLM with user_id='test_user'...")
    llm = LLMFactory.create_llm(user_id="test_user")
    print("   [OK] LLM created successfully")

    # Test 2: Send a test prompt
    print("\n2. Sending test prompt...")
    test_prompt = "What is 2 + 2? Please answer briefly."

    try:
        response = await llm.ainvoke(test_prompt)
        print(f"   [OK] Response received: {response.content[:50]}...")
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        return False

    # Test 3: Check if log file was created
    print("\n3. Checking if log file was created...")
    from pathlib import Path

    log_file = Path("data/scratch/test_user/prompts.log")
    if log_file.exists():
        print(f"   [OK] Log file created at: {log_file}")

        # Read and display log content
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()

        print(f"\n4. Log file content ({len(content)} bytes):")
        print("-" * 80)
        # Show first 500 chars
        print(content[:500])
        if len(content) > 500:
            print(f"\n... ({len(content) - 500} more bytes)")
        print("-" * 80)

        # Verify prompt is in the log
        if test_prompt in content:
            print("\n   [OK] Test prompt found in log file!")
            return True
        else:
            print("\n   [ERROR] Test prompt NOT found in log file")
            return False
    else:
        print(f"   [ERROR] Log file NOT created at expected location: {log_file}")
        return False


if __name__ == "__main__":
    print("\nNote: This test requires Ollama to be running.")
    print("Starting test in 2 seconds...\n")

    import time
    time.sleep(2)

    result = asyncio.run(test_prompt_logging())

    print("\n" + "=" * 80)
    if result:
        print("[SUCCESS] PROMPT LOGGING TEST PASSED")
    else:
        print("[FAILED] PROMPT LOGGING TEST FAILED")
    print("=" * 80)
