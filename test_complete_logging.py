"""
Test script to verify:
1. Response logging is working
2. No truncation in prompts
"""

import asyncio
from backend.utils.llm_factory import LLMFactory
from langchain_core.messages import HumanMessage, SystemMessage


async def test_complete_logging():
    """Test that both prompts and responses are logged, and content isn't truncated"""

    print("=" * 80)
    print("TESTING COMPLETE LOGGING (Prompts + Responses)")
    print("=" * 80)

    # Create LLM
    llm = LLMFactory.create_llm(model="deepseek-r1:1.5b", user_id="test_complete")

    # Create a long message to test truncation
    long_content = """This is a very long message with multiple lines.
Line 2 of the message continues here.
Line 3 has even more content to test if truncation happens.
Line 4 adds additional context.
Line 5 provides more information.
Line 6 continues the pattern.
Line 7 keeps going.
Line 8 adds more.
Line 9 is still here.
Line 10 completes this long message that should NOT be truncated in the logs."""

    print("\n1. Sending long message...")
    print(f"   Message length: {len(long_content)} chars")

    messages = [
        SystemMessage(content="You are a helpful assistant. Please respond briefly."),
        HumanMessage(content=long_content)
    ]

    try:
        response = await llm.ainvoke(messages)
        print(f"   [OK] Response received (length: {len(response.content)} chars)")
        print(f"   Response preview: {response.content[:100]}...")
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        return False

    # Check log file
    print("\n2. Checking log file...")
    from pathlib import Path

    log_file = Path("data/scratch/test_complete/prompts.log")
    if not log_file.exists():
        print(f"   [ERROR] Log file not found at {log_file}")
        return False

    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"   [OK] Log file exists (size: {len(content)} bytes)")

    # Verify prompt was logged
    if "[SYSTEM]" in content and "[HUMAN]" in content:
        print("   [OK] Prompt logged with message labels")
    else:
        print("   [ERROR] Prompt not properly formatted")
        return False

    # Verify NO truncation - check for last line of long message
    if "Line 10 completes this long message" in content:
        print("   [OK] Full message preserved (no truncation)")
    else:
        print("   [ERROR] Message was truncated!")
        return False

    # Verify response was logged
    if "TYPE: RESPONSE" in content:
        print("   [OK] Response logged")
    else:
        print("   [ERROR] Response NOT logged")
        return False

    # Check for response content
    if response.content[:50] in content:
        print("   [OK] Response content found in log")
    else:
        print("   [ERROR] Response content NOT in log")
        return False

    print("\n3. Log file structure:")
    print("-" * 80)
    # Show last 1500 chars (should show both prompt and response)
    print(content[-1500:])
    print("-" * 80)

    return True


if __name__ == "__main__":
    result = asyncio.run(test_complete_logging())

    print("\n" + "=" * 80)
    if result:
        print("[SUCCESS] COMPLETE LOGGING TEST PASSED")
        print("Both prompts and responses are logged without truncation!")
    else:
        print("[FAILED] COMPLETE LOGGING TEST FAILED")
    print("=" * 80)
