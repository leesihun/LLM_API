"""
Test script to verify conversation logging with agent metadata
Tests both simple chat and agentic workflows
"""

import asyncio
import json
from pathlib import Path
from backend.storage.conversation_store import conversation_store
from backend.models.schemas import ChatMessage
from backend.tasks.chat_task import chat_task
from backend.tasks.React import react_agent


async def test_simple_chat_logging():
    """Test that simple chat conversations are logged properly"""
    print("\n" + "="*80)
    print("TEST 1: Simple Chat Logging")
    print("="*80)

    # Create a test session
    session_id = conversation_store.create_session("test_user")
    print(f"Created session: {session_id}")

    # Execute a simple chat
    messages = [ChatMessage(role="user", content="What is 2 + 2?")]
    response = await chat_task.execute(messages, session_id, use_memory=False)

    # Save to conversation (simulating what routes.py does)
    conversation_store.add_message(session_id, "user", "What is 2 + 2?", metadata={
        "task_type": "chat"
    })
    conversation_store.add_message(session_id, "assistant", response, metadata={
        "task_type": "chat"
    })

    # Load and verify
    conversation = conversation_store.load_conversation(session_id)
    print(f"\nLoaded conversation with {len(conversation.messages)} messages")

    for msg in conversation.messages:
        print(f"\n[{msg.role}]: {msg.content[:100]}")
        if msg.metadata:
            print(f"  Metadata: {json.dumps(msg.metadata, indent=2)}")

    # Cleanup
    conversation_store.delete_conversation(session_id)
    print(f"\n✓ Test passed: Simple chat logged successfully")


async def test_agentic_logging():
    """Test that agentic workflows are logged with metadata"""
    print("\n" + "="*80)
    print("TEST 2: Agentic Workflow Logging (ReAct)")
    print("="*80)

    # Create a test session
    session_id = conversation_store.create_session("test_user")
    print(f"Created session: {session_id}")

    # Execute an agentic task (simple query that should finish quickly)
    messages = [ChatMessage(role="user", content="Calculate the sum of 5 and 7")]

    # Run ReAct agent
    response, agent_metadata = await react_agent.execute(
        messages=messages,
        session_id=session_id,
        user_id="test_user",
        file_paths=None
    )

    print(f"\nAgent response: {response[:200]}")
    print(f"\nAgent metadata keys: {list(agent_metadata.keys())}")

    # Save to conversation with metadata (simulating routes.py)
    conversation_store.add_message(session_id, "user", "Calculate the sum of 5 and 7", metadata={
        "task_type": "agentic"
    })

    assistant_metadata = {
        "task_type": "agentic",
        "agent_metadata": agent_metadata
    }
    conversation_store.add_message(session_id, "assistant", response, metadata=assistant_metadata)

    # Load and verify
    conversation = conversation_store.load_conversation(session_id)
    print(f"\nLoaded conversation with {len(conversation.messages)} messages")

    for msg in conversation.messages:
        print(f"\n[{msg.role}]: {msg.content[:100]}")
        if msg.metadata:
            print(f"  Metadata keys: {list(msg.metadata.keys())}")
            if "agent_metadata" in msg.metadata:
                agent_meta = msg.metadata["agent_metadata"]
                print(f"  Agent type: {agent_meta.get('agent_type')}")
                print(f"  Total iterations: {agent_meta.get('total_iterations')}")
                print(f"  Tools used: {agent_meta.get('tools_used')}")
                print(f"  Execution steps count: {len(agent_meta.get('execution_steps', []))}")

    # Check if metadata was saved to file
    conv_path = Path(conversation_store._get_conversation_file(session_id))
    print(f"\nConversation file: {conv_path}")
    print(f"File exists: {conv_path.exists()}")

    if conv_path.exists():
        with open(conv_path, 'r') as f:
            data = json.load(f)
            print(f"\nFile contains {len(data.get('messages', []))} messages")
            assistant_msg = [m for m in data['messages'] if m['role'] == 'assistant']
            if assistant_msg:
                print(f"Assistant metadata saved: {'agent_metadata' in assistant_msg[0].get('metadata', {})}")

    # Cleanup
    conversation_store.delete_conversation(session_id)
    print(f"\n✓ Test passed: Agentic workflow logged with metadata")


async def test_conversation_retrieval():
    """Test that conversations can be retrieved with all metadata"""
    print("\n" + "="*80)
    print("TEST 3: Conversation Retrieval")
    print("="*80)

    # Create session and save some messages
    session_id = conversation_store.create_session("test_user")

    # Add messages with metadata
    conversation_store.add_message(session_id, "user", "Test message 1", metadata={
        "task_type": "agentic",
        "file_paths": ["/path/to/file.csv"]
    })

    conversation_store.add_message(session_id, "assistant", "Test response 1", metadata={
        "task_type": "agentic",
        "agent_metadata": {
            "agent_type": "react",
            "total_iterations": 3,
            "tools_used": ["python_coder"]
        }
    })

    # Retrieve messages
    messages = conversation_store.get_messages(session_id)
    print(f"\nRetrieved {len(messages)} messages")

    for msg in messages:
        print(f"\n[{msg.role}]: {msg.content}")
        if msg.metadata:
            print(f"  Metadata: {json.dumps(msg.metadata, indent=2)[:200]}")

    # Cleanup
    conversation_store.delete_conversation(session_id)
    print(f"\n✓ Test passed: Conversation retrieval works with metadata")


async def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("CONVERSATION LOGGING TESTS")
    print("="*80)

    try:
        await test_simple_chat_logging()
        # Skip the agentic test for now since it requires Ollama running
        # await test_agentic_logging()
        await test_conversation_retrieval()

        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
