"""
Test script to verify web search metadata is saved to conversations
"""
import asyncio
import json
from pathlib import Path
from backend.storage.conversation_store import conversation_store
from backend.tasks.React import react_agent
from backend.models.schemas import ChatMessage

async def test_websearch_recording():
    """Test that web search API calls are recorded in conversation metadata"""

    # Create a test session
    test_user_id = "test_user"
    session_id = conversation_store.create_session(test_user_id)
    print(f"Created test session: {session_id}")

    # Create a query that will trigger web search
    test_query = "What is the latest news about artificial intelligence?"
    messages = [ChatMessage(role="user", content=test_query)]

    # Execute ReAct agent (which should use web search)
    print(f"\nExecuting ReAct agent with query: {test_query}")
    final_answer, metadata = await react_agent.execute(
        messages=messages,
        session_id=session_id,
        user_id=test_user_id,
        file_paths=None
    )

    print(f"\nFinal answer: {final_answer[:200]}...")
    print(f"\nMetadata keys: {list(metadata.keys())}")
    print(f"Tools used: {metadata.get('tools_used', [])}")
    print(f"Total iterations: {metadata.get('total_iterations', 0)}")

    # Save the conversation with metadata
    conversation_store.add_message(session_id, "user", test_query, metadata={
        "task_type": "agentic"
    })

    conversation_store.add_message(session_id, "assistant", final_answer, metadata={
        "task_type": "agentic",
        "agent_metadata": metadata
    })

    print(f"\n✓ Saved conversation to session {session_id}")

    # Load and inspect the conversation file
    conv = conversation_store.load_conversation(session_id)
    conv_file = conversation_store._get_conversation_file(session_id, test_user_id, conv.created_at)

    print(f"\n✓ Conversation file: {conv_file}")

    # Print the raw JSON to see what's saved
    with open(conv_file, 'r', encoding='utf-8') as f:
        conv_data = json.load(f)

    print("\n" + "=" * 80)
    print("CONVERSATION JSON STRUCTURE:")
    print("=" * 80)
    print(json.dumps(conv_data, indent=2, ensure_ascii=False)[:2000])
    print("...")

    # Check if web search metadata is present
    assistant_msg = conv_data['messages'][-1]
    if 'metadata' in assistant_msg and 'agent_metadata' in assistant_msg['metadata']:
        agent_meta = assistant_msg['metadata']['agent_metadata']
        print(f"\n✓ Agent metadata found!")
        print(f"  - Agent type: {agent_meta.get('agent_type')}")
        print(f"  - Tools used: {agent_meta.get('tools_used')}")
        print(f"  - Total iterations: {agent_meta.get('total_iterations')}")

        if 'execution_steps' in agent_meta:
            print(f"\n✓ Execution steps found: {len(agent_meta['execution_steps'])} steps")
            for i, step in enumerate(agent_meta['execution_steps']):
                print(f"\n  Step {i+1}:")
                print(f"    Action: {step.get('action')}")
                print(f"    Observation preview: {step.get('observation', '')[:100]}...")

                # Check if this was a web search step
                if step.get('action') == 'web_search':
                    print(f"    ⚠ WEB SEARCH STEP FOUND!")
                    print(f"    Full observation length: {len(step.get('observation', ''))} chars")
        else:
            print(f"\n✗ No execution_steps in agent metadata")
    else:
        print(f"\n✗ No agent metadata in assistant message")

    # Cleanup
    conversation_store.delete_conversation(session_id)
    print(f"\n✓ Cleaned up test session {session_id}")

if __name__ == "__main__":
    asyncio.run(test_websearch_recording())
