"""
Test ReAct agent with tool calls to reproduce 500 error
"""
from backend.agents.react_agent import ReActAgent

# Initialize agent
agent = ReActAgent(model="deepseek-r1:1.5b", temperature=0.7)
agent.session_id = "test_react_500"

# Test with a simple web search
try:
    print("Testing ReAct agent with websearch...")
    response = agent.run(
        user_input="Search for 'Python tutorials'",
        conversation_history=[]
    )

    print("\n" + "=" * 80)
    print("FINAL RESPONSE:")
    print("=" * 80)
    print(response)
    print("=" * 80)

except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()
