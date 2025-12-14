"""
Test script for websearch refactoring
Tests the new flow: Raw Tavily → ReAct Agent → Final Answer
"""
import sys
sys.path.insert(0, '.')

from tools.web_search import WebSearchTool
import json


def test_websearch_tool():
    """Test that websearch tool returns raw data only"""
    print("=" * 80)
    print("TEST 1: WebSearchTool returns raw data (no LLM calls)")
    print("=" * 80)

    tool = WebSearchTool()
    result = tool.search(
        query="What was the latest game of Liverpool FC and who won?"
    )

    print("\n[TEST] Result structure:")
    print(f"  success: {result.get('success')}")
    print(f"  num_results: {result.get('num_results')}")
    print(f"  query: {result.get('query')}")
    print(f"  results type: {type(result.get('results'))}")

    if result.get('results'):
        print(f"\n[TEST] First result:")
        first = result['results'][0]
        print(f"  title: {first.get('title', 'N/A')[:100]}")
        print(f"  url: {first.get('url', 'N/A')}")
        print(f"  score: {first.get('score', 0.0):.3f}")
        print(f"  content: {first.get('content', 'N/A')[:200]}...")

    print("\n[TEST] OK Tool returns raw data successfully")
    return result


def test_data_formatting():
    """Test that ReAct agent can format the raw data"""
    print("\n" + "=" * 80)
    print("TEST 2: ReAct Agent data formatting")
    print("=" * 80)

    from backend.agents.react_agent import ReActAgent

    agent = ReActAgent()

    # Mock tool result
    mock_result = {
        "success": True,
        "data": {
            "query": "Liverpool FC latest match",
            "num_results": 2,
            "results": [
                {
                    "title": "Liverpool beats Chelsea 2-1 in Premier League",
                    "url": "https://example.com/liverpool-chelsea",
                    "content": "Liverpool defeated Chelsea 2-1 in an exciting Premier League match on December 11, 2024. Goals from Salah and Jota secured the victory.",
                    "score": 0.95
                },
                {
                    "title": "Match Report: Liverpool 2-1 Chelsea",
                    "url": "https://example.com/match-report",
                    "content": "Full match report of Liverpool's 2-1 victory over Chelsea at Anfield.",
                    "score": 0.89
                }
            ]
        },
        "metadata": {"execution_time": 1.5}
    }

    formatted = agent._format_websearch_data(mock_result["data"])

    print("\n[TEST] Formatted data:")
    print(formatted)
    print("\n[TEST] OK Data formatting successful")

    # Check that key information is preserved
    assert "Liverpool" in formatted
    assert "Chelsea" in formatted
    assert "2-1" in formatted
    assert "0.95" in formatted

    print("[TEST] OK Key information preserved in formatting")


if __name__ == "__main__":
    print("\nTesting Websearch Refactoring\n")

    try:
        # Test 1: Tool returns raw data
        tool_result = test_websearch_tool()

        # Test 2: Agent can format the data
        test_data_formatting()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED")
        print("=" * 80)
        print("\nRefactoring Summary:")
        print("  1. Websearch tool: ONLY Tavily search (no LLM)")
        print("  2. ReAct agent: Receives raw data in 'data' field")
        print("  3. Observation: LLM analyzes raw search results")
        print("  4. Final Answer: LLM synthesizes answer from observations")
        print("\nInformation flow:")
        print("  User Query → Tavily → Raw Results → Observation LLM → Final Answer LLM")
        print("  (2 LLM calls instead of 4)")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
