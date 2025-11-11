"""
Test script for LLM-based query refinement
Tests the query optimization system with various inputs
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from backend.tools.web_search import web_search_tool


async def test_query_refinement():
    """Test query refinement with various inputs"""

    test_queries = [
        # Question-style queries
        "what is the latest news about AI",
        "how does machine learning work",
        "where can I find Python tutorials",
        "why is climate change important",
        "who is the CEO of OpenAI",

        # Natural language queries
        "tell me about OpenAI GPT-4",
        "best restaurants near me",
        "weather tomorrow",

        # Comparison queries
        "Python vs JavaScript comparison",
        "electric cars versus gas cars",

        # Topic queries
        "what are the symptoms of COVID",
        "how to learn programming",
        "latest developments in quantum computing",

        # Already good queries (should improve minimally)
        "AI machine learning news 2025",
        "weather forecast Seoul",
    ]

    print("=" * 80)
    print("QUERY REFINEMENT TEST")
    print("=" * 80)
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total test queries: {len(test_queries)}")
    print("=" * 80)

    temporal_context = web_search_tool._get_temporal_context()

    improvements = 0
    no_changes = 0

    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Testing query")
        print("-" * 80)

        try:
            refined = await web_search_tool.refine_search_query(
                query,
                temporal_context,
                user_location="Seoul, Korea"
            )

            improved = refined != query
            if improved:
                improvements += 1
            else:
                no_changes += 1

            print(f"Original:  {query}")
            print(f"Refined:   {refined}")
            print(f"Changed:   {'YES' if improved else 'NO'}")

        except Exception as e:
            print(f"ERROR: {e}")

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total queries:    {len(test_queries)}")
    print(f"Improved:         {improvements}")
    print(f"No change:        {no_changes}")
    print(f"Improvement rate: {improvements/len(test_queries)*100:.1f}%")
    print("=" * 80)


async def test_full_search_with_refinement():
    """Test full search pipeline with refinement"""

    print("\n" + "=" * 80)
    print("FULL SEARCH TEST WITH REFINEMENT")
    print("=" * 80)

    test_cases = [
        {
            "query": "what's the latest AI news",
            "description": "News query with question words"
        },
        {
            "query": "how does quantum computing work",
            "description": "Explanation query"
        },
        {
            "query": "Python vs JavaScript",
            "description": "Comparison query"
        },
    ]

    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        description = test_case["description"]

        print(f"\nTest Case {i}: {description}")
        print("-" * 80)
        print(f"Query: {query}")

        try:
            results, metadata = await web_search_tool.search(
                query,
                max_results=3,
                include_context=True,
                user_location=None
            )

            print(f"\nQuery Processing:")
            print(f"  Original:   {metadata['original_query']}")

            if metadata.get('query_refinement_applied'):
                print(f"  Refined:    {metadata['refined_query']}")

            if metadata.get('enhanced_query'):
                print(f"  Enhanced:   {metadata['enhanced_query']}")

            print(f"\nResults: {len(results)} found")

            if results:
                print("\nTop 3 Results:")
                for j, result in enumerate(results[:3], 1):
                    print(f"\n  {j}. {result.title}")
                    print(f"     URL: {result.url}")
                    print(f"     Snippet: {result.content[:80]}...")
            else:
                print("\nNo results found")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("FULL SEARCH TEST COMPLETED")
    print("=" * 80)


async def test_refinement_on_off():
    """Test query refinement can be toggled on/off"""

    print("\n" + "=" * 80)
    print("ON/OFF TOGGLE TEST")
    print("=" * 80)

    query = "what is the latest news about AI"

    # Test with refinement ON
    print("\n1. With Refinement ENABLED:")
    print("-" * 80)
    web_search_tool.enable_query_refinement = True

    results_on, metadata_on = await web_search_tool.search(query, max_results=3)
    print(f"Original:  {metadata_on['original_query']}")
    print(f"Refined:   {metadata_on.get('refined_query', 'N/A')}")
    print(f"Applied:   {metadata_on.get('query_refinement_applied', False)}")
    print(f"Results:   {len(results_on)}")

    # Test with refinement OFF
    print("\n2. With Refinement DISABLED:")
    print("-" * 80)
    web_search_tool.enable_query_refinement = False

    results_off, metadata_off = await web_search_tool.search(query, max_results=3)
    print(f"Original:  {metadata_off['original_query']}")
    print(f"Refined:   {metadata_off.get('refined_query', 'N/A')}")
    print(f"Applied:   {metadata_off.get('query_refinement_applied', False)}")
    print(f"Results:   {len(results_off)}")

    # Restore default
    web_search_tool.enable_query_refinement = True

    print("\n" + "=" * 80)
    print("Toggle test completed - refinement restored to ENABLED")
    print("=" * 80)


async def main():
    """Run all tests"""

    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "QUERY REFINEMENT TEST SUITE" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")

    # Test 1: Query refinement only
    await test_query_refinement()

    # Test 2: Full search pipeline
    await test_full_search_with_refinement()

    # Test 3: On/Off toggle
    await test_refinement_on_off()

    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 28 + "ALL TESTS COMPLETED" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
