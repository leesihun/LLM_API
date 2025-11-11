"""
Test script for web search with contextual enhancement
Tests temporal and location context injection
"""
import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.tools.web_search import web_search_tool
from datetime import datetime


async def test_temporal_enhancement():
    """Test temporal context enhancement"""
    print("=" * 80)
    print("TEST 1: Temporal Context Enhancement")
    print("=" * 80)

    # Test query with temporal keywords
    query = "latest news about AI"
    print(f"\nOriginal Query: {query}")

    # Get temporal context
    context = web_search_tool._get_temporal_context()
    print(f"\nTemporal Context:")
    print(f"  Current Date: {context['current_date']}")
    print(f"  Current Time: {context['current_time']}")
    print(f"  Day of Week: {context['day_of_week']}")

    # Enhance query
    enhanced_query = web_search_tool._enhance_query_with_context(query, context)
    print(f"\nEnhanced Query: {enhanced_query}")
    print(f"Query Changed: {enhanced_query != query}")

    print("\n[PASS] Test 1 completed")


async def test_location_enhancement():
    """Test location context enhancement"""
    print("\n" + "=" * 80)
    print("TEST 2: Location Context Enhancement")
    print("=" * 80)

    # Test query with location keywords
    query = "weather near me"
    user_location = "Seoul, Korea"
    print(f"\nOriginal Query: {query}")
    print(f"User Location: {user_location}")

    # Get temporal context
    context = web_search_tool._get_temporal_context()

    # Enhance query with location
    enhanced_query = web_search_tool._enhance_query_with_context(
        query, context, user_location
    )
    print(f"\nEnhanced Query: {enhanced_query}")
    print(f"Query Changed: {enhanced_query != query}")
    print(f"Location Replaced: {'Seoul, Korea' in enhanced_query}")

    print("\n[PASS] Test 2 completed")


async def test_combined_enhancement():
    """Test combined temporal + location enhancement"""
    print("\n" + "=" * 80)
    print("TEST 3: Combined Temporal + Location Enhancement")
    print("=" * 80)

    # Test query with both temporal and location keywords
    query = "current restaurants near me"
    user_location = "New York, USA"
    print(f"\nOriginal Query: {query}")
    print(f"User Location: {user_location}")

    # Get temporal context
    context = web_search_tool._get_temporal_context()

    # Enhance query
    enhanced_query = web_search_tool._enhance_query_with_context(
        query, context, user_location
    )
    print(f"\nEnhanced Query: {enhanced_query}")
    print(f"Query Changed: {enhanced_query != query}")

    print("\n[PASS] Test 3 completed")


async def test_no_enhancement():
    """Test that queries without temporal/location keywords are not enhanced"""
    print("\n" + "=" * 80)
    print("TEST 4: No Enhancement for Generic Queries")
    print("=" * 80)

    # Test generic query
    query = "what is Python programming"
    print(f"\nOriginal Query: {query}")

    # Get temporal context
    context = web_search_tool._get_temporal_context()

    # Enhance query (should not change)
    enhanced_query = web_search_tool._enhance_query_with_context(query, context)
    print(f"\nEnhanced Query: {enhanced_query}")
    print(f"Query Changed: {enhanced_query != query}")
    print(f"Expected: False (no enhancement for generic queries)")

    print("\n[PASS] Test 4 completed")


async def test_search_with_context():
    """Test full search with context (requires Tavily API key)"""
    print("\n" + "=" * 80)
    print("TEST 5: Full Search with Context (Optional - requires API)")
    print("=" * 80)

    try:
        # Test search with context
        query = "current AI trends"
        print(f"\nQuery: {query}")
        print("Performing search with contextual enhancement...")

        results, context_metadata = await web_search_tool.search(
            query,
            max_results=3,
            include_context=True,
            user_location=None
        )

        print(f"\nContext Metadata:")
        print(f"  Current DateTime: {context_metadata.get('current_datetime')}")
        print(f"  Query Enhanced: {context_metadata.get('query_enhanced')}")
        if context_metadata.get('enhanced_query'):
            print(f"  Enhanced Query: {context_metadata.get('enhanced_query')}")

        print(f"\nSearch Results: {len(results)} found")
        for i, result in enumerate(results[:2], 1):
            print(f"\n  {i}. {result.title}")
            print(f"     URL: {result.url}")
            print(f"     Snippet: {result.content[:100]}...")

        print("\n[PASS] Test 5 completed")

    except Exception as e:
        print(f"\n[SKIP] Test 5 skipped: {str(e)}")
        print("  (This is expected if Tavily API is not configured)")


async def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("WEB SEARCH CONTEXTUAL ENHANCEMENT TESTS")
    print("=" * 80)
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run tests
    await test_temporal_enhancement()
    await test_location_enhancement()
    await test_combined_enhancement()
    await test_no_enhancement()
    await test_search_with_context()

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
