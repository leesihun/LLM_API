"""
Test script for LLM-based query refinement
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from backend.tools.web_search import web_search_tool


async def test_simple():
    """Simple test of query refinement"""
    
    print("=" * 80)
    print("SIMPLE QUERY REFINEMENT TEST")
    print("=" * 80)
    
    test_queries = [
        "what is the latest news about AI",
        "how does machine learning work",
        "Python vs JavaScript comparison",
    ]
    
    temporal_context = web_search_tool._get_temporal_context()
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 80)
        
        try:
            refined = await web_search_tool.refine_search_query(
                query,
                temporal_context,
                user_location=None
            )
            
            print(f"Original:  {query}")
            print(f"Refined:   {refined}")
            print(f"Changed:   {'YES' if refined != query else 'NO'}")
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_simple())
