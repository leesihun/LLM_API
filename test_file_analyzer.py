"""
Test script for the enhanced file_analyzer_tool.py
Tests the hybrid approach with deep structure analysis.
"""

import json
from backend.tools.file_analyzer_tool import analyze_files

def test_json_analysis():
    """Test JSON file analysis with the stats file."""

    print("=" * 80)
    print("Testing File Analyzer - Deep JSON Structure Analysis")
    print("=" * 80)

    # Test file path
    test_file = r"c:\Users\Lee\Desktop\Huni\LLM_API\data\uploads\leesihun\20251013_stats.json"

    # Run analysis
    result = analyze_files(
        file_paths=[test_file],
        user_query="What is the structure of this file?"
    )

    # Display results
    if result["success"]:
        print(f"\n[SUCCESS] Analysis successful!")
        print(f"Files analyzed: {result['files_analyzed']}")
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(result["summary"])

        print("\n" + "=" * 80)
        print("DETAILED RESULTS")
        print("=" * 80)

        for file_result in result["results"]:
            print(f"\nFile: {file_result['file']}")
            print(f"Format: {file_result['format']}")
            print(f"Size: {file_result['size_human']}")
            print(f"Structure: {file_result['structure']}")

            if 'keys' in file_result:
                print(f"Top-level keys: {file_result['keys']}")

            if 'structure_summary' in file_result:
                print("\n" + "-" * 80)
                print("STRUCTURE SUMMARY:")
                print("-" * 80)
                print(file_result['structure_summary'])

            if 'depth_analysis' in file_result:
                print("\n" + "-" * 80)
                print("DEPTH ANALYSIS (JSON):")
                print("-" * 80)
                print(json.dumps(file_result['depth_analysis'], indent=2, ensure_ascii=False))

    else:
        print(f"\n[ERROR] Analysis failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    test_json_analysis()
