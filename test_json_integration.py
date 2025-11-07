"""
Test JSON integration between file_analyzer and python_coder_tool

This test verifies that:
1. file_analyzer correctly analyzes JSON structure
2. python_coder_tool uses that analysis in metadata
3. File context includes rich structure information
"""

import json
import tempfile
import asyncio
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from backend.tools.python_coder_tool import python_coder_tool
from backend.tools.file_analyzer_tool import file_analyzer


def test_json_metadata_extraction():
    """Test that JSON metadata extraction uses file_analyzer"""

    # Create a test JSON file with nested structure
    test_data = {
        "users": [
            {"id": 1, "name": "Alice", "profile": {"age": 30, "city": "NYC"}},
            {"id": 2, "name": "Bob", "profile": {"age": 25, "city": "LA"}}
        ],
        "metadata": {
            "version": "1.0",
            "count": 2
        },
        "config": {
            "enabled": True,
            "settings": {
                "theme": "dark",
                "language": "en"
            }
        }
    }

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(test_data, f, indent=2)
        temp_path = f.name

    try:
        # Test file_analyzer directly
        print("=" * 80)
        print("1. Testing file_analyzer._analyze_json()")
        print("=" * 80)

        analysis = file_analyzer._analyze_json(temp_path)

        print(f"Structure: {analysis.get('structure')}")
        print(f"Keys: {analysis.get('keys')}")
        print(f"\nStructure Summary:")
        print(analysis.get('structure_summary', 'N/A'))

        # Verify expected results
        assert analysis.get('structure') == 'dict', "Should detect dict structure"
        assert 'users' in analysis.get('keys', []), "Should find 'users' key"
        assert 'metadata' in analysis.get('keys', []), "Should find 'metadata' key"
        assert 'config' in analysis.get('keys', []), "Should find 'config' key"
        assert 'depth_analysis' in analysis, "Should include depth analysis"

        # Calculate max depth from depth_analysis
        depth_analysis = analysis.get('depth_analysis', {})
        max_depth = file_analyzer._find_max_depth(depth_analysis) if depth_analysis else 0
        print(f"\nCalculated max depth: {max_depth}")
        assert max_depth >= 3, f"Should detect nested depth (got {max_depth})"

        print("\n✅ file_analyzer works correctly!")

        # Test python_coder_tool metadata extraction
        print("\n" + "=" * 80)
        print("2. Testing python_coder_tool._extract_file_metadata()")
        print("=" * 80)

        metadata = python_coder_tool._extract_file_metadata(Path(temp_path))

        print(f"Type: {metadata.get('type')}")
        print(f"Structure: {metadata.get('structure')}")
        print(f"Keys: {metadata.get('keys')}")
        print(f"Max depth: {metadata.get('max_depth')}")
        print(f"Items count: {metadata.get('items_count')}")

        # Verify metadata includes file_analyzer results
        assert metadata.get('type') == 'json', "Should identify as JSON"
        assert metadata.get('structure') == 'dict', "Should include structure"
        assert 'keys' in metadata, "Should include keys from file_analyzer"
        assert len(metadata.get('keys', [])) == 3, "Should have 3 top-level keys"
        assert 'structure_summary' in metadata, "Should include structure summary"

        print("\n✅ python_coder_tool metadata extraction works!")

        # Test file context building
        print("\n" + "=" * 80)
        print("3. Testing python_coder_tool._build_file_context()")
        print("=" * 80)

        validated_files = {temp_path: "test_data.json"}
        file_metadata = {temp_path: metadata}

        file_context = python_coder_tool._build_file_context(validated_files, file_metadata)

        print(file_context)

        # Verify context includes rich information
        assert "test_data.json" in file_context, "Should include filename"
        assert "Top-level keys:" in file_context, "Should show top-level keys"
        assert "users" in file_context, "Should mention 'users' key"
        assert "Structure breakdown:" in file_context, "Should include structure breakdown"
        assert "encoding='utf-8'" in file_context, "Should show proper encoding in example"
        assert "json.load(f)" in file_context, "Should show json.load usage"
        assert "with open" in file_context, "Should use context manager"

        print("\n✅ File context includes rich JSON structure info!")

        print("\n" + "=" * 80)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 80)
        print("\nThe integration successfully:")
        print("  ✅ Uses file_analyzer for deep JSON structure analysis")
        print("  ✅ Extracts nested keys, paths, and hierarchy")
        print("  ✅ Displays rich structure information in file context")
        print("  ✅ Shows proper JSON loading patterns with encoding")
        print("  ✅ Provides LLM with actionable structure details")

    finally:
        # Cleanup
        Path(temp_path).unlink()


def test_malformed_json():
    """Test handling of malformed JSON"""

    print("\n\n" + "=" * 80)
    print("4. Testing malformed JSON handling")
    print("=" * 80)

    # Create malformed JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        f.write('{"invalid": json, "missing": quotes}')
        temp_path = f.name

    try:
        metadata = python_coder_tool._extract_file_metadata(Path(temp_path))

        print(f"Type: {metadata.get('type')}")
        print(f"Error: {metadata.get('error')}")
        print(f"Parsing note: {metadata.get('parsing_note')}")

        # Should gracefully handle malformed JSON
        assert metadata.get('type') == 'json', "Should still identify as JSON"
        assert 'error' in metadata or 'parsing_note' in metadata, "Should indicate parsing issue"

        # Build context with error
        validated_files = {temp_path: "malformed.json"}
        file_metadata = {temp_path: metadata}
        file_context = python_coder_tool._build_file_context(validated_files, file_metadata)

        print("\nFile context with error:")
        print(file_context)

        assert "malformed.json" in file_context, "Should include filename"
        assert "⚠️" in file_context or "error" in file_context.lower(), "Should show warning"

        print("\n✅ Malformed JSON handled gracefully!")

    finally:
        Path(temp_path).unlink()


if __name__ == "__main__":
    print("Testing JSON Integration: file_analyzer + python_coder_tool\n")

    test_json_metadata_extraction()
    test_malformed_json()

    print("\n" + "=" * 80)
    print("✅ ALL INTEGRATION TESTS PASSED!")
    print("=" * 80)
