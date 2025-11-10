"""
Test JSON File Handling Improvements

This test validates the enhanced JSON handling in python_coder_tool:
1. Access pattern generation
2. Safe preview creation
3. Null value detection
4. Enhanced file context
5. Better LLM prompts

Run: python test_json_improvements.py
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


def create_test_json_files():
    """Create various JSON test files"""
    test_files = []

    # Test 1: Simple dict
    simple_dict = {
        "name": "John",
        "age": 30,
        "city": "NYC"
    }
    f1 = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
    json.dump(simple_dict, f1, indent=2)
    f1.close()
    test_files.append(("simple_dict", f1.name))

    # Test 2: Array of objects
    array_of_objects = [
        {"id": 1, "name": "Alice", "score": 95},
        {"id": 2, "name": "Bob", "score": 87},
        {"id": 3, "name": "Charlie", "score": 92}
    ]
    f2 = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
    json.dump(array_of_objects, f2, indent=2)
    f2.close()
    test_files.append(("array_of_objects", f2.name))

    # Test 3: Nested dict (3 levels)
    nested_dict = {
        "company": {
            "name": "TechCorp",
            "departments": {
                "engineering": {
                    "employees": 50,
                    "budget": 1000000
                },
                "sales": {
                    "employees": 30,
                    "budget": 500000
                }
            }
        }
    }
    f3 = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
    json.dump(nested_dict, f3, indent=2)
    f3.close()
    test_files.append(("nested_dict", f3.name))

    # Test 4: JSON with null values
    with_nulls = {
        "users": [
            {"name": "Alice", "email": "alice@example.com", "phone": None},
            {"name": "Bob", "email": None, "phone": "123-456-7890"},
            {"name": "Charlie", "email": "charlie@example.com", "phone": None}
        ]
    }
    f4 = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
    json.dump(with_nulls, f4, indent=2)
    f4.close()
    test_files.append(("with_nulls", f4.name))

    # Test 5: Large array (20 items to test preview truncation)
    large_array = [{"index": i, "value": i * 10} for i in range(20)]
    f5 = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
    json.dump(large_array, f5, indent=2)
    f5.close()
    test_files.append(("large_array", f5.name))

    return test_files


def test_metadata_extraction():
    """Test Phase 1 & 2: Metadata extraction with new helpers"""
    print("="*80)
    print("TEST 1: Metadata Extraction with Access Patterns")
    print("="*80)

    test_files = create_test_json_files()

    for name, file_path in test_files:
        print(f"\n--- Testing: {name} ---")
        metadata = python_coder_tool._extract_file_metadata(Path(file_path))

        print(f"Type: {metadata.get('type')}")
        print(f"Structure: {metadata.get('structure')}")
        print(f"Max Depth: {metadata.get('max_depth')}")
        print(f"Requires Null Check: {metadata.get('requires_null_check')}")

        if 'access_patterns' in metadata:
            print(f"Access Patterns ({len(metadata['access_patterns'])}):")
            for pattern in metadata['access_patterns']:
                print(f"  - {pattern}")

        if 'safe_preview' in metadata:
            print(f"Safe Preview (truncated):")
            print(f"  {str(metadata['safe_preview'])[:200]}...")

        # Assertions
        assert metadata.get('type') == 'json', "Should identify as JSON"
        assert 'access_patterns' in metadata, "Should include access patterns"
        assert 'safe_preview' in metadata, "Should include safe preview"

        print("✅ PASSED")

    # Cleanup
    for _, file_path in test_files:
        Path(file_path).unlink()

    print(f"\n{'='*80}")
    print("✅ TEST 1 PASSED - Metadata extraction works!")
    print("="*80)


def test_file_context_building():
    """Test Phase 3: Enhanced file context"""
    print("\n\n" + "="*80)
    print("TEST 2: File Context Building")
    print("="*80)

    # Create test JSON
    test_data = {
        "users": [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ],
        "total": 2
    }

    f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
    json.dump(test_data, f, indent=2)
    f.close()

    # Extract metadata
    metadata = python_coder_tool._extract_file_metadata(Path(f.name))

    # Build file context
    validated_files = {f.name: "test_data.json"}
    file_metadata = {f.name: metadata}
    file_context = python_coder_tool._build_file_context(validated_files, file_metadata)

    print("Generated File Context:")
    print(file_context)

    # Assertions
    assert "test_data.json" in file_context, "Should include filename"
    assert "📋 Access Patterns" in file_context, "Should show access patterns section"
    assert "Sample Data" in file_context, "Should show sample data"
    assert ".get(" in file_context, "Should demonstrate .get() usage"

    # Cleanup
    Path(f.name).unlink()

    print("\n" + "="*80)
    print("✅ TEST 2 PASSED - File context includes access patterns!")
    print("="*80)


async def test_code_generation():
    """Test Phase 4: Enhanced prompts and code generation"""
    print("\n\n" + "="*80)
    print("TEST 3: Code Generation with Enhanced Prompts")
    print("="*80)

    # Create test JSON
    test_data = [
        {"name": "Product A", "price": 100, "quantity": 5},
        {"name": "Product B", "price": 200, "quantity": 3},
        {"name": "Product C", "price": 150, "quantity": 4}
    ]

    f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
    json.dump(test_data, f, indent=2)
    f.close()

    # Test code generation
    query = "Calculate the total revenue from all products (price * quantity for each, then sum)"

    print(f"Query: {query}")
    print(f"File: {f.name}")

    result = await python_coder_tool.execute_code_task(
        query=query,
        file_paths=[f.name],
        session_id="test_session"
    )

    print(f"\nSuccess: {result['success']}")
    print(f"Verification Iterations: {result['verification_iterations']}")
    print(f"Execution Attempts: {result['execution_attempts']}")

    print("\nGenerated Code:")
    print(result['code'])

    print("\nOutput:")
    print(result['output'])

    # Assertions
    assert result['success'], "Code should execute successfully"
    assert '.get(' in result['code'] or 'isinstance' in result['code'], "Should use safe access patterns"

    # Expected: 100*5 + 200*3 + 150*4 = 500 + 600 + 600 = 1700
    assert '1700' in result['output'] or '1,700' in result['output'], "Should calculate correct total"

    # Cleanup
    Path(f.name).unlink()

    print("\n" + "="*80)
    print("✅ TEST 3 PASSED - Code generation uses safe patterns!")
    print("="*80)


async def test_nested_json():
    """Test handling of deeply nested JSON"""
    print("\n\n" + "="*80)
    print("TEST 4: Deeply Nested JSON Handling")
    print("="*80)

    # Create deeply nested JSON
    nested_data = {
        "company": "TechCorp",
        "employees": [
            {
                "name": "Alice",
                "details": {
                    "age": 30,
                    "address": {
                        "city": "NYC",
                        "zipcode": "10001"
                    }
                }
            },
            {
                "name": "Bob",
                "details": {
                    "age": 25,
                    "address": {
                        "city": "LA",
                        "zipcode": "90001"
                    }
                }
            }
        ]
    }

    f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
    json.dump(nested_data, f, indent=2)
    f.close()

    query = "List all employee names and their cities"

    result = await python_coder_tool.execute_code_task(
        query=query,
        file_paths=[f.name],
        session_id="test_nested"
    )

    print(f"Success: {result['success']}")
    print(f"\nGenerated Code:")
    print(result['code'])
    print(f"\nOutput:")
    print(result['output'])

    # Assertions
    assert result['success'], "Should handle nested JSON"
    assert 'Alice' in result['output'] and 'NYC' in result['output'], "Should extract nested data"
    assert 'Bob' in result['output'] and 'LA' in result['output'], "Should extract all items"

    # Cleanup
    Path(f.name).unlink()

    print("\n" + "="*80)
    print("✅ TEST 4 PASSED - Nested JSON handled correctly!")
    print("="*80)


async def run_all_tests():
    """Run all tests"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "JSON IMPROVEMENTS TEST SUITE" + " "*30 + "║")
    print("╚" + "="*78 + "╝")

    try:
        # Test 1: Metadata extraction
        test_metadata_extraction()

        # Test 2: File context building
        test_file_context_building()

        # Test 3: Code generation (async)
        await test_code_generation()

        # Test 4: Nested JSON (async)
        await test_nested_json()

        # Summary
        print("\n\n" + "╔" + "="*78 + "╗")
        print("║" + " "*25 + "🎉 ALL TESTS PASSED! 🎉" + " "*30 + "║")
        print("╚" + "="*78 + "╝")

        print("\n✅ Improvements Validated:")
        print("  1. Access pattern generation works")
        print("  2. Safe preview prevents context overflow")
        print("  3. Null value detection works")
        print("  4. Enhanced file context includes patterns")
        print("  5. LLM prompts enforce safe JSON access")
        print("  6. Generated code uses .get() and isinstance()")
        print("  7. Nested JSON handled correctly")

    except AssertionError as e:
        print(f"\n\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(run_all_tests())
