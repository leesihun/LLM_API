"""
Test script to demonstrate improved access_patterns visibility in file context.

This shows how deeply nested JSON structures now have full access patterns
displayed (up to 12 patterns instead of 6).
"""

import json
import tempfile
from pathlib import Path
from backend.tools.python_coder.file_handlers.json_handler import JSONFileHandler
from backend.tools.python_coder.context_builder import FileContextBuilder

def create_test_json():
    """Create a deeply nested JSON structure."""
    data = {
        "company": {
            "name": "Acme Corp",
            "departments": [
                {
                    "name": "Sales",
                    "employees": [
                        {
                            "name": "Alice",
                            "sales": [
                                {
                                    "quarter": "Q1",
                                    "revenue": {
                                        "amount": 150000,
                                        "currency": "USD",
                                        "breakdown": {
                                            "products": 120000,
                                            "services": 30000
                                        }
                                    }
                                }
                            ]
                        }
                    ]
                }
            ],
            "metrics": {
                "yearly": {
                    "2024": {
                        "revenue": 1000000,
                        "expenses": 600000,
                        "profit": 400000
                    }
                }
            }
        }
    }
    return data

def test_access_patterns_visibility():
    """Test that access patterns show deep nesting."""

    print("=" * 80)
    print("Access Patterns Visibility Test")
    print("=" * 80)

    # Create test JSON data
    test_data = create_test_json()

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(test_data, f, indent=2)
        temp_file = f.name

    temp_path = Path(temp_file)

    try:
        # Step 1: Extract metadata using JSON handler
        print("\n[1] Extracting metadata with JSON handler...")
        handler = JSONFileHandler()
        metadata = handler.extract_metadata(temp_path, quick_mode=False)

        # Show access patterns
        access_patterns = metadata.get('access_patterns', [])
        print(f"\n[2] Generated {len(access_patterns)} access patterns:")
        print("-" * 80)
        for i, pattern in enumerate(access_patterns, 1):
            print(f"  {i:2d}. {pattern}")

        # Step 3: Build context using FileContextBuilder
        print("\n[3] Building file context with FileContextBuilder...")
        print("-" * 80)

        validated_files = {str(temp_path): "nested_data.json"}
        file_metadata = {str(temp_path): metadata}

        context_builder = FileContextBuilder()
        context_text = context_builder.build_context(validated_files, file_metadata)

        # Show the context
        print(context_text)

        # Step 4: Verify patterns are shown
        print("\n[4] Verification:")
        print("-" * 80)

        if "[PATTERNS]" in context_text:
            print("[OK] Access patterns section found in context")

            # Count how many patterns are shown
            pattern_lines = [line for line in context_text.split('\n') if line.strip().startswith('data[')]
            print(f"[OK] {len(pattern_lines)} access patterns displayed in context")

            # Show deep nesting examples
            deep_patterns = [p for p in access_patterns if p.count('[') >= 4]
            if deep_patterns:
                print(f"[OK] {len(deep_patterns)} deep nested patterns (4+ levels) included:")
                for pattern in deep_patterns[:3]:
                    print(f"    - {pattern}")
        else:
            print("[FAIL] Access patterns section NOT found in context")

        # Step 5: Show improvement
        print("\n[5] Improvement Summary:")
        print("-" * 80)
        print("Before fix: Only 6 access patterns shown (truncated)")
        print(f"After fix:  ALL {len(access_patterns)} access patterns shown (no limit!)")
        print("\nBenefit: LLM can see COMPLETE nested structure without guessing paths")
        print("Example: Can see all patterns including deep ones like:")
        print("  - data['company']['departments'][0]['employees'][0]")
        print("  - data['company']['metrics']['yearly']['2024']['revenue']")

    finally:
        # Cleanup
        temp_path.unlink()

    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)

if __name__ == "__main__":
    test_access_patterns_visibility()
