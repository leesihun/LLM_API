"""
Test script to verify that ALL access patterns are generated (no limits).

This creates a complex nested JSON with many keys at different levels
to ensure no patterns are truncated.
"""

import json
import tempfile
from pathlib import Path
from backend.tools.python_coder.file_handlers.json_handler import JSONFileHandler
from backend.tools.python_coder.context_builder import FileContextBuilder

def create_complex_json():
    """Create a deeply nested JSON with many keys."""
    data = {
        "company": {
            "name": "Acme Corp",
            "id": 12345,
            "address": "123 Main St",
            "phone": "555-1234",
            "email": "info@acme.com",
            "website": "www.acme.com",
            "founded": 1980,
            "employees": 500,
            "departments": [
                {
                    "name": "Sales",
                    "budget": 1000000,
                    "headcount": 50,
                    "manager": "Alice Johnson",
                    "location": "Building A",
                    "employees": [
                        {
                            "name": "Bob Smith",
                            "title": "Sales Rep",
                            "salary": 60000,
                            "hire_date": "2020-01-15",
                            "sales": [
                                {
                                    "quarter": "Q1",
                                    "amount": 50000,
                                    "revenue": {
                                        "products": 40000,
                                        "services": 10000,
                                        "breakdown": {
                                            "product_a": 25000,
                                            "product_b": 15000,
                                            "service_x": 7000,
                                            "service_y": 3000
                                        }
                                    }
                                },
                                {
                                    "quarter": "Q2",
                                    "amount": 55000,
                                    "revenue": {
                                        "products": 45000,
                                        "services": 10000
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
                        "revenue": 5000000,
                        "expenses": 3000000,
                        "profit": 2000000,
                        "growth": 0.15,
                        "market_share": 0.05
                    },
                    "2023": {
                        "revenue": 4300000,
                        "expenses": 2700000,
                        "profit": 1600000
                    }
                },
                "quarterly": {
                    "Q1_2024": {
                        "revenue": 1200000,
                        "profit": 480000
                    },
                    "Q2_2024": {
                        "revenue": 1300000,
                        "profit": 520000
                    }
                }
            },
            "products": {
                "software": {
                    "product_a": {
                        "name": "Widget Pro",
                        "price": 299,
                        "sales": 1000
                    },
                    "product_b": {
                        "name": "Widget Enterprise",
                        "price": 999,
                        "sales": 500
                    }
                },
                "services": {
                    "consulting": {
                        "hourly_rate": 150,
                        "contracts": 20
                    },
                    "support": {
                        "monthly_fee": 99,
                        "subscribers": 200
                    }
                }
            }
        }
    }
    return data

def test_all_access_patterns():
    """Test that ALL access patterns are generated without limits."""

    print("=" * 80)
    print("ALL Access Patterns Generation Test")
    print("=" * 80)

    # Create complex JSON data
    test_data = create_complex_json()

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(test_data, f, indent=2)
        temp_file = f.name

    temp_path = Path(temp_file)

    try:
        # Step 1: Extract metadata
        print("\n[1] Extracting metadata from complex JSON...")
        handler = JSONFileHandler()
        metadata = handler.extract_metadata(temp_path, quick_mode=False)

        # Show access patterns
        access_patterns = metadata.get('access_patterns', [])
        print(f"\n[2] Generated {len(access_patterns)} access patterns:")
        print("-" * 80)

        # Group patterns by depth
        by_depth = {}
        for pattern in access_patterns:
            depth = pattern.count('[')
            if depth not in by_depth:
                by_depth[depth] = []
            by_depth[depth].append(pattern)

        # Show patterns grouped by depth
        for depth in sorted(by_depth.keys()):
            patterns = by_depth[depth]
            print(f"\nDepth {depth} ({len(patterns)} patterns):")
            for i, pattern in enumerate(patterns, 1):
                print(f"  {i:2d}. {pattern}")

        # Step 3: Build context
        print("\n[3] Building file context...")
        print("-" * 80)

        validated_files = {str(temp_path): "complex_data.json"}
        file_metadata = {str(temp_path): metadata}

        context_builder = FileContextBuilder()
        context_text = context_builder.build_context(validated_files, file_metadata)

        # Count patterns in context
        pattern_lines = [line for line in context_text.split('\n') if line.strip().startswith('data[')]
        patterns_in_context = len(pattern_lines)

        print(f"\nPatterns in file context: {patterns_in_context}")
        print(f"Total patterns generated: {len(access_patterns)}")

        # Step 4: Verification
        print("\n[4] Verification:")
        print("-" * 80)

        if patterns_in_context == len(access_patterns):
            print(f"[OK] ALL {len(access_patterns)} patterns shown in context!")
        else:
            print(f"[WARNING] Only {patterns_in_context}/{len(access_patterns)} patterns shown")
            print(f"[WARNING] {len(access_patterns) - patterns_in_context} patterns missing!")

        # Check deep nesting
        deep_patterns = [p for p in access_patterns if p.count('[') >= 5]
        if deep_patterns:
            print(f"\n[OK] {len(deep_patterns)} deep nested patterns (5+ levels):")
            for pattern in deep_patterns[:5]:
                print(f"    - {pattern}")
            if len(deep_patterns) > 5:
                print(f"    ... and {len(deep_patterns) - 5} more deep patterns")

        # Show sample of context
        print("\n[5] Sample of file context (access patterns section):")
        print("-" * 80)
        lines = context_text.split('\n')
        in_patterns = False
        pattern_count = 0
        for line in lines:
            if '[PATTERNS]' in line:
                in_patterns = True
            if in_patterns:
                print(line)
                if line.strip().startswith('data['):
                    pattern_count += 1
                    if pattern_count >= 20:  # Show first 20 for brevity
                        if patterns_in_context > 20:
                            print(f"      ... and {patterns_in_context - 20} more patterns")
                        break
            if in_patterns and line.strip().startswith('Sample Data'):
                break

        # Step 6: Summary
        print("\n[6] Summary:")
        print("-" * 80)
        print(f"Total keys in JSON: {count_keys(test_data)}")
        print(f"Max nesting depth: {max(by_depth.keys())}")
        print(f"Patterns generated: {len(access_patterns)}")
        print(f"Patterns in context: {patterns_in_context}")
        print(f"Match: {'YES' if patterns_in_context == len(access_patterns) else 'NO'}")

        if patterns_in_context == len(access_patterns):
            print("\n[SUCCESS] No truncation - ALL patterns included!")
        else:
            print(f"\n[ISSUE] {len(access_patterns) - patterns_in_context} patterns were truncated")

    finally:
        # Cleanup
        temp_path.unlink()

    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)

def count_keys(obj, depth=0, max_depth=10):
    """Recursively count all keys in nested structure."""
    if depth > max_depth:
        return 0

    count = 0
    if isinstance(obj, dict):
        count += len(obj)
        for value in obj.values():
            count += count_keys(value, depth + 1, max_depth)
    elif isinstance(obj, list) and len(obj) > 0:
        count += count_keys(obj[0], depth + 1, max_depth)

    return count

if __name__ == "__main__":
    test_all_access_patterns()
