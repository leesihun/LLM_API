"""
Test script to verify filename handling fix
Tests that original filenames are preserved when copied to execution directory
"""

import sys
sys.path.insert(0, '.')

from backend.tools.python_coder_tool import python_coder_tool
from pathlib import Path

def test_get_original_filename():
    """Test the _get_original_filename() helper method"""
    print("Testing _get_original_filename()...")

    test_cases = [
        ("temp_a1b2c3d4_data.csv", "data.csv"),
        ("temp_12345678_report.xlsx", "report.xlsx"),
        ("temp_abc123_20251013_stats.json", "20251013_stats.json"),
        ("normal_file.txt", "normal_file.txt"),  # No temp_ prefix
        ("temp_prefix_in_middle.csv", "temp_prefix_in_middle.csv"),  # Not at start
    ]

    for temp_name, expected in test_cases:
        result = python_coder_tool._get_original_filename(temp_name)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {temp_name} → {result} (expected: {expected})")
        if result != expected:
            print(f"    ERROR: Got '{result}', expected '{expected}'")
            return False

    print("  All tests passed!\n")
    return True

def test_prepare_files_simulation():
    """Simulate file preparation to verify original names are extracted"""
    print("Testing file preparation simulation...")

    # Create a mock temp file path
    uploads_dir = Path("data/uploads/testuser")
    uploads_dir.mkdir(parents=True, exist_ok=True)

    # Create test file with temp_ prefix
    test_file = uploads_dir / "temp_test123_sample.txt"
    test_file.write_text("Sample content for testing\nLine 2\nLine 3")

    try:
        # Prepare files
        validated_files, file_metadata = python_coder_tool._prepare_files([str(test_file)])

        # Check results
        if len(validated_files) == 1:
            original_path = str(test_file)
            result_filename = validated_files[original_path]

            print(f"  Input file: {test_file.name}")
            print(f"  Mapped to: {result_filename}")

            if result_filename == "sample.txt":
                print("  ✓ Original filename correctly extracted!")

                # Check metadata
                metadata = file_metadata[original_path]
                if metadata.get('original_filename') == "sample.txt":
                    print("  ✓ Metadata contains original_filename!")
                    print(f"  File type: {metadata.get('type')}")
                    print(f"  Lines: {metadata.get('line_count')}")
                    return True
                else:
                    print(f"  ✗ Metadata original_filename is wrong: {metadata.get('original_filename')}")
                    return False
            else:
                print(f"  ✗ Wrong filename: {result_filename} (expected: sample.txt)")
                return False
        else:
            print(f"  ✗ Expected 1 validated file, got {len(validated_files)}")
            return False

    finally:
        # Cleanup
        if test_file.exists():
            test_file.unlink()
        if uploads_dir.exists() and not list(uploads_dir.iterdir()):
            uploads_dir.rmdir()

    print()

def main():
    print("="*80)
    print("FILENAME FIX VERIFICATION TEST")
    print("="*80)
    print()

    # Test 1: Helper method
    test1_passed = test_get_original_filename()

    # Test 2: File preparation
    test2_passed = test_prepare_files_simulation()

    print("="*80)
    if test1_passed and test2_passed:
        print("✓ ALL TESTS PASSED!")
        print()
        print("The fix is working correctly:")
        print("  - Original filenames are extracted from temp_ prefixed names")
        print("  - Files will be copied to execution directory with original names")
        print("  - LLM will see correct filenames in prompts")
        print("="*80)
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("Please check the implementation")
        print("="*80)
        return 1

if __name__ == "__main__":
    sys.exit(main())
