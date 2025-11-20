"""
Test New Prompt Structure

Tests the completely restructured prompt:
1. PAST HISTORIES
2. MY ORIGINAL INPUT PROMPT
3. PLANS
4. REACTS
5. FINAL TASK FOR LLM AT THIS STAGE
6. META DATA (AVAILABLE FILES) - with access patterns only, NO templates
7. RULES
8. CHECKLISTS
"""

from backend.config.prompts.python_coder import get_python_code_generation_prompt


def test_new_structure():
    """Test the new prompt structure."""

    print("\n" + "=" * 80)
    print("TESTING NEW PROMPT STRUCTURE")
    print("=" * 80)

    query = "Calculate the sum of all sales amounts"
    file_context = """

[!!!] CRITICAL - EXACT FILENAMES REQUIRED [!!!]
ALL files are in the current working directory.
YOU MUST use the EXACT filenames shown below - NO generic names like 'file.json' or 'data.csv'!

Available files (USE THESE EXACT NAMES):

1. "sales_data_2024.json" - JSON (15KB)
   Structure: dict (3 keys)
   Top-level keys: sales, metadata, totals

   ======================================================================
   [>>>] COPY-PASTE READY: Access Patterns (Pre-Validated)
   ======================================================================
   # These patterns match your JSON structure - copy them directly:

   # Pattern 1:
   value1 = data['sales']
   print(f'Pattern 1: {value1}')

   # Pattern 2:
   value2 = data['sales'][0]
   print(f'Pattern 2: {value2}')

   # Pattern 3:
   value3 = data['sales'][0]['amount']
   print(f'Pattern 3: {value3}')

   ----------------------------------------------------------------------
   Sample Data (first few items):
      {
        "sales": [
          {"id": 1, "amount": 100},
          {"id": 2, "amount": 200}
        ]
      }
"""

    prompt = get_python_code_generation_prompt(
        query=query,
        context=None,
        file_context=file_context,
        is_prestep=False,
        has_json_files=True
    )

    # Check for new structure sections
    expected_sections = [
        "MY ORIGINAL INPUT PROMPT",
        "FINAL TASK FOR LLM AT THIS STAGE",
        "[TASK TYPE]",
        "META DATA (AVAILABLE FILES)",
        "RULES",
        "[RULE 1] EXACT FILENAMES",
        "[RULE 2] NO COMMAND-LINE ARGS",
        "[RULE 3] USE ACCESS PATTERNS",
        "[RULE 4] JSON SAFETY",
        "CHECKLISTS",
        "[1] Task Completion",
        "[2] Filename Validation",
        "[3] Safety & Error Handling",
        "[4] Access Patterns"
    ]

    # Sections that should NOT appear (templates removed)
    excluded_sections = [
        "COMPLETE TEMPLATE: Copy this entire block",
        "import json",
        "with open(filename, 'r', encoding='utf-8') as f:",
        "json.load(f)",
        "RECOMMENDED APPROACH",
        "COMMON MISTAKES TO AVOID",
        "YOUR RESPONSE"
    ]

    print("\n[TEST 1] Checking for expected sections:")
    all_found = True
    for section in expected_sections:
        found = section in prompt
        status = "[OK]" if found else "[MISSING]"
        print(f"  {status} {section}")
        if not found:
            all_found = False

    print("\n[TEST 2] Checking that templates are REMOVED:")
    none_found = True
    for section in excluded_sections:
        found = section in prompt
        status = "[X FOUND]" if found else "[OK REMOVED]"
        print(f"  {status} {section}")
        if found:
            none_found = False

    # Show complete prompt
    print("\n" + "=" * 80)
    print("COMPLETE NEW PROMPT STRUCTURE")
    print("=" * 80)
    print(prompt)
    print("=" * 80)

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    if all_found and none_found:
        print("[SUCCESS] New prompt structure is correct!")
        print("\nStructure verified:")
        print("  1. MY ORIGINAL INPUT PROMPT ✓")
        print("  2. FINAL TASK FOR LLM AT THIS STAGE ✓")
        print("  3. META DATA (AVAILABLE FILES) ✓")
        print("     - Access patterns ONLY")
        print("     - NO templates")
        print("  4. RULES ✓")
        print("  5. CHECKLISTS ✓")
        return True
    else:
        print("[FAIL] Structure issues found")
        if not all_found:
            print("  - Some expected sections missing")
        if not none_found:
            print("  - Templates still present (should be removed)")
        return False


def test_access_patterns_only():
    """Test that only access patterns are shown, no templates."""

    print("\n" + "=" * 80)
    print("TESTING ACCESS PATTERNS ONLY (NO TEMPLATES)")
    print("=" * 80)

    query = "Sum the values"
    file_context = """
1. "test.json" - JSON
   Access Patterns:
   value1 = data['field']
"""

    prompt = get_python_code_generation_prompt(
        query=query,
        context=None,
        file_context=file_context,
        has_json_files=True
    )

    # Check for access patterns
    has_patterns = "value1 = data['field']" in prompt or "[>>>] COPY-PASTE READY: Access Patterns" in file_context
    print(f"\nAccess patterns present: {'[OK]' if has_patterns else '[MISSING]'}")

    # Check that templates are NOT present
    template_markers = [
        "import json",
        "with open(filename,",
        "json.load(f)",
        "COMPLETE TEMPLATE"
    ]

    templates_removed = True
    for marker in template_markers:
        if marker in prompt:
            print(f"Template marker found (should be removed): [{marker}]")
            templates_removed = False

    if templates_removed:
        print("\n[SUCCESS] Templates correctly removed - only access patterns shown")
        return True
    else:
        print("\n[FAIL] Templates still present")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("NEW PROMPT STRUCTURE - TEST SUITE")
    print("=" * 80)

    test1 = test_new_structure()
    test2 = test_access_patterns_only()

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"New Structure Test:      {'PASSED' if test1 else 'FAILED'}")
    print(f"Access Patterns Only Test: {'PASSED' if test2 else 'FAILED'}")

    if test1 and test2:
        print("\n✓ All tests passed!")
        print("\nNew prompt order:")
        print("  1. PAST HISTORIES (placeholder)")
        print("  2. MY ORIGINAL INPUT PROMPT")
        print("  3. PLANS (placeholder)")
        print("  4. REACTS (placeholder)")
        print("  5. FINAL TASK FOR LLM AT THIS STAGE")
        print("  6. META DATA (AVAILABLE FILES)")
        print("     → Access patterns ONLY, NO templates")
        print("  7. RULES")
        print("  8. CHECKLISTS")
    else:
        print("\n✗ Some tests failed")


if __name__ == "__main__":
    main()
