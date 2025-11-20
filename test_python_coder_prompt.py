"""
Test script to verify Python Coder prompt generation with all context sections.

This script demonstrates how the python_coder prompt is constructed with:
1. Conversation history
2. Plan context (from Plan-Execute agent)
3. ReAct context (with failed attempts)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.config.prompts.python_coder import get_python_code_generation_prompt


def test_prompt_with_all_contexts():
    """Test prompt generation with all context sections."""
    
    # Sample user query
    query = "Calculate the average sales from the sales_data.json file"
    
    # Sample file context
    file_context = """
Available Files:
1. sales_data.json (15.2 KB)
   Type: JSON
   Structure: Array of objects
   Keys: date, product, quantity, price, total
   
   [PATTERNS] Access Patterns:
   - data = json.load(f)  # Returns list
   - for item in data: 
       date = item.get('date', '')
       total = item.get('total', 0)
"""
    
    # Sample conversation history
    conversation_history = [
        {
            'role': 'user',
            'content': 'Can you analyze my sales data?',
            'timestamp': '2024-11-20T10:00:00'
        },
        {
            'role': 'assistant',
            'content': 'I can help you analyze your sales data. What would you like to know?',
            'timestamp': '2024-11-20T10:00:05'
        },
        {
            'role': 'user',
            'content': 'Calculate the average sales from the sales_data.json file',
            'timestamp': '2024-11-20T10:00:15'
        }
    ]
    
    # Sample plan context (from Plan-Execute agent)
    plan_context = {
        'current_step': 2,
        'total_steps': 3,
        'plan': [
            {
                'step_number': 1,
                'goal': 'Load and validate the sales data file',
                'status': 'completed',
                'success_criteria': 'File loaded successfully without errors',
                'primary_tools': ['python_coder']
            },
            {
                'step_number': 2,
                'goal': 'Calculate average sales from the data',
                'status': 'current',
                'success_criteria': 'Average calculated and displayed',
                'primary_tools': ['python_coder']
            },
            {
                'step_number': 3,
                'goal': 'Generate summary report',
                'status': 'pending',
                'success_criteria': 'Report generated with key metrics',
                'primary_tools': ['python_coder']
            }
        ],
        'previous_results': [
            {
                'step_number': 1,
                'summary': 'Successfully loaded sales_data.json with 150 records',
                'success': True
            }
        ]
    }
    
    # Sample react context (with failed attempts)
    react_context = {
        'iteration': 3,
        'history': [
            {
                'thought': 'I need to load the JSON file and calculate the average',
                'action': 'python_coder',
                'tool_input': 'Calculate average sales',
                'code': '''import json

data = json.load(open('sales_data.json'))
total = sum([item['total'] for item in data])
avg = total / len(data)
print(f"Average sales: {avg}")''',
                'observation': 'Code execution failed: FileNotFoundError',
                'status': 'error',
                'error_reason': 'File not found - need to use proper file opening with error handling'
            },
            {
                'thought': 'Need to add error handling for file operations',
                'action': 'python_coder',
                'tool_input': 'Calculate average sales with error handling',
                'code': '''import json

try:
    with open('sales_data.json', 'r') as f:
        data = json.load(f)
    total = sum([item['total'] for item in data])
    avg = total / len(data)
    print(f"Average sales: {avg}")
except FileNotFoundError:
    print("Error: File not found")''',
                'observation': 'Code execution failed: KeyError: total',
                'status': 'error',
                'error_reason': 'KeyError - need to use .get() for safe dict access'
            }
        ]
    }
    
    # Generate prompt
    print("=" * 80)
    print("TESTING PYTHON CODER PROMPT GENERATION")
    print("=" * 80)
    print()
    
    prompt = get_python_code_generation_prompt(
        query=query,
        context="Additional context: Focus on calculating the mean value",
        file_context=file_context,
        is_prestep=False,
        has_json_files=True,
        conversation_history=conversation_history,
        plan_context=plan_context,
        react_context=react_context
    )
    
    print(prompt)
    print()
    print("=" * 80)
    print("PROMPT GENERATION COMPLETE")
    print("=" * 80)
    print()
    
    # Verify all sections are present
    sections_to_check = [
        ("PAST HISTORIES", "conversation_history"),
        ("MY ORIGINAL INPUT PROMPT", "user query"),
        ("PLANS", "plan_context"),
        ("REACTS", "react_context"),
        ("FINAL TASK FOR LLM", "task description"),
        ("META DATA", "file context"),
        ("RULES", "coding rules"),
        ("CHECKLISTS", "validation checklist")
    ]
    
    print("SECTION VERIFICATION:")
    print("-" * 80)
    all_present = True
    for section_name, description in sections_to_check:
        present = section_name in prompt
        status = "✓" if present else "✗"
        print(f"{status} {section_name:30} ({description})")
        if not present:
            all_present = False
    
    print("-" * 80)
    if all_present:
        print("✓ All sections present in prompt!")
    else:
        print("✗ Some sections missing!")
    print()
    
    # Check for specific content
    print("CONTENT VERIFICATION:")
    print("-" * 80)
    
    checks = [
        ("Conversation history included", "Can you analyze my sales data" in prompt),
        ("Plan context included", "Step 2 of 3" in prompt or "current_step" in prompt.lower()),
        ("React failed attempts included", "FileNotFoundError" in prompt or "KeyError" in prompt),
        ("File context included", "sales_data.json" in prompt),
        ("Access patterns included", "Access Patterns" in prompt or ".get(" in prompt),
        ("Rules section included", "EXACT FILENAMES" in prompt or "RULE 1" in prompt),
        ("JSON safety rules included", "JSON SAFETY" in prompt or ".get()" in prompt)
    ]
    
    all_checks_pass = True
    for check_name, result in checks:
        status = "✓" if result else "✗"
        print(f"{status} {check_name}")
        if not result:
            all_checks_pass = False
    
    print("-" * 80)
    if all_checks_pass:
        print("✓ All content checks passed!")
    else:
        print("✗ Some content checks failed!")
    print()
    
    return all_present and all_checks_pass


def test_prompt_without_contexts():
    """Test prompt generation without optional contexts."""
    
    print("=" * 80)
    print("TESTING PROMPT WITHOUT CONTEXTS (baseline)")
    print("=" * 80)
    print()
    
    query = "Calculate sum of numbers in data.csv"
    file_context = "data.csv - contains numbers in first column"
    
    prompt = get_python_code_generation_prompt(
        query=query,
        context=None,
        file_context=file_context,
        is_prestep=False,
        has_json_files=False,
        conversation_history=None,
        plan_context=None,
        react_context=None
    )
    
    # Should still have core sections
    sections = ["MY ORIGINAL INPUT PROMPT", "META DATA", "RULES", "CHECKLISTS"]
    sections_present = all(section in prompt for section in sections)
    
    # Should NOT have optional sections
    optional_sections = ["PAST HISTORIES", "PLANS", "REACTS"]
    optional_absent = all(section not in prompt for section in optional_sections)
    
    print(f"✓ Core sections present: {sections_present}")
    print(f"✓ Optional sections absent: {optional_absent}")
    print()
    
    return sections_present and optional_absent


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PYTHON CODER PROMPT GENERATION TEST")
    print("=" * 80)
    print()
    
    # Test 1: With all contexts
    test1_pass = test_prompt_with_all_contexts()
    
    print()
    
    # Test 2: Without contexts
    test2_pass = test_prompt_without_contexts()
    
    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Test 1 (with contexts): {'PASS ✓' if test1_pass else 'FAIL ✗'}")
    print(f"Test 2 (without contexts): {'PASS ✓' if test2_pass else 'FAIL ✗'}")
    print()
    
    if test1_pass and test2_pass:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        sys.exit(0)
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
        sys.exit(1)

