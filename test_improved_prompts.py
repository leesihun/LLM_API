"""
Test the improved prompt structure to verify clarity improvements.

This demonstrates how the new prompt organization presents information
in a much clearer, more actionable way for the LLM.
"""

import json
import tempfile
from pathlib import Path
from backend.config.prompts.python_coder import get_python_code_generation_prompt
from backend.tools.python_coder.file_handlers.json_handler import JSONFileHandler
from backend.tools.python_coder.context_builder import FileContextBuilder

def create_test_json():
    """Create test JSON with nested structure."""
    return {
        "company": {
            "name": "Test Corp",
            "sales": [
                {
                    "quarter": "Q1",
                    "revenue": 100000,
                    "products": {
                        "software": 80000,
                        "services": 20000
                    }
                }
            ]
        }
    }

def test_improved_prompt_structure():
    """Test that new prompt structure is clearer."""

    print("="*80)
    print("IMPROVED PROMPT STRUCTURE TEST")
    print("="*80)

    # Create test JSON
    test_data = create_test_json()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(test_data, f, indent=2)
        temp_file = f.name

    temp_path = Path(temp_file)

    try:
        # Extract metadata
        handler = JSONFileHandler()
        metadata = handler.extract_metadata(temp_path, quick_mode=False)

        # Build file context
        validated_files = {str(temp_path): "sales_data.json"}
        file_metadata = {str(temp_path): metadata}

        context_builder = FileContextBuilder()
        file_context = context_builder.build_context(validated_files, file_metadata)

        # Generate prompt
        query = "Calculate total revenue from all quarters"
        prompt = get_python_code_generation_prompt(
            query=query,
            context=None,
            file_context=file_context,
            is_prestep=False,
            has_json_files=True
        )

        # Display the prompt
        print("\n" + "="*80)
        print("GENERATED PROMPT (New Structure)")
        print("="*80 + "\n")
        print(prompt)

        # Analyze the prompt structure
        print("\n" + "="*80)
        print("PROMPT ANALYSIS")
        print("="*80 + "\n")

        lines = prompt.split('\n')
        total_lines = len(lines)

        # Find section markers
        sections = []
        current_section = None
        section_start = 0

        for i, line in enumerate(lines):
            if "="*80 in line:
                if current_section:
                    sections.append((current_section, section_start, i))
                # Next line is section title
                if i + 1 < len(lines):
                    current_section = lines[i+1].strip()
                    section_start = i

        if current_section:
            sections.append((current_section, section_start, total_lines))

        print(f"Total lines: {total_lines}")
        print(f"\nSections found: {len(sections)}")
        for name, start, end in sections:
            line_count = end - start
            print(f"  - {name}: lines {start}-{end} ({line_count} lines)")

        # Check for key elements
        print("\n" + "-"*80)
        print("KEY ELEMENTS CHECK:")
        print("-"*80)

        checks = {
            "Task shown first": "YOUR TASK" in prompt,
            "Files before instructions": prompt.index("AVAILABLE FILES") < prompt.index("TOP 3") if "AVAILABLE FILES" in prompt and "TOP 3" in prompt else False,
            "Top 3 rules present": "TOP 3 CRITICAL RULES" in prompt,
            "Access patterns as code": "value1 =" in prompt or "value2 =" in prompt,
            "Complete template": "COMPLETE TEMPLATE" in prompt,
            "JSON instructions": "JSON FILE INSTRUCTIONS" in prompt,
        }

        for check, result in checks.items():
            status = "[OK]" if result else "[FAIL]"
            print(f"  {status} {check}")

        # Count repetitions of key warnings
        print("\n" + "-"*80)
        print("REPETITION ANALYSIS (should be LOW):")
        print("-"*80)

        repetitions = {
            "sys.argv": prompt.count("sys.argv"),
            "input()": prompt.count("input()"),
            "EXACT": prompt.count("EXACT"),
            "[!!!]": prompt.count("[!!!]"),
        }

        for term, count in repetitions.items():
            print(f"  '{term}': {count} occurrences")

        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)

        improvements = []
        if checks["Files before instructions"]:
            improvements.append("✓ Files shown BEFORE long instructions")
        if checks["Top 3 rules present"]:
            improvements.append("✓ Simplified to TOP 3 rules (not 10+)")
        if checks["Access patterns as code"]:
            improvements.append("✓ Access patterns as executable code")
        if checks["Complete template"]:
            improvements.append("✓ Complete copy-paste template provided")

        print("\nImprovements:")
        for imp in improvements:
            print(f"  {imp}")

        print(f"\nPrompt is now {total_lines} lines")
        print(f"Organized into {len(sections)} clear sections")

    finally:
        temp_path.unlink()

    print("\n" + "="*80)
    print("TEST COMPLETED")
    print("="*80)

if __name__ == "__main__":
    test_improved_prompt_structure()
