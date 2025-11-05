"""
Test ReAct agent's file analysis pre-step with detailed structure information.
"""

import asyncio
from backend.tasks.React import react_agent, ReActStep
from backend.models.schemas import ChatMessage
from backend.tools.file_analyzer_tool import file_analyzer

async def test_file_analysis_context():
    """Test if file analysis provides enough context to LLM."""

    print("=" * 80)
    print("Testing File Analysis Context for LLM")
    print("=" * 80)

    # Test file
    test_file = r"c:\Users\Lee\Desktop\Huni\LLM_API\data\uploads\leesihun\20251013_stats.json"

    # Simulate the pre-step analysis
    analyzer_result = file_analyzer.analyze(
        file_paths=[test_file],
        user_query="What is the structure of this file?"
    )

    # Build observation as React agent does
    if analyzer_result.get("success"):
        obs_parts = [f"File analysis completed:\n{analyzer_result.get('summary','')}"]

        for file_result in analyzer_result.get("results", []):
            if file_result.get("structure_summary"):
                obs_parts.append(f"\nDetailed structure for {file_result.get('file', 'file')}:")
                obs_parts.append(file_result["structure_summary"])

        obs = "\n".join(obs_parts)
    else:
        obs = f"File analysis failed: {analyzer_result.get('error','Unknown error')}"

    # Display what LLM will receive
    print("\n" + "=" * 80)
    print("OBSERVATION SENT TO LLM (in ReAct context)")
    print("=" * 80)
    print(obs)
    print("\n" + "=" * 80)

    # Check if it has key information
    print("\nKEY INFORMATION CHECK:")
    print("=" * 80)
    checks = {
        "Contains 'Maximum nesting depth'": "Maximum nesting depth" in obs,
        "Contains 'Total unique paths'": "Total unique paths" in obs,
        "Contains 'Depth 4' (nested structure)": "Depth 4:" in obs,
        "Contains 'pca' nested object": "pca" in obs,
        "Contains all 12 fields at Depth 3": obs.count("Depth 3:") > 0 and obs.count("root.files[0].") >= 12,
        "Observation length > 500 chars": len(obs) > 500
    }

    for check, passed in checks.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {check}")

    print("\n" + "=" * 80)
    print(f"Total observation length: {len(obs)} characters")
    print("=" * 80)

    # Evaluate
    all_passed = all(checks.values())
    if all_passed:
        print("\n[SUCCESS] File analysis provides sufficient context for LLM!")
    else:
        print("\n[WARNING] Some key information may be missing.")

if __name__ == "__main__":
    asyncio.run(test_file_analysis_context())
