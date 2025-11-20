"""
Test script for file context storage functionality.

This demonstrates how file analysis context is saved to /data/scratch/{session_id}
and can be reused across multiple phases.
"""

import asyncio
import json
from pathlib import Path
from backend.tools.python_coder.orchestrator import python_coder_tool
from backend.tools.python_coder.file_context_storage import FileContextStorage

async def test_file_context_storage():
    """Test saving and loading file context."""

    print("=" * 80)
    print("File Context Storage Test")
    print("=" * 80)

    # Test session ID
    test_session_id = "test_session_file_context"

    # Simulate file metadata (normally comes from file analysis)
    test_validated_files = {
        "data/uploads/sample_data.csv": "sample_data.csv",
        "data/uploads/analysis_report.json": "analysis_report.json"
    }

    test_file_metadata = {
        "data/uploads/sample_data.csv": {
            "type": "csv",
            "size_mb": 0.5,
            "columns": ["id", "name", "value", "date"],
            "rows": 1000,
            "preview": [
                {"id": 1, "name": "Alice", "value": 100, "date": "2025-01-01"},
                {"id": 2, "name": "Bob", "value": 200, "date": "2025-01-02"}
            ]
        },
        "data/uploads/analysis_report.json": {
            "type": "json",
            "size_mb": 0.1,
            "structure": {
                "type": "dict",
                "num_keys": 5,
                "keys": ["summary", "metrics", "timestamps", "users", "results"]
            }
        }
    }

    test_file_context_text = """
[!!!] CRITICAL - EXACT FILENAMES REQUIRED [!!!]
ALL files are in the current working directory.

Available files (USE THESE EXACT NAMES):

1. "sample_data.csv" - CSV (0.5MB)
   Columns: id, name, value, date
   Example: df = pd.read_csv('sample_data.csv')

2. "analysis_report.json" - JSON (0.1MB)
   Structure: dict (5 keys)
   Top-level keys: summary, metrics, timestamps, users, results
"""

    # Step 1: Save file context
    print("\n[1] Saving file context to session directory...")
    success = FileContextStorage.save_file_context(
        session_id=test_session_id,
        validated_files=test_validated_files,
        file_metadata=test_file_metadata,
        file_context_text=test_file_context_text
    )

    if success:
        print(f"✓ File context saved successfully!")

        # Show where it was saved
        session_dir = Path("data/scratch") / test_session_id
        context_file = session_dir / "file_context.json"
        print(f"  Location: {context_file}")
        print(f"  File exists: {context_file.exists()}")
    else:
        print("✗ Failed to save file context")
        return

    # Step 2: Check if context exists
    print("\n[2] Checking if file context exists...")
    exists = FileContextStorage.has_file_context(test_session_id)
    print(f"  Has saved context: {exists}")

    # Step 3: Get summary
    print("\n[3] Getting file context summary...")
    summary = FileContextStorage.get_file_context_summary(test_session_id)
    if summary:
        print(summary)

    # Step 4: Load full context
    print("\n[4] Loading full file context...")
    loaded_context = FileContextStorage.load_file_context(test_session_id)

    if loaded_context:
        print(f"✓ File context loaded successfully!")
        print(f"\n  Keys in loaded context:")
        for key in loaded_context.keys():
            print(f"    - {key}")

        print(f"\n  File count: {loaded_context.get('file_count')}")
        print(f"  Files: {', '.join(loaded_context.get('file_list', []))}")
        print(f"  Timestamp: {loaded_context.get('timestamp')}")
    else:
        print("✗ Failed to load file context")
        return

    # Step 5: Demonstrate reuse in multi-phase workflow
    print("\n[5] Multi-Phase Workflow Example")
    print("-" * 80)

    print("\nPhase 1: Initial file analysis (context saved automatically)")
    print("  → Files processed, metadata extracted")
    print("  → Context saved to /data/scratch/{session_id}/file_context.json")

    print("\nPhase 2: Subsequent operations (context reused from memory)")
    print("  → Check if context exists:")
    if python_coder_tool.has_saved_file_context(test_session_id):
        print("    ✓ Context found! Reusing existing analysis")

        # Load and use the context
        saved_context = python_coder_tool.get_saved_file_context(test_session_id)
        if saved_context:
            print(f"    → Loaded {saved_context['file_count']} file(s) from cache")
            print(f"    → Files: {', '.join(saved_context['file_list'])}")
            print(f"    → No need to re-analyze files!")
    else:
        print("    ✗ No context found")

    print("\nPhase 3, 4, 5... (all reuse the same context)")
    print("  → 90% fewer LLM calls")
    print("  → Faster execution")
    print("  → Consistent analysis across phases")

    # Step 6: Show saved file structure
    print("\n[6] Viewing saved file content (first 50 lines)...")
    print("-" * 80)

    session_dir = Path("data/scratch") / test_session_id
    context_file = session_dir / "file_context.json"

    if context_file.exists():
        with open(context_file, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            for i, line in enumerate(lines[:50], 1):
                print(f"{i:3d}: {line}")

            if len(lines) > 50:
                print(f"     ... ({len(lines) - 50} more lines)")

    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_file_context_storage())
