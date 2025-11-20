# PhaseManager with Notepad Integration

**Updated:** January 2025
**Feature:** File parsing information persistence

## Overview

The `PhaseManager` now integrates with `SessionNotepad` to automatically record file parsing information and phase results for persistent memory across the session.

---

## Key Features

### 1. Automatic Notepad Recording
- Phase results automatically saved to session notepad
- File parsing information persisted
- Available to LLM in subsequent phases via notepad context

### 2. File Tracking
- Tracks which files were processed in each phase
- Prevents redundant file re-processing
- Provides file parsing summary

### 3. Persistent Memory
- Phase findings stored in notepad for session duration
- LLM can reference previous phase work from notepad
- Survives across multiple API calls

---

## Usage Examples

### Basic Usage with Notepad

```python
from backend.utils.phase_manager import PhaseManager

# Initialize with session_id for notepad integration
session_id = "abc123"
manager = PhaseManager(session_id=session_id)

# Phase 1: Analysis (with file processing)
phase1_prompt = manager.create_initial_phase_prompt(
    phase_name="Data Analysis",
    task="Analyze the attached CSV files and calculate statistics",
    expected_outputs=["Total revenue", "Product breakdown", "Regional analysis"]
)

# Execute Phase 1...
files_uploaded = ["sales_data.csv", "regions.csv"]
result1 = execute_phase(phase1_prompt, files=files_uploaded)

# Record phase result WITH file info
manager.record_phase_result(
    phase_name="Data Analysis",
    findings="Analyzed 2 CSV files with 10,000 total rows. Found top product is 'Laptop' with $1.2M revenue.",
    artifacts=["analysis_summary.json"],
    files_processed=files_uploaded,  # Track which files were parsed
    metadata={"total_rows": 10000, "total_revenue": 1200000},
    save_to_notepad=True  # Automatically saves to notepad (default)
)

# Phase 2: Visualization (reuses Phase 1 findings + notepad memory)
phase2_prompt = manager.create_handoff_phase_prompt(
    phase_name="Visualization",
    task="Create charts based on Phase 1 analysis",
    expected_outputs=["Revenue chart", "Regional breakdown chart"]
)

# Note: Files NOT re-uploaded, LLM uses Phase 1 findings from:
# 1. Conversation memory (immediate context)
# 2. Notepad memory (persistent across API calls)
result2 = execute_phase(phase2_prompt)

manager.record_phase_result(
    phase_name="Visualization",
    findings="Created 3 charts showing revenue trends and regional breakdown",
    artifacts=["revenue_chart.png", "regional_chart.png", "trends_chart.png"],
    files_processed=[],  # No new files processed in this phase
    save_to_notepad=True
)

# Get file parsing summary
file_summary = manager.get_file_parsing_summary()
print(file_summary)
# Output:
# {
#   "total_files_processed": 2,
#   "files_by_phase": {
#     "Data Analysis": ["sales_data.csv", "regions.csv"],
#     "Visualization": []
#   },
#   "all_files": ["regions.csv", "sales_data.csv"]
# }
```

---

## Notepad Integration Details

### What Gets Saved to Notepad

When you call `record_phase_result()` with `save_to_notepad=True`, the following information is saved:

```json
{
  "entry_id": 1,
  "task": "phase_data_analysis",
  "description": "Analyzed 2 CSV files with 10,000 total rows. Found top product is 'Laptop' with $1.2M revenue.",
  "code_file": null,
  "variables_saved": [],
  "key_outputs": "Files analyzed: sales_data.csv, regions.csv; Artifacts: analysis_summary.json; total_rows: 10000; total_revenue: 1200000",
  "timestamp": "2025-01-20T10:30:00"
}
```

### How LLM Accesses Notepad

The notepad context is automatically injected into the LLM's context when using the ReAct agent:

```
=== Session Memory (Notepad) ===
Previous work from this session:

Entry 1: [phase_data_analysis]
Description: Analyzed 2 CSV files with 10,000 total rows. Found top product is 'Laptop' with $1.2M revenue.
Outputs: Files analyzed: sales_data.csv, regions.csv; Artifacts: analysis_summary.json; total_rows: 10000; total_revenue: 1200000

Entry 2: [phase_visualization]
Description: Created 3 charts showing revenue trends and regional breakdown
Outputs: Artifacts: revenue_chart.png, regional_chart.png, trends_chart.png
```

---

## Complete Workflow Example

```python
from backend.utils.phase_manager import PhaseManager

def run_multi_phase_analysis(session_id: str, data_files: list):
    """
    Run a complete 3-phase analysis workflow with notepad integration.

    Args:
        session_id: Session identifier
        data_files: List of data file paths

    Returns:
        dict: Workflow summary including file parsing info
    """
    # Initialize manager with session ID
    manager = PhaseManager(session_id=session_id)

    # ============================================================
    # PHASE 1: Data Analysis
    # ============================================================
    phase1_prompt = manager.create_initial_phase_prompt(
        phase_name="Data Analysis",
        task=f"Analyze {len(data_files)} data files and extract key statistics",
        expected_outputs=[
            "Total record count",
            "Key metrics summary",
            "Outlier identification",
            "Data quality assessment"
        ]
    )

    # Execute Phase 1 (with files)
    print(f"Phase 1: Processing {len(data_files)} files...")
    result1 = execute_phase(phase1_prompt, files=data_files)

    # Record with file tracking
    manager.record_phase_result(
        phase_name="Data Analysis",
        findings=extract_summary(result1),
        artifacts=["stats.npy", "outliers.json"],
        files_processed=data_files,  # Important: track files
        metadata={
            "record_count": 50000,
            "outlier_count": 127,
            "quality_score": 0.95
        }
    )

    # ============================================================
    # PHASE 2: Visualization
    # ============================================================
    phase2_prompt = manager.create_handoff_phase_prompt(
        phase_name="Visualization",
        task="Create professional charts based on Phase 1 analysis",
        expected_outputs=[
            "Distribution charts",
            "Outlier visualization",
            "Trend analysis charts"
        ]
    )

    # Execute Phase 2 (NO files - uses notepad memory!)
    print("Phase 2: Creating visualizations from Phase 1 findings...")
    result2 = execute_phase(phase2_prompt)  # No files parameter!

    manager.record_phase_result(
        phase_name="Visualization",
        findings=extract_summary(result2),
        artifacts=["dist_chart.png", "outliers_chart.png", "trends.png"],
        files_processed=[],  # No new files
        metadata={"chart_count": 3}
    )

    # ============================================================
    # PHASE 3: Report Generation
    # ============================================================
    phase3_prompt = manager.create_handoff_phase_prompt(
        phase_name="Report Generation",
        task="Create executive PowerPoint report using Phase 1 & 2 results",
        expected_outputs=["PowerPoint presentation with charts and findings"]
    )

    # Execute Phase 3 (uses notepad for ALL previous work)
    print("Phase 3: Generating report from notepad memory...")
    result3 = execute_phase(phase3_prompt)  # No files!

    manager.record_phase_result(
        phase_name="Report Generation",
        findings=extract_summary(result3),
        artifacts=["Executive_Report.pptx"],
        files_processed=[],
        metadata={"slide_count": 12}
    )

    # ============================================================
    # Get Summary
    # ============================================================
    workflow_summary = manager.get_workflow_summary()
    file_summary = manager.get_file_parsing_summary()

    print("\n=== Workflow Complete ===")
    print(f"Total phases: {workflow_summary['total_phases']}")
    print(f"Total artifacts: {workflow_summary['total_artifacts']}")
    print(f"Files processed: {file_summary['total_files_processed']}")
    print(f"Files only processed in: {list(file_summary['files_by_phase'].keys())}")

    return {
        "workflow": workflow_summary,
        "files": file_summary,
        "notepad_context": manager.get_notepad_context()
    }

# Usage
result = run_multi_phase_analysis(
    session_id="session_abc123",
    data_files=["data1.csv", "data2.csv", "data3.csv"]
)
```

**Output:**
```
Phase 1: Processing 3 files...
[PhaseManager] Saved phase 'Data Analysis' to notepad (entry #1)

Phase 2: Creating visualizations from Phase 1 findings...
[PhaseManager] Saved phase 'Visualization' to notepad (entry #2)

Phase 3: Generating report from notepad memory...
[PhaseManager] Saved phase 'Report Generation' to notepad (entry #3)

=== Workflow Complete ===
Total phases: 3
Total artifacts: 6
Files processed: 3
Files only processed in: ['Data Analysis']
```

---

## Benefits of Notepad Integration

### 1. **Persistent Memory**
- Phase results survive across API calls
- LLM can reference previous work even after conversation context expires
- Session-based storage ensures data isolation

### 2. **File Processing Transparency**
- Clear record of which files were processed when
- Prevents accidental re-processing
- Audit trail for data lineage

### 3. **Reduced Token Usage**
- Phase findings stored in notepad (external to conversation)
- Can be selectively injected when needed
- Reduces conversation context bloat

### 4. **Better Debugging**
- Notepad JSON file shows exact phase progression
- Easy to inspect what the LLM "remembers"
- Clear audit trail for troubleshooting

---

## API Methods

### PhaseManager Constructor
```python
manager = PhaseManager(session_id: Optional[str] = None)
```
- `session_id`: If provided, enables notepad integration

### record_phase_result()
```python
manager.record_phase_result(
    phase_name: str,
    findings: str,
    artifacts: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    files_processed: Optional[List[str]] = None,  # NEW
    save_to_notepad: bool = True  # NEW
)
```

**New parameters:**
- `files_processed`: List of file paths that were parsed in this phase
- `save_to_notepad`: Whether to save phase info to notepad (default: True)

### get_file_parsing_summary()
```python
summary = manager.get_file_parsing_summary()
```

**Returns:**
```python
{
    "total_files_processed": 3,
    "files_by_phase": {
        "Data Analysis": ["file1.csv", "file2.csv"],
        "Visualization": []
    },
    "all_files": ["file1.csv", "file2.csv"]
}
```

### get_notepad_context()
```python
context = manager.get_notepad_context()
```

**Returns:** Formatted string of all notepad entries for LLM injection

---

## Notepad File Location

Notepad files are stored at:
```
data/scratch/{session_id}/notepad.json
```

**Example notepad.json:**
```json
{
  "session_id": "session_abc123",
  "created_at": "2025-01-20T10:00:00",
  "updated_at": "2025-01-20T10:45:00",
  "entries": [
    {
      "entry_id": 1,
      "task": "phase_data_analysis",
      "description": "Analyzed 3 CSV files...",
      "code_file": null,
      "variables_saved": [],
      "key_outputs": "Files analyzed: data1.csv, data2.csv, data3.csv; Artifacts: stats.npy; record_count: 50000",
      "timestamp": "2025-01-20T10:15:00"
    },
    {
      "entry_id": 2,
      "task": "phase_visualization",
      "description": "Created 3 charts...",
      "code_file": null,
      "variables_saved": [],
      "key_outputs": "Artifacts: chart1.png, chart2.png, chart3.png",
      "timestamp": "2025-01-20T10:30:00"
    }
  ]
}
```

---

## Best Practices

### 1. Always Provide session_id
```python
# Good - enables notepad
manager = PhaseManager(session_id=session_id)

# Limited - no notepad persistence
manager = PhaseManager()
```

### 2. Track Files Accurately
```python
# Good - explicit file tracking
manager.record_phase_result(
    phase_name="Analysis",
    findings="...",
    files_processed=["data.csv", "metadata.json"]
)

# Acceptable - no new files in this phase
manager.record_phase_result(
    phase_name="Visualization",
    findings="...",
    files_processed=[]  # Or omit parameter
)
```

### 3. Use Descriptive Phase Names
```python
# Good - clear, specific
phase_name="Data Quality Analysis"
phase_name="Outlier Visualization"
phase_name="Executive Report Generation"

# Bad - vague
phase_name="Phase 1"
phase_name="Analysis"
```

### 4. Include Relevant Metadata
```python
# Good - quantitative metadata
metadata={
    "rows_processed": 10000,
    "outliers_found": 42,
    "quality_score": 0.95,
    "processing_time_seconds": 12.5
}

# Less useful - qualitative only
metadata={"status": "complete"}
```

---

## Troubleshooting

### Notepad Not Saving

**Problem:** Phase results not appearing in notepad

**Solutions:**
1. Check that `session_id` was provided to `PhaseManager.__init__()`
2. Verify `save_to_notepad=True` (it's the default)
3. Check file permissions on `data/scratch/{session_id}/`
4. Review logs for notepad save errors

### File Tracking Discrepancies

**Problem:** Files processed but not tracked

**Solution:**
```python
# Ensure files_processed is set explicitly
manager.record_phase_result(
    phase_name="Analysis",
    findings="...",
    files_processed=files_uploaded  # Don't forget this!
)
```

### Notepad Context Not Injected

**Problem:** LLM doesn't see previous phase work

**Causes:**
1. Not using ReAct agent (notepad only injected in ReAct context)
2. Different session_id between phases
3. Notepad file corrupted or missing

**Solution:** Verify notepad context manually:
```python
context = manager.get_notepad_context()
print(context)  # Should show all entries
```

---

## See Also

- [backend/utils/phase_manager.py](backend/utils/phase_manager.py) - Source code
- [backend/tools/notepad.py](backend/tools/notepad.py) - SessionNotepad implementation
- [Multi_Phase_Workflow_Example.ipynb](Multi_Phase_Workflow_Example.ipynb) - Tutorial notebook
- [CLAUDE.md](CLAUDE.md#multi-phase-workflows-files-as-fallback-pattern) - Architecture docs

---

**Version:** 2.1.0 (Notepad Integration)
**Last Updated:** January 2025
