"""
Session Notepad Module

Manages persistent session memory that stores summaries of completed tasks,
code files, and variable metadata for future reference within a session.

Enhanced Features (v2.0):
- Search and filtering
- Entry editing/deletion with version history
- Tags and importance levels
- Analytics and insights
- Export to Markdown/Jupyter
- Variable dependency tracking
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from difflib import unified_diff

from backend.config.settings import settings
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SessionNotepad:
    """
    Session-specific notepad for storing task summaries, code, and variable references.
    
    Storage structure:
    - notepad.json: Main notepad file with all entries
    - {task}_{timestamp}.py: Saved code files
    - variables/: Variable storage directory (managed by VariableStorage)
    """
    
    def __init__(self, session_id: str):
        """
        Initialize notepad for a session.
        
        Args:
            session_id: Session identifier
        """
        self.session_id = session_id
        self.session_path = Path(settings.python_code_execution_dir) / session_id
        self.notepad_path = self.session_path / "notepad.json"
        self.data = {
            "session_id": session_id,
            "created_at": None,
            "updated_at": None,
            "entries": []
        }
    
    @staticmethod
    def load(session_id: str) -> 'SessionNotepad':
        """
        Load existing notepad or create new one.
        
        Args:
            session_id: Session identifier
            
        Returns:
            SessionNotepad instance
        """
        notepad = SessionNotepad(session_id)
        
        if notepad.notepad_path.exists():
            try:
                with open(notepad.notepad_path, 'r', encoding='utf-8') as f:
                    notepad.data = json.load(f)
                logger.info(f"[SessionNotepad] Loaded notepad with {len(notepad.data['entries'])} entries")
            except Exception as e:
                logger.error(f"[SessionNotepad] Failed to load notepad: {e}")
                # Keep default empty data
        else:
            # New notepad
            notepad.data["created_at"] = datetime.now().isoformat()
            logger.info(f"[SessionNotepad] Created new notepad for session {session_id}")
        
        return notepad
    
    def add_entry(
        self,
        task: str,
        description: str,
        code_file: Optional[str] = None,
        variables_saved: Optional[List[str]] = None,
        key_outputs: Optional[str] = None,
        tags: Optional[List[str]] = None,
        importance: str = "medium",
        execution_time_ms: Optional[int] = None,
        success_score: Optional[float] = None,
        tool_used: Optional[str] = None
    ) -> int:
        """
        Add new entry to notepad with enhanced metadata.

        Args:
            task: Task category (e.g., "data_analysis", "visualization")
            description: What was accomplished
            code_file: Filename of saved code (if applicable)
            variables_saved: List of variable names that were saved
            key_outputs: Summary of important outputs/results
            tags: List of tags for categorization
            importance: Importance level ("low", "medium", "high")
            execution_time_ms: Execution time in milliseconds
            success_score: Success score (0.0-1.0)
            tool_used: Primary tool used for this task

        Returns:
            Entry ID (1-indexed)
        """
        entry_id = len(self.data["entries"]) + 1

        entry = {
            "entry_id": entry_id,
            "task": task,
            "description": description,
            "code_file": code_file,
            "variables_saved": variables_saved or [],
            "key_outputs": key_outputs or "",
            "tags": tags or [],
            "importance": importance,
            "execution_time_ms": execution_time_ms,
            "success_score": success_score,
            "tool_used": tool_used,
            "timestamp": datetime.now().isoformat(),
            "is_archived": False,
            "version": 1,
            "version_history": []
        }

        self.data["entries"].append(entry)
        self.data["updated_at"] = datetime.now().isoformat()

        logger.info(f"[SessionNotepad] Added entry #{entry_id}: [{task}] {description[:50]}...")

        return entry_id
    
    def save(self) -> bool:
        """
        Persist notepad to disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            self.session_path.mkdir(parents=True, exist_ok=True)
            
            # Save notepad
            with open(self.notepad_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[SessionNotepad] Saved notepad to {self.notepad_path}")
            return True
        except Exception as e:
            logger.error(f"[SessionNotepad] Failed to save notepad: {e}")
            return False
    
    def get_full_context(self, variable_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Format ALL entries and variable metadata for LLM injection.
        
        Args:
            variable_metadata: Variable metadata from VariableStorage
            
        Returns:
            Formatted context string
        """
        if not self.data["entries"] and not variable_metadata:
            return ""
        
        lines = []
        
        # Session memory header
        if self.data["entries"]:
            lines.append("=== Session Memory (Notepad) ===")
            lines.append("Previous work from this session:")
            lines.append("")
            
            # Add all entries
            for entry in self.data["entries"]:
                entry_id = entry["entry_id"]
                task = entry["task"]
                description = entry["description"]
                code_file = entry.get("code_file", "")
                variables = entry.get("variables_saved", [])
                outputs = entry.get("key_outputs", "")
                
                # Entry header
                code_ref = f" - {code_file}" if code_file else ""
                lines.append(f"Entry {entry_id}: [{task}]{code_ref}")
                lines.append(f"Description: {description}")
                
                if variables:
                    lines.append(f"Variables saved: {', '.join(variables)}")
                
                if outputs:
                    lines.append(f"Outputs: {outputs}")
                
                lines.append("")  # Blank line between entries
        
        # Variable metadata section
        if variable_metadata:
            lines.append("=== Available Saved Variables ===")
            lines.append("You can load these variables from previous executions:")
            lines.append("")
            
            for idx, (var_name, meta) in enumerate(variable_metadata.items(), 1):
                var_type = meta.get("type", "unknown")
                load_code = meta.get("load_code", "")
                
                lines.append(f"{idx}. {var_name} ({var_type})")
                
                # Add type-specific details
                if var_type == "pandas.DataFrame":
                    shape = meta.get("shape", [])
                    columns = meta.get("columns", [])
                    lines.append(f"   Shape: {shape}, Columns: {', '.join(columns[:5])}")
                elif var_type == "numpy.ndarray":
                    shape = meta.get("shape", [])
                    dtype = meta.get("dtype", "")
                    lines.append(f"   Shape: {shape}, dtype: {dtype}")
                elif var_type == "dict":
                    keys = meta.get("keys", [])
                    lines.append(f"   Keys: {', '.join(str(k) for k in keys)}")
                elif var_type == "list":
                    length = meta.get("length", 0)
                    lines.append(f"   Length: {length}")
                
                # Add load code
                if load_code:
                    lines.append(f"   Load with: {load_code}")
                
                lines.append("")  # Blank line between variables
        
        return "\n".join(lines)
    
    def save_code_file(self, code: str, task_name: str) -> str:
        """
        Save code to a file with descriptive name.
        
        Args:
            code: Python code to save
            task_name: Task category for filename
            
        Returns:
            Filename of saved code
        """
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{task_name}_{timestamp}.py"
        filepath = self.session_path / filename
        
        try:
            # Ensure directory exists
            self.session_path.mkdir(parents=True, exist_ok=True)
            
            # Save code
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(code)
            
            logger.info(f"[SessionNotepad] Saved code to {filename}")
            return filename
        except Exception as e:
            logger.error(f"[SessionNotepad] Failed to save code file: {e}")
            return ""
    
    def get_entries_count(self) -> int:
        """Get total number of entries."""
        return len(self.data["entries"])
    
    def get_latest_entry(self) -> Optional[Dict[str, Any]]:
        """Get the most recent entry."""
        if self.data["entries"]:
            return self.data["entries"][-1]
        return None

    # ========================================================================
    # Enhanced Entry Management
    # ========================================================================

    def update_entry(self, entry_id: int, **updates) -> bool:
        """
        Update existing entry with version history tracking.

        Args:
            entry_id: Entry ID to update
            **updates: Fields to update

        Returns:
            True if successful, False otherwise
        """
        try:
            entry = self.get_entry(entry_id)
            if not entry:
                logger.error(f"[SessionNotepad] Entry #{entry_id} not found")
                return False

            # Save current version to history
            version_snapshot = {
                "version": entry.get("version", 1),
                "timestamp": datetime.now().isoformat(),
                "changes": {}
            }

            # Apply updates and track changes
            for key, value in updates.items():
                if key in entry and entry[key] != value:
                    version_snapshot["changes"][key] = {
                        "old": entry[key],
                        "new": value
                    }
                    entry[key] = value

            # Update version
            if version_snapshot["changes"]:
                if "version_history" not in entry:
                    entry["version_history"] = []
                entry["version_history"].append(version_snapshot)
                entry["version"] = entry.get("version", 1) + 1

                self.data["updated_at"] = datetime.now().isoformat()
                logger.info(f"[SessionNotepad] Updated entry #{entry_id}, version {entry['version']}")
                return True
            else:
                logger.info(f"[SessionNotepad] No changes for entry #{entry_id}")
                return True

        except Exception as e:
            logger.error(f"[SessionNotepad] Failed to update entry #{entry_id}: {e}")
            return False

    def delete_entry(self, entry_id: int, soft_delete: bool = True) -> bool:
        """
        Delete or archive an entry.

        Args:
            entry_id: Entry ID to delete
            soft_delete: If True, archive instead of delete

        Returns:
            True if successful, False otherwise
        """
        try:
            entry = self.get_entry(entry_id)
            if not entry:
                logger.error(f"[SessionNotepad] Entry #{entry_id} not found")
                return False

            if soft_delete:
                entry["is_archived"] = True
                entry["archived_at"] = datetime.now().isoformat()
                logger.info(f"[SessionNotepad] Archived entry #{entry_id}")
            else:
                self.data["entries"] = [e for e in self.data["entries"] if e["entry_id"] != entry_id]
                logger.info(f"[SessionNotepad] Deleted entry #{entry_id}")

            self.data["updated_at"] = datetime.now().isoformat()
            return True

        except Exception as e:
            logger.error(f"[SessionNotepad] Failed to delete entry #{entry_id}: {e}")
            return False

    def get_entry(self, entry_id: int) -> Optional[Dict[str, Any]]:
        """
        Get specific entry by ID.

        Args:
            entry_id: Entry ID

        Returns:
            Entry dictionary or None
        """
        for entry in self.data["entries"]:
            if entry["entry_id"] == entry_id:
                return entry
        return None

    def get_entry_history(self, entry_id: int) -> List[Dict[str, Any]]:
        """
        Get version history for an entry.

        Args:
            entry_id: Entry ID

        Returns:
            List of version snapshots
        """
        entry = self.get_entry(entry_id)
        if entry:
            return entry.get("version_history", [])
        return []

    # ========================================================================
    # Search and Filtering
    # ========================================================================

    def search_entries(
        self,
        query: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        include_archived: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search notepad entries with optional filters.

        Args:
            query: Text query to search in description/task/outputs
            filters: Dictionary of filters:
                - task: Task category
                - tags: List of tags (any match)
                - importance: Importance level
                - date_from: Start date (ISO format)
                - date_to: End date (ISO format)
                - variables: List of variable names (any match)
                - tool_used: Tool name
            include_archived: Include archived entries

        Returns:
            List of matching entries
        """
        results = []
        filters = filters or {}

        for entry in self.data["entries"]:
            # Skip archived entries unless requested
            if entry.get("is_archived", False) and not include_archived:
                continue

            # Text query matching
            if query:
                query_lower = query.lower()
                searchable_text = f"{entry.get('task', '')} {entry.get('description', '')} {entry.get('key_outputs', '')}".lower()
                if query_lower not in searchable_text:
                    continue

            # Apply filters
            if "task" in filters and entry.get("task") != filters["task"]:
                continue

            if "tags" in filters:
                entry_tags = entry.get("tags", [])
                if not any(tag in entry_tags for tag in filters["tags"]):
                    continue

            if "importance" in filters and entry.get("importance") != filters["importance"]:
                continue

            if "date_from" in filters:
                entry_date = datetime.fromisoformat(entry["timestamp"])
                filter_date = datetime.fromisoformat(filters["date_from"])
                if entry_date < filter_date:
                    continue

            if "date_to" in filters:
                entry_date = datetime.fromisoformat(entry["timestamp"])
                filter_date = datetime.fromisoformat(filters["date_to"])
                if entry_date > filter_date:
                    continue

            if "variables" in filters:
                entry_vars = entry.get("variables_saved", [])
                if not any(var in entry_vars for var in filters["variables"]):
                    continue

            if "tool_used" in filters and entry.get("tool_used") != filters["tool_used"]:
                continue

            results.append(entry)

        logger.info(f"[SessionNotepad] Search returned {len(results)} entries")
        return results

    def find_entries_with_variable(self, variable_name: str) -> List[Dict[str, Any]]:
        """
        Find all entries that saved or used a specific variable.

        Args:
            variable_name: Variable name to search for

        Returns:
            List of entries
        """
        return self.search_entries(filters={"variables": [variable_name]})

    def find_entries_by_task(self, task: str) -> List[Dict[str, Any]]:
        """
        Find all entries for a specific task category.

        Args:
            task: Task category

        Returns:
            List of entries
        """
        return self.search_entries(filters={"task": task})

    # ========================================================================
    # Analytics and Insights
    # ========================================================================

    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive session statistics.

        Returns:
            Dictionary with session analytics
        """
        entries = [e for e in self.data["entries"] if not e.get("is_archived", False)]

        if not entries:
            return {
                "total_entries": 0,
                "message": "No entries yet"
            }

        # Count by task type
        task_counts = {}
        for entry in entries:
            task = entry.get("task", "unknown")
            task_counts[task] = task_counts.get(task, 0) + 1

        # Count by tool
        tool_counts = {}
        for entry in entries:
            tool = entry.get("tool_used")
            if tool:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1

        # Calculate average execution time
        exec_times = [e.get("execution_time_ms") for e in entries if e.get("execution_time_ms")]
        avg_exec_time = sum(exec_times) / len(exec_times) if exec_times else 0

        # Calculate success rate
        success_scores = [e.get("success_score") for e in entries if e.get("success_score") is not None]
        avg_success = sum(success_scores) / len(success_scores) if success_scores else None

        # Count variables
        all_variables = set()
        for entry in entries:
            all_variables.update(entry.get("variables_saved", []))

        # Time range
        timestamps = [datetime.fromisoformat(e["timestamp"]) for e in entries]
        time_span = (max(timestamps) - min(timestamps)).total_seconds() / 3600  # hours

        return {
            "total_entries": len(entries),
            "archived_entries": len([e for e in self.data["entries"] if e.get("is_archived", False)]),
            "task_breakdown": task_counts,
            "tool_usage": tool_counts,
            "most_common_task": max(task_counts.items(), key=lambda x: x[1])[0] if task_counts else None,
            "total_variables_created": len(all_variables),
            "avg_execution_time_ms": round(avg_exec_time, 2) if exec_times else None,
            "avg_success_score": round(avg_success, 3) if avg_success else None,
            "session_duration_hours": round(time_span, 2),
            "created_at": self.data.get("created_at"),
            "last_updated": self.data.get("updated_at")
        }

    def get_variable_usage_patterns(self) -> Dict[str, Any]:
        """
        Analyze variable usage patterns across entries.

        Returns:
            Dictionary with variable usage analytics
        """
        var_usage = {}

        for entry in self.data["entries"]:
            if entry.get("is_archived", False):
                continue

            for var in entry.get("variables_saved", []):
                if var not in var_usage:
                    var_usage[var] = {
                        "count": 0,
                        "entries": [],
                        "tasks": set()
                    }

                var_usage[var]["count"] += 1
                var_usage[var]["entries"].append(entry["entry_id"])
                var_usage[var]["tasks"].add(entry.get("task", "unknown"))

        # Convert sets to lists for JSON serialization
        for var in var_usage:
            var_usage[var]["tasks"] = list(var_usage[var]["tasks"])

        # Sort by usage count
        sorted_vars = dict(sorted(var_usage.items(), key=lambda x: x[1]["count"], reverse=True))

        return {
            "total_unique_variables": len(var_usage),
            "most_used_variables": list(sorted_vars.keys())[:10],
            "variable_details": sorted_vars
        }

    def get_task_sequence_patterns(self) -> List[str]:
        """
        Identify common task sequences in the session.

        Returns:
            List of task sequences
        """
        sequences = []
        entries = [e for e in self.data["entries"] if not e.get("is_archived", False)]

        for i in range(len(entries) - 1):
            task1 = entries[i].get("task", "unknown")
            task2 = entries[i + 1].get("task", "unknown")
            sequences.append(f"{task1} -> {task2}")

        # Count occurrences
        from collections import Counter
        sequence_counts = Counter(sequences)

        return [f"{seq} (x{count})" for seq, count in sequence_counts.most_common(5)]

    # ========================================================================
    # Code Management
    # ========================================================================

    def get_code_content(self, code_file: str) -> Optional[str]:
        """
        Read code content from saved file.

        Args:
            code_file: Filename of saved code

        Returns:
            Code content or None
        """
        try:
            filepath = self.session_path / code_file
            if filepath.exists():
                return filepath.read_text(encoding='utf-8')
            return None
        except Exception as e:
            logger.error(f"[SessionNotepad] Failed to read code file {code_file}: {e}")
            return None

    def compare_code_versions(self, file1: str, file2: str) -> str:
        """
        Generate unified diff between two code versions.

        Args:
            file1: First code filename
            file2: Second code filename

        Returns:
            Unified diff string
        """
        code1 = self.get_code_content(file1)
        code2 = self.get_code_content(file2)

        if not code1 or not code2:
            return "Error: One or both files not found"

        diff = unified_diff(
            code1.splitlines(keepends=True),
            code2.splitlines(keepends=True),
            fromfile=file1,
            tofile=file2
        )

        return ''.join(diff)

    def get_code_evolution(self, task: str) -> List[Dict[str, Any]]:
        """
        Track code evolution for a specific task category.

        Args:
            task: Task category

        Returns:
            List of code versions with metadata
        """
        entries = self.find_entries_by_task(task)
        code_versions = []

        for entry in entries:
            code_file = entry.get("code_file")
            if code_file:
                code_versions.append({
                    "entry_id": entry["entry_id"],
                    "code_file": code_file,
                    "timestamp": entry["timestamp"],
                    "description": entry["description"],
                    "variables": entry.get("variables_saved", [])
                })

        return code_versions

    # ========================================================================
    # Export Functionality
    # ========================================================================

    def export_to_markdown(self, include_code: bool = True, include_archived: bool = False) -> str:
        """
        Export notepad to Markdown format.

        Args:
            include_code: Include full code in export
            include_archived: Include archived entries

        Returns:
            Markdown formatted string
        """
        lines = []

        # Header
        lines.append(f"# Session Notepad: {self.session_id}")
        lines.append(f"\n**Created:** {self.data.get('created_at', 'Unknown')}")
        lines.append(f"**Last Updated:** {self.data.get('updated_at', 'Unknown')}")

        # Stats
        stats = self.get_session_stats()
        lines.append(f"\n## Session Statistics\n")
        lines.append(f"- **Total Entries:** {stats['total_entries']}")
        lines.append(f"- **Session Duration:** {stats.get('session_duration_hours', 0):.2f} hours")
        lines.append(f"- **Variables Created:** {stats['total_variables_created']}")

        if stats.get('avg_success_score'):
            lines.append(f"- **Average Success Rate:** {stats['avg_success_score']*100:.1f}%")

        # Entries
        lines.append(f"\n## Entries\n")

        for entry in self.data["entries"]:
            if entry.get("is_archived", False) and not include_archived:
                continue

            lines.append(f"### Entry #{entry['entry_id']}: {entry['task']}")
            lines.append(f"\n**Description:** {entry['description']}")
            lines.append(f"\n**Timestamp:** {entry['timestamp']}")

            if entry.get("tags"):
                lines.append(f"\n**Tags:** {', '.join(entry['tags'])}")

            if entry.get("importance"):
                lines.append(f"\n**Importance:** {entry['importance']}")

            if entry.get("variables_saved"):
                lines.append(f"\n**Variables Saved:** `{', '.join(entry['variables_saved'])}`")

            if entry.get("key_outputs"):
                lines.append(f"\n**Key Outputs:** {entry['key_outputs']}")

            if include_code and entry.get("code_file"):
                code = self.get_code_content(entry["code_file"])
                if code:
                    lines.append(f"\n**Code ({entry['code_file']}):**")
                    lines.append(f"\n```python\n{code}\n```")

            lines.append("\n---\n")

        return "\n".join(lines)

    def export_to_jupyter_notebook(self) -> Dict[str, Any]:
        """
        Export notepad to Jupyter notebook format.

        Returns:
            Jupyter notebook JSON structure
        """
        cells = []

        # Title cell
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# Session Notepad: {self.session_id}\n",
                f"\n**Created:** {self.data.get('created_at', 'Unknown')}\n",
                f"**Last Updated:** {self.data.get('updated_at', 'Unknown')}"
            ]
        })

        # Add each entry as cells
        for entry in self.data["entries"]:
            if entry.get("is_archived", False):
                continue

            # Markdown cell for entry metadata
            md_lines = [
                f"## Entry #{entry['entry_id']}: {entry['task']}\n",
                f"\n**Description:** {entry['description']}\n"
            ]

            if entry.get("variables_saved"):
                md_lines.append(f"\n**Variables:** `{', '.join(entry['variables_saved'])}`\n")

            if entry.get("key_outputs"):
                md_lines.append(f"\n**Outputs:** {entry['key_outputs']}\n")

            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": md_lines
            })

            # Code cell
            if entry.get("code_file"):
                code = self.get_code_content(entry["code_file"])
                if code:
                    cells.append({
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": code.splitlines(keepends=True)
                    })

        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.9.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }

        return notebook

    def export_to_json(self, include_archived: bool = False) -> Dict[str, Any]:
        """
        Export notepad to JSON format.

        Args:
            include_archived: Include archived entries

        Returns:
            Complete notepad data as dictionary
        """
        export_data = self.data.copy()

        if not include_archived:
            export_data["entries"] = [
                e for e in export_data["entries"]
                if not e.get("is_archived", False)
            ]

        return export_data

