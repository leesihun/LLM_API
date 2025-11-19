"""
Session Notepad Module

Manages persistent session memory that stores summaries of completed tasks,
code files, and variable metadata for future reference within a session.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

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
        key_outputs: Optional[str] = None
    ) -> int:
        """
        Add new entry to notepad.
        
        Args:
            task: Task category (e.g., "data_analysis", "visualization")
            description: What was accomplished
            code_file: Filename of saved code (if applicable)
            variables_saved: List of variable names that were saved
            key_outputs: Summary of important outputs/results
            
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
            "timestamp": datetime.now().isoformat()
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

