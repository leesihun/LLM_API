import logging
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any
from dataclasses import dataclass
import re

from backend.config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class FileEntry:
    """Represents a file loaded from a session directory."""
    path: Path
    content: str
    timestamp: float
    metadata: Dict[str, Any]

class SessionFileLoader:
    """
    Centralized session file loading utilities.
    
    Handles loading, filtering, and sorting files from session directories.
    Replaces duplicate logic in execution.py and python_coder tools.
    """

    def __init__(self, session_id: str):
        """
        Initialize with session ID.

        Args:
            session_id: The unique session identifier
        """
        self.session_id = session_id
        self.session_dir = Path(settings.python_code_execution_dir) / session_id

    def load_files_by_pattern(
        self,
        pattern: str,
        processor: Optional[Callable[[Path, str], Dict[str, Any]]] = None,
        sort_reverse: bool = True
    ) -> List[FileEntry]:
        """
        Load files matching pattern from session directory.

        Args:
            pattern: Glob pattern (e.g., "script_*.py")
            processor: Optional function to extract metadata from path and content
            sort_reverse: Sort by mtime descending (default: True)

        Returns:
            List of FileEntry objects
        """
        if not self.session_dir.exists():
            logger.debug(f"Session directory does not exist: {self.session_dir}")
            return []

        try:
            files = list(self.session_dir.glob(pattern))
        except Exception as e:
            logger.warning(f"Failed to glob pattern {pattern} in {self.session_dir}: {e}")
            return []

        if not files:
            return []

        # Sort by modification time
        try:
            files.sort(key=lambda p: p.stat().st_mtime, reverse=sort_reverse)
        except OSError as e:
            logger.warning(f"Error accessing file stats during sort: {e}")
            # Fallback to name sort if stat fails
            files.sort(key=lambda p: p.name, reverse=sort_reverse)

        entries = []
        for file_path in files:
            try:
                content = file_path.read_text(encoding='utf-8')
                metadata = processor(file_path, content) if processor else {}

                entries.append(FileEntry(
                    path=file_path,
                    content=content,
                    timestamp=file_path.stat().st_mtime,
                    metadata=metadata
                ))
            except Exception as e:
                logger.warning(f"Failed to read {file_path.name}: {e}")

        return entries

    def extract_attempt_number(self, filename: str) -> int:
        """
        Extract attempt number from filename.
        
        Expected format: ..._attempt(\d+)...
        
        Args:
            filename: Name of the file
            
        Returns:
            Attempt number (default 1 if not found)
        """
        match = re.search(r'attempt(\d+)', filename)
        if match:
            return int(match.group(1))
        return 1

