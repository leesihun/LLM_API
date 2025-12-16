"""
Base interface for Python code executors
All executors must implement this interface for consistent behavior
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class BasePythonExecutor(ABC):
    """
    Abstract base class for Python code execution backends

    All implementations must:
    - Accept session_id for workspace isolation
    - Return standardized result dictionaries
    - Support the same execution interface
    """

    def __init__(self, session_id: str):
        """
        Initialize executor with session ID

        Args:
            session_id: Session ID for workspace isolation
        """
        self.session_id = session_id

    @abstractmethod
    def execute(
        self,
        code: str,
        timeout: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute Python code

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds (optional)
            context: Additional context dictionary (optional)

        Returns:
            Execution result dictionary with standardized format:
            {
                "success": bool,
                "stdout": str,
                "stderr": str,
                "returncode": int,
                "execution_time": float,
                "files": dict,
                "workspace": str,
                "error": Optional[str]
            }
        """
        pass

    @abstractmethod
    def read_file(self, filename: str) -> Optional[str]:
        """
        Read a file from workspace

        Args:
            filename: Name of file to read

        Returns:
            File contents as string, or None if file not found
        """
        pass

    @abstractmethod
    def list_files(self) -> List[str]:
        """
        List all files in workspace

        Returns:
            List of filenames in workspace
        """
        pass

    @abstractmethod
    def clear_workspace(self):
        """
        Clear all files from workspace

        Returns:
            None
        """
        pass
