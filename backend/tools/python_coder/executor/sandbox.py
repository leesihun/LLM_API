"""Sandbox configuration for secure code execution."""

from pathlib import Path
from typing import List

# Security: Blocked imports for code execution
BLOCKED_IMPORTS = [
    "socket", "subprocess", "os.system", "eval", "exec",
    "__import__", "importlib", "shutil.rmtree", "pickle",
]

# Supported file types for input files
SUPPORTED_FILE_TYPES = [
    ".txt", ".md", ".log", ".rtf",  # Text
    ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml",  # Data
    ".xlsx", ".xls", ".xlsm", ".docx", ".doc",  # Office
    ".pdf",  # PDF
    ".dat", ".h5", ".hdf5", ".nc", ".parquet", ".feather",  # Scientific
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".svg",  # Images
    ".zip", ".tar", ".gz", ".bz2", ".7z",  # Compressed
]


class SandboxConfig:
    """
    Configuration for sandboxed code execution.
    Controls security restrictions, resource limits, and execution environment.
    """

    def __init__(
        self,
        timeout: int = 30,
        max_memory_mb: int = 512,
        execution_base_dir: str = "./data/scratch",
        use_persistent_repl: bool = True
    ):
        """
        Initialize sandbox configuration.

        Args:
            timeout: Maximum execution time in seconds
            max_memory_mb: Maximum memory usage in MB (not enforced yet)
            execution_base_dir: Base directory for code execution
            use_persistent_repl: Enable persistent REPL for faster retries
        """
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.execution_base_dir = Path(execution_base_dir).resolve()
        self.use_persistent_repl = use_persistent_repl

        # Ensure base directory exists
        self.execution_base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def blocked_imports(self) -> List[str]:
        """Get list of blocked import modules."""
        return BLOCKED_IMPORTS.copy()

    @property
    def supported_file_types(self) -> List[str]:
        """Get list of supported file extensions."""
        return SUPPORTED_FILE_TYPES.copy()

    def is_file_type_supported(self, file_path: str) -> bool:
        """
        Check if file type is supported.

        Args:
            file_path: Path to file

        Returns:
            True if file type is supported
        """
        ext = Path(file_path).suffix.lower()
        return ext in SUPPORTED_FILE_TYPES

    def get_execution_dir(self, session_id: str) -> Path:
        """
        Get execution directory for a session.

        Args:
            session_id: Session identifier

        Returns:
            Path to execution directory
        """
        execution_dir = self.execution_base_dir / session_id
        execution_dir.mkdir(parents=True, exist_ok=True)
        return execution_dir

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "timeout": self.timeout,
            "max_memory_mb": self.max_memory_mb,
            "execution_base_dir": str(self.execution_base_dir),
            "use_persistent_repl": self.use_persistent_repl,
            "blocked_imports": self.blocked_imports,
            "supported_file_types_count": len(SUPPORTED_FILE_TYPES)
        }
