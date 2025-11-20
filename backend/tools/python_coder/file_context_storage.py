"""
File Context Storage for Python Coder Tool

Saves and loads analyzed file context to/from session directories.
This enables multi-phase workflows where file analysis is done once
and reused across multiple phases.

Created: 2025-01-20
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

from backend.utils.logging_utils import get_logger
from backend.config.settings import settings

logger = get_logger(__name__)


class FileContextStorage:
    """
    Manages file context persistence in session directories.

    Stores file metadata and context analysis results so that subsequent
    phases can reuse the analysis instead of re-processing files.
    """

    FILE_CONTEXT_FILENAME = "file_context.json"

    @staticmethod
    def save_file_context(
        session_id: str,
        validated_files: Dict[str, str],
        file_metadata: Dict[str, Any],
        file_context_text: str
    ) -> bool:
        """
        Save file context to session directory.

        Args:
            session_id: Session ID for directory path
            validated_files: Dict mapping original file paths to basenames
            file_metadata: Detailed metadata for each file
            file_context_text: Human-readable file context string

        Returns:
            True if saved successfully, False otherwise
        """
        if not session_id:
            logger.debug("[FileContextStorage] No session_id provided, skipping save")
            return False

        try:
            # Get session directory
            session_dir = Path(settings.python_code_execution_dir) / session_id
            session_dir.mkdir(parents=True, exist_ok=True)

            # Build context data structure
            context_data = {
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "validated_files": validated_files,
                "file_metadata": FileContextStorage._serialize_metadata(file_metadata),
                "file_context_text": file_context_text,
                "file_count": len(validated_files),
                "file_list": list(validated_files.values())
            }

            # Save to JSON file
            context_file = session_dir / FileContextStorage.FILE_CONTEXT_FILENAME
            with open(context_file, 'w', encoding='utf-8') as f:
                json.dump(context_data, f, indent=2, ensure_ascii=False)

            logger.success(
                f"[FileContextStorage] Saved file context: {len(validated_files)} files → {context_file.name}"
            )
            return True

        except Exception as e:
            logger.error(f"[FileContextStorage] Failed to save file context: {e}")
            return False

    @staticmethod
    def load_file_context(session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load file context from session directory.

        Args:
            session_id: Session ID for directory path

        Returns:
            Dict with file context data, or None if not found
        """
        if not session_id:
            logger.debug("[FileContextStorage] No session_id provided, cannot load")
            return None

        try:
            # Get context file path
            session_dir = Path(settings.python_code_execution_dir) / session_id
            context_file = session_dir / FileContextStorage.FILE_CONTEXT_FILENAME

            if not context_file.exists():
                logger.debug(f"[FileContextStorage] No saved context found for session {session_id[:8]}")
                return None

            # Load from JSON
            with open(context_file, 'r', encoding='utf-8') as f:
                context_data = json.load(f)

            logger.success(
                f"[FileContextStorage] Loaded file context: {context_data.get('file_count', 0)} files from {context_file.name}"
            )
            return context_data

        except Exception as e:
            logger.error(f"[FileContextStorage] Failed to load file context: {e}")
            return None

    @staticmethod
    def has_file_context(session_id: str) -> bool:
        """
        Check if file context exists for session.

        Args:
            session_id: Session ID to check

        Returns:
            True if context file exists, False otherwise
        """
        if not session_id:
            return False

        session_dir = Path(settings.python_code_execution_dir) / session_id
        context_file = session_dir / FileContextStorage.FILE_CONTEXT_FILENAME
        return context_file.exists()

    @staticmethod
    def get_file_context_summary(session_id: str) -> Optional[str]:
        """
        Get a brief summary of saved file context.

        Args:
            session_id: Session ID

        Returns:
            Summary string or None if not found
        """
        context_data = FileContextStorage.load_file_context(session_id)
        if not context_data:
            return None

        file_count = context_data.get('file_count', 0)
        file_list = context_data.get('file_list', [])
        timestamp = context_data.get('timestamp', 'unknown')

        summary = f"Saved file context from {timestamp}:\n"
        summary += f"  - {file_count} file(s): {', '.join(file_list[:5])}"
        if file_count > 5:
            summary += f" ... (+{file_count - 5} more)"

        return summary

    @staticmethod
    def _serialize_metadata(file_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize file metadata for JSON storage.

        Converts non-serializable objects to strings.

        Args:
            file_metadata: Original metadata dict

        Returns:
            Serialized metadata dict
        """
        serialized = {}

        for file_path, metadata in file_metadata.items():
            serialized[file_path] = FileContextStorage._serialize_value(metadata)

        return serialized

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """
        Recursively serialize a value for JSON storage.

        Args:
            value: Value to serialize

        Returns:
            Serialized value
        """
        if isinstance(value, dict):
            return {k: FileContextStorage._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [FileContextStorage._serialize_value(v) for v in value]
        elif isinstance(value, (str, int, float, bool, type(None))):
            return value
        else:
            # Convert non-serializable objects to string
            return str(value)


# Convenience functions for direct use
def save_file_context(
    session_id: str,
    validated_files: Dict[str, str],
    file_metadata: Dict[str, Any],
    file_context_text: str
) -> bool:
    """Save file context to session directory."""
    return FileContextStorage.save_file_context(
        session_id, validated_files, file_metadata, file_context_text
    )


def load_file_context(session_id: str) -> Optional[Dict[str, Any]]:
    """Load file context from session directory."""
    return FileContextStorage.load_file_context(session_id)


def has_file_context(session_id: str) -> bool:
    """Check if file context exists for session."""
    return FileContextStorage.has_file_context(session_id)


def get_file_context_summary(session_id: str) -> Optional[str]:
    """Get summary of saved file context."""
    return FileContextStorage.get_file_context_summary(session_id)
