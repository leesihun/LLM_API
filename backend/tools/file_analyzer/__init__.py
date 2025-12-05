"""
Lightweight File Analyzer Tool
==============================
Single-file implementation that relies on the shared file handler registry.

Goals:
- Minimal surface area (one module, one class)
- No optional fallbacks or extra abstractions
- Fast summaries with just enough structure for agents
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.core import BaseTool, ToolResult
from backend.core.file_handlers import file_handler_registry
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class FileAnalyzerTool(BaseTool):
    """Simple wrapper around the shared file handler registry."""

    async def execute(
        self,
        query: str,
        context: Optional[str] = None,
        *,
        file_paths: Optional[List[str]] = None,
        user_query: Optional[str] = None,
        **_: Any,
    ) -> ToolResult:
        """Async entrypoint for BaseTool compatibility."""
        if not self.validate_inputs(file_paths=file_paths):
            return self._handle_validation_error(
                "file_paths must be a non-empty list of paths",
                parameter="file_paths",
            )

        try:
            result = self.analyze(file_paths=file_paths or [], user_query=user_query or query)
            return ToolResult.success_result(
                output=result["summary"],
                metadata=result,
                execution_time=self._elapsed_time(),
            )
        except Exception as exc:
            return self._handle_error(exc, "execute")

    def validate_inputs(self, **kwargs: Any) -> bool:
        file_paths = kwargs.get("file_paths")
        return isinstance(file_paths, list) and len(file_paths) > 0

    def analyze(self, file_paths: List[str], user_query: str = "") -> Dict[str, Any]:
        """Synchronous helper used by agents."""
        normalized_paths = self._normalize_paths(file_paths)
        if not normalized_paths:
            return {
                "success": False,
                "summary": "No valid files to analyze.",
                "results": [],
                "files_analyzed": 0,
                "metadata": {"analysis_id": uuid.uuid4().hex[:8], "user_query": user_query},
                "error": "No files provided",
            }

        results = [self._analyze_single(path) for path in normalized_paths]
        summary = self._build_summary(results, user_query)
        success = any(item["success"] for item in results)

        metadata = {
            "analysis_id": uuid.uuid4().hex[:8],
            "files_analyzed": len(results),
            "successful_files": sum(1 for item in results if item["success"]),
            "failed_files": sum(1 for item in results if not item["success"]),
            "user_query": user_query,
        }

        return {
            "success": success,
            "summary": summary,
            "results": results,
            "files_analyzed": len(results),
            "metadata": metadata,
            "error": None if success else "All files failed to analyze",
        }

    def _normalize_paths(self, file_paths: List[str]) -> List[str]:
        seen = set()
        normalized = []
        for path in file_paths:
            if not path:
                continue
            resolved = str(Path(path))
            if resolved not in seen:
                seen.add(resolved)
                normalized.append(resolved)
        return normalized

    def _analyze_single(self, file_path: str) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            return {
                "file": path.name or file_path,
                "full_path": str(path),
                "success": False,
                "format": "unknown",
                "error": "File not found",
            }

        handler = file_handler_registry.get_handler(path)
        if handler is None:
            return {
                "file": path.name,
                "full_path": str(path.resolve()),
                "success": False,
                "format": "unsupported",
                "error": f"No handler for *.{path.suffix.lstrip('.') or 'unknown'} files",
            }

        try:
            details = handler.analyze(path) or {}
        except Exception as exc:
            logger.error(f"[FileAnalyzer] Handler error for {path.name}: {exc}")
            return {
                "file": path.name,
                "full_path": str(path.resolve()),
                "success": False,
                "format": handler.__class__.__name__.replace("Handler", ""),
                "error": str(exc),
            }

        size_bytes = path.stat().st_size
        result = {
            "file": path.name,
            "full_path": str(path.resolve()),
            "extension": path.suffix.lstrip(".").lower(),
            "size_bytes": size_bytes,
            "size_human": self._human_size(size_bytes),
            "format": details.get("format") or handler.__class__.__name__.replace("Handler", ""),
            "success": not bool(details.get("error")),
        }
        result.update(details)
        return result

    def _build_summary(self, results: List[Dict[str, Any]], user_query: str) -> str:
        lines = [f"Analyzed {len(results)} file(s)."]
        if user_query:
            lines.append(f"Focus: {user_query}")

        successful = [item for item in results if item["success"]]
        failed = [item for item in results if not item["success"]]

        if successful:
            lines.append("Successful analyses:")
            for item in successful[:5]:
                line = f"- {item['file']} ({item.get('format', 'unknown')}, {item.get('size_human', 'unknown size')})"
                columns = item.get("columns")
                if isinstance(columns, list) and columns:
                    preview = ", ".join(str(c) for c in columns[:3])
                    if len(columns) > 3:
                        preview += "..."
                    line += f" | Columns: {preview}"
                lines.append(line)

        if failed:
            lines.append("Failed analyses:")
            for item in failed[:3]:
                lines.append(f"- {item['file']}: {item.get('error', 'Unknown error')}")

        return "\n".join(lines)

    def _human_size(self, size_bytes: int) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        value = float(size_bytes)
        for unit in units:
            if value < 1024 or unit == units[-1]:
                return f"{value:.2f} {unit}"
            value /= 1024
        return f"{value:.2f} TB"


file_analyzer = FileAnalyzerTool()


def analyze_files(file_paths: List[str], user_query: str = "") -> Dict[str, Any]:
    """Backward compatible helper."""
    return file_analyzer.analyze(file_paths, user_query)


__all__ = ["FileAnalyzerTool", "file_analyzer", "analyze_files"]
