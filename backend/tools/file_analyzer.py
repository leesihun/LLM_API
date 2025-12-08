"""
File Analyzer Tool
==================
Analyzes attached files and generates comprehensive structure information for LLMs.

This tool provides detailed metadata, schema, and content previews to help LLMs
understand file contents and generate appropriate code/responses.

Version: 1.1.0
Created: 2025-12-05
Updated: 2025-12-08 - Added BaseTool-compatible FileAnalyzerTool wrapper
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from backend.core import BaseTool, ToolResult
from backend.core.file_handlers import file_handler_registry
from backend.utils.logging_utils import get_logger
from backend.runtime import extract_original_filename

logger = get_logger(__name__)


class FileAnalyzer:
    """
    Analyzes files and generates structured information for LLM consumption.

    Provides:
    - File metadata (size, type, structure)
    - Schema information (columns, data types)
    - Content previews (sample data)
    - Data quality metrics (nulls, duplicates)
    - Formatted context strings for prompts
    """

    def __init__(self):
        self.registry = file_handler_registry
        logger.info("[FileAnalyzer] Initialized")

    def analyze(
        self,
        file_paths: List[str],
        user_query: Optional[str] = None,
        quick_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze multiple files and generate comprehensive report.

        Args:
            file_paths: List of file paths to analyze
            user_query: Optional user query for context
            quick_mode: If True, only extract essential metadata (faster)

        Returns:
            Dict with success status, summary, and detailed analysis
        """
        if not file_paths:
            return {
                "success": False,
                "error": "No files provided",
                "summary": "No files to analyze"
            }

        try:
            analyses = []
            for file_path in file_paths:
                path = Path(file_path)
                if not path.exists():
                    logger.warning(f"[FileAnalyzer] File not found: {file_path}")
                    analyses.append({
                        "file_path": file_path,
                        "error": "File not found",
                        "success": False
                    })
                    continue

                # Extract original filename
                original_name = extract_original_filename(str(path))

                # Get metadata
                metadata = self.registry.extract_metadata(
                    path,
                    quick_mode=quick_mode,
                    use_cache=True
                )

                # Perform detailed analysis if not quick mode
                if not quick_mode:
                    detailed = self.registry.analyze(path)
                else:
                    detailed = {}

                analyses.append({
                    "file_path": str(path),
                    "original_name": original_name,
                    "metadata": metadata,
                    "detailed_analysis": detailed,
                    "success": metadata.get('error') is None
                })

            # Generate summary
            summary = self._generate_summary(analyses, user_query)

            # Generate LLM-friendly context
            llm_context = self._generate_llm_context(analyses)

            return {
                "success": True,
                "summary": summary,
                "llm_context": llm_context,
                "analyses": analyses,
                "file_count": len(file_paths)
            }

        except Exception as e:
            logger.error(f"[FileAnalyzer] Analysis failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "summary": f"File analysis failed: {str(e)}"
            }

    def _generate_summary(
        self,
        analyses: List[Dict[str, Any]],
        user_query: Optional[str] = None
    ) -> str:
        """Generate human-readable summary of file analysis."""
        lines = []

        if user_query:
            lines.append(f"Query: {user_query}\n")

        lines.append(f"Analyzed {len(analyses)} file(s):\n")

        for i, analysis in enumerate(analyses, 1):
            original_name = analysis.get('original_name', 'unknown')
            metadata = analysis.get('metadata', {})
            file_type = metadata.get('file_type', 'unknown')

            if not analysis.get('success'):
                error = analysis.get('error', metadata.get('error', 'Unknown error'))
                lines.append(f"{i}. {original_name} - ERROR: {error}")
                continue

            # File type specific summary
            if file_type == 'csv' or file_type == 'excel':
                rows = metadata.get('rows', 'unknown')
                columns = metadata.get('columns', [])
                lines.append(
                    f"{i}. {original_name} ({file_type.upper()}): "
                    f"{rows:,} rows × {len(columns)} columns"
                )
                lines.append(f"   Columns: {', '.join(str(c) for c in columns[:6])}")
                if len(columns) > 6:
                    lines.append(f"   ... and {len(columns) - 6} more")

            elif file_type == 'json':
                structure = metadata.get('structure', 'unknown')
                keys = metadata.get('top_level_keys', [])
                lines.append(
                    f"{i}. {original_name} (JSON): "
                    f"Structure={structure}"
                )
                if keys:
                    lines.append(f"   Keys: {', '.join(str(k) for k in keys[:6])}")

            elif file_type == 'image':
                dimensions = metadata.get('dimensions', ())
                format_type = metadata.get('format', 'unknown')
                if dimensions:
                    lines.append(
                        f"{i}. {original_name} (Image): "
                        f"{format_type} {dimensions[0]}×{dimensions[1]}px"
                    )
                else:
                    lines.append(f"{i}. {original_name} (Image): {format_type}")

            elif file_type == 'pdf':
                pages = metadata.get('page_count', 'unknown')
                lines.append(f"{i}. {original_name} (PDF): {pages} pages")

            elif file_type == 'docx':
                paragraphs = metadata.get('paragraph_count', 'unknown')
                lines.append(f"{i}. {original_name} (DOCX): {paragraphs} paragraphs")

            else:
                size = metadata.get('file_size_human', metadata.get('file_size', 'unknown'))
                lines.append(f"{i}. {original_name} ({file_type}): {size}")

        return "\n".join(lines)

    def _generate_llm_context(self, analyses: List[Dict[str, Any]]) -> str:
        """
        Generate structured context string optimized for LLM consumption.

        This provides the LLM with all necessary information to:
        - Understand file structures
        - Generate appropriate code
        - Reference correct column names and data types
        """
        lines = ["# File Structure Information"]
        lines.append("")

        for i, analysis in enumerate(analyses, 1):
            original_name = analysis.get('original_name', 'unknown')
            file_path = analysis.get('file_path', 'unknown')
            metadata = analysis.get('metadata', {})
            file_type = metadata.get('file_type', 'unknown')

            lines.append(f"## File {i}: {original_name}")
            lines.append(f"- **Path**: `{file_path}`")
            lines.append(f"- **Type**: {file_type.upper()}")

            if not analysis.get('success'):
                error = analysis.get('error', metadata.get('error', 'Unknown error'))
                lines.append(f"- **Error**: {error}")
                lines.append("")
                continue

            # Type-specific details
            if file_type == 'csv' or file_type == 'excel':
                self._add_tabular_context(lines, metadata)

            elif file_type == 'json':
                self._add_json_context(lines, metadata)

            elif file_type == 'image':
                self._add_image_context(lines, metadata)

            elif file_type == 'pdf':
                self._add_pdf_context(lines, metadata)

            elif file_type == 'docx':
                self._add_docx_context(lines, metadata)

            elif file_type == 'text':
                self._add_text_context(lines, metadata)

            lines.append("")

        return "\n".join(lines)

    def _add_tabular_context(self, lines: List[str], metadata: Dict[str, Any]):
        """Add context for CSV/Excel files."""
        rows = metadata.get('rows', 'unknown')
        columns = metadata.get('columns', [])
        dtypes = metadata.get('dtypes', {})

        if rows != 'unknown':
            lines.append(f"- **Rows**: {rows:,}")
        lines.append(f"- **Columns**: {len(columns)}")

        # Column details
        if columns:
            lines.append("- **Column Schema**:")
            for col in columns[:20]:  # Limit to 20 columns
                dtype = dtypes.get(col, 'unknown')
                lines.append(f"  - `{col}`: {dtype}")
            if len(columns) > 20:
                lines.append(f"  - ... and {len(columns) - 20} more columns")

        # Null analysis
        null_analysis = metadata.get('null_analysis', {})
        if null_analysis:
            lines.append("- **Null Values**:")
            for col, count in list(null_analysis.items())[:10]:
                lines.append(f"  - `{col}`: {count} nulls")

        # Preview
        preview = metadata.get('preview', [])
        if preview:
            lines.append("- **Sample Data** (first 3 rows):")
            lines.append(f"```")
            lines.append(json.dumps(preview, indent=2, ensure_ascii=False))
            lines.append(f"```")

        # Statistics
        numeric_stats = metadata.get('numeric_stats', {})
        if numeric_stats:
            lines.append("- **Numeric Statistics** (top 5 columns):")
            for col, stats in list(numeric_stats.items())[:5]:
                mean = stats.get('mean', 0)
                std = stats.get('std', 0)
                min_val = stats.get('min', 0)
                max_val = stats.get('max', 0)
                lines.append(
                    f"  - `{col}`: mean={mean:.2f}, std={std:.2f}, "
                    f"range=[{min_val:.2f}, {max_val:.2f}]"
                )

    def _add_json_context(self, lines: List[str], metadata: Dict[str, Any]):
        """Add context for JSON files."""
        structure = metadata.get('structure', 'unknown')
        lines.append(f"- **Structure**: {structure}")

        # Item count for lists
        item_count = metadata.get('item_count')
        if item_count is not None:
            lines.append(f"- **Items**: {item_count:,}")

        # Key count for dicts
        key_count = metadata.get('key_count')
        if key_count is not None:
            lines.append(f"- **Keys**: {key_count}")

        # Top-level keys
        keys = metadata.get('top_level_keys', [])
        if keys:
            lines.append(f"- **Top-level keys**: {', '.join(str(k) for k in keys[:15])}")
            if len(keys) > 15:
                lines.append(f"  ... and {len(keys) - 15} more keys")

        # Value types
        value_types = metadata.get('value_types', {})
        if value_types:
            lines.append("- **Value Types**:")
            for key, vtype in list(value_types.items())[:10]:
                lines.append(f"  - `{key}`: {vtype}")
            if len(value_types) > 10:
                lines.append(f"  ... and {len(value_types) - 10} more")

        # Max depth
        max_depth = metadata.get('max_depth')
        if max_depth is not None:
            lines.append(f"- **Nesting Depth**: {max_depth}")

        # Item types (for list of primitives)
        item_types = metadata.get('item_types')
        if item_types:
            lines.append(f"- **Item Types**: {', '.join(item_types)}")

        # Sample data
        sample = metadata.get('sample', {})
        if sample:
            lines.append("- **Sample**:")
            lines.append("```json")
            sample_str = json.dumps(sample, indent=2, ensure_ascii=False)
            # Truncate if too long
            if len(sample_str) > 1000:
                sample_str = sample_str[:1000] + "\n  ... (truncated)"
            lines.append(sample_str)
            lines.append("```")

    def _add_image_context(self, lines: List[str], metadata: Dict[str, Any]):
        """Add context for image files."""
        dimensions = metadata.get('dimensions', ())
        format_type = metadata.get('format', 'unknown')
        mode = metadata.get('mode', 'unknown')

        lines.append(f"- **Format**: {format_type}")
        if dimensions:
            lines.append(f"- **Dimensions**: {dimensions[0]}×{dimensions[1]} pixels")
        lines.append(f"- **Color Mode**: {mode}")

        size_human = metadata.get('file_size_human', '')
        if size_human:
            lines.append(f"- **Size**: {size_human}")

    def _add_pdf_context(self, lines: List[str], metadata: Dict[str, Any]):
        """Add context for PDF files."""
        page_count = metadata.get('page_count', 'unknown')
        lines.append(f"- **Pages**: {page_count}")

        text_preview = metadata.get('text_preview', '')
        if text_preview:
            preview = text_preview[:500]
            lines.append("- **Text Preview** (first 500 chars):")
            lines.append(f"```")
            lines.append(preview)
            lines.append("```")

    def _add_docx_context(self, lines: List[str], metadata: Dict[str, Any]):
        """Add context for DOCX files."""
        para_count = metadata.get('paragraph_count', 'unknown')
        lines.append(f"- **Paragraphs**: {para_count}")

        text_preview = metadata.get('text_preview', '')
        if text_preview:
            preview = text_preview[:500]
            lines.append("- **Text Preview** (first 500 chars):")
            lines.append(f"```")
            lines.append(preview)
            lines.append("```")

    def _add_text_context(self, lines: List[str], metadata: Dict[str, Any]):
        """Add context for text files."""
        line_count = metadata.get('line_count', 'unknown')
        encoding = metadata.get('encoding', 'unknown')

        lines.append(f"- **Lines**: {line_count}")
        lines.append(f"- **Encoding**: {encoding}")

        preview = metadata.get('preview', '')
        if preview:
            preview_text = preview[:500]
            lines.append("- **Preview** (first 500 chars):")
            lines.append(f"```")
            lines.append(preview_text)
            lines.append("```")

    def get_quick_summary(self, file_paths: List[str]) -> str:
        """
        Get a quick one-line summary of files (for inline display).

        Args:
            file_paths: List of file paths

        Returns:
            One-line summary string
        """
        if not file_paths:
            return "No files"

        summaries = []
        for fp in file_paths:
            path = Path(fp)
            if not path.exists():
                continue

            original_name = extract_original_filename(str(path))
            metadata = self.registry.extract_metadata(path, quick_mode=True, use_cache=True)
            file_type = metadata.get('file_type', 'unknown')

            if file_type in ['csv', 'excel']:
                cols = metadata.get('columns', [])
                summaries.append(f"{original_name} ({len(cols)} cols)")
            else:
                summaries.append(f"{original_name} ({file_type})")

        if len(summaries) <= 3:
            return ", ".join(summaries)
        else:
            return f"{', '.join(summaries[:3])} and {len(summaries) - 3} more"


# Global singleton instance
class FileAnalyzerTool(BaseTool):
    """
    BaseTool-compatible wrapper around FileAnalyzer for agent integration.
    Keeps backward-compatible analyze signature (including quick_mode).
    """

    def __init__(self):
        super().__init__()
        self.analyzer = FileAnalyzer()
        logger.info("[FileAnalyzerTool] Initialized")

    async def execute(
        self,
        query: str,
        context: Optional[str] = None,
        *,
        file_paths: Optional[List[str]] = None,
        user_query: Optional[str] = None,
        quick_mode: bool = False,
        **_: Any
    ) -> ToolResult:
        """Async entrypoint required by BaseTool."""
        if not self.validate_inputs(file_paths=file_paths):
            return self._handle_validation_error(
                "file_paths must be a non-empty list of paths",
                parameter="file_paths"
            )

        self._log_execution_start(
            file_paths=file_paths,
            quick_mode=quick_mode,
            user_query=user_query or query
        )

        try:
            result = self.analyze(
                file_paths=file_paths or [],
                user_query=user_query or query,
                quick_mode=quick_mode
            )
            tool_result = ToolResult.success_result(
                output=result.get("summary", ""),
                metadata=result,
                execution_time=self._elapsed_time()
            )
            self._log_execution_end(tool_result)
            return tool_result
        except Exception as exc:
            error_result = self._handle_error(exc, "execute")
            self._log_execution_end(error_result)
            return error_result

    def validate_inputs(self, **kwargs: Any) -> bool:
        """Validate file_paths input for both async and sync entrypoints."""
        file_paths = kwargs.get("file_paths")
        return isinstance(file_paths, list) and len(file_paths) > 0

    def analyze(
        self,
        file_paths: List[str],
        user_query: Optional[str] = None,
        quick_mode: bool = False
    ) -> Dict[str, Any]:
        """Synchronous helper used by agents and other tools."""
        return self.analyzer.analyze(
            file_paths=file_paths,
            user_query=user_query,
            quick_mode=quick_mode
        )

    def get_quick_summary(self, file_paths: List[str]) -> str:
        """Expose quick summary helper for compatibility."""
        return self.analyzer.get_quick_summary(file_paths)


file_analyzer = FileAnalyzerTool()


__all__ = ["FileAnalyzer", "FileAnalyzerTool", "file_analyzer"]
