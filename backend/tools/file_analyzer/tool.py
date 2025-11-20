"""
File Analyzer Tool - BaseTool Implementation
=============================================
Wraps FileAnalyzer with standardized BaseTool interface.

This provides consistent interface for ReAct agent while maintaining
backward compatibility.

Created: 2025-01-20
Version: 2.0.0 (with BaseTool)
"""

from typing import List, Dict, Any, Optional

from backend.core import BaseTool, ToolResult, FileAnalysisResult
from backend.utils.logging_utils import get_logger
from backend.tools.file_analyzer.analyzer import FileAnalyzer as FileAnalyzerCore
from backend.services.file_handler import file_handler_registry

logger = get_logger(__name__)


class FileAnalyzer(BaseTool):
    """
    File analysis tool with BaseTool interface.
    
    Features:
    - Comprehensive file analysis for multiple formats
    - Uses unified FileHandlerRegistry
    - Extracts metadata, statistics, and previews
    - Returns standardized ToolResult
    
    Supported formats: CSV, Excel, JSON, Text, PDF, DOCX, Images
    
    Usage:
        >>> tool = FileAnalyzer()
        >>> result = await tool.execute(
        ...     query="Analyze this data file",
        ...     file_paths=["data.csv"]
        ... )
        >>> print(result.output)
    """
    
    def __init__(self, use_llm_for_complex: bool = False):
        """Initialize File Analyzer Tool"""
        super().__init__()
        
        # Initialize core analyzer
        self.analyzer = FileAnalyzerCore(use_llm_for_complex=use_llm_for_complex)
        
        # Also have direct access to unified registry
        self.file_registry = file_handler_registry
        
        logger.info("[FileAnalyzer] Initialized with BaseTool interface and unified FileHandlerRegistry")
    
    async def execute(
        self,
        query: str,
        context: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        user_query: str = "",
        **kwargs
    ) -> ToolResult:
        """
        Execute file analysis.
        
        Args:
            query: User's question or analysis request
            context: Optional additional context
            file_paths: List of file paths to analyze
            user_query: User's original question (for context)
            **kwargs: Additional parameters
            
        Returns:
            ToolResult with analysis results
        """
        self._log_execution_start(
            query=query[:100],
            file_count=len(file_paths) if file_paths else 0
        )
        
        try:
            # Validate inputs
            if not self.validate_inputs(file_paths=file_paths):
                return self._handle_validation_error(
                    "file_paths must be a non-empty list",
                    parameter="file_paths"
                )
            
            # Use user_query if provided, otherwise use query
            analysis_query = user_query or query
            
            # Execute analysis via core analyzer
            result_dict = self.analyzer.analyze(
                file_paths=file_paths,
                user_query=analysis_query
            )
            
            # Convert to ToolResult
            tool_result = self._convert_to_tool_result(result_dict)
            
            self._log_execution_end(tool_result)
            return tool_result
            
        except Exception as e:
            return self._handle_error(e, "execute")
    
    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate tool inputs.
        
        Args:
            **kwargs: Must contain 'file_paths' key with non-empty list
            
        Returns:
            True if inputs are valid
        """
        file_paths = kwargs.get("file_paths")
        
        # File paths must be non-empty list
        if not isinstance(file_paths, list) or len(file_paths) == 0:
            return False
        
        # All paths must be strings
        if not all(isinstance(path, str) for path in file_paths):
            return False
        
        return True
    
    def _convert_to_tool_result(self, result_dict: Dict[str, Any]) -> ToolResult:
        """
        Convert analyzer result dictionary to ToolResult.
        
        Args:
            result_dict: Result from analyzer.analyze()
            
        Returns:
            Standardized ToolResult
        """
        success = result_dict.get("success", False)
        
        if not success:
            # Analysis failed
            return ToolResult(
                success=False,
                error=result_dict.get("error", "Unknown error"),
                error_type="FileAnalysisError"
            )
        
        # Build output from summary and results
        summary = result_dict.get("summary", "")
        results = result_dict.get("results", [])
        files_analyzed = result_dict.get("files_analyzed", 0)
        
        output = f"Analyzed {files_analyzed} file(s):\n\n"
        output += f"{summary}\n\n"
        
        # Add detailed results for each file
        output += "Detailed Analysis:\n\n"
        for i, file_result in enumerate(results, 1):
            file_name = file_result.get("file", "unknown")
            output += f"{i}. {file_name}\n"
            
            if file_result.get("success"):
                # Add key metrics
                if "rows" in file_result:
                    output += f"   - Rows: {file_result['rows']}\n"
                if "columns" in file_result:
                    cols = file_result['columns']
                    output += f"   - Columns: {len(cols) if isinstance(cols, list) else cols}\n"
                if "size_human" in file_result:
                    output += f"   - Size: {file_result['size_human']}\n"
            else:
                output += f"   - Error: {file_result.get('error', 'Unknown error')}\n"
            
            output += "\n"
        
        # Build metadata
        metadata = {
            "files_analyzed": files_analyzed,
            "results": results,
            "summary": summary,
            "all_files_success": all(r.get("success", False) for r in results)
        }
        
        return ToolResult(
            success=True,
            output=output,
            metadata=metadata
        )


# Global singleton instance for backward compatibility
file_analyzer = FileAnalyzer()


# Backward compatibility function
def analyze_files(file_paths: List[str], user_query: str = "") -> Dict[str, Any]:
    """
    Legacy function for backward compatibility.
    
    Args:
        file_paths: List of file paths to analyze
        user_query: User's question
        
    Returns:
        Analysis result dictionary
    """
    return file_analyzer.analyzer.analyze(file_paths, user_query)


# Export for backward compatibility
__all__ = [
    'FileAnalyzer',
    'file_analyzer',
    'analyze_files',
]
