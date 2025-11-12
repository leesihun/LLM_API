"""
Logging Utilities
==================
Centralized logging helpers for consistent, visually clear, and information-rich logs.

Key Features:
- Visual separators for different log levels
- Structured section logging
- Multi-line content formatting
- Deduplicated information display
- Color-coded severity indicators
"""

import logging
from typing import Any, Dict, List, Optional
from functools import wraps
import time


class LogFormatter:
    """Visual formatting helpers for logs"""

    # Visual indicators
    HEAVY_SEP = "=" * 100
    LIGHT_SEP = "-" * 100
    SECTION_SEP = "#" * 100
    DOT_SEP = "·" * 100

    # Status indicators (using ASCII art for compatibility)
    SUCCESS = "✓"
    FAILURE = "✗"
    WARNING = "⚠"
    INFO = "ℹ"
    ARROW = "→"
    BULLET = "•"

    @staticmethod
    def header(text: str, level: str = "heavy") -> List[str]:
        """Create a visually distinct header

        Args:
            text: Header text
            level: "heavy", "light", "section", or "dot"

        Returns:
            List of log lines
        """
        sep_map = {
            "heavy": LogFormatter.HEAVY_SEP,
            "light": LogFormatter.LIGHT_SEP,
            "section": LogFormatter.SECTION_SEP,
            "dot": LogFormatter.DOT_SEP
        }
        sep = sep_map.get(level, LogFormatter.LIGHT_SEP)

        return [
            "",
            sep,
            f"  {text.upper()}",
            sep,
            ""
        ]

    @staticmethod
    def section(text: str) -> List[str]:
        """Create a section header"""
        return [
            "",
            f"{LogFormatter.SECTION_SEP}",
            f"  {text}",
            f"{LogFormatter.SECTION_SEP}",
            ""
        ]

    @staticmethod
    def subsection(text: str) -> List[str]:
        """Create a subsection header"""
        return [
            "",
            f"{LogFormatter.ARROW} {text}",
            f"{LogFormatter.LIGHT_SEP}",
            ""
        ]

    @staticmethod
    def status(success: bool, text: str, details: Optional[str] = None) -> List[str]:
        """Create a status line with optional details

        Args:
            success: True for success, False for failure
            text: Status message
            details: Optional additional details

        Returns:
            List of log lines
        """
        icon = LogFormatter.SUCCESS if success else LogFormatter.FAILURE
        lines = [f"{icon} {text}"]
        if details:
            lines.append(f"  Details: {details}")
        return lines

    @staticmethod
    def key_value(data: Dict[str, Any], indent: int = 0) -> List[str]:
        """Format key-value pairs

        Args:
            data: Dictionary of key-value pairs
            indent: Indentation level (spaces)

        Returns:
            List of formatted lines
        """
        lines = []
        indent_str = " " * indent

        for key, value in data.items():
            # Handle different value types
            if isinstance(value, (list, tuple)):
                if len(value) == 0:
                    lines.append(f"{indent_str}{key}: (empty)")
                elif len(value) <= 3:
                    lines.append(f"{indent_str}{key}: {', '.join(str(v) for v in value)}")
                else:
                    lines.append(f"{indent_str}{key}: {', '.join(str(v) for v in value[:3])}... (+{len(value)-3} more)")
            elif isinstance(value, dict):
                lines.append(f"{indent_str}{key}:")
                lines.extend(LogFormatter.key_value(value, indent + 2))
            elif isinstance(value, str) and len(value) > 100:
                lines.append(f"{indent_str}{key}: {value[:100]}...")
            else:
                lines.append(f"{indent_str}{key}: {value}")

        return lines

    @staticmethod
    def multiline(content: str, prefix: str = "", max_lines: Optional[int] = None) -> List[str]:
        """Format multiline content with optional prefix

        Args:
            content: Content to format
            prefix: Prefix for each line
            max_lines: Maximum number of lines to show

        Returns:
            List of formatted lines
        """
        lines = content.splitlines()

        if max_lines and len(lines) > max_lines:
            formatted = [f"{prefix}{line}" for line in lines[:max_lines]]
            formatted.append(f"{prefix}... ({len(lines) - max_lines} more lines)")
        else:
            formatted = [f"{prefix}{line}" for line in lines]

        return formatted

    @staticmethod
    def list_items(items: List[str], bullet: str = "•", indent: int = 2) -> List[str]:
        """Format a list of items

        Args:
            items: List of items to format
            bullet: Bullet character
            indent: Indentation level

        Returns:
            List of formatted lines
        """
        indent_str = " " * indent
        return [f"{indent_str}{bullet} {item}" for item in items]

    @staticmethod
    def table(headers: List[str], rows: List[List[str]], max_col_width: int = 30) -> List[str]:
        """Format data as a simple text table

        Args:
            headers: Column headers
            rows: Data rows
            max_col_width: Maximum column width

        Returns:
            List of formatted lines
        """
        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)[:max_col_width]))

        # Limit to max_col_width
        col_widths = [min(w, max_col_width) for w in col_widths]

        # Format header
        header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        separator = "-+-".join("-" * w for w in col_widths)

        lines = [
            header_line,
            separator
        ]

        # Format rows
        for row in rows:
            formatted_cells = []
            for i, cell in enumerate(row):
                cell_str = str(cell)[:max_col_width]
                formatted_cells.append(cell_str.ljust(col_widths[i]))
            lines.append(" | ".join(formatted_cells))

        return lines


class StructuredLogger:
    """Wrapper around standard logger with structured logging helpers"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._execution_stack = []

    def _log_lines(self, level: int, lines: List[str]):
        """Log multiple lines at specified level"""
        for line in lines:
            self.logger.log(level, line)

    def header(self, text: str, level: str = "heavy"):
        """Log a header"""
        self._log_lines(logging.INFO, LogFormatter.header(text, level))

    def section(self, text: str):
        """Log a section header"""
        self._log_lines(logging.INFO, LogFormatter.section(text))

    def subsection(self, text: str):
        """Log a subsection"""
        self._log_lines(logging.INFO, LogFormatter.subsection(text))

    def success(self, text: str, details: Optional[str] = None):
        """Log a success status"""
        self._log_lines(logging.INFO, LogFormatter.status(True, text, details))

    def failure(self, text: str, details: Optional[str] = None):
        """Log a failure status"""
        self._log_lines(logging.ERROR, LogFormatter.status(False, text, details))

    def key_values(self, data: Dict[str, Any], title: Optional[str] = None, indent: int = 2):
        """Log key-value pairs with optional title"""
        lines = []
        if title:
            lines.append(f"{title}:")
        lines.extend(LogFormatter.key_value(data, indent))
        self._log_lines(logging.INFO, lines)

    def multiline(self, content: str, title: Optional[str] = None, max_lines: Optional[int] = None):
        """Log multiline content with optional title"""
        lines = []
        if title:
            lines.append(f"{title}:")
        lines.extend(LogFormatter.multiline(content, prefix="  ", max_lines=max_lines))
        self._log_lines(logging.INFO, lines)

    def list(self, items: List[str], title: Optional[str] = None):
        """Log a list of items with optional title"""
        lines = []
        if title:
            lines.append(f"{title}:")
        lines.extend(LogFormatter.list_items(items))
        self._log_lines(logging.INFO, lines)

    def table(self, headers: List[str], rows: List[List[str]], title: Optional[str] = None):
        """Log a table with optional title"""
        lines = []
        if title:
            lines.append(f"{title}:")
        lines.extend(LogFormatter.table(headers, rows))
        self._log_lines(logging.INFO, lines)

    def execution_start(self, name: str, params: Optional[Dict[str, Any]] = None):
        """Log start of an execution block"""
        self._execution_stack.append({"name": name, "start_time": time.time()})

        lines = LogFormatter.header(f"{name} - START", "heavy")
        if params:
            lines.extend(LogFormatter.key_value(params, indent=2))
        self._log_lines(logging.INFO, lines)

    def execution_end(self, result: Optional[Dict[str, Any]] = None):
        """Log end of an execution block"""
        if not self._execution_stack:
            self.logger.warning("execution_end called without matching execution_start")
            return

        exec_info = self._execution_stack.pop()
        duration = time.time() - exec_info["start_time"]

        lines = []
        lines.extend(LogFormatter.header(f"{exec_info['name']} - COMPLETED", "heavy"))

        summary = {"Execution Time": f"{duration:.2f}s"}
        if result:
            summary.update(result)

        lines.extend(LogFormatter.key_value(summary, indent=2))
        self._log_lines(logging.INFO, lines)

    def progress(self, current: int, total: int, item_name: str = "item"):
        """Log progress"""
        percentage = (current / total * 100) if total > 0 else 0
        self.logger.info(f"Progress: {current}/{total} {item_name}(s) ({percentage:.1f}%)")

    # Delegate standard logging methods
    def debug(self, msg: str):
        self.logger.debug(msg)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str, exc_info: bool = False):
        self.logger.error(msg, exc_info=exc_info)

    def critical(self, msg: str, exc_info: bool = False):
        self.logger.critical(msg, exc_info=exc_info)


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance

    Args:
        name: Logger name (usually __name__)

    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(logging.getLogger(name))


def log_execution(func_name: Optional[str] = None):
    """Decorator to automatically log function execution

    Args:
        func_name: Optional custom function name for logs

    Usage:
        @log_execution()
        async def my_function(arg1, arg2):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            name = func_name or func.__name__

            # Extract meaningful parameters (exclude self, large objects)
            params = {}
            if args and hasattr(args[0], '__class__'):
                # Skip 'self' for instance methods
                param_args = args[1:]
            else:
                param_args = args

            # Log up to 3 parameters
            for i, arg in enumerate(param_args[:3]):
                if isinstance(arg, (str, int, float, bool)):
                    params[f"arg{i+1}"] = arg
                elif isinstance(arg, (list, tuple)) and len(arg) <= 5:
                    params[f"arg{i+1}"] = arg

            # Log some kwargs
            for k, v in list(kwargs.items())[:3]:
                if isinstance(v, (str, int, float, bool)):
                    params[k] = v

            logger.execution_start(name, params)

            try:
                result = await func(*args, **kwargs)
                logger.execution_end({"Status": "Success"})
                return result
            except Exception as e:
                logger.execution_end({"Status": "Failed", "Error": str(e)[:100]})
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            name = func_name or func.__name__

            logger.execution_start(name)

            try:
                result = func(*args, **kwargs)
                logger.execution_end({"Status": "Success"})
                return result
            except Exception as e:
                logger.execution_end({"Status": "Failed", "Error": str(e)[:100]})
                raise

        # Return appropriate wrapper based on function type
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
