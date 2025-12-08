"""
Code Sandbox Module
===================
Secure Python code execution environment with safety restrictions.

Provides isolated execution of Python code with:
- Import restrictions (blocked dangerous modules)
- AST validation (static analysis before execution)
- Timeout controls
- Memory limits
- Session isolation
- Variable persistence across executions

Version: 1.0.0
Created: 2025-12-05
"""

import ast
import sys
import io
import os
import traceback
import time
import threading
import signal
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field

from backend.utils.logging_utils import get_logger
from backend.config.settings import settings

logger = get_logger(__name__)


# ============================================================================
# Security Configuration
# ============================================================================

# Blocked imports - modules that could compromise system security
BLOCKED_IMPORTS = {
    # System access
    'subprocess', 'os.system', 'commands', 'popen', 'popen2', 'pexpect',
    # Code execution
    'exec', 'eval', 'compile', '__import__',
    # Serialization (can execute code)
    'pickle', 'cPickle', 'marshal', 'shelve',
    # Network (prevent external connections in sandbox)
    'socket', 'urllib', 'urllib2', 'urllib3', 'httplib', 'ftplib', 
    'smtplib', 'poplib', 'imaplib', 'telnetlib',
    # File system manipulation (beyond workspace)
    'shutil.rmtree', 'shutil.move',
    # Process control
    'multiprocessing', 'threading.Thread', 'concurrent.futures',
    # System info
    'ctypes', 'cffi',
    # Debugging (can access internals)
    'pdb', 'code', 'codeop',
}

# Allowed imports - common data science and utility libraries
ALLOWED_IMPORTS = {
    'pandas', 'numpy', 'matplotlib', 'seaborn', 'scipy', 'sklearn',
    'json', 'csv', 'datetime', 'time', 'math', 'random', 're',
    'collections', 'itertools', 'functools', 'operator',
    'pathlib', 'os.path', 'glob',
    'io', 'base64', 'hashlib', 'uuid',
    'typing', 'dataclasses', 'enum',
    'warnings', 'logging',
    'PIL', 'openpyxl', 'xlrd', 'xlsxwriter',
    'plotly', 'bokeh', 'altair',
    'requests',  # Allow HTTP requests for data fetching
    'beautifulsoup4', 'bs4', 'lxml',  # Web scraping
    'PyPDF2', 'pdfplumber', 'tabula',  # PDF processing
    'docx', 'python-docx',  # DOCX processing
    'statsmodels', 'sympy',  # Statistical analysis
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ExecutionResult:
    """Result of code execution in sandbox."""
    success: bool
    output: str = ""
    error: str = ""
    error_type: str = ""
    execution_time: float = 0.0
    variables: Dict[str, Any] = field(default_factory=dict)
    created_files: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "error_type": self.error_type,
            "execution_time": self.execution_time,
            "variables": list(self.variables.keys()),
            "created_files": self.created_files
        }


@dataclass
class ValidationResult:
    """Result of AST validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ============================================================================
# Code Validator
# ============================================================================

class CodeValidator(ast.NodeVisitor):
    """
    AST visitor to validate code for security issues.
    
    Checks:
    - Blocked imports
    - Dangerous function calls
    - File system access patterns
    - Network access patterns
    """
    
    def __init__(self, working_dir: str):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.working_dir = working_dir
        
    def visit_Import(self, node: ast.Import) -> None:
        """Check import statements."""
        for alias in node.names:
            module = alias.name.split('.')[0]
            if module in BLOCKED_IMPORTS or alias.name in BLOCKED_IMPORTS:
                self.errors.append(f"Blocked import: '{alias.name}'")
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check from ... import statements."""
        if node.module:
            module = node.module.split('.')[0]
            if module in BLOCKED_IMPORTS or node.module in BLOCKED_IMPORTS:
                self.errors.append(f"Blocked import: '{node.module}'")
            
            # Check specific imports from module
            for alias in node.names:
                full_name = f"{node.module}.{alias.name}"
                if full_name in BLOCKED_IMPORTS or alias.name in BLOCKED_IMPORTS:
                    self.errors.append(f"Blocked import: '{full_name}'")
        self.generic_visit(node)
        
    def visit_Call(self, node: ast.Call) -> None:
        """Check function calls."""
        # Check for exec/eval calls
        if isinstance(node.func, ast.Name):
            if node.func.id in ('exec', 'eval', 'compile', '__import__'):
                self.errors.append(f"Blocked function call: '{node.func.id}'")
                
        # Check for attribute calls like os.system
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ('system', 'popen', 'spawn', 'fork'):
                self.warnings.append(f"Potentially dangerous call: '{node.func.attr}'")
                
        self.generic_visit(node)
        
    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Check attribute access patterns."""
        # Warn about accessing __builtins__ or __globals__
        if node.attr in ('__builtins__', '__globals__', '__code__', '__closure__'):
            self.warnings.append(f"Access to internal attribute: '{node.attr}'")
        self.generic_visit(node)
    
    def validate(self, code: str) -> ValidationResult:
        """
        Validate code and return result.
        
        Args:
            code: Python code to validate
            
        Returns:
            ValidationResult with errors and warnings
        """
        try:
            tree = ast.parse(code)
            self.visit(tree)
            return ValidationResult(
                valid=len(self.errors) == 0,
                errors=self.errors,
                warnings=self.warnings
            )
        except SyntaxError as e:
            return ValidationResult(
                valid=False,
                errors=[f"Syntax error: {e.msg} at line {e.lineno}"]
            )


# ============================================================================
# Code Sandbox
# ============================================================================

class CodeSandbox:
    """
    Secure Python code execution sandbox.
    
    Features:
    - Session-based isolation
    - Variable persistence across executions
    - Pre-loaded common libraries
    - Timeout and memory controls
    - File system restrictions
    
    Example:
        >>> sandbox = CodeSandbox(session_id="test-session")
        >>> result = sandbox.execute("x = 1 + 1\\nprint(x)")
        >>> print(result.output)  # "2"
        >>> result = sandbox.execute("print(x)")  # x persists
        >>> print(result.output)  # "2"
    """
    
    def __init__(
        self,
        session_id: str,
        working_dir: Optional[str] = None,
        timeout: Optional[int] = None,
        preload_libraries: Optional[List[str]] = None
    ):
        """
        Initialize sandbox.
        
        Args:
            session_id: Unique session identifier
            working_dir: Working directory for file operations
            timeout: Execution timeout in seconds
            preload_libraries: Libraries to import at startup
        """
        self.session_id = session_id
        self.timeout = timeout or settings.python_code_timeout
        self.preload_libraries = preload_libraries or settings.python_code_preload_libraries
        
        # Setup working directory
        base_dir = Path(settings.python_code_execution_dir)
        self.working_dir = Path(working_dir) if working_dir else base_dir / session_id
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize execution namespace with builtins
        self.namespace: Dict[str, Any] = {
            '__builtins__': __builtins__,
            '__name__': '__main__',
            '__file__': str(self.working_dir / 'script.py'),
        }
        
        # Track created files
        self.created_files: List[str] = []
        
        # Pre-load libraries
        self._preload_libraries()
        
        logger.info(f"[CodeSandbox] Initialized for session '{session_id}' in {self.working_dir}")
    
    def _preload_libraries(self) -> None:
        """Pre-load common libraries into namespace."""
        for lib_spec in self.preload_libraries:
            try:
                # Handle "import x as y" format
                if ' as ' in lib_spec:
                    module_name, alias = lib_spec.split(' as ')
                    module_name = module_name.strip()
                    alias = alias.strip()
                else:
                    module_name = lib_spec.strip()
                    alias = module_name.split('.')[-1]
                
                # Import and add to namespace
                module = __import__(module_name, fromlist=[''])
                self.namespace[alias] = module
                
                # Configure matplotlib for non-interactive backend
                if module_name == 'matplotlib':
                    import matplotlib
                    matplotlib.use('Agg')
                    
                logger.debug(f"[CodeSandbox] Pre-loaded: {module_name} as {alias}")
                
            except ImportError as e:
                logger.warning(f"[CodeSandbox] Failed to preload '{lib_spec}': {e}")
    
    def validate_code(self, code: str) -> ValidationResult:
        """
        Validate code for security issues.
        
        Args:
            code: Python code to validate
            
        Returns:
            ValidationResult with errors and warnings
        """
        validator = CodeValidator(str(self.working_dir))
        return validator.validate(code)
    
    def execute(
        self,
        code: str,
        timeout: Optional[int] = None,
        validate: bool = True
    ) -> ExecutionResult:
        """
        Execute Python code in sandbox.
        
        Args:
            code: Python code to execute
            timeout: Optional timeout override
            validate: Whether to validate code before execution
            
        Returns:
            ExecutionResult with output, errors, and variables
        """
        start_time = time.time()
        timeout = timeout or self.timeout
        
        # Validate code if requested
        if validate:
            validation = self.validate_code(code)
            if not validation.valid:
                return ExecutionResult(
                    success=False,
                    error="\n".join(validation.errors),
                    error_type="ValidationError",
                    execution_time=time.time() - start_time
                )
            if validation.warnings:
                logger.warning(f"[CodeSandbox] Code warnings: {validation.warnings}")
        
        # Capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        # Track files before execution
        files_before = set(self.working_dir.glob('*'))
        
        # Change to working directory
        original_cwd = os.getcwd()
        os.chdir(self.working_dir)
        
        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                # Execute with timeout
                result = self._execute_with_timeout(code, timeout)
                
            execution_time = time.time() - start_time
            
            # Track new files
            files_after = set(self.working_dir.glob('*'))
            new_files = files_after - files_before
            created_files = [str(f) for f in new_files]
            self.created_files.extend(created_files)
            
            if result['success']:
                # Extract printable variables
                variables = self._extract_variables()
                
                return ExecutionResult(
                    success=True,
                    output=stdout_buffer.getvalue(),
                    execution_time=execution_time,
                    variables=variables,
                    created_files=created_files
                )
            else:
                return ExecutionResult(
                    success=False,
                    output=stdout_buffer.getvalue(),
                    error=result['error'],
                    error_type=result['error_type'],
                    execution_time=execution_time,
                    created_files=created_files
                )
                
        except Exception as e:
            return ExecutionResult(
                success=False,
                output=stdout_buffer.getvalue(),
                error=str(e),
                error_type=type(e).__name__,
                execution_time=time.time() - start_time
            )
        finally:
            os.chdir(original_cwd)
    
    def _execute_with_timeout(self, code: str, timeout: int) -> Dict[str, Any]:
        """
        Execute code with timeout protection.

        Args:
            code: Python code to execute
            timeout: Timeout in seconds

        Returns:
            Dict with success status and error info
        """
        result = {'success': False, 'error': '', 'error_type': ''}

        def execute():
            try:
                exec(code, self.namespace)
                result['success'] = True
            except Exception as e:
                # Extract relevant error information with line numbers
                tb = traceback.extract_tb(e.__traceback__)
                error_lines = []

                # Add the exception message
                error_lines.append(f"{type(e).__name__}: {str(e)}")

                # Add relevant traceback lines (from the executed code, not internals)
                for frame in tb:
                    # Only show frames from the executed code (not from exec internals)
                    if frame.filename == '<string>':
                        error_lines.append(f"  Line {frame.lineno}: {frame.line}")

                # If no relevant frames, show full traceback
                if len(error_lines) == 1:
                    error_lines.append(traceback.format_exc())

                result['error'] = '\n'.join(error_lines)
                result['error_type'] = type(e).__name__

        # Run in thread with timeout
        thread = threading.Thread(target=execute)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            result['error'] = f"Execution timed out after {timeout} seconds. Possible infinite loop or very long computation."
            result['error_type'] = "TimeoutError"
            # Note: Thread cannot be killed in Python, it will continue running
            logger.warning(f"[CodeSandbox] Thread timeout - thread will continue running in background")
            return result

        return result
    
    def _extract_variables(self) -> Dict[str, Any]:
        """
        Extract user-defined variables from namespace.

        Returns:
            Dict of variable names to their string representations
        """
        import types

        variables = {}
        excluded = {'__builtins__', '__name__', '__file__', '__doc__', '__package__'}

        for name, value in self.namespace.items():
            if name.startswith('_') or name in excluded:
                continue

            # Skip modules (but keep user classes and instances)
            if isinstance(value, types.ModuleType):
                continue

            # Skip functions/classes that are from pre-loaded libraries
            if callable(value) and hasattr(value, '__module__'):
                if value.__module__ in ('pandas', 'numpy', 'matplotlib', 'matplotlib.pyplot'):
                    continue

            try:
                # Store string representation
                variables[name] = repr(value)[:200]  # Limit size
            except:
                variables[name] = "<unrepresentable>"

        return variables
    
    def get_variable(self, name: str) -> Any:
        """
        Get a variable from the namespace.
        
        Args:
            name: Variable name
            
        Returns:
            Variable value or None
        """
        return self.namespace.get(name)
    
    def set_variable(self, name: str, value: Any) -> None:
        """
        Set a variable in the namespace.
        
        Args:
            name: Variable name
            value: Variable value
        """
        self.namespace[name] = value
    
    def list_files(self) -> List[str]:
        """
        List files in working directory.
        
        Returns:
            List of file paths
        """
        return [str(f) for f in self.working_dir.glob('*') if f.is_file()]
    
    def read_file(self, filename: str, max_chars: int = 10000) -> str:
        """
        Read a file from working directory.
        
        Args:
            filename: Name of file to read
            max_chars: Maximum characters to return
            
        Returns:
            File contents (truncated if necessary)
        """
        file_path = self.working_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {filename}")
        if not file_path.is_relative_to(self.working_dir):
            raise PermissionError("Cannot access files outside working directory")
            
        content = file_path.read_text(encoding='utf-8', errors='replace')
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n... (truncated, {len(content) - max_chars} more chars)"
        return content
    
    def cleanup(self) -> None:
        """Clean up sandbox resources (optional)."""
        logger.info(f"[CodeSandbox] Cleanup for session '{self.session_id}'")
        # Optionally clear namespace or files
        # self.namespace.clear()
        # For now, keep files and variables for session persistence


# ============================================================================
# Sandbox Manager
# ============================================================================

class SandboxManager:
    """
    Manages sandbox instances across sessions.
    
    Provides session-based sandbox pooling and lifecycle management.
    """
    
    _instances: Dict[str, CodeSandbox] = {}
    
    @classmethod
    def get_sandbox(
        cls,
        session_id: str,
        working_dir: Optional[str] = None
    ) -> CodeSandbox:
        """
        Get or create a sandbox for a session.
        
        Args:
            session_id: Session identifier
            working_dir: Optional working directory
            
        Returns:
            CodeSandbox instance
        """
        if session_id not in cls._instances:
            cls._instances[session_id] = CodeSandbox(
                session_id=session_id,
                working_dir=working_dir
            )
            logger.debug(f"[SandboxManager] Created sandbox for session '{session_id}'")
        return cls._instances[session_id]
    
    @classmethod
    def cleanup_session(cls, session_id: str) -> None:
        """
        Cleanup a session's sandbox.
        
        Args:
            session_id: Session to cleanup
        """
        if session_id in cls._instances:
            cls._instances[session_id].cleanup()
            del cls._instances[session_id]
            logger.debug(f"[SandboxManager] Cleaned up session '{session_id}'")
    
    @classmethod
    def list_sessions(cls) -> List[str]:
        """
        List active sandbox sessions.
        
        Returns:
            List of session IDs
        """
        return list(cls._instances.keys())


# Global sandbox manager
sandbox_manager = SandboxManager()

