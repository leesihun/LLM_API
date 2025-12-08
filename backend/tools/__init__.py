"""
Tools Package
=============
Collection of tools for the LLM API agent system.

Available Tools:
- python_coder: LLM-powered Python code generation and execution
- shell_tool: Safe shell navigation and inspection within sandbox
- web_search: Real-time web search via Tavily API
- rag_retriever: FAISS-based document retrieval
- file_analyzer: File metadata and structure analysis
- vision_analyzer: Image understanding with multimodal LLMs

Code Execution:
- code_sandbox: Secure Python execution environment

Version: 1.1.0
Created: 2025-01-20
Updated: 2025-12-05 - Added python_coder and code_sandbox
"""

from backend.tools.python_coder import python_coder_tool, PythonCoderTool
from backend.tools.shell_tool import shell_tool, ShellTool
from backend.tools.code_sandbox import (
    CodeSandbox,
    SandboxManager,
    sandbox_manager,
    ExecutionResult,
    ValidationResult,
    CodeValidator,
    BLOCKED_IMPORTS,
    ALLOWED_IMPORTS
)
from backend.tools.web_search import web_search_tool
from backend.tools.rag_retriever import rag_retriever_tool
from backend.tools.file_analyzer import file_analyzer, FileAnalyzerTool
from backend.tools.vision_analyzer import vision_analyzer_tool

__all__ = [
    # Python Coder
    "python_coder_tool",
    "PythonCoderTool",
    # Shell
    "shell_tool",
    "ShellTool",
    
    # Code Sandbox
    "CodeSandbox",
    "SandboxManager",
    "sandbox_manager",
    "ExecutionResult",
    "ValidationResult",
    "CodeValidator",
    "BLOCKED_IMPORTS",
    "ALLOWED_IMPORTS",
    
    # Web Search
    "web_search_tool",
    
    # RAG Retriever
    "rag_retriever_tool",
    
    # File Analyzer
    "file_analyzer",
    "FileAnalyzerTool",
    
    # Vision Analyzer
    "vision_analyzer_tool",
]

