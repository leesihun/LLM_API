"""
Tool Metadata Models
====================
Standardized Pydantic models for tool execution responses and metadata.
Provides consistent data structures across all backend tools.

Version: 1.0.0
Created: 2025-01-13
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


# ============================================================================
# Base Metadata Model
# ============================================================================

class ToolMetadata(BaseModel):
    """
    Base metadata model for all tool executions.

    Common fields across all tools.
    """

    success: bool = Field(description="Whether the tool execution succeeded")
    execution_time: float = Field(default=0.0, description="Execution time in seconds")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    timestamp: datetime = Field(default_factory=datetime.now, description="Execution timestamp")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolMetadata":
        """Create from dictionary representation."""
        return cls(**data)


# ============================================================================
# Python Coder Metadata
# ============================================================================

class VerificationHistory(BaseModel):
    """Record of a single verification iteration."""

    iteration: int
    issues: List[str] = Field(default_factory=list)
    action: str  # "approved", "needs_modification", "modified"


class ExecutionAttempt(BaseModel):
    """Record of a single execution attempt."""

    attempt: int
    success: bool
    error: Optional[str] = None
    execution_time: float


class PythonCoderMetadata(ToolMetadata):
    """
    Metadata for Python code generation and execution.

    Includes verification history, execution attempts, and code details.
    """

    code: str = Field(description="Generated Python code")
    output: str = Field(default="", description="Execution output")
    verification_iterations: int = Field(default=0, description="Number of verification iterations")
    execution_attempts: int = Field(default=0, description="Number of execution attempts")
    modifications: List[str] = Field(default_factory=list, description="List of code modifications")
    input_files: List[str] = Field(default_factory=list, description="List of input file paths")
    file_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata about input files")
    verification_history: List[VerificationHistory] = Field(default_factory=list)
    execution_attempts_history: List[ExecutionAttempt] = Field(default_factory=list)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the execution."""
        return {
            "status": "SUCCESS" if self.success else "FAILED",
            "verification_iterations": self.verification_iterations,
            "execution_attempts": self.execution_attempts,
            "code_modifications": len(self.modifications),
            "execution_time": round(self.execution_time, 2),
            "output_length": len(self.output)
        }


# ============================================================================
# Web Search Metadata
# ============================================================================

class SearchResult(BaseModel):
    """Single web search result."""

    title: str
    url: str
    content: str
    score: Optional[float] = None


class WebSearchMetadata(ToolMetadata):
    """
    Metadata for web search operations.

    Includes search results, context, and answer synthesis details.
    """

    query: str = Field(description="Original search query")
    refined_query: Optional[str] = Field(default=None, description="LLM-refined query")
    results: List[SearchResult] = Field(default_factory=list, description="Search results")
    answer: str = Field(default="", description="LLM-generated answer from results")
    sources_used: List[str] = Field(default_factory=list, description="URLs used in answer")
    context_used: Optional[Dict[str, Any]] = Field(default=None, description="Temporal/contextual data")
    num_results: int = Field(default=0, description="Number of results returned")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the search."""
        return {
            "status": "SUCCESS" if self.success else "FAILED",
            "query": self.query,
            "refined_query": self.refined_query,
            "num_results": self.num_results,
            "sources_used": len(self.sources_used),
            "answer_length": len(self.answer)
        }


# ============================================================================
# RAG Metadata
# ============================================================================

class RAGDocument(BaseModel):
    """Single RAG retrieval result."""

    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float


class RAGMetadata(ToolMetadata):
    """
    Metadata for RAG (Retrieval-Augmented Generation) operations.

    Includes document retrieval results and context information.
    """

    query: str = Field(description="Search query")
    documents: List[RAGDocument] = Field(default_factory=list, description="Retrieved documents")
    num_documents: int = Field(default=0, description="Number of documents retrieved")
    document_ids: Optional[List[str]] = Field(default=None, description="Document IDs used")
    top_k: int = Field(default=5, description="Number of top results requested")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the retrieval."""
        return {
            "status": "SUCCESS" if self.success else "FAILED",
            "query": self.query,
            "num_documents": self.num_documents,
            "top_k": self.top_k
        }


# ============================================================================
# ReAct Agent Metadata
# ============================================================================

class ReActStep(BaseModel):
    """Single step in ReAct agent execution."""

    step_num: int
    thought: str
    action: str
    tool: str
    observation: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ReActMetadata(ToolMetadata):
    """
    Metadata for ReAct agent execution.

    Includes step-by-step thought-action-observation history.
    """

    query: str = Field(description="User query")
    steps: List[ReActStep] = Field(default_factory=list, description="ReAct execution steps")
    final_answer: str = Field(default="", description="Final synthesized answer")
    tools_used: List[str] = Field(default_factory=list, description="List of tools used")
    total_steps: int = Field(default=0, description="Total number of steps")
    max_iterations: int = Field(default=10, description="Maximum allowed iterations")
    early_exit: bool = Field(default=False, description="Whether early exit was triggered")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the execution."""
        return {
            "status": "SUCCESS" if self.success else "FAILED",
            "query": self.query[:100],
            "total_steps": self.total_steps,
            "tools_used": list(set(self.tools_used)),
            "early_exit": self.early_exit,
            "answer_length": len(self.final_answer)
        }


# ============================================================================
# Plan-Execute Metadata
# ============================================================================

class PlanStep(BaseModel):
    """Single step in execution plan."""

    step_num: int
    goal: str
    primary_tools: List[str] = Field(default_factory=list)
    fallback_tools: List[str] = Field(default_factory=list)
    success_criteria: str
    context: Optional[str] = None


class StepResult(BaseModel):
    """Result of executing a plan step."""

    step_num: int
    goal: str
    success: bool
    tool_used: Optional[str] = None
    attempts: int = 0
    observation: str
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PlanExecuteMetadata(ToolMetadata):
    """
    Metadata for Plan-Execute agent workflow.

    Includes planning phase and execution phase results.
    """

    query: str = Field(description="User query")
    plan_steps: List[PlanStep] = Field(default_factory=list, description="Planned execution steps")
    step_results: List[StepResult] = Field(default_factory=list, description="Step execution results")
    final_answer: str = Field(default="", description="Final synthesized answer")
    total_steps: int = Field(default=0, description="Total planned steps")
    successful_steps: int = Field(default=0, description="Number of successful steps")
    architecture: str = Field(default="Plan-and-Execute with ReAct")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the execution."""
        success_rate = (self.successful_steps / self.total_steps * 100) if self.total_steps > 0 else 0
        return {
            "status": "SUCCESS" if self.success else "FAILED",
            "query": self.query[:100],
            "total_steps": self.total_steps,
            "successful_steps": self.successful_steps,
            "success_rate": round(success_rate, 1),
            "architecture": self.architecture
        }


# ============================================================================
# File Analysis Metadata
# ============================================================================

class FileMetadata(BaseModel):
    """Metadata about a single file."""

    filename: str
    extension: str
    size_mb: float
    file_type: str  # "csv", "excel", "json", "text", "pdf", "docx", "image"
    columns: Optional[List[str]] = None
    dtypes: Optional[Dict[str, str]] = None
    structure: Optional[str] = None  # For JSON: "list", "dict", "nested"
    preview: Optional[str] = None


class FileAnalysisMetadata(ToolMetadata):
    """
    Metadata for file analysis operations.

    Includes structural information and content previews.
    """

    files_analyzed: int = Field(default=0, description="Number of files analyzed")
    file_details: List[FileMetadata] = Field(default_factory=list, description="Details for each file")
    summary: str = Field(default="", description="Analysis summary")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the analysis."""
        return {
            "status": "SUCCESS" if self.success else "FAILED",
            "files_analyzed": self.files_analyzed,
            "file_types": [f.file_type for f in self.file_details]
        }
