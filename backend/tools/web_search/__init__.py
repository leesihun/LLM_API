"""
Web Search Tool Module
======================
Modular web search with LLM-enhanced query refinement and answer generation.

This module provides a complete web search pipeline:
- Query refinement using LLM
- Tavily API search with fallback
- Result processing and ranking
- Answer generation from multiple sources

Main Components:
- WebSearchTool: Main orchestrator class
- QueryRefiner: LLM-based query optimization
- ResultProcessor: Result filtering and formatting
- AnswerGenerator: Multi-source answer synthesis

Version: 1.0.0
Created: 2025-01-13
"""

from .searcher import WebSearchTool, web_search_tool
from .query_refiner import QueryRefiner
from .result_processor import ResultProcessor
from .answer_generator import AnswerGenerator

__all__ = [
    # Main tool
    'WebSearchTool',
    'web_search_tool',  # Global instance for backward compatibility

    # Component modules (for advanced usage)
    'QueryRefiner',
    'ResultProcessor',
    'AnswerGenerator',
]

__version__ = '1.0.0'
