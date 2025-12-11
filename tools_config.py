"""
Tool Configuration and Schemas
Define all available tools with their schemas for LLM understanding
"""
from typing import List, Dict, Any


# ============================================================================
# Tool Schemas
# ============================================================================

TOOL_SCHEMAS = {
    "websearch": {
        "name": "websearch",
        "description": "Search the web for current information and get answers to questions. Use this when you need up-to-date information, facts, news, or information not in your knowledge base.",
        "endpoint": "/api/tools/websearch",
        "method": "POST",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find information about"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of search results to return (default: 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        },
        "returns": {
            "success": "True or False Boolean indicating if search succeeded",
            "answer": "LLM-generated answer based on search results",
            "data": "Dictionary with search_query, results, and num_results",
            "metadata": "Execution metadata including timing"
        }
    },

    "python_coder": {
        "name": "python_coder",
        "description": "Execute Python code in a sandboxed environment. Can access files created in previous executions within the same session. Use for calculations, data analysis, visualizations, or any computational task.",
        "endpoint": "/api/tools/python_coder",
        "method": "POST",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute"
                },
                "session_id": {
                    "type": "string",
                    "description": "Session ID for workspace isolation (required)"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Execution timeout in seconds (optional)",
                    "default": 30
                }
            },
            "required": ["code", "session_id"]
        },
        "returns": {
            "success": "Boolean indicating if execution succeeded",
            "answer": "Human-readable execution result",
            "data": "Dictionary with stdout, stderr, files, workspace, returncode",
            "metadata": "Execution metadata including timing"
        }
    },

    "rag": {
        "name": "rag",
        "description": "Retrieve relevant information from document collections using semantic search. Query user-specific document collections. Documents must be uploaded first using the RAG upload API.",
        "endpoint": "/api/tools/rag/query",
        "method": "POST",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search for in the document database"
                },
                "collection_name": {
                    "type": "string",
                    "description": "Name of the document collection to search in (required)"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of documents to retrieve (default: 5)",
                    "default": 5
                }
            },
            "required": ["query", "collection_name"]
        },
        "returns": {
            "success": "Boolean indicating if retrieval succeeded",
            "answer": "LLM-generated answer based on retrieved documents",
            "data": "Dictionary with optimized_query, documents, and num_results",
            "metadata": "Execution metadata including timing and collection name"
        }
    }
}


# ============================================================================
# Helper Functions
# ============================================================================

def get_tool_schema(tool_name: str) -> Dict[str, Any]:
    """
    Get schema for a specific tool

    Args:
        tool_name: Name of the tool

    Returns:
        Tool schema dictionary
    """
    return TOOL_SCHEMAS.get(tool_name)


def get_all_tool_schemas() -> Dict[str, Dict[str, Any]]:
    """
    Get all tool schemas

    Returns:
        Dictionary of all tool schemas
    """
    return TOOL_SCHEMAS


def get_available_tools() -> List[str]:
    """
    Get list of available tool names

    Returns:
        List of tool names
    """
    return list(TOOL_SCHEMAS.keys())


def format_tools_for_llm() -> str:
    """
    Format tool schemas as a string for LLM context
    Now emphasizes string input format instead of JSON parameters

    Returns:
        Formatted string describing all available tools
    """
    tool_descriptions = []

    for tool_name, schema in TOOL_SCHEMAS.items():
        desc = f"Tool: {tool_name}\n"
        desc += f"Description: {schema['description']}\n"

        # Add string input format guidance instead of JSON schema
        if tool_name == "websearch":
            desc += "Action Input Format: Provide the search query as plain text\n"
            desc += "Example: Action Input: machine learning tutorials\n"
            desc += "Example: Action Input: latest AI news"

        elif tool_name == "python_coder":
            desc += "Action Input Format: Provide the actual Python code to execute\n"
            desc += "Example: Action Input: import math\\nprint(math.factorial(10))\n"
            desc += "Example: Action Input: import numpy as np\\nprint(np.array([1,2,3]).mean())"

        elif tool_name == "rag":
            desc += "Action Input Format: Provide the search query as plain text\n"
            desc += "Example: Action Input: project requirements\n"
            desc += "Example: Action Input: API documentation"

        tool_descriptions.append(desc)

    return "\n\n".join(tool_descriptions)


def format_tools_for_ollama_native() -> List[Dict[str, Any]]:
    """
    Format tool schemas for Ollama native tool calling

    Returns:
        List of tool schemas in Ollama format
    """
    tools = []

    for tool_name, schema in TOOL_SCHEMAS.items():
        tool = {
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema["parameters"]
            }
        }
        tools.append(tool)

    return tools
