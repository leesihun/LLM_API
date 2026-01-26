"""
Tool Configuration and Schemas
Define all available tools with their schemas for LLM understanding
"""
from typing import List, Dict, Any

import config


# ============================================================================
# Mode-Aware Tool Description Functions
# ============================================================================

def get_python_coder_description() -> str:
    """Get python_coder description based on executor mode"""
    if config.PYTHON_EXECUTOR_MODE in ["nanocoder", "opencode"]:
        return (
            "Execute code tasks using an AI coding agent. Provide detailed natural language "
            "instructions describing what code to write and execute. The agent will generate "
            "Python code, execute it, handle errors, and iterate as needed. "
            "Example: 'Create a function to calculate factorial of numbers 1-10, test it, "
            "and save the results to a file called results.txt'"
        )
    else:  # native mode
        return (
            "Execute Python code in a sandboxed environment. Provide the complete Python "
            "code to execute. Can access files created in previous executions within the "
            "same session. Use for calculations, data analysis, visualizations, or any "
            "computational task."
        )


def get_python_coder_input_description() -> str:
    """Get python_coder input parameter description based on executor mode"""
    if config.PYTHON_EXECUTOR_MODE in ["nanocoder", "opencode"]:
        return (
            "Detailed natural language instruction describing the code task. "
            "Be specific about: what the code should do, expected outputs, "
            "files to create, and any constraints."
        )
    else:  # native mode
        return "The Python code to execute"


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
        "description": get_python_coder_description(),
        "endpoint": "/api/tools/python_coder",
        "method": "POST",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": get_python_coder_input_description()
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
    },

    "ppt_maker": {
        "name": "ppt_maker",
        "description": "Create professional presentations from natural language descriptions. Generates PDF and PPTX files with customizable themes and layouts. Use this when user asks to create slides, presentations, or decks.",
        "endpoint": "/api/tools/ppt_maker",
        "method": "POST",
        "parameters": {
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "Natural language description of the presentation to create (e.g., 'Create a 5-slide presentation about machine learning basics')"
                },
                "session_id": {
                    "type": "string",
                    "description": "Session ID for workspace isolation (required)"
                },
                "theme": {
                    "type": "string",
                    "description": "Marp theme to use: 'default', 'gaia', or 'uncover' (optional, default: gaia)",
                    "default": "gaia"
                },
                "footer": {
                    "type": "string",
                    "description": "Footer text for all slides (optional)",
                    "default": ""
                },
                "header": {
                    "type": "string",
                    "description": "Header text for all slides (optional)",
                    "default": ""
                }
            },
            "required": ["instruction", "session_id"]
        },
        "returns": {
            "success": "Boolean indicating if presentation was created successfully",
            "answer": "Human-readable summary of presentation creation",
            "data": "Dictionary with markdown, pdf_path, pptx_path, num_slides, files",
            "metadata": "Execution metadata including timing and theme"
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
    tools = []
    tools.append('websearch, python_coder, rag, ppt_maker')

    return "\n\n".join(tools)


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
