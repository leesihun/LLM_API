"""
Tool-specific prompt templates.
"""

PYTHON_CODER_PROMPT = (
    "Write Python code to accomplish the task. "
    "Return only one fenced code block with language python. "
    "Keep code self-contained and avoid file I/O unless required."
    "\nTask: {task}"
)

WEB_SEARCH_ANSWER_PROMPT = """Based on the following search results, provide a concise and accurate answer to the user's query.

User Query: {query}

Search Results:
{context}

Answer:"""

