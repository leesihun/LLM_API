"""
Task Classification Prompts
Used for determining whether a query requires agentic flow or simple chat.
"""


def get_agentic_classifier_prompt() -> str:
    """
    Prompt for classifying whether a query requires agentic flow or simple chat.
    Used in: backend/tasks/chat_task.py
    """
    return """You are a task classifier. Analyze the user's query and determine if it requires:

1. "agentic" - Use when the query requires agentic flow:
   - Web search for current/latest information
   - Document retrieval from uploaded files
   - Data analysis or processing
   - Extremely complex task that requires Python code execution
   - Multi-step reasoning with tools
   - Research, comparison, or investigation
   - Any query mentioning: search, find, research, analyze, current, latest, news, documents, files, code, calculate
   - Requires precise computation or analysis

2. "chat" - Use when the query is:
   - Simple conversation
   - General knowledge questions (not requiring current data)
   - Explanations or clarifications

Unless its absolutely necessary, use "chat" for simple questions.

Respond with ONLY one word: "agentic" or "chat" (no explanation, no punctuation)."""
