"""
Prompts for LangGraph agent workflow.
"""

from typing import Optional


def get_planning_prompt(user_message: str) -> str:
    """
    Generate prompt for planning step.

    Args:
        user_message: The user's query

    Returns:
        Planning prompt string
    """
    return f"""You are a planning assistant. Analyze this user query and create a step-by-step plan.

User Query: {user_message}

Consider:
1. Does this require web search for current information?
2. Does this require document retrieval (RAG)?
3. Does this absolutely require Python code generation?
4. Is this a straightforward chat response?
5. What tools are needed?

Create a concise plan (2-10 steps) explaining how to answer this query."""


def get_reasoning_prompt(conversation: str, context: Optional[str] = None) -> str:
    """
    Generate prompt for reasoning step.

    Args:
        conversation: Chat conversation history
        context: Optional context information from previous steps

    Returns:
        Reasoning prompt string
    """
    if context:
        return f"""You are a helpful AI assistant. Use the provided context to answer the user's question.

Context Information:
{context}

Conversation:
{conversation}

Provide a clear, accurate, and helpful response based on the context and conversation."""
    else:
        return f"""You are a helpful AI assistant. Engage in a natural conversation.

Conversation:
{conversation}

Provide a clear, accurate, and helpful response."""


def get_verification_prompt(user_message: str, final_output: str) -> str:
    """
    Generate prompt for verification step.

    Args:
        user_message: The user's original question
        final_output: The generated response to verify

    Returns:
        Verification prompt string
    """
    return f"""Verify if this response adequately answers the user's question.

User Question: {user_message}

Response: {final_output}

Answer with "YES" if the response is adequate and complete, or "NO" if it needs improvement.
Provide brief reasoning."""
