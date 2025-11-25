"""
Task Classification Prompts
Used for determining which agent type should handle a query: chat, react, or plan_execute.
"""


def get_agent_type_classifier_prompt() -> str:
    """
    Prompt for classifying queries into three agent types: chat, react, or plan_execute.
    
    Returns:
        Prompt string for 3-way agent classification
    """
    return """You are an agent type classifier. Classify user queries into one of three types: "chat", "react", or "plan_execute".
CHAT - Very simple questions answerable from easy general knowledge base (NO tools needed):
REACT - A little bit complicated, single-goal tasks requiring tools (web search, code execution, simple analysis):
PLAN_EXECUTE - Multi-step complex tasks requiring planning and structured execution:
Respond with ONLY one word: "chat", "react", or "plan_execute" (no explanation, no punctuation)."""
