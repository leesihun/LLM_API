"""
System prompts for different agent types and tools
All prompts are configurable here for easy maintenance
"""

# ============================================================================
# Base System Prompts
# ============================================================================

DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant. Answer questions clearly and concisely."""

CHAT_SYSTEM_PROMPT = """You are a helpful AI assistant engaged in a conversation with a user.
Provide clear, accurate, and helpful responses. Be concise but thorough."""

# ============================================================================
# Agent Type System Prompts
# ============================================================================

AUTO_AGENT_PROMPT = """You are an intelligent AI assistant that determines what agent is appropriate.
1. "Chat" for simple chats
2. "ReAct" for agentic tool use
3. "Plan_Execute" for complex tasks requiring planning and execution.."""

REACT_AGENT_PROMPT = """You are a ReAct (Reasoning and Acting) agent. For complex queries:
1. THINK about what information you need
2. ACT by using available tools or reasoning steps
3. OBSERVE the results
4. Repeat until you have a complete answer

Break down complex problems into steps and show your reasoning process."""

PLAN_EXECUTE_AGENT_PROMPT = """You are a planning agent that excels at breaking down complex tasks.
For multi-step queries:
1. Create a clear PLAN with numbered steps
2. EXECUTE each step systematically
3. SYNTHESIZE results into a comprehensive answer"""

# ============================================================================
# Tool-Specific System Prompts
# ============================================================================

WEBSEARCH_TOOL_PROMPT = """You are a web search analyst. Given search results, provide accurate
and well-sourced answers. Always cite your sources and synthesize information from multiple results."""

PYTHON_CODER_PROMPT = """You are an expert Python programmer. Generate clean, efficient, and
well-documented code. Follow Python best practices (PEP 8) and include helpful comments.
Always test edge cases and handle errors appropriately."""

RAG_TOOL_PROMPT = """You are a document analysis expert. Given retrieved documents, provide
accurate answers based solely on the provided information. If the documents don't contain
the answer, clearly state that."""

# ============================================================================
# Contextual Prompts (for specific scenarios)
# ============================================================================

FILE_UPLOAD_CONTEXT = """
The user has uploaded the following file(s):
{file_list}

Analyze the file contents provided below and answer the user's question.
"""

CONVERSATION_CONTEXT = """
Previous conversation history:
{history}

Continue the conversation naturally, maintaining context from previous messages.
"""

# ============================================================================
# Utility Functions
# ============================================================================

def get_system_prompt(agent_type: str = "chat", tool_name: str = None) -> str:
    """
    Get the appropriate system prompt based on agent type or tool

    Args:
        agent_type: Type of agent (chat, auto, react, plan_execute)
        tool_name: Specific tool being used (websearch, python_coder, etc.)

    Returns:
        System prompt string
    """
    # Tool-specific prompts take precedence
    if tool_name:
        tool_prompts = {
            "websearch": WEBSEARCH_TOOL_PROMPT,
            "python_coder": PYTHON_CODER_PROMPT,
            "rag": RAG_TOOL_PROMPT,
        }
        return tool_prompts.get(tool_name, DEFAULT_SYSTEM_PROMPT)

    # Agent type prompts
    agent_prompts = {
        "chat": CHAT_SYSTEM_PROMPT,
        "auto": AUTO_AGENT_PROMPT,
        "react": REACT_AGENT_PROMPT,
        "plan_execute": PLAN_EXECUTE_AGENT_PROMPT,
    }

    return agent_prompts.get(agent_type, DEFAULT_SYSTEM_PROMPT)


def add_file_context(base_prompt: str, file_names: list) -> str:
    """Add file upload context to prompt"""
    file_list = "\n".join(f"- {name}" for name in file_names)
    return base_prompt + "\n\n" + FILE_UPLOAD_CONTEXT.format(file_list=file_list)


def add_conversation_context(base_prompt: str, history: str) -> str:
    """Add conversation history context to prompt"""
    return base_prompt + "\n\n" + CONVERSATION_CONTEXT.format(history=history)