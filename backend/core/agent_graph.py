"""
LangGraph Agentic Controller
Implements the multi-step reasoning workflow with tools
"""

import logging
from typing import Dict, Any, List, TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from backend.config.settings import settings
from backend.config.prompts import PromptRegistry
from backend.models.schemas import ChatMessage
from backend.tools.web_search import web_search_tool
from backend.tools.rag_retriever import rag_retriever
from backend.tools.python_coder import python_coder_tool

logger = logging.getLogger(__name__)


# ============================================================================
# Agent State Definition
# ============================================================================

class AgentState(TypedDict):
    """State passed through the graph"""
    messages: Annotated[List[ChatMessage], operator.add]
    session_id: str
    user_id: str
    plan: str
    tools_used: List[str]
    search_results: str
    rag_context: str
    python_coder_results: str
    current_agent: str
    final_output: str
    verification_passed: bool
    iteration_count: int
    max_iterations: int


# ============================================================================
# LLM Setup
# ============================================================================

def get_llm():
    """Get configured Ollama LLM"""
    from backend.utils.llm_factory import LLMFactory
    return LLMFactory.create_llm()


# ============================================================================
# Graph Nodes
# ============================================================================

async def planning_node(state: AgentState) -> Dict[str, Any]:
    """
    Step 1: Analyze the query and create a plan
    """
    logger.info("[AGENT: Planning] Creating execution plan")
    llm = get_llm()

    # Extract user query
    user_message = state["messages"][-1].content

    planning_prompt = PromptRegistry.get('agent_graph_planning', user_message=user_message)

    response = await llm.ainvoke([HumanMessage(content=planning_prompt)])
    logger.info(f"[AGENT: Planning] Plan created: {response.content[:]}...")

    return {
        "plan": response.content,
        "current_agent": "planning",
        "iteration_count": state.get("iteration_count", 0) + 1
    }


async def tool_selection_node(state: AgentState) -> Dict[str, Any]:
    """
    Step 2: Determine which tools to use based on the plan
    """
    logger.info("[AGENT: Tool Selection] Selecting appropriate tools")
    plan = state["plan"]
    user_message = state["messages"][-1].content

    tools_used = []

    # Simple keyword-based tool selection
    plan_lower = plan.lower()
    query_lower = user_message.lower()

    # Check for search needs
    search_keywords = ["search", "current", "latest", "news", "web", "find", "look up"]
    if any(keyword in plan_lower or keyword in query_lower for keyword in search_keywords):
        tools_used.append("web_search")

    # Check for RAG needs
    rag_keywords = ["document", "file", "pdf", "uploaded", "provided", "text"]
    if any(keyword in plan_lower or keyword in query_lower for keyword in rag_keywords):
        tools_used.append("rag")

    # Check for Python code generation needs
    python_keywords = ["write code", "generate code", "python", "script", "implement", "calculate", "compute", "process file", "csv", "excel", "pandas"]
    if any(keyword in plan_lower or keyword in query_lower for keyword in python_keywords):
        tools_used.append("python_coder")

    # If no tools needed, it's just chat
    if not tools_used:
        tools_used.append("chat")

    logger.info(f"[AGENT: Tool Selection] Selected tools: {', '.join(tools_used)}")
    return {"tools_used": tools_used, "current_agent": "tool_selection"}


async def web_search_node(state: AgentState) -> Dict[str, Any]:
    """
    Step 3a: Execute web search if needed
    """
    if "web_search" not in state.get("tools_used", []):
        logger.info("[AGENT: Web Search] Skipping - not needed")
        return {"search_results": "", "current_agent": "web_search"}

    logger.info("[AGENT: Web Search] Performing web search")
    user_message = state["messages"][-1].content

    # Extract search query (use original message or refined from plan)
    search_results, context_metadata = await web_search_tool.search(
        user_message,
        max_results=5,
        include_context=True,
        user_location=None
    )

    formatted_results = web_search_tool.format_results(search_results)
    logger.info(f"[AGENT: Web Search] Found {len(search_results)} results")
    if context_metadata.get('query_enhanced'):
        logger.info(f"[AGENT: Web Search] Query enhanced with context")

    return {"search_results": formatted_results, "current_agent": "web_search"}


async def rag_retrieval_node(state: AgentState) -> Dict[str, Any]:
    """
    Step 3b: Execute RAG retrieval if needed
    """
    if "rag" not in state.get("tools_used", []):
        logger.info("[AGENT: RAG Retrieval] Skipping - not needed")
        return {"rag_context": "", "current_agent": "rag_retrieval"}

    logger.info("[AGENT: RAG Retrieval] Retrieving relevant documents")
    user_message = state["messages"][-1].content

    # Retrieve relevant documents
    rag_results = await rag_retriever.retrieve(user_message, top_k=5)

    formatted_context = rag_retriever.format_results(rag_results)
    logger.info(f"[AGENT: RAG Retrieval] Found {len(rag_results)} relevant documents")

    return {"rag_context": formatted_context, "current_agent": "rag_retrieval"}


async def python_coder_node(state: AgentState) -> Dict[str, Any]:
    """
    Step 3d: Execute Python code generation if needed
    """
    if "python_coder" not in state.get("tools_used", []):
        logger.info("[AGENT: Python Coder] Skipping - not needed")
        return {"python_coder_results": "", "current_agent": "python_coder"}

    logger.info("[AGENT: Python Coder] Generating and executing Python code")
    logger.info(f"[AGENT: Python Coder] User message: {state['messages'][-1].content[:]}\n\n\n\n")
    user_message = state["messages"][-1].content

    # Execute code generation task
    result = await python_coder_tool.execute_code_task(user_message)

    if result["success"]:
        formatted_result = f"Code executed successfully:\n{result['output']}\n\nExecution details: {result['iterations']} iterations, {result['execution_time']:.2f}s"
    else:
        formatted_result = f"Code execution failed: {result.get('error', 'Unknown error')}"

    logger.info(f"[AGENT: Python Coder] Execution completed")

    return {"python_coder_results": formatted_result, "current_agent": "python_coder"}


async def reasoning_node(state: AgentState) -> Dict[str, Any]:
    """
    Step 4: Generate response using LLM with retrieved information
    """
    logger.info("[AGENT: Reasoning] Generating final response")
    llm = get_llm()

    # Build context from tools
    context_parts = []

    if state.get("search_results"):
        context_parts.append(f"Web Search Results:\n{state['search_results']}")

    if state.get("rag_context"):
        context_parts.append(f"Document Context:\n{state['rag_context']}")

    if state.get("python_coder_results"):
        context_parts.append(f"Python Code Execution Results:\n{state['python_coder_results']}")

    context = "\n\n".join(context_parts) if context_parts else ""

    # Get conversation history
    conversation = "\n".join([
        f"{msg.role}: {msg.content}"
        for msg in state["messages"]
    ])

    # Build prompt
    prompt = PromptRegistry.get('agent_graph_reasoning', conversation=conversation, context=context)

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    logger.info("[AGENT: Reasoning] Response generated")

    return {"final_output": response.content, "current_agent": "reasoning"}


async def verification_node(state: AgentState) -> Dict[str, Any]:
    """
    Step 5: Verify the response quality
    """
    logger.info("[AGENT: Verification] Verifying response quality")
    llm = get_llm()

    final_output = state["final_output"]
    user_message = state["messages"][-1].content

    # Simple verification: check if response is relevant and complete
    verification_prompt = PromptRegistry.get('agent_graph_verification',
                                              user_message=user_message,
                                              final_output=final_output)

    response = await llm.ainvoke([HumanMessage(content=verification_prompt)])

    verification_passed = "yes" in response.content.lower()[:]
    logger.info(f"[AGENT: Verification] Verification {'passed' if verification_passed else 'failed'}")

    return {"verification_passed": verification_passed, "current_agent": "verification"}


def should_continue(state: AgentState) -> str:
    """
    Routing function: decide if we need another iteration
    """
    # Check if verification passed
    if state.get("verification_passed", False):
        return "end"

    # Check iteration limit
    if state.get("iteration_count", 0) >= state.get("max_iterations", 3):
        return "end"

    # Continue refining
    return "continue"


# ============================================================================
# Build the Graph
# ============================================================================

def create_agent_graph():
    """
    Create the LangGraph workflow
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("planning", planning_node)
    workflow.add_node("tool_selection", tool_selection_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("rag_retrieval", rag_retrieval_node)
    workflow.add_node("python_coder", python_coder_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("verification", verification_node)

    # Define edges
    workflow.set_entry_point("planning")
    workflow.add_edge("planning", "tool_selection")
    workflow.add_edge("tool_selection", "web_search")
    workflow.add_edge("web_search", "rag_retrieval")
    workflow.add_edge("rag_retrieval", "python_coder")
    workflow.add_edge("python_coder", "reasoning")
    workflow.add_edge("reasoning", "verification")

    # Conditional routing from verification
    workflow.add_conditional_edges(
        "verification",
        should_continue,
        {
            "end": END,
            "continue": "planning"  # Loop back for refinement
        }
    )

    return workflow.compile()


# Global graph instance
agent_graph = create_agent_graph()
