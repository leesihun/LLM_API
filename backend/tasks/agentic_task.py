"""
Agentic Task
Uses LangGraph for complex multi-step reasoning with tools
"""

import logging
from typing import List, Optional

from backend.models.schemas import ChatMessage
from backend.core.agent_graph import agent_graph, AgentState

logger = logging.getLogger(__name__)


class AgenticTask:
    """Handles complex agentic workflows using LangGraph"""

    async def execute(
        self,
        messages: List[ChatMessage],
        session_id: Optional[str],
        user_id: str,
        max_iterations: int = 3
    ) -> str:
        """
        Execute agentic workflow with planning, tool use, and verification

        Args:
            messages: List of chat messages
            session_id: Session ID for conversation tracking
            user_id: User identifier
            max_iterations: Maximum refinement iterations

        Returns:
            Final AI response
        """
        # Prepare initial state
        initial_state: AgentState = {
            "messages": messages,
            "session_id": session_id or "",
            "user_id": user_id,
            "plan": "",
            "tools_used": [],
            "search_results": "",
            "rag_context": "",
            "data_analysis_results": "",
            "current_agent": "",
            "final_output": "",
            "verification_passed": False,
            "iteration_count": 0,
            "max_iterations": max_iterations
        }

        logger.info(f"Starting agentic workflow for user: {user_id}, session: {session_id}")

        # Execute the graph
        result = await agent_graph.ainvoke(initial_state)

        # Log completion
        tools_used = result.get("tools_used", [])
        logger.info(f"Agentic workflow completed. Tools used: {', '.join(tools_used)}")

        # Return final output
        return result.get("final_output", "I apologize, but I couldn't generate a response.")


# Global agentic task instance
agentic_task = AgenticTask()
