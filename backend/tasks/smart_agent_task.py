"""
Smart Agent Task Router
Automatically routes to ReAct or Plan-and-Execute based on query characteristics
"""

import logging
from typing import List, Optional
from enum import Enum

from backend.models.schemas import ChatMessage
from backend.tasks.agentic_task import agentic_task  # Plan-and-Execute
from backend.core.react_agent import react_agent


logger = logging.getLogger(__name__)


class AgentType(str, Enum):
    """Available agent types"""
    REACT = "react"
    PLAN_EXECUTE = "plan_execute"
    AUTO = "auto"


class SmartAgentTask:
    """
    Intelligent agent router that selects the best agent based on query type

    ReAct Agent:
    - Best for: Exploratory queries, multi-step reasoning, dynamic tool selection
    - Example: "Find the capital of France, then search for its population, then calculate if it's larger than London"

    Plan-and-Execute Agent:
    - Best for: Complex queries requiring upfront planning, batch operations
    - Example: "Search for weather in Seoul AND analyze uploaded data AND retrieve documents about climate"
    """

    async def execute(
        self,
        messages: List[ChatMessage],
        session_id: Optional[str],
        user_id: str,
        agent_type: AgentType = AgentType.AUTO
    ) -> tuple[str, dict]:
        """
        Execute task using the appropriate agent

        Args:
            messages: Conversation messages
            session_id: Session ID
            user_id: User identifier
            agent_type: Which agent to use (auto, react, plan_execute)

        Returns:
            Tuple of (AI response, metadata)
        """
        # Auto-select agent if not specified
        selected_agent = agent_type
        if agent_type == AgentType.AUTO:
            selected_agent = self._select_agent(messages[-1].content)

        logger.info(f"[Smart Agent] Selected agent: {selected_agent}")

        # Route to appropriate agent
        if selected_agent == AgentType.REACT:
            logger.info(f"[Smart Agent] Using ReAct (Reasoning + Acting)")
            response, metadata = await react_agent.execute(messages, session_id, user_id)
        else:
            logger.info(f"[Smart Agent] Using Plan-and-Execute")
            response, metadata = await agentic_task.execute(messages, session_id, user_id)

        # Add selection info to metadata
        metadata["agent_selected"] = selected_agent.value
        metadata["agent_selection_mode"] = "auto" if agent_type == AgentType.AUTO else "manual"

        return response, metadata

    def _select_agent(self, query: str) -> AgentType:
        """
        Automatically select the best agent based on query characteristics

        ReAct is better for:
        - Exploratory queries ("find X, then Y, then Z")
        - Sequential dependencies ("after finding X, do Y")
        - Iterative refinement needs
        - Single-tool queries that may need follow-up

        Plan-and-Execute is better for:
        - Multi-tool parallel queries ("search AND analyze AND retrieve")
        - Complex batch operations
        - Well-defined comprehensive tasks
        """
        query_lower = query.lower()

        # Indicators for ReAct
        react_indicators = [
            "then", "after that", "next", "followed by",
            "step by step", "first", "second", "third",
            "if", "depending on", "based on",
            "explore", "investigate", "find out",
        ]

        # Indicators for Plan-and-Execute
        plan_indicators = [
            " and ", " also ", " plus ",
            "both", "all", "multiple",
            "comprehensive", "complete analysis",
            "summarize everything", "full report",
        ]

        # Count indicators
        react_score = sum(1 for indicator in react_indicators if indicator in query_lower)
        plan_score = sum(1 for indicator in plan_indicators if indicator in query_lower)

        # Additional heuristics
        if "?" in query and query_lower.count("?") == 1 and len(query.split()) < 15:
            # Simple single question - prefer ReAct for flexibility
            react_score += 1

        if query_lower.count(",") >= 2 or query_lower.count(" and ") >= 2:
            # Multiple requirements - prefer Plan-and-Execute
            plan_score += 2

        # Decision
        if react_score > plan_score:
            return AgentType.REACT
        elif plan_score > react_score:
            return AgentType.PLAN_EXECUTE
        else:
            # Default to ReAct for better transparency and flexibility
            return AgentType.REACT


# Global smart agent instance
smart_agent_task = SmartAgentTask()
