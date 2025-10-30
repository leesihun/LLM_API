"""
Smart Agent Task Router
Automatically routes to ReAct or Plan-and-Execute based on query characteristics
"""

import logging
from typing import List, Optional
from enum import Enum

from backend.models.schemas import ChatMessage
from backend.tasks.Plan_execute import plan_execute_task  # Plan-and-Execute
from backend.tasks.React import react_agent  # ReAct
from backend.config.settings import settings


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
            selected_agent = AgentType.PLAN_EXECUTE

        logger.info(f"[Smart Agent] Selected agent: {selected_agent}")

        # Route to appropriate agent
        if selected_agent == AgentType.REACT:
            logger.info(f"[Smart Agent] Using ReAct (Reasoning + Acting)")
            response, metadata = await react_agent.execute(messages, session_id, user_id)
        else:
            logger.info(f"[Smart Agent] Using Plan-and-Execute")
            response, metadata = await plan_execute_task.execute(messages, session_id, user_id)

        # Add selection info to metadata
        metadata["agent_selected"] = selected_agent.value
        metadata["agent_selection_mode"] = "auto" if agent_type == AgentType.AUTO else "manual"

        return response, metadata


# Global smart agent instance
smart_agent_task = SmartAgentTask()
