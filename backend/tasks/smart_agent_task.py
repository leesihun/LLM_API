from typing import List, Optional
from enum import Enum

from backend.models.schemas import ChatMessage
from backend.tasks.Plan_execute import plan_execute_task
from backend.tasks.react import react_agent
from backend.config.settings import settings
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class AgentType(str, Enum):
    """Available agent types"""
    REACT = "react"
    PLAN_EXECUTE = "plan_execute"
    AUTO = "auto"


class SmartAgentTask:
    async def execute(
        self,
        messages: List[ChatMessage],
        session_id: Optional[str],
        user_id: str,
        agent_type: AgentType = AgentType.AUTO,
        file_paths: Optional[List[str]] = None
    ) -> tuple[str, dict]:
    
        # Validate agent type - AUTO should not reach here
        if agent_type == AgentType.AUTO:
            logger.warning("[Smart Agent] AUTO agent type received - should be classified in chat.py. Defaulting to PLAN_EXECUTE.")
            agent_type = AgentType.PLAN_EXECUTE

        logger.info(f"[Smart Agent] Using agent: {agent_type}")

        # Route to appropriate agent
        if agent_type == AgentType.REACT:
            logger.info("[Smart Agent] Executing ReAct (Reasoning + Acting)")
            response, metadata = await react_agent.execute(messages, session_id, user_id, file_paths)
        elif agent_type == AgentType.PLAN_EXECUTE:
            logger.info("[Smart Agent] Executing Plan-and-Execute")
            response, metadata = await plan_execute_task.execute(messages, session_id, user_id, file_paths)
        else:
            raise ValueError(f"Invalid agent type: {agent_type}")

        # Add agent info to metadata
        metadata["agent_type"] = agent_type.value

        return response, metadata


# Global smart agent instance
smart_agent_task = SmartAgentTask()
