"""
Smart Agent Task Router

Automatically routes to ReAct or Plan-and-Execute based on query characteristics.
Provides intelligent agent selection for optimal task execution.
"""

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
        agent_type: AgentType = AgentType.AUTO,
        file_paths: Optional[List[str]] = None
    ) -> tuple[str, dict]:
        """
        Execute task using the appropriate agent

        Args:
            messages: Conversation messages
            session_id: Session ID
            user_id: User identifier
            agent_type: Which agent to use (auto, react, plan_execute)
            file_paths: Optional list of file paths for code execution

        Returns:
            Tuple of (AI response, metadata)
        """
        # Auto-select agent if not specified
        selected_agent = agent_type
        if agent_type == AgentType.AUTO:
            selected_agent = self._select_agent(messages, file_paths)

        logger.info(f"[Smart Agent] Selected agent: {selected_agent}")

        # Route to appropriate agent
        if selected_agent == AgentType.REACT:
            logger.info(f"[Smart Agent] Using ReAct (Reasoning + Acting)")
            response, metadata = await react_agent.execute(messages, session_id, user_id, file_paths)
        else:
            logger.info(f"[Smart Agent] Using Plan-and-Execute")
            response, metadata = await plan_execute_task.execute(messages, session_id, user_id, file_paths)

        # Add selection info to metadata
        metadata["agent_selected"] = selected_agent.value
        metadata["agent_selection_mode"] = "auto" if agent_type == AgentType.AUTO else "manual"

        return response, metadata

    def _select_agent(
        self,
        messages: List[ChatMessage],
        file_paths: Optional[List[str]] = None
    ) -> AgentType:
        """
        Intelligently select agent based on query characteristics.

        Heuristic:
        - Use Plan-Execute if:
          - Multiple files (>2) are attached (complex data analysis)
          - Query contains multi-step keywords (AND, then, after, followed by)
          - Query is very long (>500 chars, likely complex)
        - Use ReAct for:
          - Simple queries or exploratory tasks
          - Single file or no files
          - Short, focused queries

        Args:
            messages: Conversation messages
            file_paths: Optional list of attached files

        Returns:
            Selected agent type (REACT or PLAN_EXECUTE)
        """
        # Extract user query
        user_query = messages[-1].content.lower() if messages else ""

        # Heuristic 1: Multiple files suggest complex analysis -> Plan-Execute
        file_count = len(file_paths) if file_paths else 0
        if file_count > 2:
            logger.info(f"[SmartAgent] Selecting Plan-Execute: {file_count} files attached")
            return AgentType.PLAN_EXECUTE

        # Heuristic 2: Multi-step keywords suggest structured workflow -> Plan-Execute
        multi_step_keywords = [
            ' and then ', ' then ', ' after that', ' followed by',
            ' next ', ' subsequently', ' finally', ' first', ' second',
            'step 1', 'step 2', 'multiple steps', 'several steps'
        ]
        if any(keyword in user_query for keyword in multi_step_keywords):
            logger.info("[SmartAgent] Selecting Plan-Execute: multi-step keywords detected")
            return AgentType.PLAN_EXECUTE

        # Heuristic 3: Very long query suggests complexity -> Plan-Execute
        if len(user_query) > 500:
            logger.info(f"[SmartAgent] Selecting Plan-Execute: long query ({len(user_query)} chars)")
            return AgentType.PLAN_EXECUTE

        # Heuristic 4: Explicit planning keywords -> Plan-Execute
        planning_keywords = [
            'plan', 'organize', 'structure', 'comprehensive', 'detailed analysis',
            'break down', 'systematically', 'thorough'
        ]
        if any(keyword in user_query for keyword in planning_keywords):
            logger.info("[SmartAgent] Selecting Plan-Execute: planning keywords detected")
            return AgentType.PLAN_EXECUTE

        # Default: ReAct for simple, exploratory tasks
        logger.info("[SmartAgent] Selecting ReAct: simple/exploratory query")
        return AgentType.REACT


# Global smart agent instance
smart_agent_task = SmartAgentTask()
