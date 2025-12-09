"""
Agent orchestrator that routes between ReAct and Plan-Execute.
"""

from __future__ import annotations

import asyncio
from typing import Dict, Any, List, Tuple

from backend.config.settings import settings
from backend.config.prompts import AGENT_CLASSIFIER_PROMPT
from backend.models.schemas import ChatMessage
from backend.agents.react_agent import ReactAgent
from backend.agents.plan_execute_agent import PlanExecuteAgent
from backend.utils.llm_factory import LLMFactory
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class AgentSystem:
    """Entry point for chat completions."""

    def __init__(self):
        self.react_agent = ReactAgent()
        self.plan_agent = PlanExecuteAgent(self.react_agent)

    async def execute(
        self,
        messages: List[ChatMessage],
        session_id: str,
        user_id: str,
        file_paths: List[str] | None = None,
        agent_type: str = "auto",
    ) -> Tuple[str, Dict[str, Any]]:
        agent_choice = agent_type
        if agent_type == "auto":
            agent_choice = await self._classify_agent(messages, user_id)

        if agent_choice == "plan_execute":
            answer, meta = await self.plan_agent.run(messages, session_id, user_id)
        else:
            answer, meta = await self.react_agent.run(messages, session_id, user_id)

        return answer, {
            "agent_type": agent_choice,
            "metadata": meta,
            "file_paths": file_paths or [],
        }

    async def _classify_agent(self, messages: List[ChatMessage], user_id: str) -> str:
        """Use classifier LLM to choose between react and plan_execute."""
        llm = LLMFactory.create_classifier_llm(user_id=user_id)
        user_text = messages[-1].content if messages else ""
        prompt = f"{AGENT_CLASSIFIER_PROMPT}\n\nUser request:\n{user_text}\nAnswer:"

        if hasattr(llm, "ainvoke"):
            result = await llm.ainvoke(prompt)
            prediction = getattr(result, "content", str(result))
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: llm.invoke(prompt))
            prediction = getattr(result, "content", str(result))

        choice = prediction.strip().lower()
        if "plan" in choice:
            return "plan_execute"
        return "react"


# Singleton used by routes
agent_system = AgentSystem()

