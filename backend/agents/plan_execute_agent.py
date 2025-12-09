"""
Plan-then-execute agent built on top of the ReAct executor.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from backend.config.prompts import PLAN_CREATE_PROMPT, PLAN_EXECUTION_SYSTEM_PROMPT
from backend.models.schemas import ChatMessage
from backend.utils.llm_factory import LLMFactory
from backend.utils.logging_utils import get_logger
from backend.agents.react_agent import ReactAgent, ReActMetadata

logger = get_logger(__name__)


@dataclass
class PlanStepResult:
    step_num: int
    goal: str
    success: bool
    observation: str
    metadata: Optional[ReActMetadata] = None


@dataclass
class PlanMetadata:
    agent_type: str = "plan_execute"
    plan_steps: List[str] = field(default_factory=list)
    step_results: List[PlanStepResult] = field(default_factory=list)
    final_answer: Optional[str] = None


class PlanExecuteAgent:
    """Generates a plan then executes each step with the ReactAgent."""

    def __init__(self, react_agent: ReactAgent):
        self.react_agent = react_agent

    async def run(
        self,
        messages: List[ChatMessage],
        session_id: str,
        user_id: str,
    ) -> Tuple[str, PlanMetadata]:
        metadata = PlanMetadata()
        llm = LLMFactory.create_llm(user_id=user_id)

        plan_prompt = self._build_plan_prompt(messages)
        plan_text = await self._call_llm(llm, plan_prompt)
        steps = self._parse_plan(plan_text)
        metadata.plan_steps = steps

        if not steps:
            metadata.final_answer = "Planner failed to produce steps."
            return metadata.final_answer, metadata

        final_answer = ""
        for idx, step in enumerate(steps, start=1):
            step_context = f"Step {idx}: {step}"
            answer, react_meta = await self.react_agent.run(
                messages=messages,
                session_id=session_id,
                user_id=user_id,
                step_context=step_context,
            )

            success = True if answer else False
            metadata.step_results.append(
                PlanStepResult(
                    step_num=idx,
                    goal=step,
                    success=success,
                    observation=answer,
                    metadata=react_meta,
                )
            )
            final_answer = answer

            if not success:
                break

        metadata.final_answer = final_answer
        return final_answer, metadata

    def _build_plan_prompt(self, messages: List[ChatMessage]) -> str:
        conversation = []
        for msg in messages:
            prefix = "User" if msg.role == "user" else "Assistant"
            conversation.append(f"{prefix}: {msg.content}")
        convo_text = "\n".join(conversation)
        return f"{PLAN_EXECUTION_SYSTEM_PROMPT}\n\n{PLAN_CREATE_PROMPT}\n\nConversation:\n{convo_text}\nPlan:"

    async def _call_llm(self, llm, prompt: str) -> str:
        if hasattr(llm, "ainvoke"):
            result = await llm.ainvoke(prompt)
            return getattr(result, "content", str(result))
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: llm.invoke(prompt))
        return getattr(result, "content", str(result))

    def _parse_plan(self, text: str) -> List[str]:
        lines = text.splitlines()
        steps = []
        for line in lines:
            clean = line.strip()
            if not clean:
                continue
            m = re.match(r"^\d+[\).]\s*(.+)", clean)
            if m:
                steps.append(m.group(1).strip())
            elif clean.startswith("- "):
                steps.append(clean[2:].strip())
        return steps

