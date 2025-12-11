"""
ReAct agent implementation.

Supports two modes:
- Text-based ReAct (Thought/Action/Observation) for any backend
- Native tool-calling prompt hint for Ollama function-calling
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from backend.config.settings import settings
from backend.config.prompts import (
    REACT_SYSTEM_PROMPT,
    REACT_STEP_PROMPT,
    REACT_NATIVE_TOOL_PROMPT,
)
from backend.models.schemas import ChatMessage
from backend.tools import (
    web_search_tool,
    rag_retriever_tool,
    vision_analyzer_tool,
    shell_tool,
)
from backend.tools.python_coder import python_coder_tool
from backend.core.base_tool import BaseTool
from backend.utils.llm_factory import LLMFactory
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ReActStep:
    step: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[str] = None
    observation: Optional[str] = None


@dataclass
class ReActMetadata:
    agent_type: str = "react"
    tool_calling_mode: str = "react"
    steps: List[ReActStep] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    final_answer: Optional[str] = None


class ReactAgent:
    """Lightweight ReAct executor."""

    def __init__(self):
        # registry is intentionally explicit; extend cautiously
        self.tool_registry: Dict[str, BaseTool] = {
            "web_search": web_search_tool,
            "rag": rag_retriever_tool,
            "vision_analyzer": vision_analyzer_tool,
            "shell": shell_tool,
            "python_coder": python_coder_tool,
        }

    async def run(
        self,
        messages: List[ChatMessage],
        session_id: str,
        user_id: str,
        step_context: Optional[str] = None,
    ) -> Tuple[str, ReActMetadata]:
        """Execute ReAct loop until Final Answer or iteration cap."""
        llm = LLMFactory.create_llm(user_id=user_id)
        tool_calling_mode = settings.tool_calling_mode
        max_iters = settings.react_max_iterations
        metadata = ReActMetadata(
            tool_calling_mode=tool_calling_mode,
        )

        scratchpad = ""

        for iter_idx in range(1, max_iters + 1):
            prompt = self._build_prompt(messages, scratchpad, step_context, tool_calling_mode)
            llm_response = await self._call_llm(llm, prompt)
            thought, action, action_input, final_answer = self._parse_response(llm_response)

            metadata.steps.append(
                ReActStep(
                    step=iter_idx,
                    thought=thought or "",
                    action=action,
                    action_input=action_input,
                )
            )

            if final_answer:
                metadata.final_answer = final_answer
                return final_answer, metadata

            if not action:
                # No action, conclude with last thought
                metadata.final_answer = thought or "No action produced."
                return metadata.final_answer, metadata

            observation = await self._execute_action(
                action=action,
                action_input=action_input or "",
                session_id=session_id,
                user_id=user_id,
            )

            metadata.steps[-1].observation = observation
            metadata.tools_used.append(action)

            scratchpad += f"\nThought: {thought}\nAction: {action}\nAction Input: {action_input}\nObservation: {observation}\n"

        metadata.final_answer = "Max iterations reached without final answer."
        return metadata.final_answer, metadata

    def _build_prompt(
        self,
        messages: List[ChatMessage],
        scratchpad: str,
        step_context: Optional[str],
        tool_calling_mode: str,
    ) -> str:
        conversation = []
        for msg in messages:
            prefix = "User" if msg.role == "user" else "Assistant"
            conversation.append(f"{prefix}: {msg.content}")
        convo_text = "\n".join(conversation)

        tools_desc = "\n".join([f"- {name}" for name in self.tool_registry.keys()])

        mode_hint = ""
        if tool_calling_mode == "native":
            mode_hint = f"\n\n{REACT_NATIVE_TOOL_PROMPT}"

        step_hint = f"\n\nStep Context: {step_context}" if step_context else ""

        prompt = (
            f"{REACT_SYSTEM_PROMPT}{mode_hint}\n\nAvailable Tools:\n{tools_desc}"
            f"{step_hint}\n\nConversation so far:\n{convo_text}\n\nScratchpad:\n{scratchpad}\nAssistant:"
        )
        if step_context:
            prompt = f"{REACT_STEP_PROMPT}\n\n{prompt}"
        return prompt

    async def _call_llm(self, llm, prompt: str) -> str:
        """Call LLM asynchronously, regardless of backend sync model."""
        if hasattr(llm, "ainvoke"):
            result = await llm.ainvoke(prompt)
            return getattr(result, "content", str(result))
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: llm.invoke(prompt))
        return getattr(result, "content", str(result))

    def _parse_response(self, text: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
        final_answer = None
        thought_match = re.search(r"Thought:\s*(.*)", text, re.IGNORECASE)
        thought = thought_match.group(1).strip() if thought_match else ""

        final_match = re.search(r"Final Answer:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        if final_match:
            final_answer = final_match.group(1).strip()

        action_match = re.search(r"Action:\s*(.*)", text, re.IGNORECASE)
        action = action_match.group(1).strip() if action_match else None

        action_input_match = re.search(r"Action Input:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        action_input = action_input_match.group(1).strip() if action_input_match else None

        return thought, action, action_input, final_answer

    async def _execute_action(
        self,
        action: str,
        action_input: str,
        session_id: str,
        user_id: str,
    ) -> str:
        tool = self.tool_registry.get(action)
        if not tool:
            return f"Unknown tool '{action}'. Available: {list(self.tool_registry.keys())}"

        try:
            payload = self._safe_load(action_input)
            result = await tool.execute(query=payload if isinstance(payload, str) else action_input, session_id=session_id, user_id=user_id)
            if result.success:
                return result.output
            return f"Tool error: {result.error}"
        except Exception as exc:  # pragma: no cover - runtime safeguard
            logger.error(f"[ReActAgent] Tool '{action}' failed: {exc}")
            return f"Tool execution failed: {exc}"

    def _safe_load(self, text: str):
        try:
            return json.loads(text)
        except Exception:
            return text

