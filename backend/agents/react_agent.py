import re
import json
import logging
from typing import List, Optional, Tuple, Dict, Any, Protocol, Callable
from enum import Enum
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from backend.config.settings import settings
from backend.models.schemas import ChatMessage, PlanStep, StepResult
from backend.utils.logging_utils import get_logger
from backend.utils.llm_manager import LLMManager, SimpleLLMManager
from backend.utils.llm_response_parser import LLMResponseParser
from backend.utils.session_file_loader import SessionFileLoader
from backend.utils.conversation_loader import ConversationLoader
from backend.storage.conversation_store import conversation_store

# Tools
from backend.tools.web_search import web_search_tool
from backend.tools.rag_retriever import rag_retriever_tool
from backend.tools.python_coder import python_coder_tool
from backend.tools.file_analyzer import file_analyzer
from backend.tools.vision_analyzer import vision_analyzer_tool

logger = get_logger(__name__)


class AgentType(str, Enum):
    """High-level routing modes."""
    CHAT = "chat"
    REACT = "react"
    PLAN_EXECUTE = "plan_execute"
    AUTO = "auto"

class ToolName(str, Enum):
    WEB_SEARCH = "web_search"
    RAG_RETRIEVAL = "rag_retrieval"
    PYTHON_CODER = "python_coder"
    FILE_ANALYZER = "file_analyzer"
    VISION_ANALYZER = "vision_analyzer"
    FINISH = "finish"


class ReActStep:
    """Represents a single Thought-Action-Observation cycle."""

    def __init__(self, step_num: int):
        self.step_num = step_num
        self.thought: str = ""
        self.action: str = ""
        self.action_input: str = ""
        self.observation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_num": self.step_num,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
        }

    def __str__(self) -> str:
        return (
            f"Step {self.step_num}:\n"
            f"Thought: {self.thought}\n"
            f"Action: {self.action}\n"
            f"Input: {self.action_input}\n"
            f"Observation: {self.observation}"
        )

class SimpleChatAgent:
    """LLM-only chat handler used when no tools are required."""

    def __init__(self):
        self.llm_manager = SimpleLLMManager()

    async def run(
        self,
        messages: List[ChatMessage],
        session_id: Optional[str],
        user_id: str,
    ) -> str:
        llm = self.llm_manager.get_llm(user_id)
        conversation = self._build_conversation(messages, session_id)
        response = await llm.ainvoke(conversation)
        return response.content

    def _build_conversation(
        self,
        messages: List[ChatMessage],
        session_id: Optional[str],
    ) -> List[BaseMessage]:
        conversation: List[BaseMessage] = []

        if session_id:
            history = conversation_store.get_messages(session_id, limit=500)
            for msg in history:
                if msg.role == "user":
                    conversation.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    conversation.append(AIMessage(content=msg.content))

        for msg in messages:
            if msg.role == "user":
                conversation.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                conversation.append(AIMessage(content=msg.content))

        return conversation

class AgentOrchestrator:
    def __init__(self, max_iterations: int = 6):
        self.chat_agent = SimpleChatAgent()
        self.react_agent = ReActAgentFactory.create(max_iterations=max_iterations)

    async def execute(
        self,
        messages: List[ChatMessage],
        session_id: Optional[str],
        user_id: str,
        file_paths: Optional[List[str]] = None,
        agent_type: str = "auto",
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        resolved_type = self._resolve_agent_type(agent_type, messages, file_paths)

        if resolved_type == AgentType.CHAT:
            response = await self.chat_agent.run(messages, session_id, user_id)
            return response, None

        if resolved_type == AgentType.REACT:
            return await self.react_agent.execute(messages, session_id, user_id, file_paths)

        plan_steps = await self._create_plan(messages, file_paths, user_id)
        final_answer, step_results = await self.react_agent.execute_with_plan(
            plan_steps=plan_steps,
            messages=messages,
            session_id=session_id or "temp_session",
            user_id=user_id,
            file_paths=file_paths or [],
        )
        metadata = self.react_agent.plan_executor._build_metadata(plan_steps, step_results)
        return final_answer, metadata

    def _resolve_agent_type(
        self,
        agent_type: str,
        messages: List[ChatMessage],
        file_paths: Optional[List[str]],
    ) -> AgentType:
        try:
            requested = AgentType(agent_type)
        except ValueError:
            requested = AgentType.AUTO

        if requested == AgentType.AUTO:
            if file_paths:
                return AgentType.REACT
            last_message = messages[-1].content if messages else ""
            if len(last_message) > 120 or any(
                keyword in last_message.lower() for keyword in ("analyze", "search", "report")
            ):
                return AgentType.REACT
            return AgentType.CHAT

        return requested

    async def _create_plan(
        self,
        messages: List[ChatMessage],
        file_paths: Optional[List[str]],
        user_id: str,
    ) -> List[PlanStep]:
        self.react_agent.llm_manager.ensure_user_id(user_id)
        llm = self.react_agent.llm_manager.llm

        user_query = messages[-1].content if messages else ""
        conversation_history = self._build_conversation_history(messages[:-1])
        planning_prompt = _build_plan_prompt(
            query=user_query,
            conversation_history=conversation_history or "No previous conversation",
            available_tools=settings.available_tools,
            has_files=bool(file_paths),
        )

        response = await llm.ainvoke([HumanMessage(content=planning_prompt)])

        try:
            plan_data = json.loads(response.content.strip())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Plan generation returned invalid JSON: {exc}") from exc

        if not isinstance(plan_data, list) or not plan_data:
            raise ValueError("Plan generation must return a list of steps.")

        plan_steps: List[PlanStep] = []
        for idx, step_data in enumerate(plan_data, start=1):
            plan_steps.append(
                PlanStep(
                    step_num=step_data.get("step_num", idx),
                    goal=step_data.get("goal", "").strip(),
                    primary_tools=step_data.get("primary_tools", []),
                    success_criteria=step_data.get("success_criteria", "").strip(),
                    context=step_data.get("context"),
                )
            )

        return plan_steps

    def _build_conversation_history(self, messages: List[ChatMessage]) -> str:
        if not messages:
            return ""
        return "\n".join(f"{msg.role}: {msg.content}" for msg in messages)


def _build_thought_action_prompt(
    query: str,
    context: str,
    file_guidance: str,
    include_finish_check: bool,
    latest_observation: str,
) -> str:
    finish_section = ""
    if include_finish_check and latest_observation:
        snippet = latest_observation[:500]
        if len(latest_observation) > 500:
            snippet += "... (truncated)"
        finish_section = f"""
## Latest Observation
{snippet}

## Completion Assessment
Decide if the latest observation fully answers the query.
- If YES → choose **finish** and draft the final answer.
- If NO → pick the single tool that will move you closer."""

    guidance_section = f"\n{file_guidance}\n" if file_guidance else ""

    return f"""You are a focused ReAct agent. Think, pick ONE tool, supply its input.
{guidance_section}
## Query
{query}

## Context
{context if context else 'No prior steps yet.'}
{finish_section}
## Tools
- web_search → realtime info
- rag_retrieval → uploaded documents
- python_coder → run python / inspect files
- vision_analyzer → answer image questions (only if images attached)
- finish → only when you already have the final answer

## Response Format
THOUGHT: short reasoning, including whether you can finish
ACTION: tool name
ACTION INPUT: input for the tool"""


def _build_final_answer_prompt(query: str, context: str) -> str:
    return f"""You are a synthesis assistant.

## Query
{query}

## Observations
{context if context else 'No observations.'}

## Instructions
1. Answer the query directly in the first sentence.
2. Support with concrete observations, numbers, or file findings.
3. Keep the response concise but complete."""


def _build_plan_prompt(
    query: str,
    conversation_history: str,
    available_tools: List[str],
    has_files: bool,
) -> str:
    tools_line = ", ".join(available_tools) if available_tools else "python_coder, web_search, rag_retrieval, vision_analyzer"
    history = conversation_history or "No previous conversation."
    files_note = "yes" if has_files else "no"
    return f"""You are a planning assistant.

## User Query
{query}

## Conversation History
{history}

## Tools
{tools_line}

Files Attached: {files_note}

Return a JSON array (1-4 steps). Each step must include:
- "step_num": integer
- "goal": short description
- "primary_tools": list of tool names
- "success_criteria": how we know the step worked
- "context": optional helper text

Example:
[
  {{
    "step_num": 1,
    "goal": "Load the CSV and inspect columns",
    "primary_tools": ["python_coder"],
    "success_criteria": "Columns listed and preview generated",
    "context": "Focus on columns with revenue"
  }}
]"""
# ============================================================================
# Utilities & Generators
# ============================================================================

class ContextFormatter:
    """Formats execution history for LLM context."""
    
    def format(self, steps: List[ReActStep]) -> str:
        if not steps:
            return ""
        
        # Compact format for > 3 steps to save tokens
        if len(steps) > 3:
            return self._format_compact(steps)
        return self._format_detailed(steps)

    def _format_detailed(self, steps: List[ReActStep]) -> str:
        parts = ["Previous Steps:"]
        for step in steps:
            parts.extend([
                f"Step {step.step_num}:",
                f"- Thought: {step.thought}",
                f"- Action: {step.action}",
                f"- Input: {step.action_input}",
                f"- Result: {step.observation}",
                ""
            ])
        return "\n".join(parts)

    def _format_compact(self, steps: List[ReActStep]) -> str:
        parts = ["[HISTORY]"]
        tools_used = ", ".join(set(str(s.action) for s in steps))
        parts.append(f"Tools used so far: {tools_used}")
        parts.append("")

        for step in steps:
            obs = step.observation
            if len(obs) > 300:
                obs = obs[:300] + "... [truncated]"
            
            parts.extend([
                f"[Step {step.step_num}]",
                f"Thought: {step.thought}",
                f"Action: {step.action}",
                f"Result: {obs}",
                ""
            ])
        return "\n".join(parts)


class ThoughtActionGenerator:
    """Generates thought and action from LLM."""
    
    VALID_ACTIONS = {tool.value for tool in ToolName}

    def __init__(self, llm, file_paths: Optional[List[str]] = None):
        self.llm = llm
        self.file_paths = file_paths

    async def generate(self, user_query: str, steps: List[ReActStep], context: str, include_finish_check: bool = False) -> Tuple[str, str, str]:
        file_guidance = self._build_file_guidance()
        latest_observation = steps[-1].observation if steps else ""

        prompt = _build_thought_action_prompt(
            query=user_query,
            context=context,
            file_guidance=file_guidance,
            include_finish_check=include_finish_check,
            latest_observation=latest_observation
        )

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return self._parse_response(response.content.strip())

    def _parse_response(self, response: str) -> Tuple[str, str, str]:
        thought_match = re.search(r'THOUGHT:\s*(.+?)(?=ACTION:|$)', response, re.IGNORECASE | re.DOTALL)
        action_match = re.search(r'ACTION:\s*(\w+)', response, re.IGNORECASE)
        input_match = re.search(r'ACTION\s+INPUT:\s*(.+?)(?=\n\n|\Z)', response, re.IGNORECASE | re.DOTALL)
        if not thought_match or not action_match or not input_match:
            raise ValueError("Incomplete thought/action response from LLM.")

        thought = thought_match.group(1).strip()
        action = action_match.group(1).strip().lower()
        action_input = input_match.group(1).strip()

        if action not in self.VALID_ACTIONS:
            raise ValueError(f"Unsupported action requested: {action}")

        return thought, action, action_input

    def _build_file_guidance(self) -> str:
        if not self.file_paths: return ""
        return "\nGuidelines:\n- Files available. Attempt local analysis (python_coder) first.\n- Use web_search only if local analysis fails."


class AnswerGenerator:
    """Generates final answer."""
    def __init__(self, llm):
        self.llm = llm
        self.formatter = ContextFormatter()

    async def generate(self, user_query: str, steps: List[ReActStep]) -> str:
        context = self.formatter.format(steps)
        prompt = _build_final_answer_prompt(query=user_query, context=context)
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()


# ============================================================================
# Execution Engine
# ============================================================================

class ToolExecutor:
    """Executes tools and routes requests."""
    def __init__(self, llm, user_id: str = "default"):
        self.llm = llm
        self.user_id = user_id

    async def execute(
        self, action: str, action_input: str, 
        file_paths: Optional[List[str]] = None, 
        session_id: Optional[str] = None,
        steps: Optional[List[ReActStep]] = None,
        plan_context: Optional[dict] = None
    ) -> str:
        try:
            if action == ToolName.WEB_SEARCH:
                return await self._execute_web_search(action_input)
            elif action == ToolName.RAG_RETRIEVAL:
                return await self._execute_rag(action_input)
            elif action == ToolName.PYTHON_CODER:
                return await self._execute_python(action_input, file_paths, session_id, steps, plan_context)
            elif action == ToolName.FILE_ANALYZER:
                return await self._execute_file_analysis(action_input, file_paths)
            elif action == ToolName.VISION_ANALYZER:
                return await self._execute_vision(action_input, file_paths)
            return "Invalid action."
        except Exception as e:
            logger.error(f"Tool execution error ({action}): {e}")
            return f"Error executing action: {str(e)}"

    async def _execute_web_search(self, query: str) -> str:
        results, _ = await web_search_tool.search(query, max_results=5, include_context=True)
        return web_search_tool.format_results(results) or "No web search results found."

    async def _execute_rag(self, query: str) -> str:
        results = await rag_retriever_tool.retrieve(query, top_k=5)
        return rag_retriever_tool.format_results(results) or "No relevant documents found."

    async def _execute_python(self, query: str, file_paths: List[str], session_id: str, steps: List[ReActStep], plan_context: dict) -> str:
        # Build robust context
        context_str = self._build_python_context(steps)
        react_context = self._build_react_history(steps, session_id)
        
        conversation_history = ConversationLoader.load_as_dicts(session_id, limit=10)
        current_step_num = len(steps) + 1 if steps else 1
        
        plan_step_num = plan_context.get('current_step') if plan_context else None

        result = await python_coder_tool.execute_code_task(
            query=query, file_paths=file_paths, session_id=session_id,
            context=context_str, stage_prefix=f"step{current_step_num}",
            react_context=react_context, plan_context=plan_context,
            conversation_history=conversation_history,
            react_step=current_step_num, plan_step=plan_step_num
        )
        
        if result["success"]:
            return f"Code executed successfully:\n{result['output']}"
        return f"Code execution failed: {result.get('error', 'Unknown error')}"

    async def _execute_file_analysis(self, query: str, file_paths: List[str]) -> str:
        if not file_paths: return "No files attached."
        result = file_analyzer.analyze(file_paths=file_paths, user_query=query)
        return result.get('summary') if result.get('success') else f"Analysis failed: {result.get('error')}"

    async def _execute_vision(self, query: str, file_paths: List[str]) -> str:
        if not file_paths: return "No files attached."
        result = await vision_analyzer_tool(query=query, file_paths=file_paths, user_id=self.user_id)
        return result.get('analysis') if result.get('success') else f"Vision failed: {result.get('error')}"

    def _build_python_context(self, steps: List[ReActStep]) -> str:
        if not steps: return ""
        return "\n".join([f"Step {s.step_num}: {s.thought} -> {s.action} -> {s.observation[:200]}..." for s in steps[-3:]])

    def _build_react_history(self, steps: List[ReActStep], session_id: str) -> Dict:
        """Builds simplified react context with previous code attempts."""
        if not steps and not session_id: return None
        
        # Load historical code files
        all_session_code = []
        if session_id:
            loader = SessionFileLoader(session_id)
            entries = loader.load_files_by_pattern("script_*.py", sort_reverse=True)
            for e in entries:
                all_session_code.append({
                    'code': e.content,
                    'filename': e.path.name,
                    'timestamp': e.timestamp
                })

        history = []
        for step in (steps or []):
            item = {
                'thought': step.thought,
                'action': step.action,
                'observation': step.observation,
                'status': 'error' if 'error' in step.observation.lower() else 'success'
            }
            history.append(item)

        return {
            'iteration': len(steps) + 1 if steps else 1,
            'history': history,
            'session_code_history': all_session_code
        }


# ============================================================================
# Planning Engine
# ============================================================================

class PlanExecutor:
    """Executes structured plans."""
    def __init__(self, tool_executor, llm):
        self.tool_executor = tool_executor
        self.llm = llm

    async def execute_plan(self, plan_steps: List[PlanStep], user_query: str, file_paths: List[str], session_id: str) -> Tuple[str, List[StepResult]]:
        results = []
        accumulated_obs = []
        react_steps_history = []
        
        for i, plan_step in enumerate(plan_steps):
            logger.info(f"[PlanExecutor] Step {plan_step.step_num}: {plan_step.goal}")
            
            step_result, new_react_steps = await self._execute_step_with_retry(
                plan_step, user_query, accumulated_obs, file_paths, session_id, plan_steps, results, react_steps_history
            )
            
            results.append(step_result)
            react_steps_history.extend(new_react_steps)
            
            if step_result.observation:
                accumulated_obs.append(f"Step {plan_step.step_num}: {step_result.observation}")

            # Simple Replanning: If step failed heavily, consider adapting (simplified)
            # Ideally we just fail or retry. Here we just proceed but mark as failed.
            if not step_result.success and i < len(plan_steps) - 1:
                logger.warning(f"Step {plan_step.step_num} failed. Continuing with best effort.")

        final_answer = await self._generate_final_answer(user_query, results, accumulated_obs)
        return final_answer, results

    async def _execute_step_with_retry(
        self, plan_step: PlanStep, user_query: str, accumulated_obs: List[str], 
        file_paths: List[str], session_id: str, all_plan_steps: List[PlanStep], 
        step_results: List[StepResult], react_steps_history: List[ReActStep]
    ) -> Tuple[StepResult, List[ReActStep]]:
        
        tool_name = plan_step.primary_tools[0] if plan_step.primary_tools else "web_search"
        # Map common names to enum
        tool_map = {"python_code": ToolName.PYTHON_CODER, "python_coder": ToolName.PYTHON_CODER, "web_search": ToolName.WEB_SEARCH}
        tool_enum = tool_map.get(tool_name, ToolName.WEB_SEARCH)

        max_retries = settings.react_step_max_retries
        current_react_steps = []
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                # Build action input
                context = "\n".join(accumulated_obs[-3:]) # Last 3 obs
                action_input = f"Task: {plan_step.goal}\nContext: {context}\nUser Query: {user_query}"
                if attempt > 1:
                    action_input += f"\nRetry Attempt {attempt}. Previous error: {last_error}"

                # Plan context for python tool
                plan_context = {
                    "current_step": plan_step.step_num,
                    "total_steps": len(all_plan_steps),
                    "previous_results": [{"step": r.step_num, "success": r.success} for r in step_results]
                }

                # Execute directly (simulating a ReAct step)
                step = ReActStep(len(react_steps_history) + len(current_react_steps) + 1)
                step.thought = f"Executing plan step {plan_step.step_num} (Attempt {attempt})"
                step.action = tool_enum
                step.action_input = action_input
                
                observation = await self.tool_executor.execute(
                    tool_enum, action_input, file_paths, session_id, 
                    steps=react_steps_history + current_react_steps, plan_context=plan_context
                )
                step.observation = observation
                current_react_steps.append(step)

                success = "error" not in observation.lower() and "failed" not in observation.lower()
                
                if success:
                    return StepResult(
                        step_num=plan_step.step_num, goal=plan_step.goal, success=True, 
                        tool_used=tool_name, attempts=attempt, observation=observation, error=None, metadata={}
                    ), current_react_steps
                
                last_error = observation

            except Exception as e:
                last_error = str(e)
                logger.error(f"Step execution error: {e}")

        # Failed after retries
        return StepResult(
            step_num=plan_step.step_num, goal=plan_step.goal, success=False, 
            tool_used=tool_name, attempts=max_retries, observation=last_error, error=last_error, metadata={}
        ), current_react_steps

    async def _generate_final_answer(self, user_query: str, results: List[StepResult], obs: List[str]) -> str:
        context = "\n".join([f"Step {r.step_num}: {r.observation}" for r in results])
        prompt = _build_final_answer_prompt(query=user_query, context=context)
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()

    def _build_metadata(self, plan_steps: List[PlanStep], step_results: List[StepResult]) -> Dict[str, Any]:
        """Build metadata for plan execution results."""
        return {
            "agent_type": "plan_execute",
            "plan": [
                {
                    "step_num": step.step_num,
                    "goal": step.goal,
                    "primary_tools": step.primary_tools,
                    "success_criteria": step.success_criteria
                }
                for step in plan_steps
            ],
            "results": [
                {
                    "step_num": result.step_num,
                    "goal": result.goal,
                    "success": result.success,
                    "tool_used": result.tool_used,
                    "attempts": result.attempts,
                    "observation": result.observation[:500] if result.observation else None,  # Truncate for size
                    "error": result.error
                }
                for result in step_results
            ],
            "total_steps": len(plan_steps),
            "successful_steps": sum(1 for r in step_results if r.success),
            "failed_steps": sum(1 for r in step_results if not r.success)
        }


# ============================================================================
# ReAct Agent (Main Class)
# ============================================================================

class ReActAgent:
    """Unified ReAct Agent."""
    def __init__(self, max_iterations: int = 20, user_id: str = "default"):
        self.max_iterations = max_iterations
        self.llm_manager = LLMManager(user_id=user_id)
        self.llm = self.llm_manager.llm
        
        self.thought_generator = ThoughtActionGenerator(self.llm)
        self.answer_generator = AnswerGenerator(self.llm)
        self.tool_executor = ToolExecutor(self.llm, user_id)
        self.plan_executor = PlanExecutor(self.tool_executor, self.llm)
        
        self.steps: List[ReActStep] = []

    async def execute(self, messages: List[ChatMessage], session_id: Optional[str], user_id: str, file_paths: Optional[List[str]] = None) -> Tuple[str, Dict[str, Any]]:
        self.llm_manager.ensure_user_id(user_id)
        self.llm = self.llm_manager.llm
        
        # Update components
        self.thought_generator = ThoughtActionGenerator(self.llm, file_paths)
        self.answer_generator.llm = self.llm
        self.tool_executor.llm = self.llm
        self.tool_executor.user_id = user_id
        self.plan_executor.llm = self.llm
        self.plan_executor.tool_executor = self.tool_executor

        user_query = messages[-1].content
        self.steps = []
        final_answer = ""
        
        # Standard ReAct Loop
        for i in range(self.max_iterations):
            step = ReActStep(i + 1)
            context = ContextFormatter().format(self.steps)
            
            # Generate
            try:
                thought, action, action_input = await self.thought_generator.generate(
                    user_query, self.steps, context, include_finish_check=(i > 2)
                )
            except ValueError as exc:
                step.observation = f"Failed to parse thought/action: {exc}"
                self.steps.append(step)
                break
            
            step.thought = thought
            step.action = action
            step.action_input = action_input
            
            if action == ToolName.FINISH:
                final_answer = await self.answer_generator.generate(user_query, self.steps)
                step.observation = "Task completed"
                self.steps.append(step)
                break
                
            # Execute
            observation = await self.tool_executor.execute(
                action, action_input, file_paths, session_id, steps=self.steps
            )
            step.observation = observation
            self.steps.append(step)

        if not final_answer:
            final_answer = await self.answer_generator.generate(user_query, self.steps)

        # Build Metadata
        metadata = {
            "agent_type": "react",
            "steps": [s.to_dict() for s in self.steps],
            "total_steps": len(self.steps)
        }
        return final_answer, metadata

    async def execute_with_plan(self, plan_steps: List[PlanStep], messages: List[ChatMessage], session_id: str, user_id: str, file_paths: List[str], max_iterations_per_step: int = 3) -> Tuple[str, List[StepResult]]:
        """Execute a pre-defined plan."""
        self.llm_manager.ensure_user_id(user_id)
        self.plan_executor.llm = self.llm_manager.llm
        self.plan_executor.tool_executor.llm = self.llm_manager.llm
        
        user_query = messages[-1].content
        final_answer, step_results = await self.plan_executor.execute_plan(
            plan_steps, user_query, file_paths, session_id
        )
        return final_answer, step_results


# Global Factory
class ReActAgentFactory:
    @staticmethod
    def create(max_iterations: int = 6) -> ReActAgent:
        return ReActAgent(max_iterations=max_iterations)

react_agent = ReActAgentFactory.create()

# Global agent system (must be defined after ReActAgentFactory)
agent_system = AgentOrchestrator()


# ============================================================================
# Simple Chat Agent
# ============================================================================

class SimpleChatAgent:
    """Lightweight chat agent without tool usage."""

    def __init__(self):
        self.llm_manager = SimpleLLMManager()

    async def run(
        self,
        messages: List[ChatMessage],
        session_id: Optional[str],
        user_id: str
    ) -> str:
        llm = self.llm_manager.get_llm(user_id)
        conversation = self._build_conversation(messages, session_id)
        response = await llm.ainvoke(conversation)
        return response.content

    def _build_conversation(
        self,
        messages: List[ChatMessage],
        session_id: Optional[str]
    ) -> List[BaseMessage]:
        conversation: List[BaseMessage] = []

        if session_id:
            history = conversation_store.get_messages(session_id, limit=500)
            for msg in history:
                if msg.role == "user":
                    conversation.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    conversation.append(AIMessage(content=msg.content))

        for msg in messages:
            if msg.role == "user":
                conversation.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                conversation.append(AIMessage(content=msg.content))

        return conversation


# ============================================================================
# Agent Orchestrator
# ============================================================================
