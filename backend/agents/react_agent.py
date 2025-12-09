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
from backend.storage.conversation_store import conversation_store

# Tools
from backend.tools.web_search import web_search_tool
from backend.tools.rag_retriever import rag_retriever_tool
from backend.tools.python_coder import python_coder_tool
from backend.tools.shell_tool import shell_tool
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
    SHELL = "shell"
    FILE_ANALYZER = "file_analyzer"
    VISION_ANALYZER = "vision_analyzer"
    NO_TOOLS = "no_tools"
    FINISH = "finish"


# FINISH detection helpers (used for both ReAct and plan_execute paths)
_FINISH_PATTERN = re.compile(r"\bfinish\s*:\s*(true|yes|1)\b", re.IGNORECASE)


def observation_signals_finish(observation: str) -> bool:
    """Return True if observation explicitly signals FINISH: true/yes/1."""
    if not observation:
        return False
    return bool(_FINISH_PATTERN.search(observation))


# ============================================================================
# Native Tool Definitions (for Ollama/OpenAI function calling)
# ============================================================================

NATIVE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for real-time information. Use for current events, news, or facts that may have changed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rag_retrieval",
            "description": "Search through uploaded documents and knowledge base for specific information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for document retrieval"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "python_coder",
            "description": "Generate and execute Python code based on a task description. Use for calculations, data analysis, file processing, and creating visualizations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Clear description of what you want the code to accomplish. The system will generate and execute the appropriate Python code."
                    }
                },
                "required": ["task"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vision_analyzer",
            "description": "Analyze images to understand their content, extract text, or answer questions about them.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Question or task about the image(s)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run safe shell commands (ls/dir, pwd, cd, cat/head/tail, find, grep, wc, echo) inside the sandbox working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run for navigation or inspection"
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "no_tools",
            "description": "Think and reason without using any external tools. Use this when no suitable tool is available or when you need to synthesize information from previous steps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Your reasoning or analysis of the current situation"
                    }
                },
                "required": ["reasoning"]
            }
        }
    }
]


class ReActStep:
    """Represents a single Thought-Action-Observation cycle."""

    def __init__(self, step_num: int):
        self.step_num = step_num
        self.thought: str = ""
        self.action: str = ""
        self.action_input: str = ""
        self.observation: str = ""
        self.finish: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_num": self.step_num,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
            "finish": self.finish,
        }

    def __str__(self) -> str:
        return (
            f"Step {self.step_num}:\n"
            f"Thought: {self.thought}\n"
            f"Action: {self.action}\n"
            f"Input: {self.action_input}\n"
            f"Finish: {self.finish}\n"
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
        resolved_type = await self._resolve_agent_type(agent_type, messages, file_paths, user_id)

        if resolved_type == AgentType.CHAT:
            response = await self.chat_agent.run(messages, session_id, user_id)
            return response, None

        if resolved_type == AgentType.REACT:
            return await self.react_agent.execute(messages, session_id, user_id, file_paths)

        plan_steps = await self._create_plan(messages, file_paths, user_id)
        final_answer, step_results = await self.react_agent.execute_with_plan(
            plan_steps=plan_steps,
            messages=messages,
            session_id=session_id or "session",
            user_id=user_id,
            file_paths=file_paths or [],
        )
        metadata = self.react_agent.plan_executor._build_metadata(plan_steps, step_results)
        return final_answer, metadata

    async def _resolve_agent_type(
        self,
        agent_type: str,
        messages: List[ChatMessage],
        file_paths: Optional[List[str]],
        user_id: str,
    ) -> AgentType:
        try:
            requested = AgentType(agent_type)
        except ValueError:
            requested = AgentType.AUTO

        if requested != AgentType.AUTO:
            return requested

        return await self._llm_route_agent(messages, file_paths, user_id)

    async def _llm_route_agent(
        self,
        messages: List[ChatMessage],
        file_paths: Optional[List[str]],
        user_id: str,
    ) -> AgentType:
        """Use the LLM to choose chat vs react vs plan_execute when in AUTO."""
        user_query = messages[-1].content if messages else ""
        has_files = bool(file_paths)
        serialized_query = json.dumps(user_query)

        routing_prompt = (
            "You are an agent router. Choose exactly one execution mode.\n"
            "- chat: direct, single-step answer with no external tools required.\n"
            "- react: needs tool use for more specific information (web search, python code, shell, file/vision analysis) "
            "or ad-hoc multi-hop reasoning without a pre-plan.\n"
            "- plan_execute: A detailed task requires a structured multi-step plan (reports, multi-file edits, "
            "pipelines, research + synthesis) before tool execution.\n"
            "Prefer plan_execute when the task is not simple & trivial.\n"
            "Respond ONLY with JSON: {\"agent_type\": \"chat|react|plan_execute\"}.\n"
            f"Has_files: {has_files}\n"
            f"User_request: {serialized_query}\n"
        )

        self.react_agent.llm_manager.ensure_user_id(user_id)
        llm = self.react_agent.llm_manager.llm
        response = await llm.ainvoke([HumanMessage(content=routing_prompt)])

        content = getattr(response, "content", "") or ""
        content = self._strip_markdown_code_blocks(content)

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Routing LLM returned invalid JSON: {content}") from exc

        agent_value = parsed.get("agent_type")
        if agent_value not in {AgentType.CHAT.value, AgentType.REACT.value, AgentType.PLAN_EXECUTE.value}:
            raise ValueError(f"Routing LLM returned unsupported agent_type: {agent_value}")

        return AgentType(agent_value)

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

        # Extract content using the same method as ReAct (handles tool_calls)
        response_content = self._extract_plan_response_content(response)

        # Strip markdown code blocks (```json ... ```)
        response_content = self._strip_markdown_code_blocks(response_content)

        try:
            plan_data = json.loads(response_content.strip())
        except json.JSONDecodeError as exc:
            logger.error(f"[Planner] Failed to parse JSON from response: {response_content[:500]}")
            raise ValueError(f"Plan generation returned invalid JSON: {exc}") from exc

        # Handle common structured wrappers: {"steps": [...]} or {"plan": [...]}
        if isinstance(plan_data, str):
            try:
                plan_data = json.loads(plan_data)
            except json.JSONDecodeError:
                pass  # keep original for error message below

        if isinstance(plan_data, dict):
            # Preferred: explicit steps/plan field
            for key in ("steps", "plan"):
                if key in plan_data and isinstance(plan_data[key], list):
                    plan_data = plan_data[key]
                    break
            else:
                # Fallback: single list-valued entry (e.g., {"0": [{...}]})
                list_values = [v for v in plan_data.values() if isinstance(v, list)]
                if len(list_values) == 1:
                    plan_data = list_values[0]

        if not isinstance(plan_data, list) or not plan_data:
            raise ValueError(
                f"Plan generation must return a non-empty list of steps; got {type(plan_data).__name__} with keys {getattr(plan_data, 'keys', lambda: [])()}."
            )

        plan_steps: List[PlanStep] = []
        for idx, step_data in enumerate(plan_data, start=1):
            if not isinstance(step_data, dict):
                raise ValueError(f"Plan step {idx} is not an object: {step_data!r}")

            goal = step_data.get("goal", "").strip()
            primary_tools = step_data.get("primary_tools", [])
            success_criteria = step_data.get("success_criteria", "").strip()

            if not goal or not isinstance(primary_tools, list):
                raise ValueError(
                    f"Plan step {idx} is missing required fields (goal, primary_tools)."
                )

            plan_steps.append(
                PlanStep(
                    step_num=step_data.get("step_num", idx),
                    goal=goal,
                    primary_tools=primary_tools,
                    success_criteria=success_criteria,
                    context=step_data.get("context"),
                )
            )

        return plan_steps

    def _build_conversation_history(self, messages: List[ChatMessage]) -> str:
        if not messages:
            return ""
        return "\n".join(f"{msg.role}: {msg.content}" for msg in messages)

    def _strip_markdown_code_blocks(self, content: str) -> str:
        """Remove markdown code blocks (```json ... ``` or ```...```)."""
        if not content:
            return content

        # Pattern: ```json\n...\n``` or ```\n...\n```
        # Remove opening fence
        content = re.sub(r'^```(?:json)?\s*\n', '', content, flags=re.IGNORECASE)
        # Remove closing fence
        content = re.sub(r'\n```\s*$', '', content)

        return content.strip()

    def _extract_plan_response_content(self, response) -> str:
        """Extract text content from LLM response (handles tool_calls, content, etc.)."""
        # Direct content
        if hasattr(response, 'content') and response.content:
            return response.content

        # Tool calls (Ollama tool calling) - extract JSON from args
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_calls = response.tool_calls
            logger.info(f"[Planner] Found tool_calls: {tool_calls}")

            if isinstance(tool_calls, list) and len(tool_calls) > 0:
                tool_call = tool_calls[0]

                # Extract args
                if isinstance(tool_call, dict):
                    args = tool_call.get('args', {})
                else:
                    args = getattr(tool_call, 'args', {})

                # If args contains a 'plan' or 'steps' field with JSON
                if isinstance(args, dict):
                    for key in ['plan', 'steps', 'response', 'output']:
                        if key in args:
                            value = args[key]
                            # If it's already a list/dict, convert to JSON string
                            if isinstance(value, (list, dict)):
                                return json.dumps(value)
                            # If it's a string, return as-is
                            return str(value)

                # Fallback: return entire args as JSON
                if isinstance(args, (list, dict)):
                    return json.dumps(args)
                return str(args)

        # Text attribute
        if hasattr(response, 'text') and response.text:
            return response.text

        # Additional kwargs
        if hasattr(response, 'additional_kwargs') and response.additional_kwargs:
            kwargs = response.additional_kwargs
            for key in ['content', 'text', 'message', 'response', 'output']:
                if key in kwargs and kwargs[key]:
                    return str(kwargs[key])

        # Response metadata
        if hasattr(response, 'response_metadata') and response.response_metadata:
            meta = response.response_metadata
            if isinstance(meta, dict):
                if 'message' in meta and isinstance(meta['message'], dict):
                    if 'content' in meta['message'] and meta['message']['content']:
                        return meta['message']['content']
                if 'content' in meta and meta['content']:
                    return meta['content']

        # Fallback
        logger.error(f"[Planner] Could not extract content from response!")
        return str(response)


def _build_thought_action_prompt(
    query: str,
    context: str,
    file_guidance: str,
    include_finish_check: bool,
    latest_observation: str,
) -> str:
    finish_section = ""
    if include_finish_check and latest_observation:
        snippet = latest_observation[:50000]
        if len(latest_observation) > 50000:
            snippet += "... (truncated)"
        finish_section = (
            "----------------------------------------------------------\n"
            "Latest observation (for finish check):\n"
            f"{snippet}\n"
            "- If this fully answers the query → set FINISH: true and do NOT call a tool.\n"
            "- Otherwise → FINISH: false and choose one tool that adds new information.\n"
        )

    context_section = context if context else "No prior steps yet."

    return f"""You are a focused ReAct agent. Choose at most ONE tool per turn. Set FINISH to true only when you can answer directly; otherwise keep FINISH false and call the best single tool.

General Guidelines:
- Follow the Response format exactly; no extra prose.
- Use only the tool names listed under Tools; never invent new ones.
- ACTION INPUT must be a concrete command/query:
python_coder -> plain-language task
web_search/rag_retrieval -> search query
shell -> one safe command
vision_analyzer -> question
no_tools -> reasoning
- Prefer local analysis before web_search unless the request clearly needs live/external info.{file_guidance}

{finish_section}
----------------------------------------------------------

## User Query (Original inquire)
{query}

----------------------------------------------------------

## Context (Overall plans)
{context_section}

----------------------------------------------------------

## Tools
- web_search → realtime or complex/specific external info
- rag_retrieval → very specific info from the knowledge base (only when explicitly needed)
- python_coder → generate and execute Python code based on a natural-language task
- shell → run safe shell commands (ls/dir/pwd/cd/cat/head/tail/find/grep/wc/echo) inside the sandbox
- vision_analyzer → answer image questions (only if images attached)
- no_tools → think and reason without external tools

## Response format (STRICTLY FOLLOW THE BELOW FORMAT)

THOUGHT: reasoning and next step, including if you can finish
ACTION: tool name (empty when FINISH is true)
ACTION INPUT: exact input for the chosen tool
FINISH: true or false
"""


def _build_final_answer_prompt(query: str, context: str) -> str:
    return f"""You are a assistant that answers the user's qurey based on observations The qury and observations are provided below.

## Query
{query}

## Observations
{context if context else 'No observations.'}

## Instructions: Keep the response concise but complete."""


def _build_plan_prompt(
    query: str,
    conversation_history: str,
    available_tools: List[str],
    has_files: bool,
) -> str:
    tools_line = ", ".join(available_tools) if available_tools else "python_coder, web_search, rag_retrieval, vision_analyzer"
    history = conversation_history or "No previous conversation."
    files_note = "yes" if has_files else "no"
    return f"""You are a planning agent. Design a THOROUGH and comprehensive multi-step plan the agents will execute.

Inputs:
----------------------------------------------------------

- User Query (Original inquire): {query}

----------------------------------------------------------

- Conversation history (Past history): {history}

----------------------------------------------------------

- Files attached: {files_note}

----------------------------------------------------------

- Available tools: {tools_line}

----------------------------------------------------------

Guidelines:
- Output ONLY valid JSON (no prose) representing THOROUGH and comprehensive multi-step plan.
- Use only the available tools; prefer python_coder for local/file analysis and reserve web_search for live or external data.
- When given file, always inspect the file first, THOROUGHLY inspect the file access patterns, especiallly for JSON files with keys.

----------------------------------------------------------
JSON schema:
[
  {{
    "step_num": 1,
    "goal": "short, outcome-focused objective",
    "primary_tools": ["tool_name"],
    "success_criteria": "objective check that confirms success",
    "context": "concise guidance for the next step"
  }},
  {{
    "step_num": 2,
    "goal": "short, outcome-focused objective",
    "primary_tools": ["tool_name"],
    "success_criteria": "objective check that confirms success",
    "context": "concise guidance for the next step"
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
        return ""



class ThoughtActionGenerator:
    """Generates thought and action from LLM.
    
    Supports two modes:
    - 'react': Text-based THOUGHT/ACTION/OBSERVATION prompting (works with all backends)
    - 'native': Ollama/OpenAI function calling with structured JSON (best with Ollama)
    """
    
    VALID_ACTIONS = {tool.value for tool in ToolName}

    def __init__(self, llm, file_paths: Optional[List[str]] = None):
        self.llm = llm
        self.file_paths = file_paths
        self.mode = settings.tool_calling_mode  # 'react' or 'native'
        logger.info(f"[ThoughtActionGenerator] Initialized with LLM: {type(llm).__name__}, mode: {self.mode}")
        
        # For native mode, bind tools to LLM
        if self.mode == 'native':
            self._setup_native_tools()
    
    def _setup_native_tools(self):
        """Setup native tool calling for Ollama/OpenAI."""
        try:
            # Get the underlying LLM (unwrap interceptor if present)
            underlying_llm = getattr(self.llm, 'llm', self.llm)
            
            # Check if LLM supports tool binding
            if hasattr(underlying_llm, 'bind_tools'):
                self.llm_with_tools = underlying_llm.bind_tools(NATIVE_TOOLS)
                logger.info(f"[ThoughtActionGenerator] Native tools bound successfully ({len(NATIVE_TOOLS)} tools)")
            else:
                logger.warning(f"[ThoughtActionGenerator] LLM does not support bind_tools, falling back to 'react' mode")
                self.mode = 'react'
                self.llm_with_tools = None
        except Exception as e:
            logger.error(f"[ThoughtActionGenerator] Failed to bind tools: {e}, falling back to 'react' mode")
            self.mode = 'react'
            self.llm_with_tools = None

    async def generate(self, user_query: str, steps: List[ReActStep], context: str, include_finish_check: bool = False) -> Tuple[str, str, str, bool]:
        """Generate thought, action, input, and finish flag using configured mode."""
        if self.mode == 'native':
            return await self._generate_native(user_query, steps, context, include_finish_check)
        else:
            return await self._generate_react(user_query, steps, context, include_finish_check)
    
    async def _generate_native(self, user_query: str, steps: List[ReActStep], context: str, include_finish_check: bool = False) -> Tuple[str, str, str, bool]:
        """Generate using native Ollama/OpenAI function calling."""
        file_guidance = self._build_file_guidance()
        
        # Build a simpler prompt for native mode (tools are defined in schema)
        prompt = self._build_native_prompt(user_query, steps, context, file_guidance, include_finish_check)
        
        logger.info(f"[ReAct-Native] Invoking LLM with tools...")
        logger.info(f"[ReAct-Native] Prompt length: {len(prompt)} chars")
        
        try:
            response = await self.llm_with_tools.ainvoke([HumanMessage(content=prompt)])
            
            # Extract tool call from response
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_call = response.tool_calls[0]
                
                # Extract action and arguments
                if isinstance(tool_call, dict):
                    action = tool_call.get('name', 'finish')
                    args = tool_call.get('args', {})
                else:
                    action = getattr(tool_call, 'name', 'finish')
                    args = getattr(tool_call, 'args', {})
                
                finish_flag = action == 'finish'
                # Build action_input based on tool type
                if action == 'python_coder':
                    action_input = args.get('task', args.get('query', ''))
                elif action == 'finish':
                    action_input = args.get('answer', '')
                else:
                    action_input = args.get('query', str(args))
                
                # Generate thought from content or synthesize
                thought = response.content if response.content else f"Using {action} tool to complete the task."
                
                logger.info(f"[ReAct-Native] Tool call: {action}, input length: {len(action_input)}, finish={finish_flag}")
                return thought, action, action_input, finish_flag
            
            # No tool call - check if it's a direct response (treat as finish)
            if response.content:
                logger.info(f"[ReAct-Native] No tool call, treating as finish")
                return response.content, "finish", response.content, True
            
            raise ValueError("No tool call or content in response")
            
        except Exception as e:
            logger.error(f"[ReAct-Native] Error: {e}, falling back to react mode for this call")
            return await self._generate_react(user_query, steps, context, include_finish_check)
    
    def _build_native_prompt(self, query: str, steps: List[ReActStep], context: str, file_guidance: str, include_finish_check: bool) -> str:
        """Build prompt for native tool calling mode."""
        parts = [f"Answer the user's query by calling the appropriate tool.\n"]
        
        if file_guidance:
            parts.append(f"## Available Files\n{file_guidance}\n")
        
        if context:
            parts.append(f"## Previous Steps\n{context}\n")
        
        if include_finish_check and steps:
            latest = steps[-1].observation[:2000]
            parts.append(f"## Latest Result\n{latest}\n")
            parts.append("If this answers the query, respond directly (no tool call). Otherwise, call another tool.\n")
        
        parts.append(f"## User Query\n{query}")
        
        return "\n".join(parts)

    async def _generate_react(self, user_query: str, steps: List[ReActStep], context: str, include_finish_check: bool = False) -> Tuple[str, str, str, bool]:
        """Generate using ReAct text-based prompting."""
        file_guidance = self._build_file_guidance()
        latest_observation = steps[-1].observation if steps else ""

        prompt = _build_thought_action_prompt(
            query=user_query,
            context=context,
            file_guidance=file_guidance,
            include_finish_check=include_finish_check,
            latest_observation=latest_observation
        )

        logger.info(f"[ReAct] LLM type: {type(self.llm).__name__}")
        logger.info(f"[ReAct] Prompt length: {len(prompt)} chars (~{len(prompt)//4} tokens)")
        
        # Check if LLM has an underlying llm (for interceptor)
        underlying_llm = getattr(self.llm, 'llm', self.llm)
        logger.info(f"[ReAct] Underlying LLM: {type(underlying_llm).__name__}")
        if hasattr(underlying_llm, 'model'):
            logger.info(f"[ReAct] Model: {underlying_llm.model}")
        
        # Retry logic for empty responses (up to 3 attempts)
        max_retries = 3
        last_error = None
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"[ReAct] Attempt {attempt}: Invoking LLM...")
                response = await self.llm.ainvoke([HumanMessage(content=prompt)])
                logger.info(f"[ReAct] Attempt {attempt}: Response type: {type(response).__name__}")
                
                response_text = self._extract_response_content(response)
                
                if response_text and response_text.strip():
                    logger.info(f"[ReAct] LLM response received: {len(response_text)} chars (attempt {attempt})")
                    logger.info(f"[ReAct] Full response text:\n{response_text}")
                    
                    # Parse and return
                    try:
                        thought, action, action_input, finish_flag = self._parse_response(response_text.strip())
                        logger.info(f"[ReAct] Successfully parsed - thought: {thought[:50]}..., action: {action}, finish: {finish_flag}, input: {action_input[:50]}...")

                        return thought, action, action_input, finish_flag
                    except ValueError as parse_error:
                        logger.error(f"[ReAct] Parse failed: {parse_error}")
                        logger.error(f"[ReAct] Response that failed parsing:\n{response_text}")
                        raise
                else:
                    logger.warning(f"[ReAct] Empty response on attempt {attempt}/{max_retries}")
                    last_error = "Empty response"
                    
            except Exception as e:
                logger.error(f"[ReAct] LLM invoke error on attempt {attempt}/{max_retries}: {e}")
                import traceback
                logger.error(f"[ReAct] Traceback: {traceback.format_exc()}")
                last_error = str(e)
        
        # All retries failed
        logger.error(f"[ReAct] All {max_retries} attempts failed. Last error: {last_error}")
        logger.error(f"[ReAct] Prompt ({len(prompt)} chars): {prompt[:1000]}...")
        logger.error("[ReAct] Check: 1) Is Ollama running? 2) Is model loaded? 3) Check data/scratch/prompts.log")
        raise ValueError(f"LLM failed after {max_retries} attempts - {last_error}")

    def _extract_response_content(self, response) -> str:
        """Extract text content from LLM response object."""
        # Log ALL attributes for debugging
        logger.info(f"[ReAct] === EXTRACTING RESPONSE CONTENT ===")
        logger.info(f"[ReAct] Response type: {type(response)}")
        
        # Check each possible attribute
        has_content = hasattr(response, 'content') and response.content
        has_tool_calls = hasattr(response, 'tool_calls') and response.tool_calls
        has_additional = hasattr(response, 'additional_kwargs') and response.additional_kwargs
        
        logger.info(f"[ReAct] Has content: {has_content} (value: {repr(response.content)[:100] if hasattr(response, 'content') else 'N/A'})")
        logger.info(f"[ReAct] Has tool_calls: {has_tool_calls} (value: {response.tool_calls if hasattr(response, 'tool_calls') else 'N/A'})")
        logger.info(f"[ReAct] Has additional_kwargs: {has_additional}")
        
        # PRIORITY: Check for tool_calls first - Ollama returns function calls here
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_calls = response.tool_calls
            logger.info(f"[ReAct] Found tool_calls: {tool_calls}")
            logger.info(f"[ReAct] tool_calls type: {type(tool_calls)}, first item type: {type(tool_calls[0]) if tool_calls else 'empty'}")

            # Convert tool call to THOUGHT/ACTION/ACTION INPUT format
            if isinstance(tool_calls, list) and len(tool_calls) > 0:
                tool_call = tool_calls[0]

                # Handle both dict-like and object-like tool calls
                if isinstance(tool_call, dict):
                    action = tool_call.get('name', 'unknown')
                    args = tool_call.get('args', {})
                else:
                    # Object with attributes (LangChain ToolCall)
                    action = getattr(tool_call, 'name', None) or getattr(tool_call, 'function', {}).get('name', 'unknown')
                    args = getattr(tool_call, 'args', None) or getattr(tool_call, 'function', {}).get('arguments', {})
                    logger.info(f"[ReAct] Extracted from object - action: {action}, args: {args}")

                # Build action input from args
                if isinstance(args, dict):
                    # For web_search, use the query
                    if 'query' in args:
                        action_input = args['query']
                    else:
                        # Stringify all args
                        action_input = ', '.join(f"{k}={v}" for k, v in args.items())
                elif isinstance(args, str):
                    # Args might be a JSON string
                    try:
                        import json
                        parsed_args = json.loads(args)
                        if isinstance(parsed_args, dict) and 'query' in parsed_args:
                            action_input = parsed_args['query']
                        else:
                            action_input = args
                    except:
                        action_input = args
                else:
                    action_input = str(args)

                # Construct the expected format
                formatted = f"THOUGHT: Using {action} tool to answer the query.\nACTION: {action}\nACTION INPUT: {action_input}"
                logger.info(f"[ReAct] Converted tool_call to format: {formatted}")
                return formatted
        
        # Try direct content attribute
        if hasattr(response, 'content') and response.content:
            logger.debug(f"[ReAct] Found content attribute: {len(response.content)} chars")
            return response.content
        
        # Try text attribute
        if hasattr(response, 'text') and response.text:
            logger.debug(f"[ReAct] Found text attribute: {len(response.text)} chars")
            return response.text
        
        # Check additional_kwargs - Ollama sometimes puts response here
        if hasattr(response, 'additional_kwargs') and response.additional_kwargs:
            kwargs = response.additional_kwargs
            logger.info(f"[ReAct] Checking additional_kwargs: {kwargs}")
            
            # Common keys where content might be
            for key in ['content', 'text', 'message', 'response', 'output', 'result']:
                if key in kwargs and kwargs[key]:
                    logger.info(f"[ReAct] Found content in additional_kwargs['{key}']")
                    return str(kwargs[key])
            
            # If additional_kwargs has tool_calls, extract from there
            if 'tool_calls' in kwargs:
                tool_calls = kwargs['tool_calls']
                logger.info(f"[ReAct] Found tool_calls in additional_kwargs: {tool_calls}")
                # Tool calls might contain the response
                if isinstance(tool_calls, list) and tool_calls:
                    return str(tool_calls[0])
            
            # Last resort: stringify the whole additional_kwargs if it has content
            if kwargs:
                kwargs_str = str(kwargs)
                if len(kwargs_str) > 10:  # Has meaningful content
                    logger.warning(f"[ReAct] Using stringified additional_kwargs as content")
                    return kwargs_str
        
        # Check response_metadata - Ollama puts message here
        if hasattr(response, 'response_metadata') and response.response_metadata:
            meta = response.response_metadata
            logger.info(f"[ReAct] Checking response_metadata keys: {meta.keys() if isinstance(meta, dict) else type(meta)}")
            logger.info(f"[ReAct] Full response_metadata: {meta}")
            
            # Try message.content
            if 'message' in meta:
                msg = meta['message']
                logger.info(f"[ReAct] Found message in metadata: {msg}")
                if isinstance(msg, dict) and 'content' in msg and msg['content']:
                    return msg['content']
            
            # Try direct content in metadata
            if 'content' in meta and meta['content']:
                return meta['content']
            
            # Try response field
            if 'response' in meta and meta['response']:
                return meta['response']
        
        # Fallback: convert to string and try to extract
        response_str = str(response)
        logger.debug(f"[ReAct] Fallback to string: {response_str[:500]}")
        
        # Try to extract content from string representation
        import re
        # Handle: content='...' or AIMessage(content='...')
        match = re.search(r"content=['\"](.+?)['\"]", response_str, re.DOTALL)
        if match:
            return match.group(1)
        
        # Handle: AIMessage(content=..., where content might not be quoted
        match = re.search(r"content=([^,\)]+)", response_str)
        if match:
            content = match.group(1).strip().strip("'\"")
            if content and content != 'None' and content != '':
                return content
        
        logger.error(f"[ReAct] Could not extract content from response!")
        logger.error(f"[ReAct] Response __dict__: {getattr(response, '__dict__', 'N/A')}")
        return ""

    def _parse_response(self, response: str) -> Tuple[str, str, str, bool]:
        """Parse LLM response for THOUGHT, ACTION, ACTION INPUT, and FINISH flag.
        
        Handles various LLM output formats including:
        - THOUGHT: / Thought: / thought:
        - ACTION: / Action: / action:
        - ACTION INPUT: / Action Input: / ACTION_INPUT: / Input:
        - Multi-line content
        - Various whitespace patterns
        """
        logger.debug(f"[ReAct] Parsing response ({len(response)} chars): {response[:300]}...")
        
        if not response or not response.strip():
            logger.error("[ReAct] Empty response from LLM")
            raise ValueError("Empty response from LLM")
        
        # Normalize: convert various action input formats to consistent marker
        normalized = response
        # Handle: "ACTION INPUT:", "Action Input:", "ACTION_INPUT:", "action input:"
        normalized = re.sub(r'action[\s_]*input\s*:', 'ACTION_INPUT:', normalized, flags=re.IGNORECASE)
        
        # Strategy 1: Try structured extraction with flexible patterns
        thought = action = action_input = None
        finish_flag = False

        # FINISH: Parse bool flag if present
        finish_match = re.search(r'finish\s*:\s*(true|false|yes|no|1|0)', normalized, re.IGNORECASE)
        if finish_match:
            finish_value = finish_match.group(1).lower()
            finish_flag = finish_value in ['true', 'yes', '1']
        
        # THOUGHT: Extract everything between THOUGHT: and ACTION: (case insensitive)
        thought_patterns = [
            r'thought\s*:\s*(.+?)(?=\s*action\s*:)',  # THOUGHT: ... ACTION:
            r'thought\s*:\s*(.+?)$',  # THOUGHT: ... (at end)
        ]
        for pattern in thought_patterns:
            match = re.search(pattern, normalized, re.IGNORECASE | re.DOTALL)
            if match:
                thought = match.group(1).strip()
                break
        
        # ACTION: Extract the tool name (word after ACTION:, but not ACTION_INPUT)
        action_patterns = [
            r'(?<!_)action\s*:\s*(\w+)',  # ACTION: tool_name (not ACTION_INPUT)
        ]
        for pattern in action_patterns:
            match = re.search(pattern, normalized, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip().lower()
                # Make sure we didn't accidentally match "input" from "ACTION INPUT"
                if candidate != 'input':
                    action = candidate
                    break
        
        # ACTION_INPUT: Extract everything after ACTION_INPUT: to end or next section
        input_patterns = [
            r'action_input\s*:\s*(.+?)(?=\s*thought\s*:|$)',  # Until next THOUGHT or end
            r'action_input\s*:\s*(.+)$',  # Until end
            r'input\s*:\s*(.+?)(?=\s*thought\s*:|$)',  # Fallback: INPUT:
            r'input\s*:\s*(.+)$',  # INPUT: until end
        ]
        for pattern in input_patterns:
            match = re.search(pattern, normalized, re.IGNORECASE | re.DOTALL)
            if match:
                action_input = match.group(1).strip()
                break
        
        # Strategy 2: If structured parsing failed, try line-by-line parsing
        if not thought or not action or not action_input:
            logger.debug("[ReAct] Structured parsing incomplete, trying line-by-line")
            lines = normalized.split('\n')
            current_section = None
            sections = {'thought': [], 'action': [], 'input': []}
            
            for line in lines:
                line_lower = line.lower().strip()
                if line_lower.startswith('thought'):
                    current_section = 'thought'
                    content = re.sub(r'^thought\s*:\s*', '', line, flags=re.IGNORECASE).strip()
                    if content:
                        sections['thought'].append(content)
                elif re.match(r'^action\s*:', line_lower) and 'input' not in line_lower:
                    current_section = 'action'
                    content = re.sub(r'^action\s*:\s*', '', line, flags=re.IGNORECASE).strip()
                    if content:
                        sections['action'].append(content)
                elif 'input' in line_lower and ':' in line:
                    current_section = 'input'
                    content = re.sub(r'^.*input\s*:\s*', '', line, flags=re.IGNORECASE).strip()
                    if content:
                        sections['input'].append(content)
                elif current_section and line.strip():
                    sections[current_section].append(line.strip())
            
            if not thought and sections['thought']:
                thought = ' '.join(sections['thought'])
            if not action and sections['action']:
                action = sections['action'][0].split()[0].lower()  # First word only
            if not action_input and sections['input']:
                action_input = ' '.join(sections['input'])
        
        # Validate we have all parts
        if not thought:
            logger.warning(f"[ReAct] Could not extract THOUGHT from: {response[:300]}")
            raise ValueError("Could not parse THOUGHT from LLM response")
        if action and action.lower() == 'finish':
            finish_flag = True

        if finish_flag and (not action or action.lower() == 'none'):
            action = ToolName.FINISH.value

        if not action:
            logger.warning(f"[ReAct] Could not extract ACTION from: {response[:300]}")
            raise ValueError("Could not parse ACTION from LLM response")

        # ACTION INPUT is optional for finish when finish_flag is True
        if not action_input:
            if finish_flag or action == 'finish':
                logger.info("[ReAct] Finish indicated, using empty ACTION INPUT")
                action_input = ""  # Allow empty input for finish
            else:
                logger.warning(f"[ReAct] Could not extract ACTION INPUT from: {response[:300]}")
                raise ValueError("Could not parse ACTION INPUT from LLM response")
        
        # Validate action is a known tool
        if action not in self.VALID_ACTIONS:
            logger.warning(f"[ReAct] Unknown action '{action}'. Valid: {self.VALID_ACTIONS}")
            raise ValueError(f"Unsupported action requested: {action}")
        
        logger.debug(f"[ReAct] Parsed - Thought: {thought[:50]}... Action: {action}, Finish: {finish_flag}, Input: {action_input[:50]}...")
        return thought, action, action_input, finish_flag

    def _build_file_guidance(self) -> str:
        """Build file guidance showing original filenames (without temp prefix)."""
        if not self.file_paths:
            return ""

        # Import here to avoid circular dependency
        from backend.runtime import extract_original_filename

        # Extract original filenames
        original_names = [extract_original_filename(fp) for fp in self.file_paths]
        filenames_list = ", ".join(original_names)

        return f"\n\nAttached files:\n- Files available: {filenames_list}"


class AnswerGenerator:
    """Generates final answer."""
    def __init__(self, llm):
        self.llm = llm
        self.formatter = ContextFormatter()

    async def generate(self, user_query: str, steps: List[ReActStep]) -> str:
        context = self.formatter.format(steps)
        prompt = _build_final_answer_prompt(query=user_query, context=context)
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            content = self._extract_response_content(response)
            
            if content and content.strip():
                return content.strip()
            
            # Fallback: synthesize from observations if LLM returns empty
            logger.warning("[AnswerGenerator] LLM returned empty content, using fallback synthesis")
            return self._fallback_synthesis(user_query, steps)
            
        except Exception as e:
            logger.error(f"[AnswerGenerator] Error generating answer: {e}")
            return self._fallback_synthesis(user_query, steps)

    def _extract_response_content(self, response) -> str:
        """Extract text content from LLM response object (handles various formats)."""
        # Direct content attribute
        if hasattr(response, 'content') and response.content:
            return response.content
        
        # Text attribute
        if hasattr(response, 'text') and response.text:
            return response.text
        
        # Check additional_kwargs
        if hasattr(response, 'additional_kwargs') and response.additional_kwargs:
            kwargs = response.additional_kwargs
            for key in ['content', 'text', 'message', 'response', 'output']:
                if key in kwargs and kwargs[key]:
                    return str(kwargs[key])
        
        # Check response_metadata
        if hasattr(response, 'response_metadata') and response.response_metadata:
            meta = response.response_metadata
            if isinstance(meta, dict):
                if 'message' in meta and isinstance(meta['message'], dict):
                    if 'content' in meta['message'] and meta['message']['content']:
                        return meta['message']['content']
                if 'content' in meta and meta['content']:
                    return meta['content']
        
        # Fallback to string
        return str(response) if response else ""

    def _fallback_synthesis(self, user_query: str, steps: List[ReActStep]) -> str:
        """Create a fallback answer from the observations when LLM fails."""
        if not steps:
            return "I was unable to generate a response. Please try again."
        
        # Collect all observations
        observations = []
        for step in steps:
            if step.observation and step.observation != "Task completed":
                observations.append(step.observation)
        
        if not observations:
            return "The task was completed but no detailed observations were recorded."
        
        # Return a summary of observations
        return f"Based on the analysis:\n\n" + "\n\n".join(observations)


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
        plan_context: Optional[dict] = None,
        current_thought: Optional[str] = None
    ) -> str:
        observation = ""
        try:
            if action == ToolName.WEB_SEARCH:
                observation = await self._execute_web_search(action_input)
            elif action == ToolName.RAG_RETRIEVAL:
                observation = await self._execute_rag(action_input)
            elif action == ToolName.PYTHON_CODER:
                observation = await self._execute_python(action_input, file_paths, session_id, steps, plan_context, current_thought, action)
            elif action == ToolName.SHELL:
                observation = await self._execute_shell(action_input, session_id)
            elif action == ToolName.FILE_ANALYZER:
                observation = await self._execute_file_analysis(action_input, file_paths)
            elif action == ToolName.VISION_ANALYZER:
                observation = await self._execute_vision(action_input, file_paths)
            elif action == ToolName.NO_TOOLS:
                observation = await self._execute_no_tools(action_input, steps)
            else:
                observation = "Invalid action."
        except Exception as e:
            logger.error(f"Tool execution error ({action}): {e}")
            observation = f"Error executing action: {str(e)}"
        finally:
            self._log_tool_output(action, action_input, observation)

        return observation

    async def _execute_web_search(self, query: str) -> str:
        results, _ = await web_search_tool.search(query, max_results=5, include_context=True)
        return web_search_tool.format_results(results) or "No web search results found."

    async def _execute_rag(self, query: str) -> str:
        results = await rag_retriever_tool.retrieve(query, top_k=5)
        return rag_retriever_tool.format_results(results) or "No relevant documents found."

    async def _execute_python(
        self,
        task_description: str,
        file_paths: List[str],
        session_id: str,
        steps: List[ReActStep],
        plan_context: dict,
        current_thought: Optional[str] = None,
        current_action: Optional[str] = None
    ) -> str:
        """Execute Python code task with LLM-based code generation."""
        current_step_num = len(steps) + 1 if steps else 1

        # Build conversation history from session
        conversation_history = None
        if session_id:
            from backend.storage.conversation_store import conversation_store
            messages = conversation_store.get_messages(session_id, limit=10)
            conversation_history = [{"role": msg.role, "content": msg.content} for msg in messages]

        # Build react context from previous steps + current step
        react_context = None
        if steps or current_thought:
            react_context = {
                "previous_steps": [
                    {
                        "step": s.step_num,
                        "action": s.action,
                        "observation": s.observation[:500] if s.observation else ""
                    }
                    for s in (steps[-3:] if steps else [])  # Last 3 steps
                ],
                "total_steps": len(steps) if steps else 0,
                "current_step": {
                    "thought": current_thought or "",
                    "action": str(current_action) if current_action else "python_coder",
                    "action_input": task_description
                }
            }

        # Call the updated python_coder tool with task description
        tool_result = await python_coder_tool.execute(
            query=task_description,
            context=None,
            file_paths=file_paths,
            session_id=session_id,
            stage_prefix=f"step{current_step_num}",
            conversation_history=conversation_history,
            plan_context=plan_context,
            react_context=react_context
        )

        # Format observation from ToolResult
        if tool_result.success:
            result = tool_result.output
            observation_text = ""
            output = ""
            created_files = []
            attempts = 1

            if isinstance(result, dict):
                observation_text = result.get('observation', '') or ''
                output = result.get('output', '') or ''
                created_files = result.get('created_files', []) or []
                attempts = result.get('attempts', 1) or 1
            else:
                # Gracefully handle unexpected shapes by stringifying
                output = result if isinstance(result, str) else str(result)

            parts = []
            if observation_text:
                parts.append(observation_text)

            parts.append(f"Code generation and execution successful (attempt {attempts}/{settings.python_code_max_iterations}).")

            if output:
                parts.append(f"Output:\n{output}")

            if created_files:
                parts.append(f"Created files: {', '.join(created_files)}")

            return "\n\n".join(parts).strip()
        else:
            error_msg = tool_result.error or "Unknown error"
            attempt_history = (tool_result.metadata or {}).get('attempt_history', [])

            response = f"Code generation/execution failed: {error_msg}"
            if attempt_history:
                response += f"\n\nAttempted {len(attempt_history)} time(s). Last attempt error: {attempt_history[-1].get('error', 'Unknown')}"
            return response

    async def _execute_shell(self, command: str, session_id: Optional[str]) -> str:
        """Execute safe shell command inside the sandbox working directory."""
        if isinstance(command, dict):
            command = command.get("command") or command.get("query") or ""

        tool_result = await shell_tool.execute(command, session_id=session_id)
        if tool_result.success:
            return tool_result.output or "(no output)"
        return f"Shell command failed: {tool_result.error or 'Unknown error'}"

    async def _execute_file_analysis(self, query: str, file_paths: List[str]) -> str:
        if not file_paths:
            return "No files attached."

        result = file_analyzer.analyze(file_paths=file_paths, user_query=query)
        if not result.get("success"):
            return f"Analysis failed: {result.get('error')}"

        summary = result.get("summary") or ""
        access_help = self._build_file_access_help(result, file_paths)
        llm_context = result.get("llm_context") or result.get("context") or ""

        sections = [section for section in (summary, access_help, llm_context) if section]
        return "\n\n".join(sections) if sections else "File analysis completed but no details were returned."

    async def _execute_vision(self, query: str, file_paths: List[str]) -> str:
        if not file_paths: return "No files attached."
        result = await vision_analyzer_tool(query=query, file_paths=file_paths, user_id=self.user_id)
        return result.get('analysis') if result.get('success') else f"Vision failed: {result.get('error')}"

    async def _execute_no_tools(self, reasoning: str, steps: Optional[List[ReActStep]] = None) -> str:
        """
        Execute no_tools action - just return the LLM's reasoning without using external tools.
        This allows the agent to think and synthesize information from previous steps.
        """
        logger.info(f"[NO_TOOLS] LLM reasoning: {reasoning[:200]}...")

        # Format the observation to show this was a thinking step
        observation = f"Reasoning: {reasoning}"

        # Optionally, you could enhance this with context from previous steps
        if steps and len(steps) > 0:
            observation += f"\n(Based on {len(steps)} previous step(s))"

        return observation

    def _log_tool_output(self, action: str, action_input: str, observation: str):
        """Write tool inputs/outputs to prompts.log via the LLM interceptor when available."""
        log_fn = getattr(self.llm, "log_tool_output", None)
        if not callable(log_fn):
            return

        tool_name = action.value if isinstance(action, Enum) else str(action)
        try:
            log_fn(tool_name, action_input or "", observation or "")
        except Exception as e:
            logger.warning(f"[ToolExecutor] Failed to log tool output for {tool_name}: {e}")

    def _build_file_access_help(self, result: Dict[str, Any], file_paths: List[str]) -> str:
        """
        Provide concrete guidance on how to read the analyzed files.
        Uses whichever structure the analyzer returns (analyses/results).
        """
        analyses = result.get("analyses") or result.get("results") or []
        if not analyses:
            return ""

        lines = [
            "How to access the attached files:",
            "- Files are already present in the sandbox; read them by path without downloading again.",
        ]

        for item in analyses:
            if not isinstance(item, dict) or not item.get("success", True):
                continue

            metadata = item.get("metadata") or {}
            path = item.get("file_path") or item.get("full_path") or metadata.get("file_path")
            name = item.get("original_name") or item.get("file") or path or "file"
            file_type = (metadata.get("file_type")
                         or item.get("format")
                         or item.get("extension")
                         or "file").lower()

            hint = self._reader_hint(file_type)
            target = path or name
            lines.append(f"- {target}: {hint}")

        return "\n".join(lines)

    def _reader_hint(self, file_type: str) -> str:
        """Return a short, direct instruction for loading the file type."""
        ft = file_type.lower()
        if ft in ("csv",):
            return "load with pandas.read_csv(path) and remember to set encoding if needed."
        if ft in ("excel", "xls", "xlsx"):
            return "use pandas.read_excel(path); specify sheet_name when multiple sheets exist."
        if ft == "json":
            return "open with json.load(open(path, 'r', encoding='utf-8')) or pandas.read_json(path) for records."
        if ft in ("parquet",):
            return "read via pandas.read_parquet(path)."
        if ft in ("pdf",):
            return "extract text with pdfplumber or PyPDF2 using the provided path."
        if ft in ("docx", "word"):
            return "load with python-docx (Document(path)) to read paragraphs and tables."
        if ft in ("image", "png", "jpg", "jpeg", "gif", "webp"):
            return "open with PIL.Image.open(path) or cv2.imread(path)."
        if ft in ("text", "txt", "md"):
            return "read with open(path, 'r', encoding='utf-8') for plain text."
        return "use standard file IO (open(path, 'rb' or 'r')) appropriate to the format."



# ============================================================================
# Planning Engine
# ============================================================================

class PlanExecutor:
    """Executes structured plans with per-step ReAct loops."""
    def __init__(self, tool_executor, llm):
        self.tool_executor = tool_executor
        self.llm = llm

    async def execute_plan(
        self,
        plan_steps: List[PlanStep],
        user_query: str,
        file_paths: List[str],
        session_id: str,
        max_iterations_per_step: int = 3,
    ) -> Tuple[str, List[StepResult]]:
        results = []
        accumulated_obs = []
        react_steps_history = []
        
        for i, plan_step in enumerate(plan_steps):
            logger.info(f"[PlanExecutor] Step {plan_step.step_num}: {plan_step.goal}")
            
            step_result, new_react_steps = await self._execute_step_with_retry(
                plan_step,
                user_query,
                accumulated_obs,
                file_paths,
                session_id,
                plan_steps,
                results,
                react_steps_history,
                max_iterations_per_step,
            )
            
            results.append(step_result)
            react_steps_history.extend(new_react_steps)
            
            if step_result.success and step_result.observation:
                accumulated_obs.append(f"Step {plan_step.step_num}: {step_result.observation}")

            if not step_result.success and i < len(plan_steps) - 1:
                logger.warning(f"Step {plan_step.step_num} failed. Continuing with best effort.")

        final_answer = await self._generate_final_answer(user_query, plan_steps, results, accumulated_obs, react_steps_history)
        return final_answer, results

    async def _execute_step_with_retry(
        self,
        plan_step: PlanStep,
        user_query: str,
        accumulated_obs: List[str], 
        file_paths: List[str],
        session_id: str,
        all_plan_steps: List[PlanStep],
        step_results: List[StepResult],
        react_steps_history: List[ReActStep],
        max_iterations_per_step: int,
    ) -> Tuple[StepResult, List[ReActStep]]:
        
        allowed_tools = self._resolve_allowed_tools(plan_step.primary_tools)
        current_react_steps: List[ReActStep] = []
        last_error = None

        for iteration in range(1, max_iterations_per_step + 1):
            try:
                plan_context = {
                    "current_step": plan_step.step_num,
                    "total_steps": len(all_plan_steps),
                    "goal": plan_step.goal,
                    "success_criteria": plan_step.success_criteria,
                    "primary_tools": plan_step.primary_tools,
                    "overall_goal": user_query,
                    "all_plan_steps": [
                        {
                            "step_num": ps.step_num,
                            "goal": ps.goal,
                            "primary_tools": ps.primary_tools,
                            "success_criteria": ps.success_criteria
                        }
                        for ps in all_plan_steps
                    ],
                    "previous_results": [
                        {
                            "step": r.step_num,
                            "goal": all_plan_steps[r.step_num - 1].goal if r.step_num <= len(all_plan_steps) else "Unknown",
                            "success": r.success,
                            "observation": r.observation[:300] if r.observation else "No observation",
                            "tool_used": r.tool_used
                        }
                        for r in step_results
                    ],
                }

                plan_prompt = self._format_plan_query(user_query, plan_step, all_plan_steps)
                context = self._format_plan_context(
                    plan_step,
                    accumulated_obs,
                    react_steps_history + current_react_steps,
                    all_plan_steps,
                )

                thought_generator = ThoughtActionGenerator(self.llm, file_paths)
                thought, action, action_input, finish_flag = await thought_generator.generate(
                    plan_prompt,
                    react_steps_history + current_react_steps,
                    context,
                    include_finish_check=(iteration > 1),
                )

                step = ReActStep(len(react_steps_history) + len(current_react_steps) + 1)
                step.thought = thought
                step.action_input = action_input
                step.finish = bool(finish_flag or action in (ToolName.FINISH, "finish"))

                if step.finish:
                    step.action = ToolName.FINISH
                    step.observation = thought or f"Step {plan_step.step_num} marked complete."
                    current_react_steps.append(step)
                    return StepResult(
                        step_num=plan_step.step_num,
                        goal=plan_step.goal,
                        success=True,
                        tool_used="finish",
                        attempts=iteration,
                        observation=step.observation,
                        error=None,
                        metadata={"react_trace": self._serialize_trace(current_react_steps)},
                    ), current_react_steps

                resolved_action = self._resolve_action(action, allowed_tools)
                step.action = resolved_action

                if not step.action_input:
                    step.action_input = f"Task: {plan_step.goal}\nUser Query: {user_query}"

                observation = await self.tool_executor.execute(
                    step.action,
                    step.action_input,
                    file_paths,
                    session_id,
                    steps=react_steps_history + current_react_steps,
                    plan_context=plan_context,
                    current_thought=thought
                )
                step.observation = observation
                # Allow observation to signal completion (e.g., python_coder returning FINISH)
                if observation_signals_finish(step.observation):
                    step.finish = True
                    current_react_steps.append(step)
                    return StepResult(
                        step_num=plan_step.step_num,
                        goal=plan_step.goal,
                        success=True,
                        tool_used=step.action.value if isinstance(step.action, Enum) else str(step.action),
                        attempts=iteration,
                        observation=step.observation,
                        error=None,
                        metadata={"react_trace": self._serialize_trace(current_react_steps), "finish": True},
                    ), current_react_steps
                current_react_steps.append(step)

                # Check for explicit errors - if found, store for potential retry
                has_error = "error" in observation.lower() or "failed" in observation.lower()
                if has_error:
                    last_error = observation
                    # Continue to next iteration to allow retry
                    logger.warning(f"[PlanExecutor] Step {plan_step.step_num} iteration {iteration}: Error detected in observation, will retry")
                else:
                    # Observation is clean - continue to next iteration to let LLM decide if more work needed
                    # The LLM will set FINISH: true when truly done
                    logger.info(f"[PlanExecutor] Step {plan_step.step_num} iteration {iteration}: Clean observation, continuing to next iteration for LLM decision")
                    last_error = None  # Clear error since this iteration succeeded

            except Exception as e:
                last_error = str(e)
                logger.error(f"Step execution error: {e}")

        # Max iterations reached without explicit FINISH
        # Determine success based on whether we have useful observations and no errors
        has_useful_output = bool(current_react_steps and current_react_steps[-1].observation and not last_error)
        final_observation = current_react_steps[-1].observation if current_react_steps else "No iterations completed"

        if has_useful_output:
            # We completed max iterations with clean output - consider it successful
            logger.info(f"[PlanExecutor] Step {plan_step.step_num} completed {max_iterations_per_step} iterations with clean output (no explicit FINISH)")
            return StepResult(
                step_num=plan_step.step_num,
                goal=plan_step.goal,
                success=True,
                tool_used=current_react_steps[-1].action.value if current_react_steps and isinstance(current_react_steps[-1].action, Enum) else (allowed_tools[0].value if allowed_tools else None),
                attempts=max_iterations_per_step,
                observation=final_observation,
                error=None,
                metadata={"react_trace": self._serialize_trace(current_react_steps), "reason": "max_iterations_reached"},
            ), current_react_steps
        else:
            # Failed - either had errors or no useful output
            logger.warning(f"[PlanExecutor] Step {plan_step.step_num} failed after {max_iterations_per_step} iterations")
            return StepResult(
                step_num=plan_step.step_num,
                goal=plan_step.goal,
                success=False,
                tool_used=allowed_tools[0].value if allowed_tools else None,
                attempts=max_iterations_per_step,
                observation=last_error or final_observation or "Max iterations reached without success",
                error=last_error,
                metadata={"react_trace": self._serialize_trace(current_react_steps)},
            ), current_react_steps

    def _format_plan_query(self, user_query: str, plan_step: PlanStep, all_plan_steps: List[PlanStep]) -> str:
        overview = self._plan_overview(all_plan_steps)
        return (
            f"Overall goal: {user_query}\n"
            f"Current plan step ({plan_step.step_num}/{len(all_plan_steps)}): {plan_step.goal}\n"
            f"Success criteria: {plan_step.success_criteria}\n"
            f"Tools: {', '.join(plan_step.primary_tools) if plan_step.primary_tools else 'any standard tool'}\n"
            # f"Full plan:\n{overview}"
        )

    def _format_plan_context(
        self,
        plan_step: PlanStep,
        accumulated_obs: List[str],
        react_steps: List[ReActStep],
        all_plan_steps: List[PlanStep],
    ) -> str:
        recent_success = "\n".join(accumulated_obs[-3:]) if accumulated_obs else "None"
        plan_outline = self._plan_overview(all_plan_steps)
        formatter = ContextFormatter()
        react_context = formatter.format(react_steps) if react_steps else "No ReAct steps yet for this subgoal."
        return (
            f"Prior plan observations (success only):\n{recent_success}\n\n"
            f"Original plan overview:\n{plan_outline}\n\n"
            f"Current step: \n{plan_step.step_num}/{len(all_plan_steps)} -> {plan_step.goal}\n\n"
            f"Success criteria for this step: {plan_step.success_criteria or 'Not specified'}\n"
            f"Tools for this step: {', '.join(plan_step.primary_tools) if plan_step.primary_tools else 'any standard tool'}\n"
            f"ReAct trace so far for this subgoal:\n{react_context}"
        )

    def _resolve_allowed_tools(self, primary_tools: List[str]) -> List[ToolName]:
        tool_map = {
            "python_code": ToolName.PYTHON_CODER,
            "python_coder": ToolName.PYTHON_CODER,
            "web_search": ToolName.WEB_SEARCH,
            "rag": ToolName.RAG_RETRIEVAL,
            "rag_retrieval": ToolName.RAG_RETRIEVAL,
            "vision_analyzer": ToolName.VISION_ANALYZER,
            "shell": ToolName.SHELL,
            "no_tools": ToolName.NO_TOOLS,
            "file_analyzer": ToolName.FILE_ANALYZER,
        }
        enums = []
        for t in primary_tools:
            mapped = tool_map.get(t)
            if mapped:
                enums.append(mapped)
        return enums

    def _resolve_action(self, action: str, allowed_tools: List[ToolName]) -> ToolName:
        # Normalize to ToolName
        if isinstance(action, ToolName):
            resolved = action
        else:
            try:
                resolved = ToolName(action)
            except Exception:
                tool_map = {
                    "python_code": ToolName.PYTHON_CODER,
                    "python_coder": ToolName.PYTHON_CODER,
                    "web_search": ToolName.WEB_SEARCH,
                    "rag": ToolName.RAG_RETRIEVAL,
                    "rag_retrieval": ToolName.RAG_RETRIEVAL,
                    "vision_analyzer": ToolName.VISION_ANALYZER,
                    "shell": ToolName.SHELL,
                    "no_tools": ToolName.NO_TOOLS,
                }
                resolved = tool_map.get(str(action).lower(), ToolName.WEB_SEARCH)

        if allowed_tools and resolved not in allowed_tools:
            logger.info(f"[PlanExecutor] Action {resolved} not in primary tools {allowed_tools}. Enforcing {allowed_tools[0]}.")
            return allowed_tools[0]
        return resolved

    def _plan_overview(self, plan_steps: List[PlanStep]) -> str:
        lines = []
        for step in plan_steps:
            tools = ", ".join(step.primary_tools) if step.primary_tools else "none specified"
            lines.append(f"{step.step_num}. {step.goal} (tools: {tools}, success: {step.success_criteria})")
        return "\n".join(lines)

    def _serialize_trace(self, steps: List[ReActStep]) -> List[Dict[str, Any]]:
        trace = []
        for s in steps:
            item = s.to_dict()
            action_value = s.action.value if isinstance(s.action, Enum) else str(s.action)
            item["action"] = action_value
            trace.append(item)
        return trace

    async def _generate_final_answer(
        self,
        user_query: str,
        plan_steps: List[PlanStep],
        results: List[StepResult],
        obs: List[str],
        react_steps: List[ReActStep],
    ) -> str:
        plan_outline = self._plan_overview(plan_steps)
        execution_summary = "\n".join([f"Step {r.step_num} [{r.goal}]: {r.observation}" for r in results])
        obs_summary = "\n".join(obs[-5:]) if obs else "None"
        react_trace_count = len(react_steps)
        prompt = (
            "You are summarizing a plan-and-execute run.\n\n"
            f"Overall goal:\n{user_query}\n\n"
            f"Plan:\n{plan_outline}\n\n"
            f"Execution observations:\n{execution_summary if execution_summary else 'None'}\n\n"
            f"Recent intermediate observations:\n{obs_summary}\n\n"
            f"ReAct trace steps captured: {react_trace_count}\n\n"
            "Provide a concise answer that references how each plan step contributed. "
            "Call out any failures or missing data explicitly."
        )
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()

    def _build_metadata(self, plan_steps: List[PlanStep], step_results: List[StepResult], react_steps: List[ReActStep] = None) -> Dict[str, Any]:
        """Build metadata for plan execution results."""
        react_trace = react_steps or []
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
                    "observation": result.observation[:500] if result.observation else None,
                    "error": result.error,
                    "react_trace": result.metadata.get("react_trace") if result.metadata else None,
                }
                for result in step_results
            ],
            "react_steps": [
                {
                    "step_num": s.step_num,
                    "action": s.action.value if isinstance(s.action, Enum) else str(s.action),
                    "thought": s.thought[:300] if s.thought else "",
                    "observation": s.observation[:300] if s.observation else "",
                }
                for s in react_trace
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

        logger.info(f"[ReActAgent] Starting execute for query: {user_query[:100]}...")

        # Step 0: Automatically run file_analyzer if files are attached
        if file_paths and len(file_paths) > 0:
            logger.info(f"[ReActAgent] Step 0: Running file_analyzer for {len(file_paths)} file(s)")
            step0 = ReActStep(0)
            step0.thought = "Files are attached. I need to analyze their structure and content first."
            step0.action = ToolName.FILE_ANALYZER
            step0.action_input = f"Analyze the attached files to understand their structure and content"

            try:
                # Run file analyzer
                analysis_result = file_analyzer.analyze(file_paths=file_paths, user_query=user_query, quick_mode=False)

                if analysis_result.get('success'):
                    # Use the LLM-friendly context for the observation
                    step0.observation = analysis_result.get('llm_context', analysis_result.get('summary', 'File analysis completed'))
                    logger.info(f"[ReActAgent] Step 0 completed: {len(step0.observation)} chars of file analysis")
                else:
                    step0.observation = f"File analysis failed: {analysis_result.get('error', 'Unknown error')}"
                    logger.warning(f"[ReActAgent] Step 0 failed: {step0.observation}")

                self.steps.append(step0)

            except Exception as e:
                logger.error(f"[ReActAgent] Step 0 error: {e}")
                step0.observation = f"File analysis error: {str(e)}"
                self.steps.append(step0)

        # Standard ReAct Loop
        for i in range(self.max_iterations):
            step = ReActStep(i + 1)
            context = ContextFormatter().format(self.steps)
            
            logger.info(f"[ReActAgent] === Iteration {i + 1} ===")
            
            # Generate
            try:
                thought, action, action_input, finish_flag = await self.thought_generator.generate(
                    user_query, self.steps, context, include_finish_check=(i > 2)
                )
                logger.info(f"[ReActAgent] Generated - Thought: {thought[:50]}..., Action: {action}, Finish: {finish_flag}, Input: {action_input[:50]}...")
            except ValueError as exc:
                logger.error(f"[ReActAgent] Failed to parse thought/action: {exc}")
                step.observation = f"Failed to parse thought/action: {exc}"
                self.steps.append(step)
                break
            
            step.thought = thought
            step.action_input = action_input
            step.finish = bool(finish_flag or action == ToolName.FINISH or action == "finish")
            
            if step.finish:
                step.action = ToolName.FINISH
                logger.info(f"[ReActAgent] FINISH flag detected. Current steps count: {len(self.steps)}")
                final_answer = await self.answer_generator.generate(user_query, self.steps)
                logger.info(f"[ReActAgent] Generated final answer ({len(final_answer)} chars): {final_answer[:200]}...")
                step.observation = "Task completed"
                self.steps.append(step)
                break
                
            step.action = action
            # Execute
            logger.info(f"[ReActAgent] Executing tool: {action}")
            observation = await self.tool_executor.execute(
                action, action_input, file_paths, session_id, steps=self.steps, current_thought=thought
            )
            step.observation = observation
            # Detect completion signaled inside observation (e.g., from python_coder)
            if observation_signals_finish(step.observation):
                step.finish = True
                logger.info(f"[ReActAgent] FINISH detected in observation at step {step.step_num}.")
                self.steps.append(step)
                final_answer = await self.answer_generator.generate(user_query, self.steps)
                logger.info(f"[ReActAgent] Generated final answer ({len(final_answer)} chars) via observation finish.")
                break

            self.steps.append(step)
            logger.info(f"[ReActAgent] Step {step.step_num} completed. Observation: {observation[:200]}...")

        if not final_answer:
            logger.info(f"[ReActAgent] No final answer yet, generating from {len(self.steps)} steps")
            final_answer = await self.answer_generator.generate(user_query, self.steps)
            logger.info(f"[ReActAgent] Generated final answer ({len(final_answer)} chars)")

        logger.info(f"[ReActAgent] Total steps recorded: {len(self.steps)}")
        for idx, s in enumerate(self.steps):
            logger.info(f"[ReActAgent] Step {idx + 1}: action={s.action}, thought={s.thought[:50]}...")

        # Build Metadata (steps included here for debugging/analysis)
        metadata = {
            "agent_type": "react",
            "steps": [s.to_dict() for s in self.steps],
            "total_steps": len(self.steps)
        }

        # Return only final answer (not the full reasoning process)
        return final_answer, metadata

    async def execute_with_plan(self, plan_steps: List[PlanStep], messages: List[ChatMessage], session_id: str, user_id: str, file_paths: List[str], max_iterations_per_step: int = 3) -> Tuple[str, List[StepResult]]:
        """Execute a pre-defined plan."""
        self.llm_manager.ensure_user_id(user_id)
        self.plan_executor.llm = self.llm_manager.llm
        self.plan_executor.tool_executor.llm = self.llm_manager.llm
        
        user_query = messages[-1].content
        final_answer, step_results = await self.plan_executor.execute_plan(
            plan_steps, user_query, file_paths, session_id, max_iterations_per_step
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
