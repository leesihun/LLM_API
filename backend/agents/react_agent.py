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
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Complete the task and provide the final answer. Only use when you have gathered all necessary information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final answer to the user's question"
                    }
                },
                "required": ["answer"]
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
            session_id=session_id or "session",
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

        # Extract content using the same method as ReAct (handles tool_calls)
        response_content = self._extract_plan_response_content(response)

        # Strip markdown code blocks (```json ... ```)
        response_content = self._strip_markdown_code_blocks(response_content)

        try:
            plan_data = json.loads(response_content.strip())
        except json.JSONDecodeError as exc:
            logger.error(f"[Planner] Failed to parse JSON from response: {response_content[:500]}")
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
        finish_section = f"""
## Latest Observation
{snippet}

## Completion Assessment
Decide if the latest observation fully answers the query.
- If YES → you must choose **finish** and draft the final answer. Do not rerun tools.
- If NO → pick the single tool that will move you closer, but only if it gathers new information."""

    guidance_section = f"\n{file_guidance}\n" if file_guidance else ""

    return f"""You are a focused ReAct agent. Think, pick ONE tool or FINISH, supply its input. If you have enough information, you must finish. Do not call the same tool with the same or trivially modified input unless you clearly need new information and explain what that is.
{guidance_section}

## Context
{context if context else 'No prior steps yet.'}
{finish_section}
## Tools
- web_search → realtime or complex or specific info
- rag_retrieval → very specific infomation.... Choose only when specifically asked
- python_coder → generate and execute Python code based on task description
- shell → run safe shell commands (ls/dir/pwd/cd/cat/head/tail/find/grep/wc/echo) inside the sandbox
- vision_analyzer → answer image questions (only if images attached)
- no_tools → think and reason without external tools (use when no tool fits or need to synthesize)
- finish → only when you already have the final answer

## python_coder Example
ACTION: python_coder
ACTION INPUT: Load the CSV file and print its shape, column names, and first 5 rows

## shell Example
ACTION: shell
ACTION INPUT: ls


## IMPORTANT:
## Response Format: Strictly follow the format.
THOUGHT: Detailed reasoning, future recommendations after the action, including whether you can finish. If you consider repeating a tool, state what new information you need and why the prior result was insufficient.
ACTION: tool name
ACTION INPUT: For web_search and rag_retrieval, write the search query. For python_coder, write a clear task description (what you want the code to do). For vision_analyzer, write the question. For no_tools, write your reasoning. Do not repeat the same tool/input unless justified as above.


## Query
{query}
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

    async def generate(self, user_query: str, steps: List[ReActStep], context: str, include_finish_check: bool = False) -> Tuple[str, str, str]:
        """Generate thought and action using configured mode."""
        if self.mode == 'native':
            return await self._generate_native(user_query, steps, context, include_finish_check)
        else:
            return await self._generate_react(user_query, steps, context, include_finish_check)
    
    async def _generate_native(self, user_query: str, steps: List[ReActStep], context: str, include_finish_check: bool = False) -> Tuple[str, str, str]:
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
                
                # Build action_input based on tool type
                if action == 'python_coder':
                    action_input = args.get('task', args.get('query', ''))
                elif action == 'finish':
                    action_input = args.get('answer', '')
                else:
                    action_input = args.get('query', str(args))
                
                # Generate thought from content or synthesize
                thought = response.content if response.content else f"Using {action} tool to complete the task."
                
                logger.info(f"[ReAct-Native] Tool call: {action}, input length: {len(action_input)}")
                return thought, action, action_input
            
            # No tool call - check if it's a direct response (treat as finish)
            if response.content:
                logger.info(f"[ReAct-Native] No tool call, treating as finish")
                return response.content, "finish", response.content
            
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
            parts.append("If this answers the query, call 'finish'. Otherwise, call another tool.\n")
        
        parts.append(f"## User Query\n{query}")
        
        return "\n".join(parts)

    async def _generate_react(self, user_query: str, steps: List[ReActStep], context: str, include_finish_check: bool = False) -> Tuple[str, str, str]:
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
                        parsed = self._parse_response(response_text.strip())
                        logger.info(f"[ReAct] Successfully parsed - thought: {parsed[0][:50]}..., action: {parsed[1]}, input: {parsed[2][:50]}...")

                        return parsed
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

    def _parse_response(self, response: str) -> Tuple[str, str, str]:
        """Parse LLM response for THOUGHT, ACTION, and ACTION INPUT.
        
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
        if not action:
            logger.warning(f"[ReAct] Could not extract ACTION from: {response[:300]}")
            raise ValueError("Could not parse ACTION from LLM response")

        # ACTION INPUT is optional for "finish" action
        if not action_input:
            if action == 'finish':
                logger.info("[ReAct] ACTION is 'finish', using empty ACTION INPUT")
                action_input = ""  # Allow empty input for finish
            else:
                logger.warning(f"[ReAct] Could not extract ACTION INPUT from: {response[:300]}")
                raise ValueError("Could not parse ACTION INPUT from LLM response")
        
        # Validate action is a known tool
        if action not in self.VALID_ACTIONS:
            logger.warning(f"[ReAct] Unknown action '{action}'. Valid: {self.VALID_ACTIONS}")
            raise ValueError(f"Unsupported action requested: {action}")
        
        logger.debug(f"[ReAct] Parsed - Thought: {thought[:50]}... Action: {action}, Input: {action_input[:50]}...")
        return thought, action, action_input

    def _build_file_guidance(self) -> str:
        """Build file guidance showing original filenames (without temp prefix)."""
        if not self.file_paths:
            return ""

        # Import here to avoid circular dependency
        from backend.runtime import extract_original_filename

        # Extract original filenames
        original_names = [extract_original_filename(fp) for fp in self.file_paths]
        filenames_list = ", ".join(original_names)

        return f"\nGuidelines:\n- Files available: {filenames_list}\n- Attempt local analysis (python_coder) first.\n- Use web_search only if local analysis fails."


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
        plan_context: Optional[dict] = None
    ) -> str:
        try:
            if action == ToolName.WEB_SEARCH:
                return await self._execute_web_search(action_input)
            elif action == ToolName.RAG_RETRIEVAL:
                return await self._execute_rag(action_input)
            elif action == ToolName.PYTHON_CODER:
                return await self._execute_python(action_input, file_paths, session_id, steps, plan_context)
            elif action == ToolName.SHELL:
                return await self._execute_shell(action_input, session_id)
            elif action == ToolName.FILE_ANALYZER:
                return await self._execute_file_analysis(action_input, file_paths)
            elif action == ToolName.VISION_ANALYZER:
                return await self._execute_vision(action_input, file_paths)
            elif action == ToolName.NO_TOOLS:
                return await self._execute_no_tools(action_input, steps)
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

    async def _execute_python(self, task_description: str, file_paths: List[str], session_id: str, steps: List[ReActStep], plan_context: dict) -> str:
        """Execute Python code task with LLM-based code generation."""
        current_step_num = len(steps) + 1 if steps else 1

        # Build conversation history from session
        conversation_history = None
        if session_id:
            from backend.storage.conversation_store import conversation_store
            messages = conversation_store.get_messages(session_id, limit=10)
            conversation_history = [{"role": msg.role, "content": msg.content} for msg in messages]

        # Build react context from previous steps
        react_context = None
        if steps:
            react_context = {
                "previous_steps": [
                    {
                        "step": s.step_num,
                        "action": s.action,
                        "observation": s.observation[:500] if s.observation else ""
                    }
                    for s in steps[-3:]  # Last 3 steps
                ],
                "total_steps": len(steps)
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
            output = result.get('output', '')
            created_files = result.get('created_files', [])
            attempts = result.get('attempts', 1)

            response = f"Code generation and execution successful (attempt {attempts}/{settings.python_code_max_iterations}):\n{output}"
            if created_files:
                response += f"\n\nCreated files: {', '.join(created_files)}"
            return response
        else:
            error_msg = tool_result.error or "Unknown error"
            attempt_history = tool_result.metadata.get('attempt_history', [])

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
        if not file_paths: return "No files attached."
        result = file_analyzer.analyze(file_paths=file_paths, user_query=query)
        return result.get('summary') if result.get('success') else f"Analysis failed: {result.get('error')}"

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
        tool_map = {
            "python_code": ToolName.PYTHON_CODER,
            "python_coder": ToolName.PYTHON_CODER,
            "web_search": ToolName.WEB_SEARCH,
            "rag_retrieval": ToolName.RAG_RETRIEVAL,
            "vision_analyzer": ToolName.VISION_ANALYZER,
            "shell": ToolName.SHELL,
            "no_tools": ToolName.NO_TOOLS
        }
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
                thought, action, action_input = await self.thought_generator.generate(
                    user_query, self.steps, context, include_finish_check=(i > 2)
                )
                logger.info(f"[ReActAgent] Generated - Thought: {thought[:50]}..., Action: {action}, Input: {action_input[:50]}...")
            except ValueError as exc:
                logger.error(f"[ReActAgent] Failed to parse thought/action: {exc}")
                step.observation = f"Failed to parse thought/action: {exc}"
                self.steps.append(step)
                break
            
            step.thought = thought
            step.action = action
            step.action_input = action_input
            
            if action == ToolName.FINISH or action == "finish":
                logger.info(f"[ReActAgent] FINISH action detected. Current steps count: {len(self.steps)}")
                final_answer = await self.answer_generator.generate(user_query, self.steps)
                logger.info(f"[ReActAgent] Generated final answer ({len(final_answer)} chars): {final_answer[:200]}...")
                step.observation = "Task completed"
                self.steps.append(step)
                break
                
            # Execute
            logger.info(f"[ReActAgent] Executing tool: {action}")
            observation = await self.tool_executor.execute(
                action, action_input, file_paths, session_id, steps=self.steps
            )
            step.observation = observation
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
