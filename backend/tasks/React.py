"""
ReAct Agent Implementation
Implements the Reasoning + Acting pattern with iterative Thought-Action-Observation loops
"""

import logging
from typing import Dict, Any, List, Optional
from enum import Enum

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from backend.config.settings import settings
from backend.models.schemas import ChatMessage
from backend.tools.web_search import web_search_tool
from backend.tools.rag_retriever import rag_retriever
from backend.tools.data_analysis import data_analysis_tool
from backend.tools.python_executor import python_executor
from backend.tools.python_coder_tool import python_coder_tool
from backend.tools.math_calculator import math_calculator

logger = logging.getLogger(__name__)

class ToolName(str, Enum):
    """Available tools for ReAct agent"""
    WEB_SEARCH = "web_search"
    RAG_RETRIEVAL = "rag_retrieval"
    DATA_ANALYSIS = "data_analysis"
    PYTHON_CODE = "python_code"
    PYTHON_CODER = "python_coder"
    MATH_CALC = "math_calc"
    FINISH = "finish"


class ReActStep:
    """Represents a single Thought-Action-Observation cycle"""

    def __init__(self, step_num: int):
        self.step_num = step_num
        self.thought: str = ""
        self.action: str = ""
        self.action_input: str = ""
        self.observation: str = ""

    def __str__(self):
        return f"""
Step {self.step_num}:
Thought: {self.thought}
Action: {self.action}
Action Input: {self.action_input}
Observation: {self.observation}
"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for serialization"""
        return {
            "step_num": self.step_num,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation
        }


class ReActAgent:
    """
    ReAct Agent: Combines Reasoning and Acting in an iterative loop

    Pattern:
    1. Thought: Reason about what to do next
    2. Action: Select and execute a tool
    3. Observation: Observe the result
    4. Repeat until answer is ready
    """

    def __init__(self, max_iterations: int = 100):
        import httpx
        self.max_iterations = max_iterations

        # Use AsyncClient for async operations
        async_client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.ollama_timeout / 1000, connect=60.0),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
        )

        self.llm = ChatOllama(
            base_url=settings.ollama_host,
            model=settings.ollama_model,
            temperature=settings.ollama_temperature,
            num_ctx=settings.ollama_num_ctx,
            top_p=settings.ollama_top_p,
            top_k=settings.ollama_top_k,
            timeout=settings.ollama_timeout / 1000,
            async_client=async_client
        )
        self.steps: List[ReActStep] = []

    async def execute(
        self,
        messages: List[ChatMessage],
        session_id: Optional[str],
        user_id: str
    ) -> tuple[str, Dict[str, Any]]:
        """
        Execute ReAct loop

        Args:
            messages: Conversation messages
            session_id: Session ID
            user_id: User identifier

        Returns:
            Tuple of (final_answer, metadata)
        """
        logger.info(f"[ReAct Agent] Starting for user: {user_id}, session: {session_id}")

        # Extract user query
        user_query = messages[-1].content

        # Initialize
        self.steps = []
        iteration = 0
        final_answer = ""

        # ReAct loop
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"[ReAct Agent] Iteration {iteration}/{self.max_iterations}")

            step = ReActStep(iteration)

            # Step 1: Thought - What should I do next?
            thought = await self._generate_thought(user_query, self.steps)
            step.thought = thought
            logger.info(f"[ReAct Agent] Thought: {thought[:]}...")

            # Step 2: Action - Select tool and input
            action, action_input = await self._select_action(user_query, thought, self.steps)
            step.action = action
            step.action_input = action_input
            logger.info(f"[ReAct Agent] Action: {action}, Input: {action_input[:100]}...")

            # Check if we're done
            if action == ToolName.FINISH:
                final_answer = action_input
                step.observation = "Task completed"
                self.steps.append(step)  # Store the final step before breaking
                logger.info(f"[ReAct Agent] Finished with answer: {final_answer[:100]}...")
                break

            # Step 3: Observation - Execute action and observe result
            observation = await self._execute_action(action, action_input)
            step.observation = observation
            logger.info(f"[ReAct Agent] Observation: {observation[:200]}...")

            # Store step
            self.steps.append(step)

        # If we didn't finish naturally, generate final answer
        if not final_answer:
            logger.info(f"[ReAct Agent] Max iterations reached, generating final answer")
            final_answer = await self._generate_final_answer(user_query, self.steps)

        # Final validation: ensure we always have an answer
        if not final_answer or not final_answer.strip():
            logger.error(f"[ReAct Agent] Empty final answer detected! Generating fallback response...")
            final_answer = await self._generate_final_answer(user_query, self.steps)
            if not final_answer or not final_answer.strip():
                final_answer = "I apologize, but I was unable to generate a proper response. Please try rephrasing your question."

        logger.info(f"[ReAct Agent] Completed after {len(self.steps)} steps")
        logger.info("=" * 80)
        logger.info(f"[ReAct Agent] FINAL ANSWER:")
        logger.info(f"{final_answer}")
        logger.info("=" * 80)

        # Build metadata
        metadata = self._build_metadata()

        return final_answer, metadata

    async def _generate_thought(self, query: str, steps: List[ReActStep]) -> str:
        """
        Generate reasoning about what to do next
        """
        # Build context from previous steps
        context = self._format_steps_context(steps)

        prompt = f"""You are a helpful AI assistant using the ReAct (Reasoning + Acting) framework.

Question: {query}

{context}

Think step-by-step about what you need to do to answer this question. Do not skip any steps.
What information do you need? What should you do next? Try to avoid python coder if possible.

Provide your reasoning:"""

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()

    async def _select_action(
        self,
        query: str,
        thought: str,
        steps: List[ReActStep]
    ) -> tuple[str, str]:
        """
        Select which tool to use and what input to provide

        Returns:
            (action_name, action_input)
        """
        context = self._format_steps_context(steps)

        prompt = f"""You are a helpful AI assistant. Based on your reasoning, select the next action.

Question: {query}

{context}

Your Thought: {thought}

Available Actions (choose EXACTLY one of these names):
1. web_search - Search the web for current information (use for: news, current events, latest data)
2. rag_retrieval - Retrieve relevant documents from uploaded files (use for: document queries, file content)
3. data_analysis - Analyze JSON data with statistics (use for: min, max, mean, count, sum)
4. python_code - Execute simple Python code (use for: quick calculations, simple scripts)
5. python_coder - Generate, verify, and execute complex Python code with file processing (use for: data analysis, file processing, complex calculations, working with CSV/Excel/PDF files)
6. math_calc - Perform advanced math calculations (use for: algebra, calculus, equations, symbolic math)
7. finish - Provide the final answer (use when you have enough information)

CRITICAL: You MUST respond with EXACTLY this format (no extra text):
Action: <one_of_the_action_names_above>
Action Input: <your_input_here>

Good example:
Action: web_search
Action Input: current weather in Seoul

Bad example (don't do this):
I think we should search the web.
Action: search the web
Action Input: weather

Now provide your action (follow the format exactly):"""

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])

        # Parse response
        action, action_input = self._parse_action_response(response.content)
        return action, action_input

    def _parse_action_response(self, response: str) -> tuple[str, str]:
        """
        Parse LLM response to extract action and input
        Enhanced parsing with better error handling and fuzzy matching
        """
        import re

        lines = response.strip().split('\n')
        action = ""
        action_input = ""

        # Try strict parsing first
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.lower().startswith("action:"):
                action = line_stripped.split(":", 1)[1].strip().lower()
            elif line_stripped.lower().startswith("action input:"):
                action_input = line_stripped.split(":", 1)[1].strip()

        # If strict parsing failed, try regex-based extraction
        if not action:
            # Look for "Action: <value>" pattern anywhere in response
            action_match = re.search(r'action\s*:\s*(\w+)', response, re.IGNORECASE)
            if action_match:
                action = action_match.group(1).strip().lower()

        if not action_input:
            # Look for "Action Input: <value>" pattern
            input_match = re.search(r'action\s+input\s*:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
            if input_match:
                action_input = input_match.group(1).strip()

        # Log raw response for debugging if parsing failed
        if not action:
            logger.warning(f"[ReAct Agent] Failed to parse action from response: {response[:200]}...")

        if not action_input:
            logger.warning(f"[ReAct Agent] Failed to parse action input from response: {response[:200]}...")

        # Validate action
        valid_actions = [e.value for e in ToolName]
        if action not in valid_actions:
            # Try fuzzy matching for common typos/variations
            action_mapping = {
                "web": ToolName.WEB_SEARCH,
                "search": ToolName.WEB_SEARCH,
                "rag": ToolName.RAG_RETRIEVAL,
                "retrieval": ToolName.RAG_RETRIEVAL,
                "retrieve": ToolName.RAG_RETRIEVAL,
                "document": ToolName.RAG_RETRIEVAL,
                "data": ToolName.DATA_ANALYSIS,
                "analyze": ToolName.DATA_ANALYSIS,
                "analysis": ToolName.DATA_ANALYSIS,
                "python": ToolName.PYTHON_CODE,
                "code": ToolName.PYTHON_CODE,
                "coder": ToolName.PYTHON_CODER,
                "generate": ToolName.PYTHON_CODER,
                "generate_code": ToolName.PYTHON_CODER,
                "math": ToolName.MATH_CALC,
                "calculator": ToolName.MATH_CALC,
                "calc": ToolName.MATH_CALC,
                "wiki": ToolName.WIKIPEDIA,
                "done": ToolName.FINISH,
                "answer": ToolName.FINISH,
                "complete": ToolName.FINISH,
            }

            matched_action = action_mapping.get(action, None)
            if matched_action:
                logger.info(f"[ReAct Agent] Fuzzy matched '{action}' to '{matched_action}'")
                action = matched_action
            else:
                # Default to finish if invalid - use full response as fallback
                logger.warning(f"[ReAct Agent] Invalid action '{action}', defaulting to finish with full response")
                action = ToolName.FINISH
                if not action_input:
                    # Use the full response as action_input if parsing completely failed
                    action_input = response.strip() if response.strip() else "I don't have enough information to answer this question."

        # Additional check: if action is FINISH but action_input is empty, use full response
        if action == ToolName.FINISH and not action_input:
            logger.warning(f"[ReAct Agent] FINISH action with empty input, using full response")
            action_input = response.strip() if response.strip() else "I don't have enough information to answer this question."

        return action, action_input

    async def _execute_action(self, action: str, action_input: str) -> str:
        """
        Execute the selected action and return observation
        """
        try:
            if action == ToolName.WEB_SEARCH:
                logger.info(f"[ReAct Agent] Executing web search: {action_input}")
                results = await web_search_tool.search(action_input, max_results=5)
                observation = web_search_tool.format_results(results)
                return observation if observation else "No web search results found."

            elif action == ToolName.RAG_RETRIEVAL:
                logger.info(f"[ReAct Agent] Executing RAG retrieval: {action_input}")
                results = await rag_retriever.retrieve(action_input, top_k=5)
                observation = rag_retriever.format_results(results)
                return observation if observation else "No relevant documents found."

            elif action == ToolName.DATA_ANALYSIS:
                logger.info(f"[ReAct Agent] Executing data analysis: {action_input}")
                result = await data_analysis_tool.analyze_json(action_input)
                return result

            elif action == ToolName.PYTHON_CODE:
                logger.info(f"[ReAct Agent] Executing Python code: {action_input[:50]}...")
                result = await python_executor.execute(action_input)
                return python_executor.format_result(result)

            elif action == ToolName.PYTHON_CODER:
                logger.info(f"[ReAct Agent] Executing Python coder: {action_input[:50]}...")
                result = await python_coder_tool.execute_code_task(action_input)
                if result["success"]:
                    return f"Code executed successfully:\n{result['output']}\n\nExecution details: {result['iterations']} iterations, {result['execution_time']:.2f}s"
                else:
                    return f"Code execution failed: {result.get('error', 'Unknown error')}"

            elif action == ToolName.MATH_CALC:
                logger.info(f"[ReAct Agent] Calculating: {action_input}")
                result = await math_calculator.calculate(action_input)
                return result

            else:
                return "Invalid action."

        except Exception as e:
            logger.error(f"[ReAct Agent] Error executing action {action}: {e}")
            return f"Error executing action: {str(e)}"

    async def _generate_final_answer(self, query: str, steps: List[ReActStep]) -> str:
        """
        Generate final answer based on all observations
        """
        context = self._format_steps_context(steps)

        prompt = f"""You are a helpful AI assistant. Based on all your reasoning and observations, provide a final answer.

Question: {query}

{context}

Based on all the information you've gathered, provide a clear, concise, and accurate final answer to the question:"""

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()

    def _format_steps_context(self, steps: List[ReActStep]) -> str:
        """
        Format previous steps into context string
        """
        if not steps:
            return ""

        context_parts = ["Previous Steps:"]
        for step in steps:
            context_parts.append(f"""
Step {step.step_num}:
- Thought: {step.thought}
- Action: {step.action}
- Action Input: {step.action_input}
- Observation: {step.observation[:]}{"..." if len(step.observation) > 200 else ""}
""")

        return "\n".join(context_parts)

    def _build_metadata(self) -> Dict[str, Any]:
        """
        Build metadata dictionary with execution details
        """
        # Collect unique tools used
        tools_used = list(set([
            step.action for step in self.steps
            if step.action != ToolName.FINISH
        ]))

        # Build execution steps
        execution_steps = [step.to_dict() for step in self.steps]

        return {
            "agent_type": "react",
            "total_iterations": len(self.steps),
            "max_iterations": self.max_iterations,
            "tools_used": tools_used,
            "execution_steps": execution_steps,
            "execution_trace": self.get_trace()
        }

    def get_trace(self) -> str:
        """
        Get full trace of ReAct execution for debugging
        """
        if not self.steps:
            return "No steps executed."

        trace = ["=== ReAct Execution Trace ===\n"]
        for step in self.steps:
            trace.append(str(step))

        return "\n".join(trace)


# Global ReAct agent instance
react_agent = ReActAgent(max_iterations=5)
