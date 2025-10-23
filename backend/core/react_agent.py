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
from backend.tools.math_calculator import math_calculator
from backend.tools.wikipedia_tool import wikipedia_tool
from backend.tools.weather_tool import weather_tool
from backend.tools.sql_query_tool import sql_query_tool

logger = logging.getLogger(__name__)


class ToolName(str, Enum):
    """Available tools for ReAct agent"""
    WEB_SEARCH = "web_search"
    RAG_RETRIEVAL = "rag_retrieval"
    DATA_ANALYSIS = "data_analysis"
    PYTHON_CODE = "python_code"
    MATH_CALC = "math_calc"
    WIKIPEDIA = "wikipedia"
    WEATHER = "weather"
    SQL_QUERY = "sql_query"
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

    def __init__(self, max_iterations: int = 5):
        import httpx
        self.max_iterations = max_iterations

        # Use AsyncClient for async operations
        async_client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.ollama_timeout / 1000, connect=60.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
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
            logger.info(f"[ReAct Agent] Thought: {thought[:100]}...")

            # Step 2: Action - Select tool and input
            action, action_input = await self._select_action(user_query, thought, self.steps)
            step.action = action
            step.action_input = action_input
            logger.info(f"[ReAct Agent] Action: {action}, Input: {action_input[:50]}...")

            # Check if we're done
            if action == ToolName.FINISH:
                final_answer = action_input
                logger.info(f"[ReAct Agent] Finished with answer")
                break

            # Step 3: Observation - Execute action and observe result
            observation = await self._execute_action(action, action_input)
            step.observation = observation
            logger.info(f"[ReAct Agent] Observation: {observation[:100]}...")

            # Store step
            self.steps.append(step)

        # If we didn't finish naturally, generate final answer
        if not final_answer:
            logger.info(f"[ReAct Agent] Max iterations reached, generating final answer")
            final_answer = await self._generate_final_answer(user_query, self.steps)

        logger.info(f"[ReAct Agent] Completed after {len(self.steps)} steps")

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

Think step-by-step about what you need to do to answer this question.
What information do you need? What should you do next?

Provide your reasoning (1-2 sentences):"""

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

Available Actions:
1. web_search - Search the web for current information (use for: news, current events, latest data)
2. rag_retrieval - Retrieve relevant documents from uploaded files (use for: document queries, file content)
3. data_analysis - Analyze JSON data with statistics (use for: min, max, mean, count, sum)
4. python_code - Execute Python code (use for: calculations, data transformations, code examples)
5. math_calc - Perform advanced math calculations (use for: algebra, calculus, equations, symbolic math)
6. wikipedia - Search Wikipedia for factual information (use for: definitions, facts, history)
7. weather - Get current weather information (use for: weather queries, forecasts)
8. sql_query - Query SQL database (use for: structured data queries)
9. finish - Provide the final answer (use when you have enough information)

Select ONE action and provide the input.

Format your response EXACTLY as:
Action: <action_name>
Action Input: <input_for_the_action>

Example:
Action: web_search
Action Input: current weather in Seoul

Your response:"""

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])

        # Parse response
        action, action_input = self._parse_action_response(response.content)
        return action, action_input

    def _parse_action_response(self, response: str) -> tuple[str, str]:
        """
        Parse LLM response to extract action and input
        """
        lines = response.strip().split('\n')
        action = ""
        action_input = ""

        for line in lines:
            if line.startswith("Action:"):
                action = line.replace("Action:", "").strip().lower()
            elif line.startswith("Action Input:"):
                action_input = line.replace("Action Input:", "").strip()

        # Validate action
        valid_actions = [e.value for e in ToolName]
        if action not in valid_actions:
            # Default to finish if invalid
            logger.warning(f"[ReAct Agent] Invalid action '{action}', defaulting to finish")
            action = ToolName.FINISH
            action_input = "I don't have enough information to answer this question."

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

            elif action == ToolName.MATH_CALC:
                logger.info(f"[ReAct Agent] Calculating: {action_input}")
                result = await math_calculator.calculate(action_input)
                return result

            elif action == ToolName.WIKIPEDIA:
                logger.info(f"[ReAct Agent] Searching Wikipedia: {action_input}")
                result = await wikipedia_tool.search_and_summarize(action_input, sentences=3)
                return result

            elif action == ToolName.WEATHER:
                logger.info(f"[ReAct Agent] Getting weather: {action_input}")
                result = await weather_tool.get_weather(action_input)
                return result

            elif action == ToolName.SQL_QUERY:
                logger.info(f"[ReAct Agent] Executing SQL query: {action_input[:50]}...")
                result = await sql_query_tool.execute_query(action_input)
                return sql_query_tool.format_results(result)

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
- Observation: {step.observation[:200]}{"..." if len(step.observation) > 200 else ""}
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
