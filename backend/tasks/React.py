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
from backend.tools.python_executor import python_executor
from backend.tools.python_coder_tool import python_coder_tool

logger = logging.getLogger(__name__)

class ToolName(str, Enum):
    """Available tools for ReAct agent"""
    WEB_SEARCH = "web_search"
    RAG_RETRIEVAL = "rag_retrieval"
    PYTHON_CODE = "python_code"
    PYTHON_CODER = "python_coder"
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
            timeout=httpx.Timeout(settings.ollama_timeout / 1000, connect=600.0),
            limits=httpx.Limits(max_keepalive_connections=100, max_connections=200)
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
        self.file_paths: Optional[List[str]] = None  # Store file paths for python_coder
        self.session_id: Optional[str] = None  # Store session_id for python_coder

    async def execute(
        self,
        messages: List[ChatMessage],
        session_id: Optional[str],
        user_id: str,
        file_paths: Optional[List[str]] = None
    ) -> tuple[str, Dict[str, Any]]:
        """
        Execute ReAct loop

        Args:
            messages: Conversation messages
            session_id: Session ID
            user_id: User identifier
            file_paths: Optional list of file paths for code execution

        Returns:
            Tuple of (final_answer, metadata)
        """
        # Store file paths and session_id for use in python_coder actions
        self.file_paths = file_paths
        self.session_id = session_id

        logger.info("\n\n\n\n\n" + "=" * 100)
        logger.info("[ReAct Agent] EXECUTION STARTED")
        if file_paths:
            logger.info(f"Attached Files: {len(file_paths)} files")
        logger.info("-" * 100)

        # Extract user query
        user_query = messages[-1].content
        logger.info(f"USER QUERY:\n{user_query}")
        logger.info("=" * 100 + "\n\n\n\n\n")

        # Initialize
        self.steps = []
        iteration = 0
        final_answer = ""

        # ReAct loop
        while iteration < self.max_iterations:
            iteration += 1
            logger.info("\n" + "#" * 100)
            logger.info(f"ITERATION {iteration}/{self.max_iterations}")
            logger.info("#" * 100 + "\n")

            step = ReActStep(iteration)

            # Step 1: Thought - What should I do next?
            logger.info("")
            logger.info("PHASE 1: THOUGHT GENERATION")
            logger.info("")
            thought = await self._generate_thought(user_query, self.steps)
            step.thought = thought
            logger.info("Generated Thought:")
            for _line in thought.splitlines():
                logger.info(_line)
            logger.info("")

            # Step 2: Action - Select tool and input
            logger.info("")
            logger.info("PHASE 2: ACTION SELECTION")
            logger.info("")
            action, action_input = await self._select_action(user_query, thought, self.steps)
            step.action = action
            step.action_input = action_input

            # Check if we're done
            if action == ToolName.FINISH:
                logger.info("")
                logger.info("FINISH ACTION DETECTED - GENERATING FINAL ANSWER")
                logger.info("")

                # ALWAYS regenerate final answer using all observations to prevent information loss
                final_answer = await self._generate_final_answer(user_query, self.steps)

                # If regenerated answer is insufficient, use action_input as fallback
                if not final_answer or len(final_answer.strip()) < 10:
                    logger.warning("\n" + "!" * 100)
                    logger.warning("WARNING: Generated answer insufficient, using fallback")
                    logger.warning("!" * 100 + "\n")
                    if action_input and len(action_input.strip()) >= 10:
                        final_answer = action_input
                        logger.info("Using action_input as fallback answer")
                    else:
                        # Last resort: extract from observations
                        final_answer = self._extract_answer_from_steps(user_query, self.steps)
                        logger.info("Extracted answer from previous observations")

                step.observation = "Task completed"
                self.steps.append(step)  # Store the final step before breaking
                logger.info("")
                logger.info("Final Answer Generated:")
                for _line in str(final_answer).splitlines():
                    logger.info(_line)
                logger.info("")
                break

            # Step 3: Observation - Execute action and observe result
            logger.info("")
            logger.info("PHASE 3: ACTION EXECUTION & OBSERVATION")
            logger.info("")
            observation = await self._execute_action(action, action_input)
            step.observation = observation
            logger.info("Observation Result:")
            for _line in str(observation).splitlines():
                logger.info(_line)
            logger.info("")

            # Store step
            self.steps.append(step)

        # If we didn't finish naturally, generate final answer
        if not final_answer:
            logger.info("")
            logger.info("MAX ITERATIONS REACHED - GENERATING FINAL ANSWER")
            logger.info("")
            final_answer = await self._generate_final_answer(user_query, self.steps)

        # Final validation: ensure we always have an answer
        if not final_answer or not final_answer.strip():
            logger.error("\n" + "X" * 100)
            logger.error("ERROR: EMPTY FINAL ANSWER DETECTED - GENERATING FALLBACK")
            logger.error("X" * 100 + "\n")
            final_answer = await self._generate_final_answer(user_query, self.steps)
            if not final_answer or not final_answer.strip():
                final_answer = "I apologize, but I was unable to generate a proper response. Please try rephrasing your question."

        logger.info("")
        logger.info("[ReAct Agent] EXECUTION COMPLETED")
        logger.info("")
        logger.info(f"Total Steps: {len(self.steps)}")
        logger.info(f"Total Iterations: {iteration}")
        logger.info("FINAL ANSWER:")
        for _line in str(final_answer).splitlines():
            logger.info(_line)
        logger.info("")

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
Break down the task into smaller baby steps. Such as
if the question is "Analyze the data", the baby steps could be

1. Write down the data to a scratch file.
2. Load the data from the scratch file.
3. Use math tools to calculate mean, median, etc.
4. Acquire results from the tools
5. Append the results to the scratch file.
6. Read the scratch file and answer the question.
7. Make sure the answers are adequate to the query.
8. Finish the task.

These are available tools:
{settings.available_tools}
What information do you need? What should you do next?

Provide your reasoning:"""

        # Intentionally do not log system prompt. Inputs are already captured elsewhere.
        logger.info("")
        logger.info("Thought generation requested")
        logger.info("")

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
3. python_code - Execute simple Python code (use for: quick calculations, simple scripts)
4. python_coder - Generate, verify, and execute complex Python code with file processing (use for: data analysis, file processing, complex calculations, working with CSV/Excel/PDF files)
5. finish - Provide the final answer (use ONLY when you have complete information to answer the question)

RESPONSE FORMAT - You can think briefly, but you MUST end your response with these two lines:
Action: <action_name>
Action Input: <input_for_the_action>

Examples:

Example 1 (Good):
Action: web_search
Action Input: current weather in Seoul

Example 2 (Good - with brief reasoning):
I need current data for this query.
Action: web_search
Action Input: latest news about AI

Example 3 (Good - finish action):
I now have all the information needed.
Action: finish
Action Input: The capital of France is Paris, with a population of approximately 2.2 million people.

Example 4 (Bad - don't do this):
I think we should search the web.
Action: search the web

Now provide your action:"""

        # Intentionally do not log system prompt for action selection
        logger.info("")
        logger.info("Action selection requested")
        logger.info("")

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])

        # Parse response
        action, action_input = self._parse_action_response(response.content)
        
        logger.info("")
        logger.info(f"Action: {action}")
        logger.info(f"Action Input: {action_input}")
        logger.info("")

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
            # Look for "Action Input: <value>" pattern - improved to capture multiline
            input_match = re.search(r'action\s+input\s*:\s*(.+?)(?=\n\n|\naction\s*:|\Z)', response, re.IGNORECASE | re.DOTALL)
            if input_match:
                action_input = input_match.group(1).strip()

        # Log raw response for debugging if parsing failed
        if not action:
            logger.warning("\n" + "!" * 100)
            logger.warning("[ReAct Agent] PARSING ERROR - Failed to parse action from response")
            logger.warning("!" * 100)
            logger.warning(f"Raw Response:\n{response}")
            logger.warning("!" * 100 + "\n")

        if not action_input:
            logger.warning("\n" + "!" * 100)
            logger.warning("[ReAct Agent] PARSING ERROR - Failed to parse action input from response")
            logger.warning("!" * 100)
            logger.warning(f"Raw Response:\n{response}")
            logger.warning("!" * 100 + "\n")

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
                "python": ToolName.PYTHON_CODE,
                "code": ToolName.PYTHON_CODE,
                "coder": ToolName.PYTHON_CODER,
                "generate": ToolName.PYTHON_CODER,
                "generate_code": ToolName.PYTHON_CODER,
                "done": ToolName.FINISH,
                "answer": ToolName.FINISH,
                "complete": ToolName.FINISH,
            }

            matched_action = action_mapping.get(action, None)
            if matched_action:
                logger.info("\n" + "~" * 100)
                logger.info(f"[ReAct Agent] FUZZY MATCH APPLIED")
                logger.info("~" * 100)
                logger.info(f"Original: '{action}'")
                logger.info(f"Matched To: '{matched_action}'")
                logger.info("~" * 100 + "\n")
                action = matched_action
            else:
                # Default to finish if invalid - try to extract answer from response
                logger.warning("\n" + "!" * 100)
                logger.warning(f"[ReAct Agent] INVALID ACTION - Defaulting to FINISH")
                logger.warning("!" * 100)
                logger.warning(f"Invalid Action: '{action}'")
                logger.warning(f"Valid Actions: {valid_actions}")
                logger.warning("!" * 100 + "\n")
                action = ToolName.FINISH
                if not action_input:
                    # Try to extract answer-like content from response
                    action_input = self._extract_answer_from_response(response)

        # Additional check: if action is FINISH but action_input is empty/short, try extraction
        if action == ToolName.FINISH and (not action_input or len(action_input.strip()) < 1):
            logger.warning("\n" + "!" * 100)
            logger.warning("[ReAct Agent] FINISH action with insufficient input")
            logger.warning("!" * 100)
            logger.warning("Attempting to extract answer from response...")
            logger.warning("!" * 100 + "\n")
            extracted = self._extract_answer_from_response(response)
            if extracted and len(extracted.strip()) >= 1:
                action_input = extracted
                logger.info(f"Extracted answer: {action_input[:200]}...")
            elif response.strip():
                action_input = response.strip()
                logger.info(f"Using full response as answer: {action_input[:200]}...")
            else:
                action_input = "I don't have enough information to answer this question."
                logger.warning("No extractable content found, using default message")

        return action, action_input

    def _extract_answer_from_response(self, response: str) -> str:
        """
        Extract answer-like content from LLM response when format parsing fails
        Looks for declarative sentences, conclusions, or final statements
        """
        import re

        # Clean up the response
        text = response.strip()

        # Try to find answer patterns
        patterns = [
            r'(?:the answer is|answer:|final answer:|conclusion:|result is?)\s*:?\s*(.+?)(?:\n\n|\Z)',
            r'(?:therefore|thus|so|hence),?\s+(.+?)(?:\n\n|\Z)',
            r'(?:in summary|to conclude|in conclusion),?\s+(.+?)(?:\n\n|\Z)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                answer = match.group(1).strip()
                if len(answer) >= 10:
                    logger.info(f"[ReAct Agent] Extracted answer from response pattern")
                    return answer

        # If no patterns match, try to get the last substantial paragraph
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if paragraphs:
            last_para = paragraphs[-1]
            # Avoid action format lines
            if not re.match(r'action\s*:', last_para, re.IGNORECASE):
                if len(last_para) >= 20:
                    logger.info(f"[ReAct Agent] Using last paragraph as answer")
                    return last_para

        # Fallback: return full response if it's reasonable length
        if len(text) >= 20:
            return text

        return ""

    async def _execute_action(self, action: str, action_input: str) -> str:
        """
        Execute the selected action and return observation
        """
        try:
            logger.info("")
            logger.info(f"EXECUTING TOOL: {action}")
            logger.info("")
            logger.info("Tool Input:")
            for _line in str(action_input).splitlines():
                logger.info(_line)
            logger.info("")

            if action == ToolName.WEB_SEARCH:
                results = await web_search_tool.search(action_input, max_results=5)
                observation = web_search_tool.format_results(results)
                final_observation = observation if observation else "No web search results found."

                logger.info("")
                logger.info("TOOL OUTPUT (Web Search):")
                for _line in str(final_observation).splitlines():
                    logger.info(_line)
                logger.info("")

                return final_observation

            elif action == ToolName.RAG_RETRIEVAL:
                results = await rag_retriever.retrieve(action_input, top_k=5)
                observation = rag_retriever.format_results(results)
                final_observation = observation if observation else "No relevant documents found."

                logger.info("")
                logger.info("TOOL OUTPUT (RAG Retrieval):")
                for _line in str(final_observation).splitlines():
                    logger.info(_line)
                logger.info("")

                return final_observation

            elif action == ToolName.PYTHON_CODE:
                result = await python_executor.execute(action_input)
                final_observation = python_executor.format_result(result)

                logger.info("")
                logger.info("TOOL OUTPUT (Python Code Execution):")
                logger.info("Execution Result:")
                for _line in str(final_observation).splitlines():
                    logger.info(_line)
                logger.info("")

                return final_observation

            elif action == ToolName.PYTHON_CODER:
                # Pass attached file paths and session_id to python_coder
                result = await python_coder_tool.execute_code_task(
                    query=action_input,
                    file_paths=self.file_paths,
                    session_id=self.session_id
                )

                logger.info("")
                logger.info("TOOL OUTPUT (Python Coder - Detailed):")
                logger.info(f"Success: {result['success']}")
                logger.info(f"Iterations: {result.get('iterations', 'N/A')}")
                logger.info(f"Execution Time: {result.get('execution_time', 'N/A'):.2f}s" if isinstance(result.get('execution_time'), (int, float)) else f"Execution Time: {result.get('execution_time', 'N/A')}")
                if self.file_paths:
                    logger.info(f"Files Used: {len(self.file_paths)} files")
                logger.info("")

                if result["success"]:
                    logger.info("Generated Code:")
                    for _line in str(result.get('code', 'N/A')).splitlines():
                        logger.info(_line)
                    logger.info("")
                    logger.info("Execution Output:")
                    for _line in str(result['output']).splitlines():
                        logger.info(_line)
                    logger.info("")
                    if result.get('verification_issues'):
                        logger.info("Verification Issues:")
                        for _line in str(result['verification_issues']).splitlines():
                            logger.info(_line)
                        logger.info("")

                    final_observation = f"Code executed successfully:\n{result['output']}\n\nExecution details: {result['iterations']} iterations, {result['execution_time']:.2f}s"
                else:
                    logger.info("Error Details:")
                    for _line in str(result.get('error', 'Unknown error')).splitlines():
                        logger.info(_line)
                    logger.info("")
                    if result.get('code'):
                        logger.info("Failed Code:")
                        for _line in str(result['code']).splitlines():
                            logger.info(_line)
                        logger.info("")

                    final_observation = f"Code execution failed: {result.get('error', 'Unknown error')}"

                logger.info("")
                return final_observation

            else:
                logger.warning(f"\n[ReAct Agent] Invalid action attempted: {action}\n")
                return "Invalid action."

        except Exception as e:
            logger.error("")
            logger.error(f"ERROR EXECUTING ACTION: {action}")
            logger.error(f"Exception Type: {type(e).__name__}")
            logger.error(f"Exception Message: {str(e)}")
            import traceback
            logger.error("Traceback:")
            for _line in traceback.format_exc().splitlines():
                logger.error(_line)
            return f"Error executing action: {str(e)}"

    async def _generate_final_answer(self, query: str, steps: List[ReActStep]) -> str:
        """
        Generate final answer based on all observations
        """
        context = self._format_steps_context(steps)

        prompt = f"""You are a helpful AI assistant. Based on all your reasoning and observations, provide a final answer.

Question: {query}

{context}

IMPORTANT: Review ALL the observations above carefully. Each observation contains critical information from tools you executed (web search results, code outputs, document content, etc.).

Your final answer MUST:
1. Incorporate ALL relevant information from the observations
2. Be comprehensive and complete
3. Directly answer the user's question
4. Include specific details, numbers, facts from the observations

Based on all the information you've gathered through your actions and observations, provide a clear, complete, and accurate final answer:"""

        # Intentionally do not log system prompt for final answer generation
        logger.info("")
        logger.info("Final answer generation requested")
        logger.info("")

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])

        logger.info("")
        logger.info("LLM OUTPUT (Final Answer Generation):")
        for _line in response.content.strip().splitlines():
            logger.info(_line)
        logger.info("")

        return response.content.strip()

    def _extract_answer_from_steps(self, query: str, steps: List[ReActStep]) -> str:
        """
        Extract answer from observation history when FINISH produces empty result
        Uses the most recent relevant observation as a fallback answer
        """
        if not steps:
            return "I apologize, but I was unable to generate a proper response. Please try rephrasing your question."

        # Try to find the most informative observation from recent steps
        for step in reversed(steps):
            obs = step.observation.strip()
            # Skip error messages and empty observations
            if obs and len(obs) >= 20 and not obs.startswith("Error") and not obs.startswith("No "):
                logger.info(f"[ReAct Agent] Extracted answer from step {step.step_num} observation")
                return f"Based on my research: {obs}"

        # If no good observation found, summarize what was attempted
        actions_taken = [step.action for step in steps if step.action != ToolName.FINISH]
        if actions_taken:
            return f"I attempted to answer your question using {', '.join(set(actions_taken))}, but was unable to find sufficient information. Please try rephrasing your question or providing more context."

        return "I apologize, but I was unable to generate a proper response. Please try rephrasing your question."

    def _format_steps_context(self, steps: List[ReActStep]) -> str:
        """
        Format previous steps into context string
        """
        if not steps:
            return ""

        context_parts = ["Previous Steps:"]
        for step in steps:
            # Increase observation limit to preserve more context (200 -> 1000 chars)
            obs_display = step.observation[:]
            context_parts.append(f"""
Step {step.step_num}:
- Thought: {step.thought}
- Action: {step.action}
- Action Input: {step.action_input}
- Observation: {obs_display}
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
react_agent = ReActAgent(max_iterations=10)
