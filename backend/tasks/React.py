"""
ReAct Agent Implementation
Implements the Reasoning + Acting pattern with iterative Thought-Action-Observation loops
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from backend.config.settings import settings
from backend.models.schemas import ChatMessage, PlanStep, StepResult
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
        self._attempted_coder: bool = False  # Track whether python_coder was attempted

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
        self._attempted_coder = False

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

        # Pre-step: If files are attached, first try python_coder once
        if self.file_paths:
            logger.info("[ReAct Agent] Pre-step: Files detected, attempting python_coder before tool loop")
            try:
                self._attempted_coder = True
                coder_result = await python_coder_tool.execute_code_task(
                    query=user_query,
                    file_paths=self.file_paths,
                    session_id=self.session_id
                )

                # Record as a step for traceability
                pre_step = ReActStep(0)
                pre_step.thought = "Files attached; attempt local code analysis first."
                pre_step.action = ToolName.PYTHON_CODER
                pre_step.action_input = user_query
                if coder_result.get("success"):
                    obs = f"Code executed successfully:\n{coder_result.get('output','')}"
                else:
                    obs = f"Code execution failed: {coder_result.get('error','Unknown error')}"
                pre_step.observation = obs
                self.steps.append(pre_step)

                if coder_result.get("success") and str(coder_result.get("output", "")).strip():
                    logger.info("[ReAct Agent] Pre-step succeeded with python_coder; returning result")
                    final_answer = str(coder_result.get("output", "")).strip()
                    metadata = self._build_metadata()
                    return final_answer, metadata
                else:
                    logger.info("[ReAct Agent] Pre-step did not yield sufficient result; continuing with tool loop")
            except Exception as e:
                logger.warning(f"[ReAct Agent] Pre-step python_coder error: {e}; continuing with tool loop")

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

    async def execute_with_plan(
        self,
        plan_steps: List[PlanStep],
        messages: List[ChatMessage],
        session_id: Optional[str],
        user_id: str,
        file_paths: Optional[List[str]] = None,
        max_iterations_per_step: int = 3
    ) -> Tuple[str, List[StepResult]]:
        """
        Execute ReAct in guided mode: execute each plan step one-by-one

        Args:
            plan_steps: Structured plan steps to execute
            messages: Conversation messages
            session_id: Session ID
            user_id: User identifier
            file_paths: Optional list of file paths for code execution
            max_iterations_per_step: Max ReAct iterations per step

        Returns:
            Tuple of (final_answer, step_results)
        """
        # Store file paths and session_id
        self.file_paths = file_paths
        self.session_id = session_id
        self._attempted_coder = False

        logger.info("\n\n\n" + "=" * 100)
        logger.info("[ReAct Agent: GUIDED MODE] EXECUTION STARTED")
        logger.info(f"Plan has {len(plan_steps)} steps")
        if file_paths:
            logger.info(f"Attached Files: {len(file_paths)} files")
        logger.info("=" * 100 + "\n\n\n")

        # Extract user query
        user_query = messages[-1].content

        # Track results from each step
        step_results: List[StepResult] = []
        accumulated_observations = []  # Collect observations from all steps

        # Execute each plan step
        for step_idx, plan_step in enumerate(plan_steps):
            logger.info("\n" + "#" * 100)
            logger.info(f"EXECUTING PLAN STEP {plan_step.step_num}/{len(plan_steps)}")
            logger.info(f"Goal: {plan_step.goal}")
            logger.info(f"Primary Tools: {plan_step.primary_tools}")
            logger.info(f"Fallback Tools: {plan_step.fallback_tools}")
            logger.info("#" * 100 + "\n")

            # Execute this step
            step_result = await self._execute_step(
                plan_step=plan_step,
                user_query=user_query,
                accumulated_observations=accumulated_observations,
                max_iterations=max_iterations_per_step
            )

            step_results.append(step_result)

            # Add observation to accumulated context
            if step_result.observation:
                accumulated_observations.append(
                    f"Step {step_result.step_num} ({step_result.goal}): {step_result.observation}"
                )

            logger.info(f"\n[ReAct Guided] Step {plan_step.step_num} {'✓ SUCCESS' if step_result.success else '✗ FAILED'}")
            logger.info(f"Tool used: {step_result.tool_used}")
            logger.info(f"Attempts: {step_result.attempts}\n")

        # Generate final answer based on all step results
        logger.info("\n" + "=" * 100)
        logger.info("[ReAct Guided] Generating final answer from all step results...")
        logger.info("=" * 100 + "\n")

        final_answer = await self._generate_final_answer_from_steps(
            user_query=user_query,
            step_results=step_results,
            accumulated_observations=accumulated_observations
        )

        logger.info("\n[ReAct Agent: GUIDED MODE] EXECUTION COMPLETED")
        logger.info(f"Total steps executed: {len(step_results)}")
        logger.info(f"Successful steps: {sum(1 for r in step_results if r.success)}/{len(step_results)}\n")

        return final_answer, step_results

    async def _execute_step(
        self,
        plan_step: PlanStep,
        user_query: str,
        accumulated_observations: List[str],
        max_iterations: int = 3
    ) -> StepResult:
        """
        Execute a single plan step using ReAct loops with tool fallback

        Args:
            plan_step: The plan step to execute
            user_query: Original user query
            accumulated_observations: Context from previous steps
            max_iterations: Max iterations for this step

        Returns:
            StepResult with execution details
        """
        logger.info(f"[ReAct Step {plan_step.step_num}] Starting execution...")

        # Build context from previous steps
        context = "\n".join(accumulated_observations) if accumulated_observations else "This is the first step."

        # Try primary tools first, then fallback tools
        all_tools = plan_step.primary_tools + plan_step.fallback_tools
        
        attempt_count = 0
        last_error = None
        last_observation = None

        for tool_idx, tool_name in enumerate(all_tools):
            attempt_count += 1
            is_primary = tool_idx < len(plan_step.primary_tools)
            tool_type = "primary" if is_primary else "fallback"

            logger.info(f"\n[ReAct Step {plan_step.step_num}] Attempt {attempt_count}: Trying {tool_type} tool '{tool_name}'")

            # Execute tool with ReAct-style reasoning
            try:
                observation, success = await self._execute_tool_for_step(
                    tool_name=tool_name,
                    plan_step=plan_step,
                    user_query=user_query,
                    context=context,
                    max_iterations=max_iterations
                )

                last_observation = observation

                # Check if step goal is achieved
                if success:
                    logger.info(f"[ReAct Step {plan_step.step_num}] SUCCESS with tool '{tool_name}'")
                    return StepResult(
                        step_num=plan_step.step_num,
                        goal=plan_step.goal,
                        success=True,
                        tool_used=tool_name,
                        attempts=attempt_count,
                        observation=observation,
                        error=None,
                        metadata={"tool_type": tool_type}
                    )
                else:
                    logger.info(f"[ReAct Step {plan_step.step_num}] Tool '{tool_name}' did not achieve goal, trying next tool...")
                    last_error = f"Tool '{tool_name}' executed but did not meet success criteria"

            except Exception as e:
                logger.error(f"[ReAct Step {plan_step.step_num}] Tool '{tool_name}' failed with error: {e}")
                last_error = str(e)
                last_observation = f"Error executing {tool_name}: {str(e)}"

        # All tools failed
        logger.warning(f"[ReAct Step {plan_step.step_num}] FAILED - All tools exhausted")
        return StepResult(
            step_num=plan_step.step_num,
            goal=plan_step.goal,
            success=False,
            tool_used=all_tools[-1] if all_tools else None,
            attempts=attempt_count,
            observation=last_observation or "No observation",
            error=last_error,
            metadata={"all_tools_tried": all_tools}
        )

    async def _execute_tool_for_step(
        self,
        tool_name: str,
        plan_step: PlanStep,
        user_query: str,
        context: str,
        max_iterations: int
    ) -> Tuple[str, bool]:
        """
        Execute a specific tool to achieve the step goal

        Args:
            tool_name: Name of tool to execute
            plan_step: Current plan step
            user_query: Original user query
            context: Context from previous steps
            max_iterations: Max iterations to try

        Returns:
            Tuple of (observation, success)
        """
        # Map tool names to ToolName enum
        tool_mapping = {
            "web_search": ToolName.WEB_SEARCH,
            "rag_retrieval": ToolName.RAG_RETRIEVAL,
            "python_code": ToolName.PYTHON_CODE,
            "python_coder": ToolName.PYTHON_CODER,
            "finish": ToolName.FINISH
        }

        tool_enum = tool_mapping.get(tool_name)
        if not tool_enum:
            return f"Unknown tool: {tool_name}", False

        # Generate action input for this tool
        action_input = await self._generate_action_input_for_step(
            tool_name=tool_enum,
            plan_step=plan_step,
            user_query=user_query,
            context=context
        )

        # Execute the tool
        observation = await self._execute_action(tool_enum, action_input)

        # Verify if goal is achieved
        success = await self._verify_step_success(
            plan_step=plan_step,
            observation=observation,
            tool_used=tool_name
        )

        return observation, success

    async def _generate_action_input_for_step(
        self,
        tool_name: ToolName,
        plan_step: PlanStep,
        user_query: str,
        context: str
    ) -> str:
        """
        Generate appropriate input for a tool based on step goal

        Args:
            tool_name: Tool to generate input for
            plan_step: Current plan step
            user_query: Original user query
            context: Context from previous steps

        Returns:
            Action input string for the tool
        """
        prompt = f"""You are executing a specific step in a plan.

Original User Query: {user_query}

Current Step Goal: {plan_step.goal}
Success Criteria: {plan_step.success_criteria}
Additional Context: {plan_step.context or 'None'}

Previous Steps Context:
{context}

Tool to use: {tool_name}

Generate the appropriate input for this tool to achieve the step goal.
Provide ONLY the tool input, no explanations:"""

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()

    async def _verify_step_success(
        self,
        plan_step: PlanStep,
        observation: str,
        tool_used: str
    ) -> bool:
        """
        Verify if the step goal was achieved based on observation

        Args:
            plan_step: Current plan step
            observation: Observation from tool execution
            tool_used: Which tool was used

        Returns:
            True if step goal achieved, False otherwise
        """
        # Check for obvious failures
        if not observation or len(observation.strip()) < 5:
            return False
        
        if "error" in observation.lower() or "failed" in observation.lower():
            # But allow if there's also success indicators
            if "success" not in observation.lower():
                return False

        # For simple cases, use heuristics
        if "success" in observation.lower() and len(observation) > 50:
            return True

        # Use LLM to verify goal achievement
        verification_prompt = f"""Verify if the step goal was achieved.

Step Goal: {plan_step.goal}
Success Criteria: {plan_step.success_criteria}

Tool Used: {tool_used}
Observation: {observation[:1000]}

Based on the observation, was the step goal achieved according to the success criteria?
Answer with "YES" or "NO" and brief reasoning:"""

        response = await self.llm.ainvoke([HumanMessage(content=verification_prompt)])
        result = response.content.strip().lower()
        
        is_success = result.startswith("yes")
        logger.info(f"[Step Verification] {'SUCCESS' if is_success else 'FAILED'} - {result[:100]}")
        
        return is_success

    async def _generate_final_answer_from_steps(
        self,
        user_query: str,
        step_results: List[StepResult],
        accumulated_observations: List[str]
    ) -> str:
        """
        Generate final answer based on all step results

        Args:
            user_query: Original user query
            step_results: Results from all steps
            accumulated_observations: All observations

        Returns:
            Final answer string
        """
        # Build context from all steps
        steps_summary = []
        for result in step_results:
            status = "✓" if result.success else "✗"
            steps_summary.append(
                f"Step {result.step_num} {status}: {result.goal}\n"
                f"  Tool: {result.tool_used}\n"
                f"  Result: {result.observation[:300]}"
            )

        steps_text = "\n\n".join(steps_summary)
        observations_text = "\n".join(accumulated_observations)

        prompt = f"""You are a helpful AI assistant. Generate a final, comprehensive answer based on the step-by-step execution results.

Original User Query: {user_query}

Execution Steps Summary:
{steps_text}

All Observations:
{observations_text}

Based on all the steps executed and their results, provide a clear, complete, and accurate final answer to the user's query.
Include specific details, numbers, and facts from the observations:"""

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()

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

        file_guidance = """\nGuidelines:\n- If any files are available, first attempt local analysis using python_coder or python_code.\n- Only use rag_retrieval or web_search if local analysis failed or is insufficient.\n- You may call different tools across iterations to complete the task.""" if self.file_paths else ""

        prompt = f"""You are a helpful AI assistant. Based on your reasoning, select the next action.{file_guidance}

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
                # Guarded fallback: if files exist and python_coder not yet attempted, try it first
                if self.file_paths and not self._attempted_coder:
                    logger.info("[ReAct Agent] Guard: Files present and coder not attempted; trying python_coder before RAG")
                    try:
                        self._attempted_coder = True
                        coder_result = await python_coder_tool.execute_code_task(
                            query=action_input,
                            file_paths=self.file_paths,
                            session_id=self.session_id
                        )
                        if coder_result.get("success") and str(coder_result.get("output", "")).strip():
                            final_observation = f"Code executed successfully:\n{coder_result.get('output','')}"
                            logger.info("")
                            logger.info("TOOL OUTPUT (Python Coder via Guard):")
                            for _line in str(final_observation).splitlines():
                                logger.info(_line)
                            logger.info("")
                            return final_observation
                        else:
                            logger.info("[ReAct Agent] Guard coder attempt insufficient; proceeding with RAG")
                    except Exception as e:
                        logger.warning(f"[ReAct Agent] Guard coder attempt failed: {e}; proceeding with RAG")

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
                self._attempted_coder = True

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
