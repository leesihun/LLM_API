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
from backend.tools.python_coder_tool import python_coder_tool
from backend.tools.file_analyzer_tool import file_analyzer

logger = logging.getLogger(__name__)

class ToolName(str, Enum):
    """Available tools for ReAct agent"""
    WEB_SEARCH = "web_search"
    RAG_RETRIEVAL = "rag_retrieval"
    PYTHON_CODER = "python_coder"
    FILE_ANALYZER = "file_analyzer"
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

        # Pre-step: If files are attached, first analyze file metadata
        if self.file_paths:
            logger.info("[ReAct Agent] Pre-step: Files detected, analyzing file metadata first")
            try:
                # Analyze files first to understand structure
                analyzer_result = file_analyzer.analyze(
                    file_paths=self.file_paths,
                    user_query=user_query
                )

                # Record file analysis as step 0
                pre_step = ReActStep(0)
                pre_step.thought = "Files attached; analyzing file metadata and structure first."
                pre_step.action = ToolName.FILE_ANALYZER
                pre_step.action_input = user_query

                if analyzer_result.get("success"):
                    # Include detailed structure information for LLM
                    obs_parts = [f"File analysis completed:\n{analyzer_result.get('summary','')}"]

                    # Add structure details if available
                    for file_result in analyzer_result.get("results", []):
                        if file_result.get("structure_summary"):
                            obs_parts.append(f"\nDetailed structure for {file_result.get('file', 'file')}:")
                            obs_parts.append(file_result["structure_summary"])

                    obs = "\n".join(obs_parts)
                else:
                    obs = f"File analysis failed: {analyzer_result.get('error','Unknown error')}"
                pre_step.observation = obs
                self.steps.append(pre_step)

                logger.info(f"[ReAct Agent] Pre-step file analysis: {analyzer_result.get('files_analyzed', 0)} files analyzed")
                logger.info(f"File Analysis Summary:\n{analyzer_result.get('summary', '')}")

            except Exception as e:
                logger.warning(f"[ReAct Agent] Pre-step file analysis error: {e}; continuing with tool loop")

        # ReAct loop
        while iteration < self.max_iterations:
            iteration += 1
            logger.info("\n" + "#" * 100)
            logger.info(f"ITERATION {iteration}/{self.max_iterations}")
            logger.info("#" * 100 + "\n")

            step = ReActStep(iteration)

            # PERFORMANCE OPTIMIZATION: Combined Thought-Action generation (1 LLM call instead of 2)
            logger.info("")
            logger.info("PHASE: THOUGHT + ACTION GENERATION (COMBINED)")
            logger.info("")
            thought, action, action_input = await self._generate_thought_and_action(user_query, self.steps)
            step.thought = thought
            step.action = action
            step.action_input = action_input
            
            logger.info("Generated Thought:")
            for _line in thought.splitlines():
                logger.info(_line)
            logger.info("")
            logger.info(f"Selected Action: {action}")
            logger.info(f"Action Input: {action_input[:200]}..." if len(action_input) > 200 else f"Action Input: {action_input}")
            logger.info("")

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

            # PERFORMANCE OPTIMIZATION: Early exit if observation contains complete answer
            if self._should_auto_finish(observation, iteration):
                logger.info("AUTO-FINISH TRIGGERED - Generating final answer from observation")
                final_answer = await self._generate_final_answer(user_query, self.steps)
                break

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
        logger.info("[ReAct Guided] Finalizing answer...")
        logger.info("=" * 100 + "\n")

        # PERFORMANCE OPTIMIZATION: Skip final answer generation if last step already contains it
        if self._is_final_answer_unnecessary(step_results, user_query):
            final_answer = step_results[-1].observation
            logger.info("⚡ SKIPPING final answer generation (last step contains sufficient answer)")
        else:
            logger.info("Generating comprehensive final answer from all steps...")
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

    def _build_context_for_plan_step(self, plan_step: PlanStep, previous_steps_context: str) -> str:
        """
        Build enhanced context for python_coder when executing in Plan-Execute mode

        Args:
            plan_step: Current plan step being executed
            previous_steps_context: Context from previous plan steps

        Returns:
            Formatted context string for python_coder
        """
        context_parts = []
        context_parts.append("=== Plan-Execute Mode Context ===\n")

        # Current step information
        context_parts.append(f"Current Step {plan_step.step_num}:")
        context_parts.append(f"  Goal: {plan_step.goal}")
        context_parts.append(f"  Success Criteria: {plan_step.success_criteria}")
        if plan_step.context:
            context_parts.append(f"  Additional Context: {plan_step.context}")
        context_parts.append("")

        # Previous steps results
        if previous_steps_context and previous_steps_context != "This is the first step.":
            context_parts.append("Previous Steps Results:")
            context_parts.append(previous_steps_context)
            context_parts.append("")

        context_parts.append("Use this context to:")
        context_parts.append("- Align code generation with the current step's goal")
        context_parts.append("- Build upon results from previous steps")
        context_parts.append("- Ensure success criteria are met")

        return "\n".join(context_parts)

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
            "python_code": ToolName.PYTHON_CODER,
            "python_coder": ToolName.PYTHON_CODER,
            "finish": ToolName.FINISH
        }

        tool_enum = tool_mapping.get(tool_name)
        if not tool_enum:
            return f"Unknown tool: {tool_name}", False

        # Special handling for FINISH tool - it doesn't execute, it generates final answer
        if tool_enum == ToolName.FINISH:
            logger.info(f"[ReAct Step {plan_step.step_num}] FINISH tool selected - generating final answer from context")

            # Generate final answer based on all previous context
            final_answer = await self._generate_final_answer_for_finish_step(
                user_query=user_query,
                plan_step=plan_step,
                context=context
            )

            # FINISH always succeeds - it's the termination condition
            return final_answer, True

        # Generate action input for this tool
        action_input = await self._generate_action_input_for_step(
            tool_name=tool_enum,
            plan_step=plan_step,
            user_query=user_query,
            context=context
        )

        # For python_coder, pass enhanced context with plan step information
        if tool_enum == ToolName.PYTHON_CODER:
            enhanced_context = self._build_context_for_plan_step(plan_step, context)
            logger.info(f"[ReAct Step {plan_step.step_num}] Passing plan context to python_coder")

            # Directly call python_coder with context
            result = await python_coder_tool.execute_code_task(
                query=action_input,
                file_paths=self.file_paths,
                session_id=self.session_id,
                context=enhanced_context
            )

            if result["success"]:
                observation = f"Code executed successfully:\n{result['output']}"
            else:
                observation = f"Code execution failed: {result.get('error', 'Unknown error')}"

            # Verify if goal is achieved
            success = await self._verify_step_success(
                plan_step=plan_step,
                observation=observation,
                tool_used=tool_name
            )

            return observation, success

        # Execute the tool (for non-python_coder tools)
        observation = await self._execute_action(tool_enum, action_input)

        # Verify if goal is achieved
        success = await self._verify_step_success(
            plan_step=plan_step,
            observation=observation,
            tool_used=tool_name
        )

        return observation, success

    async def _generate_final_answer_for_finish_step(
        self,
        user_query: str,
        plan_step: PlanStep,
        context: str
    ) -> str:
        """
        Generate final answer when FINISH tool is selected in plan execution.

        Args:
            user_query: Original user query
            plan_step: Current plan step (should be finish step)
            context: Context from all previous steps

        Returns:
            Final answer string
        """
        prompt = f"""You are completing a multi-step task. Generate a final, comprehensive answer based on all the work done so far.

Original User Query: {user_query}

Final Step Goal: {plan_step.goal}

All Previous Steps and Their Results:
{context}

Based on all the steps executed and their results above, provide a clear, complete, and accurate final answer to the user's query.
Your answer should:
1. Directly address the user's original question
2. Synthesize information from all previous steps
3. Include specific details, numbers, and facts from the observations
4. Be well-organized and easy to understand

Provide your final answer:"""

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()

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

    async def _generate_thought_and_action(
        self,
        query: str,
        steps: List[ReActStep]
    ) -> Tuple[str, str, str]:
        """
        PERFORMANCE OPTIMIZATION: Generate thought and action in single LLM call
        Reduces LLM calls by 50% in free mode execution
        
        Args:
            query: User query
            steps: Previous ReAct steps
            
        Returns:
            Tuple of (thought, action, action_input)
        """
        context = self._format_steps_context(steps)
        
        file_guidance = """\nGuidelines:\n- If any files are available, first attempt local analysis using python_coder or python_code.\n- Only use rag_retrieval or web_search if local analysis failed or is insufficient.\n- You may call different tools across iterations to complete the task.""" if self.file_paths else ""

        prompt = f"""You are a helpful AI assistant using the ReAct (Reasoning + Acting) framework.{file_guidance}

Question: {query}

{context}

Think step-by-step and then decide on an action. Provide BOTH your reasoning AND your action in this format:

THOUGHT: [Your step-by-step reasoning about what to do next]

ACTION: [Exactly one of: web_search, rag_retrieval, python_coder, finish]

ACTION INPUT: [The input for the selected action]

Available Actions:
1. web_search - Search the web for current information
2. rag_retrieval - Retrieve relevant documents from uploaded files
3. python_coder - Generate and execute Python code with file processing and data analysis
4. finish - Provide the final answer (use ONLY when you have complete information)

Note: File metadata analysis is done automatically when files are attached.

Now provide your thought and action:"""

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        response_text = response.content.strip()
        
        # Parse the combined response
        thought, action, action_input = self._parse_thought_and_action(response_text)
        
        logger.info("")
        logger.info("Combined thought-action generation completed")
        logger.info("")
        
        return thought, action, action_input

    def _parse_thought_and_action(self, response: str) -> Tuple[str, str, str]:
        """
        Parse combined thought-action response
        
        Args:
            response: LLM response containing thought and action
            
        Returns:
            Tuple of (thought, action, action_input)
        """
        import re
        
        thought = ""
        action = ""
        action_input = ""
        
        # Extract thought
        thought_match = re.search(r'THOUGHT:\s*(.+?)(?=ACTION:|$)', response, re.IGNORECASE | re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
        
        # Extract action
        action_match = re.search(r'ACTION:\s*(\w+)', response, re.IGNORECASE)
        if action_match:
            action = action_match.group(1).strip().lower()
        
        # Extract action input
        input_match = re.search(r'ACTION\s+INPUT:\s*(.+?)(?=\n\n|\Z)', response, re.IGNORECASE | re.DOTALL)
        if input_match:
            action_input = input_match.group(1).strip()
        
        # Fallback parsing if structured format not found
        if not thought:
            # Use first paragraph as thought
            paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
            if paragraphs:
                thought = paragraphs[0]
        
        if not action:
            # Try to find action using old parsing method
            action_match = re.search(r'action\s*:\s*(\w+)', response, re.IGNORECASE)
            if action_match:
                action = action_match.group(1).strip().lower()
        
        if not action_input:
            # Try old format
            input_match = re.search(r'action\s+input\s*:\s*(.+?)(?=\n\n|\Z)', response, re.IGNORECASE | re.DOTALL)
            if input_match:
                action_input = input_match.group(1).strip()
        
        # Validate and apply fuzzy matching for action
        if action:
            valid_actions = [e.value for e in ToolName]
            if action not in valid_actions:
                # Fuzzy matching
                action_mapping = {
                    "web": ToolName.WEB_SEARCH,
                    "search": ToolName.WEB_SEARCH,
                    "rag": ToolName.RAG_RETRIEVAL,
                    "retrieval": ToolName.RAG_RETRIEVAL,
                    "retrieve": ToolName.RAG_RETRIEVAL,
                    "document": ToolName.RAG_RETRIEVAL,
                    "python": ToolName.PYTHON_CODER,
                    "code": ToolName.PYTHON_CODER,
                    "coder": ToolName.PYTHON_CODER,
                    "generate": ToolName.PYTHON_CODER,
                    "done": ToolName.FINISH,
                    "answer": ToolName.FINISH,
                    "complete": ToolName.FINISH,
                }
                matched_action = action_mapping.get(action)
                if matched_action:
                    action = matched_action
                else:
                    action = ToolName.FINISH
        else:
            action = ToolName.FINISH
        
        # Default values
        if not thought:
            thought = "Proceeding with action execution."
        if not action_input:
            action_input = "No specific input provided."
        
        return thought, action, action_input

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
3. python_coder - Generate and execute Python code with file processing and data analysis (use for: data analysis, file processing, calculations, working with CSV/Excel/PDF files)
4. finish - Provide the final answer (use ONLY when you have complete information to answer the question)

Note: File metadata analysis is done automatically when files are attached.

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
                "python": ToolName.PYTHON_CODER,
                "code": ToolName.PYTHON_CODER,
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

    def _build_context_for_python_coder(self) -> str:
        """
        Build context string from ReAct execution history for python_coder

        Returns:
            Formatted context string with previous steps and observations
        """
        if not self.steps:
            return ""

        context_parts = []
        context_parts.append("=== Previous Agent Activity ===\n")

        # Include recent steps (last 3 steps or all if less than 3)
        recent_steps = self.steps[-3:] if len(self.steps) > 3 else self.steps

        for step in recent_steps:
            context_parts.append(f"Step {step.step_num}:")
            context_parts.append(f"  Thought: {step.thought[:300]}")
            context_parts.append(f"  Action: {step.action}")

            # Include observation summary
            obs_preview = step.observation[:500] if len(step.observation) > 500 else step.observation
            context_parts.append(f"  Result: {obs_preview}")

            # Highlight if there were errors
            if "error" in step.observation.lower() or "failed" in step.observation.lower():
                context_parts.append(f"  ⚠ Note: This action encountered errors")

            context_parts.append("")

        # Add summary of what's been tried
        tools_tried = [step.action for step in self.steps if step.action != ToolName.FINISH]
        if tools_tried:
            context_parts.append(f"Tools already attempted: {', '.join(set(tools_tried))}")

        context_parts.append("\nThis context shows what has already been tried. Use this information to:")
        context_parts.append("- Avoid repeating failed approaches")
        context_parts.append("- Build upon partial results from previous steps")
        context_parts.append("- Generate more targeted code based on what's already known")

        return "\n".join(context_parts)

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
                        # Build context from current execution history
                        context = self._build_context_for_python_coder()

                        coder_result = await python_coder_tool.execute_code_task(
                            query=action_input,
                            file_paths=self.file_paths,
                            session_id=self.session_id,
                            context=context
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

            elif action == ToolName.PYTHON_CODER:
                # Build context from current execution history
                context = self._build_context_for_python_coder()

                # Pass attached file paths, session_id, and execution context to python_coder
                result = await python_coder_tool.execute_code_task(
                    query=action_input,
                    file_paths=self.file_paths,
                    session_id=self.session_id,
                    context=context
                )
                self._attempted_coder = True

                logger.info("")
                logger.info("TOOL OUTPUT (Python Coder - Detailed):")
                logger.info(f"Success: {result['success']}")
                logger.info(f"Verification Iterations: {result.get('verification_iterations', 'N/A')}")
                logger.info(f"Execution Attempts: {result.get('execution_attempts', 'N/A')}")
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

                    # Build execution details string safely
                    exec_details_parts = []
                    if result.get('verification_iterations'):
                        exec_details_parts.append(f"{result['verification_iterations']} verification iterations")
                    if result.get('execution_attempts'):
                        exec_details_parts.append(f"{result['execution_attempts']} execution attempts")
                    if isinstance(result.get('execution_time'), (int, float)):
                        exec_details_parts.append(f"{result['execution_time']:.2f}s")

                    exec_details = ", ".join(exec_details_parts) if exec_details_parts else "completed"
                    final_observation = f"Code executed successfully:\n{result['output']}\n\nExecution details: {exec_details}"
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

            elif action == ToolName.FILE_ANALYZER:
                # FILE_ANALYZER should only be used in pre-step, but handle it just in case
                logger.warning("[ReAct Agent] FILE_ANALYZER called during ReAct loop (should only be in pre-step)")

                if not self.file_paths:
                    return "No files attached to analyze."

                result = file_analyzer.analyze(
                    file_paths=self.file_paths,
                    user_query=action_input
                )

                logger.info("")
                logger.info("TOOL OUTPUT (File Analyzer):")
                logger.info(f"Success: {result.get('success', False)}")
                logger.info(f"Files Analyzed: {result.get('files_analyzed', 0)}")
                logger.info("")

                if result.get("success"):
                    logger.info("Analysis Summary:")
                    for _line in str(result.get('summary', '')).splitlines():
                        logger.info(_line)
                    logger.info("")
                    final_observation = f"File analysis completed:\n{result.get('summary','')}"
                else:
                    logger.info("Error Details:")
                    for _line in str(result.get('error', 'Unknown error')).splitlines():
                        logger.info(_line)
                    logger.info("")
                    final_observation = f"File analysis failed: {result.get('error','Unknown error')}"

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
        PERFORMANCE OPTIMIZATION: Format previous steps with context pruning
        
        - If ≤3 steps: send all steps in full detail
        - If >3 steps: send summary of early steps + last 2 steps in detail
        
        This reduces context size and speeds up LLM processing
        """
        if not steps:
            return ""

        # If few steps, return all details
        if len(steps) <= 3:
            return self._format_all_steps(steps)
        
        # Context pruning: summary + recent steps
        context_parts = ["Previous Steps:\n"]
        
        # Summary of early steps
        early_steps = steps[:-2]
        tools_used = list(set([s.action for s in early_steps if s.action != ToolName.FINISH]))
        summary = f"Steps 1-{len(early_steps)} completed using: {', '.join(tools_used)}"
        context_parts.append(f"[Summary] {summary}\n")
        
        # Recent steps in full detail
        context_parts.append("\n[Recent Steps - Full Detail]")
        recent_steps = steps[-2:]
        for step in recent_steps:
            obs_display = step.observation[:500] if len(step.observation) > 500 else step.observation
            context_parts.append(f"""
Step {step.step_num}:
- Thought: {step.thought[:200]}...
- Action: {step.action}
- Action Input: {step.action_input[:200] if len(step.action_input) > 200 else step.action_input}
- Observation: {obs_display}
""")
        
        return "\n".join(context_parts)

    def _format_all_steps(self, steps: List[ReActStep]) -> str:
        """
        Format all steps in full detail (used when step count is low)
        """
        context_parts = ["Previous Steps:"]
        for step in steps:
            obs_display = step.observation[:]
            context_parts.append(f"""
Step {step.step_num}:
- Thought: {step.thought}
- Action: {step.action}
- Action Input: {step.action_input}
- Observation: {obs_display}
""")
        return "\n".join(context_parts)

    def _should_auto_finish(self, observation: str, step_num: int) -> bool:
        """
        PERFORMANCE OPTIMIZATION: Detect if observation contains complete answer
        
        Early exit optimization - automatically triggers finish if observation
        appears to contain a complete answer, saving unnecessary iterations.
        
        Args:
            observation: Latest observation from tool execution
            step_num: Current step number
            
        Returns:
            True if should auto-finish, False to continue iterations
        """
        # Need at least 2 steps before considering early exit
        if step_num < 2:
            return False
        
        # Check minimum length
        if len(observation) < 200:
            return False
        
        # Skip if observation contains errors
        if "error" in observation.lower() or "failed" in observation.lower():
            # But allow if also has success indicators
            if "success" not in observation.lower():
                return False
        
        # Check for answer indicators
        answer_phrases = [
            "the answer",
            "result is",
            "therefore",
            "in conclusion",
            "based on",
            "to summarize",
            "in summary",
            "concluded",
            "final result",
            "outcome is"
        ]
        
        observation_lower = observation.lower()
        has_answer_phrase = any(phrase in observation_lower for phrase in answer_phrases)
        
        # Check for substantive content (numbers, facts, concrete results)
        has_numbers = any(char.isdigit() for char in observation)
        is_substantial = len(observation) > 300
        
        # Auto-finish if observation looks like a complete answer
        should_finish = has_answer_phrase or (has_numbers and is_substantial)
        
        if should_finish:
            logger.info("")
            logger.info("⚡ EARLY EXIT: Observation contains complete answer")
            logger.info("")
        
        return should_finish

    def _is_final_answer_unnecessary(self, step_results: List[StepResult], user_query: str) -> bool:
        """
        PERFORMANCE OPTIMIZATION: Check if final answer generation can be skipped
        
        Skips final LLM call if the last step already contains a comprehensive answer.
        
        Args:
            step_results: Results from all executed steps
            user_query: Original user query
            
        Returns:
            True if final answer generation can be skipped, False otherwise
        """
        if not step_results:
            return False
        
        last_step = step_results[-1]
        
        # Only skip if last step was successful
        if not last_step.success:
            return False
        
        observation = last_step.observation
        
        # Check if observation is substantial
        if len(observation) < 150:
            return False
        
        # Check if observation appears to be a complete answer (not just raw data)
        observation_lower = observation.lower()
        
        # If it's just code output or raw data, we need synthesis
        raw_data_indicators = [
            "dtype:",
            "columns:",
            "shape:",
            "<class",
            "array(",
            "dataframe",
        ]
        if any(indicator in observation_lower for indicator in raw_data_indicators):
            return False
        
        # If observation contains conclusion/summary phrases, it's likely complete
        complete_indicators = [
            "the answer",
            "in conclusion",
            "to summarize",
            "based on the",
            "the result",
            "therefore",
            "this shows",
            "analysis reveals"
        ]
        has_conclusion = any(phrase in observation_lower for phrase in complete_indicators)
        
        # If only one step executed and it has a conclusive answer, we can skip
        if len(step_results) == 1 and has_conclusion:
            return True
        
        # For multi-step executions, only skip if last step is explicitly marked as final/summary
        last_goal_lower = last_step.goal.lower()
        is_final_step = any(word in last_goal_lower for word in ["final", "answer", "summary", "synthesize", "combine"])
        
        # Skip if it's the final step and has substantial output
        should_skip = is_final_step and len(observation) > 200
        
        if should_skip:
            logger.info(f"Final answer unnecessary: Last step '{last_step.goal}' contains complete answer ({len(observation)} chars)")
        
        return should_skip

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
