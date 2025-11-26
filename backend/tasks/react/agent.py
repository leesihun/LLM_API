from typing import List, Optional, Tuple, Dict, Any
from .models import ReActStep, ToolName, ReActResult
from .generators import ThoughtActionGenerator, AnswerGenerator, ContextFormatter
from .execution import ToolExecutor, StepVerifier
from .planning import PlanExecutor
from backend.models.schemas import ChatMessage, PlanStep
from backend.utils.llm_factory import LLMFactory
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ReActAgent:
    def __init__(self, max_iterations: int = 20, user_id: str = "default"):
        self.max_iterations = max_iterations
        self.user_id = user_id

        # Create LLM instance using factory with user_id for prompt logging
        self.llm = LLMFactory.create_llm(user_id=user_id)

        # Initialize specialized modules
        self.context_formatter = ContextFormatter()
        self.verifier = StepVerifier(self.llm)
        self.answer_generator = AnswerGenerator(self.llm)
        self.tool_executor = ToolExecutor(self.llm, user_id=user_id)
        self.thought_action_generator = None  # Initialized per execution (needs file_paths)
        self.plan_executor = PlanExecutor(
            tool_executor=self.tool_executor,
            context_formatter=self.context_formatter,
            verifier=self.verifier,
            answer_generator=self.answer_generator,
            llm=self.llm
        )

        # Execution state
        self.steps: List[ReActStep] = []
        self.file_paths: Optional[List[str]] = None
        self.session_id: Optional[str] = None

    async def execute(
        self,
        messages: List[ChatMessage],
        session_id: Optional[str],
        user_id: str,
        file_paths: Optional[List[str]] = None
    ) -> Tuple[str, Dict[str, Any]]:

        # Update user_id if different from initialization
        if user_id != self.user_id:
            self.user_id = user_id
            # Recreate LLM with new user_id for proper logging
            self.llm = LLMFactory.create_llm(user_id=user_id)
            self.verifier = StepVerifier(self.llm)
            self.answer_generator = AnswerGenerator(self.llm)
            self.tool_executor = ToolExecutor(self.llm, user_id=user_id)
            self.plan_executor.llm = self.llm
            self.plan_executor.tool_executor = self.tool_executor

        self.file_paths = file_paths
        self.session_id = session_id
        self.context_formatter.session_id = session_id
        self.thought_action_generator = ThoughtActionGenerator(self.llm, file_paths)

        # Extract user query
        user_query = messages[-1].content

        # Log execution start
        logger.header("REACT AGENT EXECUTION", "heavy")
        exec_params = {
            "User ID": user_id,
            "Session ID": session_id,
            "Attached Files": f"{len(file_paths)} files" if file_paths else "None",
            "Max Iterations": self.max_iterations
        }
        logger.key_values(exec_params, title="Execution Parameters")
        logger.multiline(user_query, title="User Query", max_lines=100)

        # Initialize execution
        self.steps = []
        iteration = 0
        final_answer = ""

        # NOTE: File analysis pre-step REMOVED!
        # Reason: python_coder tool already performs comprehensive file analysis
        # during its _prepare_files() phase, including metadata extraction and
        # context building. This avoids:
        # - Duplicate file analysis work
        # - File path mismatches (temp files vs scratch directory files)
        # - Extra LLM calls
        # The python_coder will analyze files from scratch directory where they're
        # actually used, ensuring consistency.

        # ReAct loop
        while iteration < self.max_iterations:
            iteration += 1
            logger.section(f"ITERATION {iteration}/{self.max_iterations}")

            step = ReActStep(iteration)

            # Generate thought and action
            logger.subsection("Thought + Action Generation")
            context = self.context_formatter.build_tool_context(self.steps)
            thought, action, action_input = await self.thought_action_generator.generate(
                user_query=user_query,
                steps=self.steps,
                context=context
            )

            step.thought = thought
            step.action = action
            step.action_input = action_input

            # Log
            logger.multiline(thought, title="Thought", max_lines=200)
            logger.key_values({
                "Action": action,
                "Input": action_input[:]
            })

            # Check if done
            if action == ToolName.FINISH:
                logger.subsection("Final Answer Generation")
                final_answer = await self.answer_generator.generate_final_answer(user_query, self.steps)

                step.observation = "Task completed"
                self.steps.append(step)
                logger.multiline(final_answer, title="Final Answer", max_lines=50)
                break

            # Execute action
            logger.subsection("Action Execution & Observation")
            observation = await self.tool_executor.execute(
                action=action,
                action_input=action_input,
                file_paths=self.file_paths,
                session_id=self.session_id,
                steps=self.steps
            )

            step.observation = observation
            logger.multiline(observation, title="Observation", max_lines=5000000)

            # Store step
            self.steps.append(step)

            # Check for early exit with enhanced auto-finish
            context = self.context_formatter.build_tool_context(self.steps)
            should_finish, confidence, reason = await self.verifier.should_auto_finish_enhanced(
                observation=observation,
                user_query=user_query,
                iteration=iteration,
                steps_context=context
            )

            if should_finish:
                logger.info(f"AUTO-FINISH ENHANCED (confidence: {confidence:.2f}) - {reason}")
                final_answer = await self.answer_generator.generate_final_answer(user_query, self.steps)
                break

        # Generate final answer if not done
        if not final_answer:
            logger.warning("Max iterations reached - generating final answer")
            final_answer = await self.answer_generator.generate_final_answer(user_query, self.steps)

        # Log completion
        logger.header("EXECUTION COMPLETED", "heavy")
        summary = {
            "Total Steps": len(self.steps),
            "Total Iterations": iteration,
            "Status": "Success",
            "Tools Used": ", ".join(set(str(s.action) for s in self.steps if s.action != ToolName.FINISH))
        }
        logger.key_values(summary, title="Execution Summary")
        logger.multiline(final_answer, title="Final Answer", max_lines=50)

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
    ) -> Tuple[str, List]:
        """
        Execute ReAct in guided mode with structured plan.

        Args:
            plan_steps: Structured plan steps to execute
            messages: Conversation messages
            session_id: Session ID
            user_id: User identifier
            file_paths: Optional list of file paths
            max_iterations_per_step: Max iterations per step

        Returns:
            Tuple of (final_answer, step_results)
        """
        # Store context
        self.file_paths = file_paths
        self.session_id = session_id
        self.context_formatter.session_id = session_id

        # Extract user query
        user_query = messages[-1].content

        # Load persisted variables for context-aware execution
        if self.session_id:
            await self._load_variables()

        # Delegate to plan executor
        final_answer, step_results = await self.plan_executor.execute_plan(
            plan_steps=plan_steps,
            user_query=user_query,
            file_paths=file_paths,
            session_id=session_id,
            max_iterations_per_step=max_iterations_per_step
        )

        return final_answer, step_results

    async def _load_variables(self) -> None:
        """
        Load persisted python_coder variable metadata for the active session.
        """
        try:
            from backend.tools.python_coder.variable_storage import VariableStorage

            var_metadata = VariableStorage.get_metadata(self.session_id)
            if var_metadata:
                self.context_formatter.set_variables(var_metadata)
                logger.info(f"[ReActAgent] Loaded {len(var_metadata)} saved variables")
        except Exception as exc:
            logger.warning(f"[ReActAgent] Failed to load variables: {exc}")

    def _build_metadata(self) -> Dict[str, Any]:
        """Build execution metadata dictionary."""
        tools_used = list(set([
            step.action for step in self.steps
            if step.action != ToolName.FINISH
        ]))

        execution_steps = [step.to_dict() for step in self.steps]

        return {
            "agent_type": "react",
            "total_iterations": len(self.steps),
            "max_iterations": self.max_iterations,
            "tools_used": tools_used,
            "execution_steps": execution_steps,
            "execution_trace": self._get_trace()
        }

    def _get_trace(self) -> str:
        """Get full trace of ReAct execution for debugging."""
        if not self.steps:
            return "No steps executed."

        trace = ["=== ReAct Execution Trace ===\n"]
        for step in self.steps:
            trace.append(str(step))

        return "\n".join(trace)


class ReActAgentFactory:
    """
    Factory for creating ReActAgent instances.

    This factory pattern ensures each request gets a fresh agent instance,
    avoiding singleton issues and state leakage.

    Usage:
        agent = ReActAgentFactory.create(max_iterations=6)
        response, metadata = await agent.execute(messages, session_id, user_id, file_paths)
    """

    @staticmethod
    def create(max_iterations: int = 6) -> ReActAgent:
        """
        Create a new ReActAgent instance.

        Args:
            max_iterations: Maximum iterations for ReAct loop (default: 6)

        Returns:
            Fresh ReActAgent instance
        """
        return ReActAgent(max_iterations=max_iterations)


# Global react_agent instance for backward compatibility
# Note: Using factory pattern (ReActAgentFactory.create()) is recommended
react_agent = ReActAgentFactory.create()
