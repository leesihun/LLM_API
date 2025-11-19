"""
ReAct Agent - Main Orchestration Module

This module contains the main ReActAgent class that orchestrates the complete
ReAct (Reasoning + Acting) execution flow by coordinating specialized modules.

The ReActAgent is the public interface for ReAct execution and delegates to:
- ThoughtActionGenerator: For generating thoughts and selecting actions
- ToolExecutor: For executing tools and managing guard logic
- AnswerGenerator: For synthesizing final answers
- ContextManager: For building and formatting context
- StepVerifier: For verification and auto-finish detection
- PlanExecutor: For guided plan-based execution

Architecture:
- Orchestration only - no inline prompts or complex logic
- Delegates to specialized modules for all major operations
- Uses llm_factory for LLM instance creation
- Uses PromptRegistry for all prompts
"""

from typing import List, Optional, Tuple, Dict, Any

from .models import ReActStep, ToolName, ReActResult
from .thought_action_generator import ThoughtActionGenerator
from .tool_executor import ToolExecutor
from .answer_generator import AnswerGenerator
from .context_manager import ContextManager
from .verification import StepVerifier
from .plan_executor import PlanExecutor
from backend.models.schemas import ChatMessage, PlanStep
from backend.utils.llm_factory import LLMFactory
from backend.utils.logging_utils import get_logger
from backend.tools.file_analyzer import file_analyzer

logger = get_logger(__name__)


class ReActAgent:
    """
    Main ReAct Agent for Reasoning + Acting execution.

    This class orchestrates the ReAct pattern:
    1. Thought: Reason about what to do next
    2. Action: Select and execute a tool
    3. Observation: Observe the result
    4. Repeat until answer is ready

    Public Methods:
        execute(): Free-form ReAct loop execution
        execute_with_plan(): Guided execution with structured plan

    Architecture:
        All logic is delegated to specialized modules. This class only
        handles initialization, coordination, and public API.
    """

    def __init__(self, max_iterations: int = 6):
        """
        Initialize ReAct agent with specialized modules.

        Args:
            max_iterations: Maximum iterations for ReAct loop (reduced to 6 for efficiency)
        """
        self.max_iterations = max_iterations

        # Create LLM instance using factory
        self.llm = LLMFactory.create_llm()

        # Initialize specialized modules
        self.context_manager = ContextManager()
        self.verifier = StepVerifier(self.llm)
        self.answer_generator = AnswerGenerator(self.llm)
        self.tool_executor = ToolExecutor(self.llm)
        self.thought_action_generator = None  # Initialized per execution (needs file_paths)
        self.plan_executor = PlanExecutor(
            tool_executor=self.tool_executor,
            context_manager=self.context_manager,
            verifier=self.verifier,
            answer_generator=self.answer_generator,
            llm=self.llm
        )

        # Execution state
        self.steps: List[ReActStep] = []
        self.file_paths: Optional[List[str]] = None
        self.session_id: Optional[str] = None
        self._attempted_coder: bool = False

    async def execute(
        self,
        messages: List[ChatMessage],
        session_id: Optional[str],
        user_id: str,
        file_paths: Optional[List[str]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Execute ReAct loop in free-form mode.

        Args:
            messages: Conversation messages
            session_id: Session ID
            user_id: User identifier
            file_paths: Optional list of file paths for code execution

        Returns:
            Tuple of (final_answer, metadata)
        """
        # Store context
        self.file_paths = file_paths
        self.session_id = session_id
        self._attempted_coder = False
        self.context_manager.session_id = session_id

        # Initialize thought-action generator with file context
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
        logger.multiline(user_query, title="User Query", max_lines=10)

        # Initialize execution
        self.steps = []
        iteration = 0
        final_answer = ""

        # Load notepad and variables for context injection
        if self.session_id:
            await self._load_notepad_and_variables()

        # Pre-step: Analyze files if attached
        if self.file_paths:
            await self._analyze_files_pre_step(user_query)

        # ReAct loop
        while iteration < self.max_iterations:
            iteration += 1
            logger.section(f"ITERATION {iteration}/{self.max_iterations}")

            step = ReActStep(iteration)

            # Generate thought and action
            logger.subsection("Thought + Action Generation")
            context = self.context_manager.build_tool_context(self.steps)
            thought, action, action_input = await self.thought_action_generator.generate(
                user_query=user_query,
                steps=self.steps,
                context=context
            )

            step.thought = thought
            step.action = action
            step.action_input = action_input

            # Log
            logger.multiline(thought, title="Thought", max_lines=20)
            logger.key_values({
                "Action": action,
                "Input": action_input[:150] + "..." if len(action_input) > 150 else action_input
            })

            # Check if done
            if action == ToolName.FINISH:
                logger.subsection("Final Answer Generation")
                final_answer = await self.answer_generator.generate_final_answer(user_query, self.steps)

                # Fallback if insufficient
                if not final_answer or len(final_answer.strip()) < 10:
                    logger.warning("Generated answer insufficient, using fallback")
                    final_answer = self.answer_generator.extract_from_steps(user_query, self.steps)

                step.observation = "Task completed"
                self.steps.append(step)
                logger.multiline(final_answer, title="Final Answer", max_lines=50)
                break

            # Execute action
            logger.subsection("Action Execution & Observation")
            observation, self._attempted_coder = await self.tool_executor.execute(
                action=action,
                action_input=action_input,
                file_paths=self.file_paths,
                session_id=self.session_id,
                attempted_coder=self._attempted_coder,
                steps=self.steps
            )

            step.observation = observation
            logger.multiline(observation, title="Observation", max_lines=30)

            # Store step
            self.steps.append(step)

            # Check for early exit
            if self.verifier.should_auto_finish(observation, iteration):
                logger.info("AUTO-FINISH TRIGGERED - Generating final answer")
                final_answer = await self.answer_generator.generate_final_answer(user_query, self.steps)
                break

        # Generate final answer if not done
        if not final_answer:
            logger.warning("Max iterations reached - generating final answer")
            final_answer = await self.answer_generator.generate_final_answer(user_query, self.steps)

        # Final validation
        if not final_answer or not final_answer.strip():
            logger.error("Empty final answer - using fallback")
            final_answer = self.answer_generator.extract_from_steps(user_query, self.steps)

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

        # Generate and save notepad entry (post-execution hook)
        if self.session_id:
            await self._generate_and_save_notepad_entry(user_query, final_answer)

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
        self.context_manager.session_id = session_id

        # Extract user query
        user_query = messages[-1].content

        # Load notepad and variables for context injection
        if self.session_id:
            await self._load_notepad_and_variables()

        # Delegate to plan executor
        final_answer, step_results = await self.plan_executor.execute_plan(
            plan_steps=plan_steps,
            user_query=user_query,
            file_paths=file_paths,
            session_id=session_id,
            max_iterations_per_step=max_iterations_per_step
        )

        # Generate and save notepad entry (post-execution hook)
        if self.session_id:
            await self._generate_and_save_notepad_entry(user_query, final_answer)

        return final_answer, step_results

    async def _load_notepad_and_variables(self):
        """
        Load session notepad and variables for context injection.
        """
        try:
            from backend.tools.notepad import SessionNotepad
            from backend.tools.python_coder.variable_storage import VariableStorage
            
            # Load notepad
            notepad = SessionNotepad.load(self.session_id)
            
            # Load variable metadata
            var_metadata = VariableStorage.get_metadata(self.session_id)
            
            # Set in context manager
            self.context_manager.set_notepad(notepad, var_metadata)
            
            if notepad.get_entries_count() > 0:
                logger.info(f"[ReActAgent] Loaded notepad with {notepad.get_entries_count()} entries")
            if var_metadata:
                logger.info(f"[ReActAgent] Loaded {len(var_metadata)} saved variables")
            
        except Exception as e:
            logger.warning(f"[ReActAgent] Failed to load notepad/variables: {e}")

    async def _analyze_files_pre_step(self, user_query: str) -> None:
        """
        Pre-step: Analyze file metadata before starting ReAct loop.

        Args:
            user_query: User's query for context
        """
        logger.subsection("PRE-STEP: File Analysis")
        try:
            analyzer_result = file_analyzer.analyze(
                file_paths=self.file_paths,
                user_query=user_query
            )

            pre_step = ReActStep(0)
            pre_step.thought = "Files attached; analyzing file metadata and structure first."
            pre_step.action = ToolName.FILE_ANALYZER
            pre_step.action_input = user_query

            if analyzer_result.get("success"):
                obs_parts = [f"File analysis completed:\n{analyzer_result.get('summary','')}"]

                # Add structure details
                for file_result in analyzer_result.get("results", []):
                    if file_result.get("structure_summary"):
                        obs_parts.append(f"\nDetailed structure for {file_result.get('file', 'file')}:")
                        obs_parts.append(file_result["structure_summary"])

                obs = "\n".join(obs_parts)
                logger.success(f"File analysis completed", f"{analyzer_result.get('files_analyzed', 0)} files")
            else:
                obs = f"File analysis failed: {analyzer_result.get('error','Unknown error')}"
                logger.failure("File analysis failed", analyzer_result.get('error','Unknown error'))

            pre_step.observation = obs
            self.steps.append(pre_step)

        except Exception as e:
            logger.warning(f"Pre-step file analysis error: {e}; continuing")

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
    
    async def _generate_and_save_notepad_entry(self, user_query: str, final_answer: str):
        """
        Generate and save notepad entry after execution completes.
        
        This method analyzes the execution steps, extracts code and variables,
        and creates a structured notepad entry for future reference.
        
        Args:
            user_query: Original user query
            final_answer: Final answer generated by the agent
        """
        try:
            from backend.config.prompts.react_agent import get_notepad_entry_generation_prompt
            from backend.tools.notepad import SessionNotepad
            from backend.tools.python_coder.variable_storage import VariableStorage
            from langchain_core.messages import HumanMessage
            import json
            
            logger.info("[ReActAgent] Generating notepad entry...")
            
            # Build steps summary
            steps_summary = []
            python_coder_result = None
            
            for step in self.steps:
                step_info = f"Step {step.step_num}: {step.action}"
                if step.observation:
                    obs_preview = step.observation[:200]
                    step_info += f" - {obs_preview}..."
                steps_summary.append(step_info)
                
                # Check if python_coder was used
                if step.action == ToolName.PYTHON_CODER and "success" in step.observation.lower():
                    # Try to extract code and namespace from observation
                    # The observation might contain result info
                    python_coder_result = step
            
            steps_text = "\n".join(steps_summary)
            
            # Generate notepad entry using LLM
            prompt = get_notepad_entry_generation_prompt(
                user_query=user_query,
                steps_summary=steps_text,
                final_answer=final_answer
            )
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            response_text = response.content.strip()
            
            # Parse JSON response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            entry_data = json.loads(response_text.strip())
            
            task = entry_data.get("task", "general_task")
            description = entry_data.get("description", "Task completed")
            code_summary = entry_data.get("code_summary", "")
            variables_list = entry_data.get("variables", [])
            key_outputs = entry_data.get("key_outputs", "")
            
            logger.info(f"[ReActAgent] Entry generated: [{task}] {description}")
            
            # Load notepad
            notepad = SessionNotepad.load(self.session_id)
            
            # If python_coder was used, save code and variables
            code_file = None
            if python_coder_result and code_summary:
                # Try to find the code in the step or in tool_executor results
                # For now, we'll save based on the session's script.py if it exists
                from pathlib import Path
                from backend.config.settings import settings
                
                session_dir = Path(settings.python_code_execution_dir) / self.session_id
                script_path = session_dir / "script.py"
                
                if script_path.exists():
                    code_content = script_path.read_text(encoding='utf-8')
                    code_file = notepad.save_code_file(code_content, task)
                    logger.info(f"[ReActAgent] Saved code as: {code_file}")
                    
                    # Save variables if they exist
                    # Try to load namespace from the last execution
                    # The tool_executor should have stored this somewhere
                    # For now we'll use the variables list from LLM
                    # In a real scenario, the python_coder would pass namespace info
                    
                    # Check if there's a namespace in memory from recent execution
                    # This would need to be passed through the observation or stored
                    # For simplicity, we'll skip actual variable saving for now
                    # and rely on the next execution to capture them properly
                    
                    logger.info(f"[ReActAgent] Variables mentioned: {variables_list}")
            
            # Add entry to notepad
            entry_id = notepad.add_entry(
                task=task,
                description=description,
                code_file=code_file,
                variables_saved=variables_list,
                key_outputs=key_outputs
            )
            
            # Save notepad
            notepad.save()
            
            logger.success(f"[ReActAgent] Notepad entry #{entry_id} saved successfully")
            
        except Exception as e:
            logger.error(f"[ReActAgent] Failed to generate notepad entry: {e}")
            # Don't fail the entire execution if notepad generation fails


# Global ReAct agent instance
react_agent = ReActAgent(max_iterations=6)
