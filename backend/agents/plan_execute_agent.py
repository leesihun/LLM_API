"""
Plan-Execute Agent
Creates a plan and executes it step-by-step using ReAct agents
"""
import json
import re
from typing import List, Dict, Optional, Any

import config
from backend.agents.base_agent import Agent
from backend.agents.react_agent import ReActAgent, ReActMaxIterationsError
from tools_config import format_tools_for_llm


class PlanExecuteAgent(Agent):
    """
    Plan-Execute agent with multi-level replanning:

    1. Creates a multi-step plan
    2. Executes each step using a ReAct agent
    3. TWO replanning mechanisms:
       a) PLAN_REPLAN_ON_FAILURE: Replans when individual steps fail (step-level)
       b) FINISHED flag iteration: Replans when answer is incomplete (plan-level)

    These work together: (a) handles execution failures, (b) handles missing information.
    """

    def __init__(self, model: str = None, temperature: float = None):
        super().__init__(model, temperature)
        self.max_steps = config.PLAN_MAX_STEPS
        self.max_iterations = config.PLAN_MAX_ITERATIONS
        self.replan_on_failure = config.PLAN_REPLAN_ON_FAILURE
        self.share_context = config.PLAN_SHARE_CONTEXT

    def _limit_history(self, conversation_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Limit conversation history based on config

        Args:
            conversation_history: Full conversation history

        Returns:
            Limited conversation history (most recent messages)
        """
        if config.PLAN_MAX_HISTORY_MESSAGES <= 0:
            # 0 or negative = no limit
            return conversation_history

        # Return last N messages
        return conversation_history[-config.PLAN_MAX_HISTORY_MESSAGES:]

    def run(
        self,
        user_input: str,
        conversation_history: List[Dict[str, str]],
        attached_files: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Run plan-execute agent with iterative replanning

        This method implements a loop that:
        1. Creates a plan
        2. Executes the plan
        3. Synthesizes answer and checks if FINISHED=True
        4. If FINISHED=False, creates ADDITIONAL plan and repeats (up to max_iterations)

        Args:
            user_input: User's message
            conversation_history: Full conversation history
            attached_files: Optional list of file metadata

        Returns:
            Final answer combining all steps
        """
        # Store attached files for passing to ReAct agents
        self.attached_files = attached_files

        # Track all plans and results across iterations
        all_results = []  # Accumulates results from ALL iterations
        current_query = user_input
        iteration = 0
        last_plan = None  # Track last successful plan for fallback
        last_synthesis_result = None  # Track last synthesis for fallback

        # ITERATIVE PLANNING LOOP
        # Each iteration creates a NEW plan, executes it, and checks if done
        while iteration < self.max_iterations:
            iteration += 1
            print(f"[PLAN-EXECUTE] Starting iteration {iteration}/{self.max_iterations}")

            # STEP 1: CREATE NEW PLAN (or first plan if iteration==1)
            plan = self._create_plan(current_query, conversation_history)

            # Replan
            if not plan:
                print(f"[PLAN-EXECUTE] Failed to create plan. Replanning...")
                plan = self._create_plan(current_query, conversation_history)

            # STEP 2: EXECUTE THE NEW PLAN
            results = self._execute_plan(plan, user_input, conversation_history)
            all_results.extend(results)  # Add to cumulative results

            # Track for fallback
            last_plan = plan

            # STEP 3: SYNTHESIZE FROM ALL RESULTS AND CHECK IF FINISHED
            synthesis_result = self._synthesize_answer(plan, all_results, user_input, conversation_history)
            last_synthesis_result = synthesis_result

            # DECISION POINT: Are we done?
            if synthesis_result['finished']:
                # FINISHED=True: We have a complete answer!
                print(f"[PLAN-EXECUTE] Task completed in {iteration} iteration(s)")
                return synthesis_result['answer']

            # FINISHED=False: Need more information, prepare for NEXT ITERATION
            print(f"[PLAN-EXECUTE] Iteration {iteration} incomplete. Preparing additional plan...")

            # Check if we've reached max iterations
            if iteration >= self.max_iterations:
                print(f"[PLAN-EXECUTE] Reached max iterations ({self.max_iterations}). Returning partial answer.")
                # Return the answer with a note that it's incomplete
                partial_answer = synthesis_result['answer']
                if synthesis_result.get('missing_info'):
                    partial_answer += f"\n\n**Note**: This answer is incomplete. {synthesis_result['missing_info']}"
                return partial_answer

            # PREPARE FOR NEXT ITERATION
            # Create a targeted query for the NEXT PLAN based on what's missing
            if synthesis_result.get('missing_info'):
                continuation_reason = f"What's still needed:\n{synthesis_result['missing_info']}"
            else:
                continuation_reason = "The previous plan was insufficient. Create additional steps to fully answer the question."

            current_query = self.load_prompt(
                "agents/plan_replan_continuation.txt",
                user_input=user_input,
                previous_steps=self._format_previous_steps(all_results),
                continuation_reason=continuation_reason
            )

            # Add context for next iteration (use copy to avoid mutating original)
            if self.share_context:
                conversation_history = conversation_history + [{
                    "role": "assistant",
                    "content": f"Iteration {iteration} partial result: {synthesis_result['answer']}"
                }]

            # Loop continues: will create ANOTHER plan in next iteration!

        # Fallback: Loop exited without returning (e.g., via break at line 95)
        print(f"[PLAN-EXECUTE] Loop exited. Returning best available synthesis.")
        if last_synthesis_result:
            # We have at least one synthesis result
            return last_synthesis_result['answer']
        elif last_plan and all_results:
            # We have results but no synthesis, create one
            final_synthesis = self._synthesize_answer(last_plan, all_results, user_input, conversation_history)
            return final_synthesis['answer']
        else:
            # No results at all (shouldn't happen, but safety fallback)
            return "I apologize, but I was unable to complete your request. Please try rephrasing your question."

    def _create_plan(
        self,
        user_input: str,
        conversation_history: List[Dict[str, str]],
        replan_reason: str = None
    ) -> Optional[Dict]:
        """
        Create execution plan

        Args:
            user_input: User's message
            conversation_history: Conversation history
            replan_reason: Optional reason for replanning (for logging)

        Returns:
            Plan dictionary or None
        """
        # Get tools description
        tools_desc = format_tools_for_llm()

        # Load planning prompt
        planning_prompt = self.load_prompt(
            "agents/plan_system.txt",
            tools=tools_desc,
            max_steps=self.max_steps,
            input=user_input
        )

        # Build messages with conversation history
        messages = [{"role": "system", "content": planning_prompt}]

        # Add conversation history (if enabled)
        if config.PLAN_INCLUDE_FULL_HISTORY and conversation_history:
            # Limit history if configured
            limited_history = self._limit_history(conversation_history)
            messages.extend(limited_history)

        # Add current user input
        messages.append({"role": "user", "content": user_input})

        # Get plan from LLM with stage tracking
        stage = f"plan_creation_{replan_reason}" if replan_reason else "plan_creation"
        response = self.call_llm(messages, temperature=0.7, stage=stage)

        # Parse JSON plan
        try:
            # Extract JSON from response (handle markdown code blocks and think tags)
            json_str = response

            # Remove <think> tags if present (Claude models use these for reasoning)
            if "<think>" in json_str and "</think>" in json_str:
                # Remove everything between <think> and </think>
                json_str = re.sub(r'<think>.*?</think>', '', json_str, flags=re.DOTALL).strip()

            # Handle markdown code blocks
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()

            plan = json.loads(json_str)

            # Validate plan structure
            if "plan" not in plan or not isinstance(plan["plan"], list):
                return None

            # Limit steps
            if len(plan["plan"]) > self.max_steps:
                plan["plan"] = plan["plan"][:self.max_steps]

            return plan

        except (json.JSONDecodeError, IndexError, KeyError) as e:
            # Log the error for debugging
            print(f"[PLAN-EXECUTE] Failed to parse plan from LLM response")
            print(f"[PLAN-EXECUTE] Error: {str(e)}")
            print(f"[PLAN-EXECUTE] Raw response preview: {response[:500]}...")
            return None

    def _execute_plan(
        self,
        plan: Dict,
        user_input: str,
        conversation_history: List[Dict[str, str]]
    ) -> List[Dict]:
        """
        Execute each step of the plan

        Args:
            plan: Plan dictionary
            user_input: Original user input
            conversation_history: Conversation history

        Returns:
            List of step results
        """
        results = []
        shared_context = conversation_history.copy() if self.share_context else []

        for step_info in plan["plan"]:
            step_num = step_info.get("step", len(results) + 1)
            description = step_info.get("description", "")
            tool_needed = step_info.get("tool")

            # Execute step using ReAct agent
            step_result = self._execute_step(
                step_num=step_num,
                step_info=step_info,
                previous_results=results,
                goal=plan.get("goal", user_input),
                context=shared_context,
                full_plan=plan,
                original_user_input=user_input
            )

            results.append({
                "step": step_num,
                "description": description,
                "tool": tool_needed,
                "result": step_result["answer"],
                "success": step_result["success"]
            })

            # If sharing context, add result to shared context
            if self.share_context:
                shared_context.append({
                    "role": "assistant",
                    "content": f"Step {step_num} result: {step_result['answer']}"
                })

            # Handle failure - only replan if React agent tried enough iterations
            if not step_result["success"] and self.replan_on_failure:
                # Check if React agent tried enough iterations before replanning
                react_iterations = step_result.get("react_iterations", 0)
                min_iterations = config.PLAN_MIN_REACT_ITERATIONS_FOR_REPLAN

                if react_iterations < min_iterations:
                    # React agent failed early (not all iterations exhausted)
                    # Don't replan - this was likely a quick failure (parsing error, tool error, etc.)
                    print(f"[PLAN-EXECUTE] Step {step_num} failed after only {react_iterations} ReAct iterations")
                    print(f"[PLAN-EXECUTE] Minimum {min_iterations} iterations required for replanning. Continuing with current plan.")
                    # Continue to next step without replanning
                    continue

                # React agent exhausted all iterations - proceed with replanning
                print(f"[PLAN-EXECUTE] Step {step_num} failed after {react_iterations} ReAct iterations (>= {min_iterations}). Replanning...")

                # Re-plan from this point with full context
                # Format previous results for prompt
                prev_results_str = json.dumps(results, indent=2)

                remaining_goal = self.load_prompt(
                    "agents/plan_failure_replan.txt",
                    step_num=step_num,
                    description=description,
                    previous_results=prev_results_str
                )

                # Build replanning context: original history + step results
                replan_context = conversation_history.copy()
                for res in results:
                    replan_context.append({
                        "role": "assistant",
                        "content": f"Step {res['step']} ({res['description']}): {res['result']}"
                    })

                # Create new plan with full context (step failure recovery)
                new_plan = self._create_plan(remaining_goal, replan_context, replan_reason="step_failure")

                if new_plan:
                    # Execute new plan with original history
                    new_results = self._execute_plan(new_plan, user_input, conversation_history)
                    results.extend(new_results)
                    break

        return results

    def _execute_step(
        self,
        step_num: int,
        step_info: Dict,
        previous_results: List[Dict],
        goal: str,
        context: List[Dict[str, str]],
        full_plan: Dict,
        original_user_input: str
    ) -> Dict:
        """
        Execute a single step - either direct LLM call (tool='null') or ReAct agent (tool-based)

        Args:
            step_num: Step number
            step_info: Step information
            goal: Overall goal
            previous_results: Results from previous steps
            context: Shared context (if enabled)
            full_plan: Complete plan dictionary
            original_user_input: Original user's question

        Returns:
            Step result dict with 'answer' and 'success'
        """
        tool = step_info.get("tool")
        description = step_info.get("description", "")

        # STRICT: Validate tool
        if tool is not None and tool not in config.AVAILABLE_TOOLS:
            return {
                "answer": f"Invalid tool '{tool}'. Must be one of {config.AVAILABLE_TOOLS} or null",
                "success": False
            }

        # Format previous steps
        prev_steps_str = self._format_previous_steps(previous_results)

        # BRANCH: null tool = direct LLM, else = ReAct with tool
        if tool is None:
            return self._execute_reasoning_step(step_num, description, prev_steps_str, goal, context)
        else:
            return self._execute_tool_step(step_num, step_info, prev_steps_str, goal, context, full_plan, original_user_input)

    def _execute_reasoning_step(
        self,
        step_num: int,
        description: str,
        prev_steps_str: str,
        goal: str,
        context: List[Dict[str, str]]
    ) -> Dict:
        """
        Execute reasoning step with direct LLM call (no tool)

        Args:
            step_num: Step number
            description: Step description
            prev_steps_str: Formatted previous steps
            goal: Overall goal
            context: Full conversation context

        Returns:
            Step result dict with 'answer' and 'success'
        """
        # Load reasoning step prompt from file
        step_prompt = self.load_prompt(
            "agents/plan_reasoning_step.txt",
            step_num=step_num,
            goal=goal,
            description=description,
            prev_steps=prev_steps_str
        )

        # Use full conversation context + current step
        messages = context + [{"role": "user", "content": step_prompt}]

        try:
            answer = self.call_llm(messages, stage=f"reasoning_step_{step_num}")
            return {
                "answer": answer,
                "success": True
            }
        except Exception as e:
            return {
                "answer": f"Reasoning step failed: {str(e)}",
                "success": False
            }

    def _execute_tool_step(
        self,
        step_num: int,
        step_info: Dict,
        prev_steps_str: str,
        goal: str,
        context: List[Dict[str, str]],
        full_plan: Dict,
        original_user_input: str
    ) -> Dict:
        """
        Execute tool-based step using ReAct agent

        Args:
            step_num: Step number
            step_info: Step information
            prev_steps_str: Formatted previous steps
            goal: Overall goal
            context: Shared context
            full_plan: Complete plan dictionary
            original_user_input: Original user's question

        Returns:
            Step result dict with 'answer' and 'success'
        """
        tools_desc = format_tools_for_llm()

        step_prompt = self.load_prompt(
            "agents/plan_execute.txt",
            step_number=step_num,
            goal=goal,
            current_step=json.dumps(step_info, indent=2),
            previous_steps=prev_steps_str or "None",
            tools=tools_desc
        )

        # Use ReAct agent to execute this step
        react_agent = ReActAgent(model=self.model, temperature=self.temperature)
        react_agent.session_id = self.session_id
        react_agent.tool_calls = self.tool_calls

        try:
            # Pass attached files to ReAct agent if available
            attached_files = getattr(self, 'attached_files', None)

            # Format plan information for ReAct agent
            plan_info = {
                'full_plan': self._format_full_plan(full_plan),
                'current_step': f"Step {step_num}: {step_info.get('description', '')}"
            }

            result = react_agent.run(
                user_input=step_prompt,
                conversation_history=context,
                attached_files=attached_files,
                original_user_input=original_user_input,
                plan_info=plan_info,
                skip_final_synthesis=True  # Skip react_final.txt in plan_execute mode
            )
            return {
                "answer": result,
                "success": True
            }
        except ReActMaxIterationsError as e:
            # ReAct agent exhausted its iterations
            # Return failure with metadata about iterations used
            print(f"[PLAN-EXECUTE] ReAct agent exhausted {e.iterations_used} iterations for step {step_num}")
            return {
                "answer": f"Tool step failed after {e.iterations_used} ReAct iterations: {str(e)}",
                "success": False,
                "react_iterations": e.iterations_used  # Track how many iterations were used
            }
        except Exception as e:
            # Other exceptions (e.g., tool errors, parsing errors)
            print(f"[PLAN-EXECUTE] Tool step {step_num} failed with exception: {str(e)}")
            return {
                "answer": f"Tool step failed: {str(e)}",
                "success": False,
                "react_iterations": 0  # Unknown iterations, treat as early failure
            }

    def _format_full_plan(self, plan: Dict) -> str:
        """
        Format the complete plan for display

        Args:
            plan: Plan dictionary

        Returns:
            Formatted string representation of the plan
        """
        if not plan or "plan" not in plan:
            return "No plan available"

        formatted = []
        formatted.append(f"Goal: {plan.get('goal', 'N/A')}")
        formatted.append("\nSteps:")
        for step_info in plan["plan"]:
            step_num = step_info.get("step", "?")
            description = step_info.get("description", "No description")
            tool = step_info.get("tool", "null")
            formatted.append(f"  {step_num}. {description} (tool: {tool})")

        return "\n".join(formatted)

    def _format_previous_steps(self, previous_results: List[Dict]) -> str:
        """
        Format previous step results as string

        Args:
            previous_results: List of previous step results

        Returns:
            Formatted string
        """
        if not previous_results:
            return "None"

        formatted = []
        for res in previous_results:
            formatted.append(f"Step {res['step']}: {res['description']}")
            formatted.append(f"Result: {res['result']}")
            formatted.append("")

        return "\n".join(formatted)

    def _synthesize_answer(
        self,
        plan: Dict,
        results: List[Dict],
        user_input: str,
        conversation_history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Synthesize final answer from all step results

        Args:
            plan: Original plan
            results: List of step results
            user_input: Original user input
            conversation_history: Full conversation history

        Returns:
            Dictionary with:
            - 'finished': bool (True if answer is complete, False if more steps needed)
            - 'answer': str (the synthesized answer)
            - 'missing_info': str (what's missing, only present if finished=False)
        """
        # Format all results
        results_summary = ""
        for res in results:
            results_summary += f"**Step {res['step']}**: {res['description']}\n"
            results_summary += f"Result: {res['result']}\n\n"

        # Load synthesis prompt from file
        synthesis_prompt = self.load_prompt(
            "agents/plan_synthesis.txt",
            user_input=user_input,
            goal=plan.get('goal', user_input),
            results_summary=results_summary.strip()
        )

        # Build messages with optional conversation history
        messages = []

        # Add conversation history if enabled
        if config.PLAN_HISTORY_IN_SYNTHESIS and conversation_history:
            limited_history = self._limit_history(conversation_history)
            messages.extend(limited_history)

        # Add synthesis prompt
        messages.append({"role": "user", "content": synthesis_prompt})

        final_answer = self.call_llm(messages, stage="synthesis")

        # Parse the response to extract FINISHED flag and content
        # Expected format:
        # Line 1: "FINISHED=True" or "FINISHED=False"
        # Rest: The answer and optional missing info section

        finished = True  # Default to True (optimistic)
        answer = final_answer
        missing_info = None

        # Check if response starts with FINISHED flag
        if final_answer.strip().startswith("FINISHED="):
            lines = final_answer.strip().split('\n', 1)

            # Extract FINISHED flag from first line
            first_line = lines[0].strip()
            if "FINISHED=False" in first_line or "FINISHED = False" in first_line:
                finished = False
            elif "FINISHED=True" in first_line or "FINISHED = True" in first_line:
                finished = True

            # Extract the rest of the content
            if len(lines) > 1:
                content = lines[1].strip()

                # If FINISHED=False, try to separate answer from missing_info
                if not finished:
                    # Look for common separators or headers
                    # The prompt asks for answer first, then missing info section
                    # We'll store everything as answer and parse missing info if clearly marked
                    answer = content

                    # Try to extract missing info section (optional)
                    # Look for markers like "Missing information:", "What's needed:", etc.
                    missing_markers = [
                        "missing information:",
                        "what's missing:",
                        "additional steps needed:",
                        "what information is missing:",
                        "to complete the answer:"
                    ]

                    content_lower = content.lower()
                    for marker in missing_markers:
                        if marker in content_lower:
                            # Split at the marker
                            split_idx = content_lower.index(marker)
                            answer = content[:split_idx].strip()
                            missing_info = content[split_idx:].strip()
                            break
                else:
                    answer = content
            else:
                # Only FINISHED flag, no content
                answer = "I've completed the analysis of the available information."

        return {
            'finished': finished,
            'answer': answer,
            'missing_info': missing_info
        }
