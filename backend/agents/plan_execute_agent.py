"""
Plan-Execute Agent
Creates a plan and executes it step-by-step using ReAct agents
"""
import json
import re
from typing import List, Dict, Optional, Any

import config
from backend.agents.base_agent import Agent
from backend.agents.react_agent import ReActAgent
from tools_config import format_tools_for_llm


class PlanExecuteAgent(Agent):
    """
    Plan-Execute agent that:
    1. Creates a multi-step plan
    2. Executes each step using a ReAct agent
    3. Re-plans if steps fail (configurable)
    """

    def __init__(self, model: str = None, temperature: float = None):
        super().__init__(model, temperature)
        self.max_steps = config.PLAN_MAX_STEPS
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
        Run plan-execute agent

        Args:
            user_input: User's message
            conversation_history: Full conversation history
            attached_files: Optional list of file metadata

        Returns:
            Final answer combining all steps
        """
        # Store attached files for passing to ReAct agents
        self.attached_files = attached_files
        # Step 1: Create plan
        plan = self._create_plan(user_input, conversation_history)

        if not plan:
            return "I apologize, but I was unable to create a plan for your request. Please try rephrasing your question."

        # Step 2: Execute plan
        results = self._execute_plan(plan, user_input, conversation_history)

        # Step 3: Synthesize final answer
        final_answer = self._synthesize_answer(plan, results, user_input, conversation_history)

        return final_answer

    def _create_plan(
        self,
        user_input: str,
        conversation_history: List[Dict[str, str]]
    ) -> Optional[Dict]:
        """
        Create execution plan

        Args:
            user_input: User's message
            conversation_history: Conversation history

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

        # Get plan from LLM
        response = self.call_llm(messages, temperature=0.7)  # Lower temp for planning

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

            # Handle failure
            if not step_result["success"] and self.replan_on_failure:
                # Re-plan from this point with full context
                remaining_goal = f"Continue from step {step_num}: {description}. Previous steps: {json.dumps(results)}"

                # Build replanning context: original history + step results
                replan_context = conversation_history.copy()
                for res in results:
                    replan_context.append({
                        "role": "assistant",
                        "content": f"Step {res['step']} ({res['description']}): {res['result']}"
                    })

                # Create new plan with full context
                new_plan = self._create_plan(remaining_goal, replan_context)

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
        # Build step prompt
        step_prompt = f"""You are executing step {step_num} of a multi-step plan.

Overall Goal: {goal}

Current Step: {description}

Previous Steps:
{prev_steps_str}

Based on the previous steps and the overall goal, complete this step and provide your answer."""

        # Use full conversation context + current step
        messages = context + [{"role": "user", "content": step_prompt}]

        try:
            answer = self.call_llm(messages)
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
        except Exception as e:
            return {
                "answer": f"Tool step failed: {str(e)}",
                "success": False
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
    ) -> str:
        """
        Synthesize final answer from all step results

        Args:
            plan: Original plan
            results: List of step results
            user_input: Original user input
            conversation_history: Full conversation history

        Returns:
            Final synthesized answer
        """
        # Format all results
        results_summary = ""
        for res in results:
            results_summary += f"**Step {res['step']}**: {res['description']}\n"
            results_summary += f"Result: {res['result']}\n\n"

        # Ask LLM to synthesize
        synthesis_prompt = f"""Based on the following step-by-step execution, provide a comprehensive final answer to the user's question.

User Question: {user_input}

Goal: {plan.get('goal', user_input)}

Step Results:
{results_summary}

Provide a clear, complete answer that addresses the user's question using the information gathered from all steps."""

        # Build messages with optional conversation history
        messages = []

        # Add conversation history if enabled
        if config.PLAN_HISTORY_IN_SYNTHESIS and conversation_history:
            limited_history = self._limit_history(conversation_history)
            messages.extend(limited_history)

        # Add synthesis prompt
        messages.append({"role": "user", "content": synthesis_prompt})

        final_answer = self.call_llm(messages)

        return final_answer
