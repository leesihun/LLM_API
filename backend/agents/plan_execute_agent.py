"""
Plan-Execute Agent
Creates a plan and executes it step-by-step using ReAct agents
"""
import json
from typing import List, Dict, Optional

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

    def run(
        self,
        user_input: str,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Run plan-execute agent

        Args:
            user_input: User's message
            conversation_history: Full conversation history

        Returns:
            Final answer combining all steps
        """
        # Step 1: Create plan
        plan = self._create_plan(user_input, conversation_history)

        if not plan:
            return "I apologize, but I was unable to create a plan for your request. Please try rephrasing your question."

        # Step 2: Execute plan
        results = self._execute_plan(plan, user_input, conversation_history)

        # Step 3: Synthesize final answer
        final_answer = self._synthesize_answer(plan, results, user_input)

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

        # Build messages
        messages = [
            {"role": "system", "content": planning_prompt},
            {"role": "user", "content": user_input}
        ]

        # Get plan from LLM
        response = self.call_llm(messages, temperature=0.3)  # Lower temp for planning

        # Parse JSON plan
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()

            plan = json.loads(json_str)

            # Validate plan structure
            if "plan" not in plan or not isinstance(plan["plan"], list):
                return None

            # Limit steps
            if len(plan["plan"]) > self.max_steps:
                plan["plan"] = plan["plan"][:self.max_steps]

            return plan

        except (json.JSONDecodeError, IndexError, KeyError):
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
                context=shared_context
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
                # Re-plan from this point
                remaining_goal = f"Continue from step {step_num}: {description}. Previous steps: {json.dumps(results)}"
                new_plan = self._create_plan(remaining_goal, shared_context)

                if new_plan:
                    # Execute new plan
                    new_results = self._execute_plan(new_plan, user_input, shared_context)
                    results.extend(new_results)
                    break

        return results

    def _execute_step(
        self,
        step_num: int,
        step_info: Dict,
        previous_results: List[Dict],
        goal: str,
        context: List[Dict[str, str]]
    ) -> Dict:
        """
        Execute a single step using ReAct agent

        Args:
            step_num: Step number
            step_info: Step information
            previous_results: Results from previous steps
            goal: Overall goal
            context: Shared context (if enabled)

        Returns:
            Step result dict with 'answer' and 'success'
        """
        # Format previous steps
        prev_steps_str = ""
        for res in previous_results:
            prev_steps_str += f"Step {res['step']}: {res['description']}\n"
            prev_steps_str += f"Result: {res['result']}\n\n"

        # Create step execution prompt
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

        try:
            result = react_agent.run(step_prompt, context)
            return {
                "answer": result,
                "success": True
            }
        except Exception as e:
            return {
                "answer": f"Step failed: {str(e)}",
                "success": False
            }

    def _synthesize_answer(
        self,
        plan: Dict,
        results: List[Dict],
        user_input: str
    ) -> str:
        """
        Synthesize final answer from all step results

        Args:
            plan: Original plan
            results: List of step results
            user_input: Original user input

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

        messages = [
            {"role": "user", "content": synthesis_prompt}
        ]

        final_answer = self.call_llm(messages)

        return final_answer
