"""
Ultrawork Agent
Executes tasks through OpenCode with iterative refinement based on LLM verification.
Only active when PYTHON_EXECUTOR_MODE="opencode".
"""
import re
from typing import List, Dict, Optional, Any

import config
from backend.agents.base_agent import Agent


class UltraworkAgent(Agent):
    """
    Ultrawork agent that:
    1. Sends user request to OpenCode via python_coder tool
    2. Verifies if response is adequate using LLM
    3. If not adequate, refines the instruction and retries
    4. Repeats until adequate or max iterations reached
    """

    def __init__(self, model: str = None, temperature: float = None):
        super().__init__(model, temperature)
        self.max_iterations = config.ULTRAWORK_MAX_ITERATIONS
        self.verify_temperature = config.ULTRAWORK_VERIFY_TEMPERATURE

    def run(
        self,
        user_input: str,
        conversation_history: List[Dict[str, str]],
        attached_files: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Run ultrawork agent with iterative refinement.

        Args:
            user_input: User's message
            conversation_history: Full conversation history
            attached_files: Optional list of file metadata

        Returns:
            Final response after task completion
        """
        print(f"\n[ULTRAWORK] Starting ultrawork agent")
        print(f"[ULTRAWORK] Max iterations: {self.max_iterations}")

        current_instruction = user_input
        iteration = 0
        all_results = []

        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n[ULTRAWORK] === Iteration {iteration}/{self.max_iterations} ===")

            # Step 1: Execute via python_coder tool
            print(f"[ULTRAWORK] Executing instruction via python_coder...")
            result = self._execute_instruction(current_instruction)
            all_results.append({
                "iteration": iteration,
                "instruction": current_instruction,
                "result": result
            })

            if not result["success"]:
                print(f"[ULTRAWORK] Execution failed: {result.get('error', 'Unknown error')}")

            # Step 2: Verify if response is adequate
            print(f"[ULTRAWORK] Verifying response adequacy...")
            verification = self._verify_response(user_input, result)

            if verification["adequate"]:
                print(f"[ULTRAWORK] Task completed successfully in {iteration} iteration(s)")
                return self._format_final_response(user_input, all_results)

            # Step 3: Generate refined instruction
            print(f"[ULTRAWORK] Response not adequate. Feedback: {verification['feedback']}")
            print(f"[ULTRAWORK] Generating refined instruction...")
            current_instruction = self._generate_refined_instruction(
                user_input,
                verification["feedback"]
            )
            print(f"[ULTRAWORK] Refined instruction: {current_instruction[:200]}...")

        # Max iterations reached
        print(f"[ULTRAWORK] Max iterations ({self.max_iterations}) reached")
        return self._format_final_response(user_input, all_results, max_iterations_reached=True)

    def _execute_instruction(self, instruction: str) -> Dict[str, Any]:
        """
        Execute instruction via python_coder tool.

        Args:
            instruction: Natural language instruction for OpenCode

        Returns:
            Tool execution result
        """
        parameters = {
            "code": instruction,
            "session_id": self.session_id,
            "timeout": config.PYTHON_CODER_TIMEOUT
        }

        result = self.call_tool("python_coder", parameters)
        return result

    def _verify_response(self, user_request: str, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify if execution result adequately fulfills user request.

        Args:
            user_request: Original user request
            execution_result: Result from python_coder tool

        Returns:
            Dict with 'adequate' (bool) and 'feedback' (str if not adequate)
        """
        # Format execution result for verification
        result_str = self._format_execution_result(execution_result)

        # Load verification prompt
        verify_prompt = self.load_prompt(
            "agents/ultrawork_verify.txt",
            user_request=user_request,
            execution_result=result_str
        )

        messages = [{"role": "user", "content": verify_prompt}]
        response = self.call_llm(messages, temperature=self.verify_temperature, stage="verify")

        # Parse verification response
        return self._parse_verification(response)

    def _parse_verification(self, response: str) -> Dict[str, Any]:
        """
        Parse verification response.

        Expected format:
        ADEQUATE: true/false
        FEEDBACK: ... (only if false)

        Args:
            response: LLM response

        Returns:
            Dict with 'adequate' and 'feedback'
        """
        response = response.strip()

        # Check for ADEQUATE: true
        if re.search(r'ADEQUATE:\s*true', response, re.IGNORECASE):
            return {"adequate": True, "feedback": None}

        # Check for ADEQUATE: false with feedback
        match = re.search(r'ADEQUATE:\s*false\s*\n*FEEDBACK:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        if match:
            feedback = match.group(1).strip()
            return {"adequate": False, "feedback": feedback}

        # Fallback: if contains "false", treat as not adequate
        if "false" in response.lower():
            # Try to extract any feedback
            lines = response.split('\n')
            feedback_lines = [l for l in lines if not l.strip().upper().startswith('ADEQUATE')]
            feedback = ' '.join(feedback_lines).strip() or "Task incomplete - needs further work"
            return {"adequate": False, "feedback": feedback}

        # Default: assume adequate if we can't parse
        print(f"[ULTRAWORK] Warning: Could not parse verification response, assuming adequate")
        return {"adequate": True, "feedback": None}

    def _generate_refined_instruction(self, user_request: str, feedback: str) -> str:
        """
        Generate refined instruction incorporating feedback.

        Args:
            user_request: Original user request
            feedback: Feedback about what's missing or wrong

        Returns:
            Refined instruction for OpenCode
        """
        refine_prompt = self.load_prompt(
            "agents/ultrawork_refine.txt",
            user_request=user_request,
            feedback=feedback
        )

        messages = [{"role": "user", "content": refine_prompt}]
        refined = self.call_llm(messages, temperature=self.temperature, stage="refine")

        return refined.strip()

    def _format_execution_result(self, result: Dict[str, Any]) -> str:
        """
        Format execution result for verification prompt.

        Args:
            result: Raw execution result from python_coder

        Returns:
            Formatted string representation
        """
        parts = []

        parts.append(f"Success: {result.get('success', False)}")

        if result.get("answer"):
            parts.append(f"\nOutput:\n{result['answer']}")

        if result.get("stdout"):
            parts.append(f"\nStdout:\n{result['stdout']}")

        if result.get("stderr"):
            parts.append(f"\nStderr:\n{result['stderr']}")

        if result.get("error"):
            parts.append(f"\nError:\n{result['error']}")

        if result.get("data", {}).get("files"):
            files = result["data"]["files"]
            parts.append(f"\nFiles created: {list(files.keys())}")

        return "\n".join(parts)

    def _format_final_response(
        self,
        user_request: str,
        all_results: List[Dict[str, Any]],
        max_iterations_reached: bool = False
    ) -> str:
        """
        Format final response to user.

        Args:
            user_request: Original user request
            all_results: All iteration results
            max_iterations_reached: Whether max iterations was reached

        Returns:
            Final formatted response
        """
        last_result = all_results[-1]["result"]

        parts = []

        if max_iterations_reached:
            parts.append(f"[Reached maximum {self.max_iterations} iterations]")
            parts.append("")

        # Include the final output
        if last_result.get("answer"):
            parts.append(last_result["answer"])
        elif last_result.get("stdout"):
            parts.append(last_result["stdout"])

        if last_result.get("error") and not last_result.get("success"):
            parts.append(f"\nNote: Final execution had an error: {last_result['error']}")

        # Summary
        total_iterations = len(all_results)
        if total_iterations > 1:
            parts.append(f"\n[Completed in {total_iterations} iteration(s)]")

        return "\n".join(parts)
