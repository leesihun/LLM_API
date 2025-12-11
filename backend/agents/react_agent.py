"""
ReAct Agent - Reasoning and Acting
Supports both prompt-based and Ollama native tool calling
"""
import re
import json
from typing import List, Dict, Optional

import config
from backend.agents.base_agent import Agent
from tools_config import format_tools_for_llm, format_tools_for_ollama_native


class ReActAgent(Agent):
    """
    ReAct (Reasoning + Acting) agent with tool calling
    Supports both prompt format and Ollama native tool calling
    """

    def __init__(self, model: str = None, temperature: float = None):
        super().__init__(model, temperature)
        self.format = config.REACT_FORMAT
        self.max_iterations = config.REACT_MAX_ITERATIONS
        self.retry_on_error = config.REACT_RETRY_ON_ERROR

    def run(
        self,
        user_input: str,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Run ReAct agent

        Args:
            user_input: User's message
            conversation_history: Full conversation history

        Returns:
            Final answer
        """
        if self.format == "native":
            return self._run_native(user_input, conversation_history)
        else:
            return self._run_prompt(user_input, conversation_history)

    def _run_prompt(
        self,
        user_input: str,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Run using prompt-based ReAct format

        Args:
            user_input: User's message
            conversation_history: Full conversation history

        Returns:
            Final answer
        """
        # Get tools description
        tools_desc = format_tools_for_llm()

        # Load system prompt
        system_prompt = self.load_prompt(
            "agents/react_system.txt",
            tools=tools_desc,
            input=user_input
        )

        # Initialize scratchpad (reasoning history)
        scratchpad = ""

        # Reasoning loop
        for iteration in range(self.max_iterations):
            # Format current state
            history_str = self.format_conversation_history(conversation_history)

            thought_prompt = self.load_prompt(
                "agents/react_thought.txt",
                history=history_str,
                input=user_input,
                scratchpad=scratchpad
            )

            # Get next action from LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": thought_prompt}
            ]

            response = self.call_llm(messages)
            scratchpad += response + "\n\n"

            # Check for Final Answer
            if "Final Answer:" in response:
                # Extract final answer
                final_answer = response.split("Final Answer:")[-1].strip()
                return final_answer

            # Parse action
            action_info = self._parse_action(response)

            if not action_info:
                # No valid action found, ask LLM to continue
                scratchpad += "Error: Invalid action format. Please follow the format:\nThought: ...\nAction: ...\nAction Input: [plain text string]\n\n"
                continue

            # Execute tool with context
            tool_name = action_info["action"]
            tool_input = action_info["action_input"]

            # Build context for tool
            context = {
                "chat_history": conversation_history,
                "react_scratchpad": scratchpad,
                "current_thought": response,
                "user_query": user_input
            }

            observation = self._execute_tool(tool_name, tool_input, context)

            # Add observation to scratchpad
            scratchpad += f"Observation: {observation}\n\n"

            # If tool failed and retry is disabled, return error
            if not self.retry_on_error and "error" in str(observation).lower():
                return f"Tool execution failed: {observation}"

        # Max iterations reached
        return "I apologize, but I was unable to complete the task within the allowed reasoning steps. Please try rephrasing your question or breaking it into smaller parts."

    def _run_native(
        self,
        user_input: str,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Run using Ollama native tool calling

        Args:
            user_input: User's message
            conversation_history: Full conversation history

        Returns:
            Final answer
        """
        # Get tools in Ollama format
        tools = format_tools_for_ollama_native()

        # Load system prompt
        system_prompt = self.load_prompt("agents/react_system.txt", tools=str(tools), input=user_input)

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_input})

        # For native tool calling, we'd need Ollama's tool calling API
        # This is a simplified version - actual implementation depends on Ollama's API
        # For now, fall back to prompt-based approach
        # TODO: Implement actual Ollama native tool calling when API is available

        return self._run_prompt(user_input, conversation_history)

    def _parse_action(self, response: str) -> Optional[Dict[str, any]]:
        """
        Parse Action and Action Input from response
        Now expects plain string input instead of JSON

        Args:
            response: LLM response

        Returns:
            Dictionary with 'action' and 'action_input' (string) or None
        """
        # Look for Action:
        action_match = re.search(r"Action:\s*(\w+)", response)

        if not action_match:
            return None

        action = action_match.group(1)

        # Look for Action Input: - extract everything until next major section or end
        # Match pattern: "Action Input:" followed by content until we hit a newline with a section header or end
        # Also stop at "Returns:" to avoid capturing tool documentation
        input_match = re.search(
            r"Action Input:\s*(.+?)(?:\n\n|\n(?=Thought:|Action:|Observation:|Final Answer:|Returns:)|$)",
            response,
            re.DOTALL | re.IGNORECASE
        )

        if not input_match:
            return None

        # Get the raw string input
        action_input = input_match.group(1).strip()

        # Validate: Action Input must not be empty
        if not action_input:
            return None

        return {
            "action": action,
            "action_input": action_input
        }

    def _convert_string_to_params(
        self,
        tool_name: str,
        string_input: str,
        context: Dict = None
    ) -> Dict[str, any]:
        """
        Convert plain string input to tool parameters with validation

        Args:
            tool_name: Name of the tool
            string_input: Plain string input from Action Input
            context: Optional context dict

        Returns:
            Parameters dictionary for the tool

        Raises:
            ValueError: If input is invalid
        """
        # Validate input is not empty
        if not string_input or not string_input.strip():
            raise ValueError(f"Action Input for {tool_name} cannot be empty")

        # Validate input length (prevent extremely long inputs)
        MAX_INPUT_LENGTH = 50000  # 50K characters
        if len(string_input) > MAX_INPUT_LENGTH:
            raise ValueError(f"Action Input too long: {len(string_input)} characters (max: {MAX_INPUT_LENGTH})")

        clean_input = string_input.strip()

        if tool_name == "websearch":
            # For websearch, input is the search query
            if len(clean_input) < 1:
                raise ValueError("Search query cannot be empty")

            return {
                "query": clean_input,
                "max_results": 5  # default
            }

        elif tool_name == "python_coder":
            # For python_coder, input is the actual code
            if len(clean_input) < 1:
                raise ValueError("Python code cannot be empty")

            return {
                "code": clean_input,
                "session_id": context.get("session_id", self.session_id or "auto") if context else (self.session_id or "auto"),
                "timeout": 30  # default
            }

        elif tool_name == "rag":
            # For rag, input is the search query
            if len(clean_input) < 1:
                raise ValueError("Search query cannot be empty")

            return {
                "query": clean_input,
                "collection_name": context.get("collection_name", "default") if context else "default",
                "max_results": 5  # default
            }

        else:
            # Unknown tool - raise error instead of guessing
            raise ValueError(f"Unknown tool: {tool_name}")

    def _execute_tool(
        self,
        tool_name: str,
        string_input: str,
        context: Dict = None
    ) -> str:
        """
        Execute a tool with string input and return observation

        Args:
            tool_name: Tool to execute
            string_input: Plain string input for the tool
            context: Context to pass to tool

        Returns:
            Observation string
        """
        try:
            # Convert string input to proper parameters
            parameters = self._convert_string_to_params(tool_name, string_input, context)

            # Call the tool with converted parameters
            result = self.call_tool(tool_name, parameters, context)

            # Format result as observation
            if isinstance(result, dict):
                if "error" in result:
                    return f"Error: {result['error']}"

                # Format based on tool response structure
                if "answer" in result:
                    return result["answer"]
                else:
                    return json.dumps(result, indent=2)
            else:
                return str(result)

        except ValueError as e:
            # Validation error from _convert_string_to_params
            return f"Invalid input: {str(e)}"
        except Exception as e:
            return f"Tool execution error: {str(e)}"
