"""
Base Agent class with shared functionality
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path

import config
from backend.core.llm_backend import llm_backend


class Agent(ABC):
    """
    Abstract base class for all agents
    """

    def __init__(self, model: str = None, temperature: float = None):
        """
        Initialize agent

        Args:
            model: Model to use (default: config.OLLAMA_MODEL)
            temperature: Temperature for LLM (default: config.DEFAULT_TEMPERATURE)
        """
        self.model = model or config.OLLAMA_MODEL
        self.temperature = temperature or config.DEFAULT_TEMPERATURE
        self.llm = llm_backend
        self.session_id = None  # Set by caller if needed
        self.tool_calls = []  # Track tool calls for logging

    @abstractmethod
    def run(
        self,
        user_input: str,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Run the agent to process user input

        Args:
            user_input: The user's message
            conversation_history: Full conversation history

        Returns:
            Agent's response
        """
        pass

    def load_prompt(self, prompt_name: str, **kwargs) -> str:
        """
        Load a prompt template from file and format with kwargs

        Args:
            prompt_name: Name of prompt file (relative to prompts/)
            **kwargs: Variables to substitute in template

        Returns:
            Formatted prompt string
        """
        prompt_path = config.PROMPTS_DIR / prompt_name

        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        with open(prompt_path, 'r', encoding='utf-8') as f:
            template = f.read()

        return template.format(**kwargs)

    def format_conversation_history(
        self,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Format conversation history as a string

        Args:
            conversation_history: List of message dicts

        Returns:
            Formatted history string
        """
        if not conversation_history:
            return "No previous conversation."

        formatted = []
        for msg in conversation_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted.append(f"{role.upper()}: {content}")

        return "\n".join(formatted)

    def call_llm(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None
    ) -> str:
        """
        Call LLM with messages

        Args:
            messages: List of message dicts
            temperature: Temperature override (optional)

        Returns:
            LLM response
        """
        temp = temperature if temperature is not None else self.temperature
        agent_type = self.__class__.__name__.replace("Agent", "").lower()

        return self.llm.chat(
            messages,
            self.model,
            temp,
            session_id=self.session_id,
            agent_type=agent_type,
            tool_calls=self.tool_calls.copy() if self.tool_calls else None
        )

    def call_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Call a tool via its API endpoint on port 1006 with optional context

        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
            context: Optional context (chat_history, react_scratchpad, etc.)

        Returns:
            Tool response as dictionary
        """
        import httpx
        import time
        from datetime import datetime
        from tools_config import get_tool_schema

        # Get tool schema
        schema = get_tool_schema(tool_name)
        if not schema:
            result = {
                "error": f"Tool '{tool_name}' not found",
                "success": False
            }
            # Log failed tool call
            self.tool_calls.append({
                "name": tool_name,
                "input": parameters,
                "output": result,
                "success": False
            })
            self._log_tool_execution(tool_name, parameters, result, 0, False)
            return result

        # Prepare request
        endpoint = schema["endpoint"]
        method = schema.get("method", "POST")

        # Build URL using tools port (1006) to avoid deadlock
        base_url = f"http://localhost:{config.TOOLS_PORT}"
        url = f"{base_url}{endpoint}"

        # Add context to parameters if provided
        if context:
            parameters["context"] = context

        # Get tool-specific timeout or use default
        tool_timeout = config.TOOL_PARAMETERS.get(tool_name, {}).get("timeout", config.DEFAULT_TOOL_TIMEOUT)

        start_time = time.time()

        # Log tool call start
        clean_params = {k: v for k, v in parameters.items() if k != "context"}
        print(f"\n{'='*80}")
        print(f"[TOOL CALL] {tool_name.upper()}")
        print(f"{'='*80}")
        print(f"[BASE_AGENT] Preparing HTTP request to tool API")
        print(f"  URL: {url}")
        print(f"  Method: {method}")
        for key, value in clean_params.items():
            # Truncate long values for console output
            str_value = str(value)
            if len(str_value) > 150:
                str_value = str_value[:150] + "... [truncated]"
            print(f"  {key}: {str_value}")
        print(f"  Timeout: {tool_timeout}s")
        print(f"  Tools API: {base_url}")
        print(f"\n[BASE_AGENT] Making HTTP {method} request...")

        try:
            # Make HTTP request
            if method == "POST":
                print(f"[BASE_AGENT] Sending POST request...")
                response = httpx.post(url, json=parameters, timeout=tool_timeout)
                print(f"[BASE_AGENT] ✅ Received response: HTTP {response.status_code}")
            elif method == "GET":
                print(f"[BASE_AGENT] Sending GET request...")
                response = httpx.get(url, params=parameters, timeout=tool_timeout)
                print(f"[BASE_AGENT] ✅ Received response: HTTP {response.status_code}")
            else:
                print(f"[BASE_AGENT] ❌ Unsupported method: {method}")
                result = {"error": f"Unsupported method: {method}", "success": False}
                # Log failed tool call
                self.tool_calls.append({
                    "name": tool_name,
                    "input": parameters,
                    "output": result,
                    "success": False
                })
                self._log_tool_execution(tool_name, parameters, result, 0, False)
                return result

            response.raise_for_status()
            print(f"[BASE_AGENT] Parsing JSON response...")
            result = response.json()
            print(f"[BASE_AGENT] ✅ JSON parsed successfully")

            duration = time.time() - start_time

            # Log successful tool call
            self.tool_calls.append({
                "name": tool_name,
                "input": parameters,
                "output": result,
                "success": True
            })
            self._log_tool_execution(tool_name, parameters, result, duration, True)

            # Console status
            print(f"\n[RESULT] Tool completed successfully")
            print(f"  Duration: {duration:.2f}s")
            if "answer" in result:
                answer_preview = result["answer"][:200] + "..." if len(result["answer"]) > 200 else result["answer"]
                print(f"  Answer: {answer_preview}")
            print(f"{'='*80}\n")

            return result

        except httpx.HTTPStatusError as e:
            duration = time.time() - start_time
            result = {
                "error": f"Tool API error: {e.response.status_code} - {e.response.text}",
                "success": False
            }
            # Log failed tool call
            self.tool_calls.append({
                "name": tool_name,
                "input": parameters,
                "output": result,
                "success": False
            })
            self._log_tool_execution(tool_name, parameters, result, duration, False)

            # Console status
            print(f"\n[ERROR] Tool failed: HTTP {e.response.status_code}")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Error: {result['error']}")
            print(f"{'='*80}\n")

            return result

        except Exception as e:
            duration = time.time() - start_time
            result = {
                "error": f"Tool call failed: {str(e)}",
                "success": False
            }
            # Log failed tool call
            self.tool_calls.append({
                "name": tool_name,
                "input": parameters,
                "output": result,
                "success": False
            })
            self._log_tool_execution(tool_name, parameters, result, duration, False)

            # Console status
            print(f"\n[ERROR] Tool failed: {str(e)}")
            print(f"  Duration: {duration:.2f}s")
            print(f"{'='*80}\n")

            return result

    def _log_tool_execution(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        result: Dict[str, Any],
        duration: float,
        success: bool
    ):
        """
        Log tool execution to prompts.log

        Args:
            tool_name: Name of the tool
            parameters: Tool input parameters
            result: Tool output/result
            duration: Execution time in seconds
            success: Whether the tool call succeeded
        """
        from datetime import datetime
        import json

        try:
            log_path = config.PROMPTS_LOG_PATH

            # # Format the log entry
            lines = []
            # lines.append("")
            # lines.append("")
            # lines.append("=" * 80)
            # lines.append(f"TOOL EXECUTION: {tool_name}")
            # lines.append("=" * 80)
            # lines.append("")

            # # INPUT section
            # lines.append("TOOL INPUT:")
            # # Remove context from parameters for cleaner logging
            # clean_params = {k: v for k, v in parameters.items() if k != "context"}
            # for key, value in clean_params.items():
            #     # Truncate long values
            #     str_value = str(value)
            #     if len(str_value) > 200:
            #         str_value = str_value[:200] + "... [truncated]"
            #     lines.append(f"  {key}: {str_value}")
            # lines.append("")

            # # OUTPUT section
            # lines.append("TOOL OUTPUT:")
            # if "answer" in result:
            #     lines.append(f"  Answer: {result['answer']}")
            # if "data" in result:
            #     data_str = json.dumps(result["data"], indent=2)
            #     # Truncate long data
            #     if len(data_str) > 500:
            #         data_str = data_str[:500] + "... [truncated]"
            #     lines.append(f"  Data: {data_str}")
            # if "error" in result:
            #     lines.append(f"  Error: {result['error']}")
            # lines.append("")

            # STATS section
            lines.append("=" * 80)
            lines.append("STATS:")
            lines.append(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"  Tool: {tool_name}")
            lines.append(f"  Duration: {duration:.2f}s")
            lines.append(f"  Status: {'SUCCESS' if success else 'FAILED'}")
            lines.append("=" * 80)
            lines.append("")

            # Write to log file
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write('\n'.join(lines))

        except Exception as e:
            print(f"Warning: Failed to log tool execution: {e}")
