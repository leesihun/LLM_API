"""
ReAct Agent - Reasoning and Acting (Rebuilt from scratch)

Clean 2-step architecture:
1. Generate Thought/Action/Action Input
2. Execute tool and generate Observation with final_answer decision
"""
import re
import json
from typing import List, Dict, Optional, Any

import config
from backend.agents.base_agent import Agent
from tools_config import format_tools_for_llm


class ReActAgent(Agent):
    """
    ReAct (Reasoning + Acting) agent with strict 2-step execution:
    Step 1: LLM generates Thought/Action/Action Input
    Step 2: Tool executes, LLM generates Observation + decides if done
    """

    def __init__(self, model: str = None, temperature: float = None):
        super().__init__(model, temperature)
        self.max_iterations = config.REACT_MAX_ITERATIONS

    def _build_comprehensive_user_query(self, user_input: str) -> str:
        """
        Build comprehensive user query with original input, plan, and task context

        Args:
            user_input: Current user input (may be step-specific)

        Returns:
            Comprehensive user query string with all available context
        """
        user_query_parts = []

        # 1. Add original user input (if available)
        if hasattr(self, 'original_user_input') and self.original_user_input:
            user_query_parts.append(f"Original User Question:\n{self.original_user_input}\n")

        # 2. Add full plan information (if available)
        if hasattr(self, 'plan_info') and self.plan_info:
            plan_data = self.plan_info
            if 'full_plan' in plan_data:
                user_query_parts.append(f"Full Plan:\n{plan_data['full_plan']}\n")
            if 'current_step' in plan_data:
                user_query_parts.append(f"Current Step:\n{plan_data['current_step']}\n")

        # 3. Add the actual user input (which may be step-specific)
        user_query_parts.append(f"Task:\n{user_input}")

        # Combine all parts
        return "\n".join(user_query_parts)

    def run(
        self,
        user_input: str,
        conversation_history: List[Dict[str, str]],
        attached_files: Optional[List[Dict[str, Any]]] = None,
        original_user_input: Optional[str] = None,
        plan_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Run ReAct agent with strict 2-step loop

        Args:
            user_input: User's message (may be a step prompt from PlanExecute)
            conversation_history: Full conversation history
            attached_files: Optional list of file metadata
            original_user_input: Original user question (before plan conversion)
            plan_info: Optional plan information with 'full_plan' and 'current_step'

        Returns:
            Final answer

        Raises:
            ValueError: If parsing fails or max iterations reached
        """
        # Initialize scratchpad
        scratchpad = ""

        # Get tools description
        tools_desc = format_tools_for_llm()

        # Store attached files for use in prompts
        self.attached_files = attached_files

        # Store plan context for use in prompts
        self.original_user_input = original_user_input
        self.plan_info = plan_info

        # Main reasoning loop
        for iteration in range(self.max_iterations):
            print(f"\n{'='*80}")
            print(f"[REACT] ITERATION {iteration + 1}/{self.max_iterations}")
            print(f"{'='*80}")

            # STEP 1: Generate Thought/Action/Action Input
            print(f"\n[REACT] STEP 1: Generating Thought/Action/Action Input...")

            thought_action = self._step1_generate_action(
                user_input=user_input,
                conversation_history=conversation_history,
                tools_desc=tools_desc,
                scratchpad=scratchpad
            )

            # Add to scratchpad
            scratchpad += thought_action + "\n\n"

            # Parse the action
            print(f"[REACT] Parsing action from response...")
            action_info = self._parse_action(thought_action)

            # STEP 2: Execute tool
            print(f"\n[REACT] STEP 2: Executing tool '{action_info['action']}'...")

            tool_result = self._execute_tool(
                tool_name=action_info["action"],
                tool_input=action_info["action_input"]
            )

            print(f"[REACT] Tool execution complete")

            # STEP 3: Generate Observation + decide if done
            print(f"\n[REACT] STEP 3: Generating observation and checking if complete...")

            observation_result = self._step2_generate_observation(
                user_input=user_input,
                conversation_history=conversation_history,
                scratchpad=scratchpad,
                tool_result=tool_result,
                action_info=action_info
            )

            # Add observation to scratchpad
            scratchpad += f"Observation: {observation_result['observation']}\n\n"

            # Check if we're done
            if observation_result["final_answer"]:
                print(f"\n[REACT] [OK] final_answer=true, generating final response...")

                # Generate final answer based on all observations
                final_answer = self._generate_final_answer(
                    user_input=user_input,
                    conversation_history=conversation_history,
                    scratchpad=scratchpad
                )

                print(f"[REACT] [DONE] Returning final answer: {final_answer[:200]}..." if len(final_answer) > 200 else f"[REACT] [DONE] Returning final answer")
                return final_answer
            else:
                print(f"[REACT] [CONTINUE] final_answer=false, continuing to next iteration...")

        # Max iterations reached
        error_msg = f"Maximum iterations ({self.max_iterations}) reached without completing task"
        print(f"\n[REACT] [ERROR] {error_msg}")
        raise ValueError(error_msg)

    def _step1_generate_action(
        self,
        user_input: str,
        conversation_history: List[Dict[str, str]],
        tools_desc: str,
        scratchpad: str
    ) -> str:
        """
        STEP 1: Generate Thought/Action/Action Input

        Args:
            user_input: User's question
            conversation_history: Chat history
            tools_desc: Tools description
            scratchpad: Current scratchpad

        Returns:
            LLM response with Thought/Action/Action Input

        Raises:
            ValueError: If response format is invalid
        """
        # Load system prompt
        system_prompt = self.load_prompt(
            "agents/react_system.txt",
            tools=tools_desc
        )

        # Add attached files information if present
        if hasattr(self, 'attached_files') and self.attached_files:
            files_info = self.format_attached_files(self.attached_files)
            system_prompt += files_info

        # Build comprehensive user query with all context
        comprehensive_user_query = self._build_comprehensive_user_query(user_input)

        # Load thought prompt
        thought_prompt = self.load_prompt(
            "agents/react_thought.txt",
            user_query=comprehensive_user_query,
            scratchpad=scratchpad if scratchpad else "No previous actions yet."
        )

        # Build messages: conversation history + system prompt + thought prompt
        # System prompt should frame the current task, not be at the start of all history
        messages = list(conversation_history)
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": thought_prompt})

        response = self.call_llm(messages)

        # Validate response contains required fields
        if "Thought:" not in response:
            raise ValueError("LLM response missing 'Thought:' field")
        if "Action:" not in response:
            raise ValueError("LLM response missing 'Action:' field")
        if "Action Input:" not in response:
            raise ValueError("LLM response missing 'Action Input:' field")

        return response

    def _step2_generate_observation(
        self,
        user_input: str,
        conversation_history: List[Dict[str, str]],
        scratchpad: str,
        tool_result: Dict,
        action_info: Dict[str, str]
    ) -> Dict[str, any]:
        """
        STEP 2: Generate Observation and decide if task is complete

        Args:
            user_input: User's question
            conversation_history: Chat history
            scratchpad: Current scratchpad
            tool_result: Full result dict from tool with 'data' field
            action_info: Dict with 'action' and 'action_input'

        Returns:
            Dict with 'observation' (str) and 'final_answer' (bool)

        Raises:
            ValueError: If response format is invalid
        """
        # Format tool data for LLM
        tool_data_str = self._format_tool_data(action_info["action"], tool_result)

        # Build comprehensive user query with all context
        comprehensive_user_query = self._build_comprehensive_user_query(user_input)

        # Load observation prompt
        observation_prompt = self.load_prompt(
            "agents/react_observation.txt",
            user_query=comprehensive_user_query,
            scratchpad=scratchpad,
            action=action_info["action"],
            action_input=action_info["action_input"],
            tool_data=tool_data_str
        )

        # Build messages: conversation history + observation prompt
        messages = list(conversation_history)
        messages.append({"role": "user", "content": observation_prompt})

        response = self.call_llm(messages)

        # Parse the structured response
        # Expected format:
        # final_answer: true/false
        # Observation: [text]

        print(f"\n[REACT] Parsing observation response...")
        print(f"Response preview: {response[:300]}...")

        # Extract final_answer
        final_answer_match = re.search(r"final_answer:\s*(true|false)", response, re.IGNORECASE)
        if not final_answer_match:
            raise ValueError("LLM response missing 'final_answer: true/false' field")

        final_answer = final_answer_match.group(1).lower() == "true"

        # Extract observation
        observation_match = re.search(r"Observation:\s*(.+)", response, re.DOTALL | re.IGNORECASE)
        if not observation_match:
            raise ValueError("LLM response missing 'Observation:' field")

        observation = observation_match.group(1).strip()

        print(f"[REACT] [OK] Parsed: final_answer={final_answer}, observation={len(observation)} chars")

        return {
            "observation": observation,
            "final_answer": final_answer
        }

    def _generate_final_answer(
        self,
        user_input: str,
        conversation_history: List[Dict[str, str]],
        scratchpad: str
    ) -> str:
        """
        Generate final answer based on all observations

        Args:
            user_input: User's question
            conversation_history: Chat history
            scratchpad: Complete scratchpad with all observations

        Returns:
            Final answer string

        Raises:
            ValueError: If response is invalid
        """
        # Build comprehensive user query with all context
        comprehensive_user_query = self._build_comprehensive_user_query(user_input)

        # Load final answer prompt
        final_prompt = self.load_prompt(
            "agents/react_final.txt",
            user_query=comprehensive_user_query,
            scratchpad=scratchpad
        )

        # Build messages: conversation history + final prompt
        messages = list(conversation_history)
        messages.append({"role": "user", "content": final_prompt})

        response = self.call_llm(messages)

        # Validate response is not empty
        if not response or not response.strip():
            raise ValueError("LLM returned empty final answer")

        return response.strip()

    def _parse_action(self, response: str) -> Dict[str, str]:
        """
        Parse Action and Action Input from response (strict)

        Args:
            response: LLM response

        Returns:
            Dict with 'action' and 'action_input'

        Raises:
            ValueError: If parsing fails
        """
        # Extract Action
        action_match = re.search(r"Action:\s*(\w+)", response)
        if not action_match:
            raise ValueError("Failed to parse 'Action:' from response")

        action = action_match.group(1)

        # Extract Action Input - stop only at next ReAct keyword (not double newlines)
        input_match = re.search(
            r"Action Input:\s*(.+?)(?:\n(?=Thought:|Action:|Observation:|Final Answer:)|$)",
            response,
            re.DOTALL | re.IGNORECASE
        )
        if not input_match:
            raise ValueError("Failed to parse 'Action Input:' from response")

        action_input = input_match.group(1).strip()

        # Validate not empty
        if not action_input:
            raise ValueError("Action Input is empty")

        print(f"[REACT] [OK] Parsed action: {action}, input: {action_input[:100]}..." if len(action_input) > 100 else f"[REACT] [OK] Parsed action: {action}, input: {action_input}")

        return {
            "action": action,
            "action_input": action_input
        }

    def _execute_tool(self, tool_name: str, tool_input: str) -> Dict:
        """
        Execute tool and return result (with error handling based on config)

        Args:
            tool_name: Tool to execute
            tool_input: String input for tool

        Returns:
            Tool result dict - either success with 'data' field or error with 'error' field

        Raises:
            ValueError: If tool execution fails and REACT_RETRY_ON_ERROR is False
        """
        try:
            # Convert string input to tool parameters
            parameters = self._convert_string_to_params(tool_name, tool_input)

            # Call tool via base agent
            context = {
                "session_id": self.session_id or "auto"
            }

            result = self.call_tool(tool_name, parameters, context)

            # Check for errors
            if isinstance(result, dict) and "error" in result and result["error"] is not None:
                if config.REACT_RETRY_ON_ERROR:
                    # Return error result - let LLM decide what to do
                    print(f"[REACT] Tool '{tool_name}' returned error: {result['error']}")
                    return result
                else:
                    # Strict mode - raise error immediately
                    raise ValueError(f"Tool '{tool_name}' failed: {result['error']}")

            # Return successful result
            return result

        except Exception as e:
            if config.REACT_RETRY_ON_ERROR:
                # Return error as result - let LLM decide what to do
                error_msg = str(e)
                print(f"[REACT] Tool '{tool_name}' raised exception: {error_msg}")
                return {
                    "error": error_msg,
                    "data": None
                }
            else:
                # Strict mode - propagate exception
                raise

    def _format_tool_data(self, tool_name: str, tool_result: Dict) -> str:
        """
        Format tool result data for LLM consumption

        Args:
            tool_name: Name of the tool
            tool_result: Full result dict from tool

        Returns:
            Formatted string of tool data (success or error)
        """
        if not isinstance(tool_result, dict):
            return json.dumps(tool_result, indent=2)

        # Special handling for python_coder - always show stdout/stderr even on error
        if tool_name == "python_coder":
            if "data" in tool_result:
                formatted = self._format_python_coder_data(tool_result["data"])
                # If there's an error, add error indicator but keep the output
                if "error" in tool_result and tool_result["error"] is not None:
                    error_msg = tool_result["error"]
                    return f"❌ EXECUTION FAILED ❌\nError: {error_msg}\n\n{formatted}"
                return formatted
            else:
                # Fallback if no data field
                return f"❌ TOOL ERROR ❌\nError: {tool_result.get('error', 'Unknown error')}"

        # Handle error cases for other tools
        if "error" in tool_result and tool_result["error"] is not None:
            error_msg = tool_result["error"]
            return f"❌ TOOL EXECUTION FAILED ❌\nError: {error_msg}\n\nYou should analyze this error and decide your next action."

        # Handle missing data
        if "data" not in tool_result:
            return json.dumps(tool_result, indent=2)

        data = tool_result["data"]

        # Format based on tool type
        if tool_name == "websearch":
            return self._format_websearch_data(data)
        elif tool_name == "rag":
            return self._format_rag_data(data)
        else:
            return json.dumps(data, indent=2)

    def _format_websearch_data(self, data: Dict) -> str:
        """Format websearch results for LLM"""
        results = data.get("results", [])

        if not results:
            return "No search results found."

        formatted = [f"Search Query: {data.get('query', 'N/A')}\n"]
        formatted.append(f"Found {data.get('num_results', 0)} results:\n")

        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "No content")
            score = result.get("score", 0.0)

            formatted.append(f"\n[{i}] {title}")
            formatted.append(f"    URL: {url}")
            formatted.append(f"    Relevance: {score:.2f}")
            formatted.append(f"    Content: {content}")

        return "\n".join(formatted)

    def _format_python_coder_data(self, data: Dict) -> str:
        """
        Format python_coder results for LLM
        Always shows stdout and stderr, even if empty
        """
        parts = []
        
        # Always show stdout section
        stdout = data.get("stdout", "")
        if stdout:
            parts.append(f"STDOUT:\n{stdout}")
        else:
            parts.append("STDOUT:\n(empty)")
        
        # Always show stderr section if there's any error
        stderr = data.get("stderr", "")
        if stderr:
            parts.append(f"STDERR:\n{stderr}")
        
        # Show return code
        returncode = data.get("returncode")
        if returncode is not None:
            parts.append(f"Return Code: {returncode}")
        
        # Show files created
        if data.get("files"):
            file_list = ", ".join(data["files"].keys())
            parts.append(f"Files Created: {file_list}")
        
        return "\n\n".join(parts)

    def _format_rag_data(self, data: Dict) -> str:
        """Format RAG results for LLM"""
        documents = data.get("documents", [])

        if not documents:
            return "No documents found."

        formatted = [f"Found {data.get('num_results', 0)} documents:\n"]

        for i, doc in enumerate(documents, 1):
            formatted.append(f"\n[{i}] {doc.get('document', 'Unknown')}")
            formatted.append(f"    Score: {doc.get('score', 0.0):.2f}")
            formatted.append(f"    Content: {doc.get('chunk', 'No content')}")

        return "\n".join(formatted)

    def _extract_code_from_fenced_block(self, text: str) -> str:
        """
        Extract code from fenced code blocks (```python or ```)
        Supports both plain text code and fenced blocks
        Falls back to original text if extraction fails

        Args:
            text: Input text that may contain fenced code blocks

        Returns:
            Extracted code or original text
        """
        # Pattern for fenced code blocks: ```python or just ```
        # Match: ```python\ncode\n``` or ```\ncode\n```
        pattern = r"```(?:python)?\s*\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

        if match:
            # Extract code from fenced block
            code = match.group(1).strip()
            print(f"[REACT] [OK] Extracted code from fenced block ({len(code)} chars)")
            return code

        # No fenced block found - return original text (plain code)
        print(f"[REACT] No fenced block found, treating as plain code")
        return text

    def _convert_string_to_params(self, tool_name: str, string_input: str) -> Dict[str, any]:
        """
        Convert string input to tool parameters (strict)

        Args:
            tool_name: Tool name
            string_input: String input

        Returns:
            Parameters dict

        Raises:
            ValueError: If conversion fails or input invalid
        """
        # Validate input
        if not string_input or not string_input.strip():
            raise ValueError(f"Tool input for '{tool_name}' is empty")

        clean_input = string_input.strip()

        # Convert based on tool
        if tool_name == "websearch":
            return {
                "query": clean_input,
                "max_results": config.WEBSEARCH_MAX_RESULTS
            }
        elif tool_name == "python_coder":
            # Extract code from fenced code blocks if present
            extracted_code = self._extract_code_from_fenced_block(clean_input)

            return {
                "code": extracted_code,
                "session_id": self.session_id or "auto",
                "timeout": config.PYTHON_CODER_TIMEOUT
            }
        elif tool_name == "rag":
            return {
                "query": clean_input,
                "collection_name": config.RAG_DEFAULT_COLLECTION,
                "max_results": config.RAG_MAX_RESULTS
            }
        elif tool_name == "read_file":
            # Parse file path from input (may include line range)
            # Expected formats:
            # - "path/to/file.txt"
            # - "path/to/file.txt lines 10-20"
            parts = clean_input.split()
            file_path = parts[0]

            params = {"file_path": file_path}

            # Check for line range
            if len(parts) >= 3 and parts[1].lower() == "lines":
                line_range = parts[2]
                if "-" in line_range:
                    start, end = line_range.split("-")
                    params["start_line"] = int(start)
                    params["end_line"] = int(end)

            return params
        else:
            raise ValueError(f"Unknown tool: '{tool_name}'")
