"""
ReAct Agent - Reasoning and Acting (Rebuilt from scratch)

Clean 2-step architecture:
1. Generate Thought/Action/Action Input
2. Execute tool and generate Observation with final_answer decision
"""
import re
import json
from typing import List, Dict

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

    def run(
        self,
        user_input: str,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Run ReAct agent with strict 2-step loop

        Args:
            user_input: User's message
            conversation_history: Full conversation history

        Returns:
            Final answer

        Raises:
            ValueError: If parsing fails or max iterations reached
        """
        # Initialize scratchpad
        scratchpad = ""

        # Get tools description
        tools_desc = format_tools_for_llm()

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

        # Load thought prompt
        thought_prompt = self.load_prompt(
            "agents/react_thought.txt",
            user_query=user_input,
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

        # Load observation prompt
        observation_prompt = self.load_prompt(
            "agents/react_observation.txt",
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
        # Load final answer prompt
        final_prompt = self.load_prompt(
            "agents/react_final.txt",
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

        # Extract Action Input
        input_match = re.search(
            r"Action Input:\s*(.+?)(?:\n\n|\n(?=Thought:|Action:|Observation:|Final Answer:)|$)",
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

        # Handle error cases
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
        elif tool_name == "python_coder":
            return self._format_python_coder_data(data)
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
        """Format python_coder results for LLM"""
        parts = []

        if data.get("stdout"):
            parts.append(f"Output:\n{data['stdout']}")

        if data.get("stderr"):
            parts.append(f"Error:\n{data['stderr']}")

        if data.get("files"):
            parts.append(f"Files: {', '.join(data['files'].keys())}")

        return "\n\n".join(parts) if parts else "Code executed with no output"

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
            return {
                "code": clean_input,
                "session_id": self.session_id or "auto",
                "timeout": config.PYTHON_CODER_TIMEOUT
            }
        elif tool_name == "rag":
            return {
                "query": clean_input,
                "collection_name": config.RAG_DEFAULT_COLLECTION,
                "max_results": config.RAG_MAX_RESULTS
            }
        else:
            raise ValueError(f"Unknown tool: '{tool_name}'")
