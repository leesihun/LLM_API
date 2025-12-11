"""
LLM Interceptor for logging all LLM interactions
Logs in human-readable format to prompts.log
"""
import time
from datetime import datetime
from typing import List, Dict, Iterator
from pathlib import Path
import uuid

import config


class LLMInterceptor:
    """
    Wraps LLM backend calls and logs all interactions
    """

    def __init__(self, backend, log_path: Path = None):
        """
        Initialize interceptor

        Args:
            backend: The LLM backend instance to wrap
            log_path: Path to log file (default: config.PROMPTS_LOG_PATH)
        """
        self.backend = backend
        self.log_path = log_path or config.PROMPTS_LOG_PATH

        # Ensure log directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _format_human_readable(self, log_data: Dict) -> str:
        """
        Format log entry with two sections:
        1. Data section - exact messages/responses/tool calls (what LLM sees)
        2. Stats section - metadata and performance metrics

        Args:
            log_data: Dictionary containing log information

        Returns:
            Formatted string
        """
        lines = []

        # Determine if this is a request or response
        response = log_data.get("response", "")
        is_request = response in ["[WAITING FOR RESPONSE...]", "[STREAMING...]"]

        # ==================== DATA SECTION ====================

        if is_request:
            # REQUEST: Show messages being sent to LLM
            messages = log_data.get("messages", [])
            lines.append("")
            lines.append("")
            lines.append("")
            lines.append("")
            lines.append("")
            lines.append("")
            lines.append("MESSAGES TO LLM:")
            lines.append("")
            lines.append("")

            # Pretty-print each message individually for better readability
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                lines.append(f"\n--- Message {i+1} ({role}) --- (This line is for humans)")
                lines.append(content)
                lines.append("---")

            # Show tool calls if present
            tool_calls = log_data.get("tool_calls", [])
            if tool_calls:
                lines.append("")
                lines.append("TOOL CALLS: (This line is for humans)")
                for i, tool_call in enumerate(tool_calls):
                    lines.append(f"\n--- Tool Call {i+1} ---")
                    lines.append(f"Tool:  (This line is for humans)")
                    lines.append("")
                    lines.append(f"{tool_call.get('name', 'unknown')}")
                    lines.append("")
                    lines.append(f"Input: (This line is for humans)")
                    # Pretty-print the input JSON
                    tool_input = tool_call.get('input', {})
                    lines.append("")
                    if isinstance(tool_input, dict):
                        for key, value in tool_input.items():
                            lines.append(f"  {key}: {value}")
                    else:
                        lines.append(f"  {tool_input}")
                    lines.append("")
                    lines.append(f"Output: (This line is for humans)")
                    lines.append("")
                    lines.append(f"{tool_call.get('output', '')}")
                    lines.append("")
                    lines.append("---")
        else:
            lines.append("")
            lines.append("")
            lines.append("")
            lines.append("")
            lines.append("")
            lines.append("")
            # RESPONSE: Show LLM's response
            response = log_data.get("response", log_data.get("partial_response", ""))
            lines.append("LLM RESPONSE: (This line is for humans)")
            lines.append("")
            lines.append(response)

        lines.append("=" * 80)

        # ==================== STATS SECTION ====================

        lines.append("STATS:")

        if is_request:
            # REQUEST stats
            lines.append(f"  Timestamp: {log_data.get('timestamp', 'N/A')}")
            lines.append(f"  Model: {log_data.get('model', 'N/A')}")
            lines.append(f"  Backend: {log_data.get('backend', 'N/A')}")
            lines.append(f"  Temperature: {log_data.get('temperature', 'N/A')}")
            lines.append(f"  Session ID: {log_data.get('session_id', 'N/A')}")
            lines.append(f"  Agent: {log_data.get('agent_type', 'N/A')}")
            lines.append(f"  Streaming: {'Yes' if log_data.get('streaming', False) else 'No'}")
        else:
            # RESPONSE stats
            lines.append(f"  Timestamp: {log_data.get('timestamp', 'N/A')}")

            duration = log_data.get("duration_seconds", 0)
            lines.append(f"  Duration: {duration:.2f}s")

            estimated_tokens = log_data.get("estimated_tokens", {})
            tokens_in = estimated_tokens.get("input", 0)
            tokens_out = estimated_tokens.get("output", 0)
            tokens_total = estimated_tokens.get("total", 0)
            lines.append(f"  Tokens: {tokens_in} input + {tokens_out} output = {tokens_total} total")

            if duration > 0 and tokens_out > 0:
                tokens_per_sec = tokens_out / duration
                lines.append(f"  Tokens/sec: {tokens_per_sec:.2f}")

            success = log_data.get("success", False)
            error = log_data.get("error", None)
            if success:
                lines.append(f"  Status: SUCCESS")
            else:
                lines.append(f"  Status: FAILED")
                if error:
                    lines.append(f"  Error: {error}")

        lines.append("=" * 80)
        lines.append("")  # Blank line between entries

        return '\n'.join(lines)

    def _log_interaction(self, log_data: Dict):
        """
        Write log entry to file in human-readable format

        Args:
            log_data: Dictionary containing log information
        """
        try:
            # Add unique ID if not present
            if "id" not in log_data:
                log_data["id"] = str(uuid.uuid4())[:8]

            # Format as human-readable
            formatted_log = self._format_human_readable(log_data)

            # Write to file
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(formatted_log)
        except Exception as e:
            print(f"Warning: Failed to log LLM interaction: {e}")

    def chat(self, messages: List[Dict[str, str]], model: str, temperature: float = 0.7,
             session_id: str = None, agent_type: str = None, tool_calls: List[Dict] = None) -> str:
        """
        Non-streaming chat with logging

        Args:
            messages: Chat messages
            model: Model name
            temperature: Temperature parameter
            session_id: Optional session ID for tracking
            agent_type: Optional agent type (chat, react, plan_execute, auto)
            tool_calls: Optional list of tool calls made during this interaction

        Returns:
            Response text
        """
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_id = str(uuid.uuid4())[:8]

        # Console log for LLM call
        print(f"\n[LLM] Calling model: {model}")
        print(f"[LLM] Temperature: {temperature}")
        print(f"[LLM] Agent: {agent_type or 'N/A'}")
        print(f"[LLM] Messages: {len(messages)}")
        if tool_calls:
            print(f"[LLM] Tool calls in context: {len(tool_calls)}")

        # Log REQUEST immediately (real-time)
        request_log = {
            "id": log_id,
            "timestamp": timestamp,
            "type": "chat",
            "streaming": False,
            "model": model,
            "temperature": temperature,
            "messages": messages,
            "backend": self.backend.__class__.__name__,
            "session_id": session_id or "N/A",
            "agent_type": agent_type or "N/A",
            "tool_calls": tool_calls or [],
            "response": "[WAITING FOR RESPONSE...]",
            "duration_seconds": 0,
            "success": True,
            "estimated_tokens": {
                "input": int(sum(len(m.get("content", "").split()) for m in messages) * 1.3),
                "output": 0,
                "total": int(sum(len(m.get("content", "").split()) for m in messages) * 1.3)
            }
        }
        self._log_interaction(request_log)

        # Prepare response log entry
        response_log = {
            "id": log_id,
            "timestamp": timestamp,
            "type": "chat",
            "streaming": False,
            "model": model,
            "temperature": temperature,
            "messages": messages,
            "backend": self.backend.__class__.__name__,
            "session_id": session_id or "N/A",
            "agent_type": agent_type or "N/A",
            "tool_calls": tool_calls or []
        }

        try:
            # Make the actual LLM call
            response = self.backend.chat(messages, model, temperature)

            # Calculate timing
            duration = time.time() - start_time

            # Add response to log
            response_log["response"] = response
            response_log["duration_seconds"] = duration
            response_log["success"] = True

            # Estimate token usage (rough approximation)
            input_tokens = sum(len(m.get("content", "").split()) for m in messages) * 1.3
            output_tokens = len(response.split()) * 1.3
            response_log["estimated_tokens"] = {
                "input": int(input_tokens),
                "output": int(output_tokens),
                "total": int(input_tokens + output_tokens)
            }

            # Console log for LLM response
            print(f"[LLM] Response received in {duration:.2f}s")
            print(f"[LLM] Tokens: ~{int(output_tokens)} output")
            response_preview = response[:150] + "..." if len(response) > 150 else response
            print(f"[LLM] Response preview: {response_preview}")

            return response

        except Exception as e:
            # Console log for error
            print(f"[LLM] ERROR: {str(e)}")

            # Log error
            response_log["success"] = False
            response_log["error"] = str(e)
            response_log["duration_seconds"] = time.time() - start_time
            response_log["response"] = ""
            response_log["estimated_tokens"] = {
                "input": int(sum(len(m.get("content", "").split()) for m in messages) * 1.3),
                "output": 0,
                "total": int(sum(len(m.get("content", "").split()) for m in messages) * 1.3)
            }
            raise

        finally:
            # Log RESPONSE (real-time)
            self._log_interaction(response_log)

    def chat_stream(self, messages: List[Dict[str, str]], model: str, temperature: float = 0.7,
                   session_id: str = None, agent_type: str = None, tool_calls: List[Dict] = None) -> Iterator[str]:
        """
        Streaming chat with logging

        Args:
            messages: Chat messages
            model: Model name
            temperature: Temperature parameter
            session_id: Optional session ID for tracking
            agent_type: Optional agent type (chat, react, plan_execute, auto)
            tool_calls: Optional list of tool calls made during this interaction

        Yields:
            Response tokens
        """
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_id = str(uuid.uuid4())[:8]

        # Log REQUEST immediately (real-time)
        request_log = {
            "id": log_id,
            "timestamp": timestamp,
            "type": "chat",
            "streaming": True,
            "model": model,
            "temperature": temperature,
            "messages": messages,
            "backend": self.backend.__class__.__name__,
            "session_id": session_id or "N/A",
            "agent_type": agent_type or "N/A",
            "tool_calls": tool_calls or [],
            "response": "[STREAMING...]",
            "duration_seconds": 0,
            "success": True,
            "estimated_tokens": {
                "input": int(sum(len(m.get("content", "").split()) for m in messages) * 1.3),
                "output": 0,
                "total": int(sum(len(m.get("content", "").split()) for m in messages) * 1.3)
            }
        }
        self._log_interaction(request_log)

        # Prepare response log entry
        response_log = {
            "id": log_id,
            "timestamp": timestamp,
            "type": "chat",
            "streaming": True,
            "model": model,
            "temperature": temperature,
            "messages": messages,
            "backend": self.backend.__class__.__name__,
            "session_id": session_id or "N/A",
            "agent_type": agent_type or "N/A",
            "tool_calls": tool_calls or []
        }

        collected_response = ""

        try:
            # Stream from backend
            for token in self.backend.chat_stream(messages, model, temperature):
                collected_response += token
                yield token

            # Calculate timing
            duration = time.time() - start_time

            # Add complete response to log
            response_log["response"] = collected_response
            response_log["duration_seconds"] = duration
            response_log["success"] = True

            # Estimate token usage
            input_tokens = sum(len(m.get("content", "").split()) for m in messages) * 1.3
            output_tokens = len(collected_response.split()) * 1.3
            response_log["estimated_tokens"] = {
                "input": int(input_tokens),
                "output": int(output_tokens),
                "total": int(input_tokens + output_tokens)
            }

        except Exception as e:
            # Log error
            response_log["success"] = False
            response_log["error"] = str(e)
            response_log["duration_seconds"] = time.time() - start_time
            response_log["partial_response"] = collected_response
            response_log["estimated_tokens"] = {
                "input": int(sum(len(m.get("content", "").split()) for m in messages) * 1.3),
                "output": int(len(collected_response.split()) * 1.3) if collected_response else 0,
                "total": int(sum(len(m.get("content", "").split()) for m in messages) * 1.3) + (int(len(collected_response.split()) * 1.3) if collected_response else 0)
            }
            raise

        finally:
            # Log RESPONSE after stream completes (real-time)
            self._log_interaction(response_log)

    def list_models(self) -> List[str]:
        """
        Pass-through to backend list_models (no logging needed)

        Returns:
            List of model names
        """
        return self.backend.list_models()

    def is_available(self) -> bool:
        """
        Pass-through to backend is_available (no logging needed)

        Returns:
            True if backend is available
        """
        return self.backend.is_available()
