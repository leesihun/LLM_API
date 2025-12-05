"""
LLM Interceptor - Prompt/Response Logging
=========================================
Wrapper for LLM instances that intercepts and logs all prompts/responses.

Features:
- Structured, easy-to-read log format
- Request/Response pairing with unique call IDs
- Multiple output formats (structured, JSON, compact)
- Token estimation for tracking usage
- Duration tracking for performance analysis
- Clear visual hierarchy with message role separation

Version: 1.0.0
Created: 2025-12-03
"""

from typing import Optional, List
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json
import uuid
import logging

logger = logging.getLogger(__name__)


class LogFormat(Enum):
    """Output format for LLM logs."""
    STRUCTURED = "structured"  # Human-readable structured format
    JSON = "json"              # JSON Lines format for parsing
    COMPACT = "compact"        # Minimal format for quick scanning


@dataclass
class LogMessage:
    """Represents a single message in the conversation."""
    role: str
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class LogEntry:
    """Structured log entry for LLM interactions."""
    call_id: str
    timestamp: str
    entry_type: str  # "REQUEST" or "RESPONSE"
    model: str
    user_id: str
    messages: List[LogMessage]
    token_estimate: int
    duration_ms: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "timestamp": self.timestamp,
            "type": self.entry_type,
            "model": self.model,
            "user": self.user_id,
            "messages": [m.to_dict() for m in self.messages],
            "token_estimate": self.token_estimate,
            "duration_ms": self.duration_ms
        }


class LLMInterceptor:
    """
    Wrapper for LLM instances that intercepts and logs all prompts/responses.

    Features:
    - Structured, easy-to-read log format
    - Request/Response pairing with unique call IDs
    - Multiple output formats (structured, JSON, compact)
    - Token estimation for tracking usage
    - Duration tracking for performance analysis
    - Clear visual hierarchy with message role separation

    Log Format Options:
    - STRUCTURED: Human-readable with clear visual sections
    - JSON: JSON Lines format for programmatic parsing
    - COMPACT: Minimal format for quick scanning
    """

    def __init__(
        self,
        llm,
        user_id: str = "default",
        log_format: LogFormat = LogFormat.STRUCTURED,
        log_file: Optional[Path] = None
    ):
        """
        Initialize the interceptor.

        Args:
            llm: The LLM instance to wrap
            user_id: User ID for organizing log files (defaults to "default")
            log_format: Output format for logs (default: STRUCTURED)
            log_file: Custom log file path (defaults to data/scratch/prompts.log)
        """
        self.llm = llm
        self.user_id = user_id
        self.log_format = log_format
        self._current_call_id: Optional[str] = None
        self._call_start_time: Optional[datetime] = None

        # Create log file path
        self.log_file = log_file or Path("data/scratch/prompts.log")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize log file with header if it doesn't exist
        if not self.log_file.exists():
            self._write_header()

    def _write_header(self):
        """Write log file header based on format."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if self.log_format == LogFormat.JSON:
            header = json.dumps({
                "log_type": "llm_prompt_log",
                "created": timestamp,
                "user_id": self.user_id,
                "format_version": "1.0"
            }) + "\n"
        else:
            header = f"""┌{'─'*78}┐
│{'LLM PROMPT LOG'.center(78)}│
├{'─'*78}┤
│  User: {self.user_id:<69}│
│  Created: {timestamp:<66}│
│  Format: {self.log_format.value:<67}│
└{'─'*78}┘

"""
        self.log_file.write_text(header, encoding='utf-8')

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (approx 4 chars per token)."""
        return len(text) // 4

    def _parse_messages(self, prompt) -> List[LogMessage]:
        """Parse prompt into structured messages."""
        messages = []

        if isinstance(prompt, str):
            messages.append(LogMessage(role="USER", content=prompt))
        elif isinstance(prompt, list):
            for msg in prompt:
                if hasattr(msg, 'type') and hasattr(msg, 'content'):
                    role = msg.type.upper()
                    messages.append(LogMessage(role=role, content=msg.content))
                else:
                    messages.append(LogMessage(role="UNKNOWN", content=str(msg)))
        else:
            messages.append(LogMessage(role="RAW", content=str(prompt)))

        return messages

    def _format_structured(self, entry: LogEntry) -> str:
        """Format entry in human-readable structured format."""
        lines = []

        # Header box
        type_icon = "📤" if entry.entry_type == "REQUEST" else "📥"
        type_color = "REQUEST " if entry.entry_type == "REQUEST" else "RESPONSE"

        lines.append(f"\n{'═'*80}")
        lines.append(f"  {type_icon} {type_color}  │  ID: {entry.call_id[:8]}  │  {entry.timestamp}")
        lines.append(f"{'─'*80}")
        lines.append(f"  Model: {entry.model:<30}  User: {entry.user_id}")

        # Build the tokens/duration line
        tokens_line = f"  Tokens: ~{entry.token_estimate:<25}"
        if entry.duration_ms:
            tokens_line += f"  Duration: {entry.duration_ms:.0f}ms"
        lines.append(tokens_line)

        lines.append(f"{'─'*80}")

        # Messages section
        for msg in entry.messages:
            role_label = f"[{msg.role}]"
            lines.append(f"\n  {role_label}")
            lines.append(f"  {'·'*40}")

            # Indent content lines
            content_lines = msg.content.split('\n')
            for line in content_lines:
                # Wrap long lines
                if len(line) > 100:
                    wrapped = [line[i:i+100] for i in range(0, len(line), 100)]
                    for w in wrapped:
                        lines.append(f"    {w}")
                else:
                    lines.append(f"    {line}")

        lines.append(f"\n{'═'*80}\n")

        return '\n'.join(lines)

    def _format_json(self, entry: LogEntry) -> str:
        """Format entry as JSON line."""
        return json.dumps(entry.to_dict(), ensure_ascii=False) + "\n"

    def _format_compact(self, entry: LogEntry) -> str:
        """Format entry in compact format."""
        type_marker = ">>>" if entry.entry_type == "REQUEST" else "<<<"
        content_preview = entry.messages[0].content[:100] if entry.messages else ""
        content_preview = content_preview.replace('\n', ' ')
        if len(entry.messages[0].content if entry.messages else "") > 100:
            content_preview += "..."

        return f"[{entry.timestamp}] {type_marker} {entry.call_id[:8]} | {entry.model} | {content_preview}\n"

    def _format_entry(self, entry: LogEntry) -> str:
        """Format entry based on configured format."""
        if self.log_format == LogFormat.JSON:
            return self._format_json(entry)
        elif self.log_format == LogFormat.COMPACT:
            return self._format_compact(entry)
        else:
            return self._format_structured(entry)

    def _log_request(self, prompt, model: str = None) -> str:
        """Log a request and return the call ID."""
        call_id = str(uuid.uuid4())
        self._current_call_id = call_id
        self._call_start_time = datetime.now()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        model_name = model or getattr(self.llm, 'model', 'unknown')
        messages = self._parse_messages(prompt)

        total_content = ' '.join(m.content for m in messages)
        token_estimate = self._estimate_tokens(total_content)

        entry = LogEntry(
            call_id=call_id,
            timestamp=timestamp,
            entry_type="REQUEST",
            model=model_name,
            user_id=self.user_id,
            messages=messages,
            token_estimate=token_estimate
        )

        formatted = self._format_entry(entry)

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(formatted)

        logger.debug(f"[LLMInterceptor] Logged request {call_id[:8]} for user '{self.user_id}'")

        return call_id

    def _log_response(self, response, model: str = None):
        """Log a response with timing information."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            model_name = model or getattr(self.llm, 'model', 'unknown')

            # Calculate duration
            duration_ms = None
            if self._call_start_time:
                duration_ms = (datetime.now() - self._call_start_time).total_seconds() * 1000

            # Extract response content - try multiple sources
            response_content = ""

            # First, check if content is already populated
            if hasattr(response, 'content') and response.content:
                response_content = response.content

            # If content is empty, check for tool_calls (Ollama tool calling)
            elif hasattr(response, 'tool_calls') and response.tool_calls:
                logger.debug(f"[LLMInterceptor] Response has tool_calls but empty content - formatting tool calls")
                tool_calls = response.tool_calls

                if isinstance(tool_calls, list) and len(tool_calls) > 0:
                    tool_call = tool_calls[0]

                    # Extract action and args
                    if isinstance(tool_call, dict):
                        action = tool_call.get('name', 'unknown')
                        args = tool_call.get('args', {})
                    else:
                        # Object with attributes
                        action = getattr(tool_call, 'name', 'unknown')
                        args = getattr(tool_call, 'args', {})

                    # Build action_input from args
                    # Check if args contains plan/steps (JSON plan generation)
                    if isinstance(args, dict):
                        # Plan generation: args contains 'plan', 'steps', etc.
                        if 'plan' in args or 'steps' in args:
                            plan_data = args.get('plan') or args.get('steps')
                            if isinstance(plan_data, (list, dict)):
                                response_content = json.dumps(plan_data, indent=2)
                                logger.debug(f"[LLMInterceptor] Formatted tool_call as JSON plan")
                            else:
                                response_content = str(plan_data)
                        # ReAct: args contains 'query'
                        elif 'query' in args:
                            action_input = args['query']
                            response_content = f"THOUGHT: Using {action} tool to answer the query.\nACTION: {action}\nACTION INPUT: {action_input}"
                            logger.debug(f"[LLMInterceptor] Formatted tool_call as THOUGHT/ACTION/INPUT")
                        # Generic: stringify all args
                        else:
                            action_input = ', '.join(f"{k}={v}" for k, v in args.items())
                            response_content = f"THOUGHT: Using {action} tool.\nACTION: {action}\nACTION INPUT: {action_input}"
                    elif isinstance(args, str):
                        try:
                            parsed_args = json.loads(args)
                            if isinstance(parsed_args, dict) and 'query' in parsed_args:
                                action_input = parsed_args['query']
                            else:
                                action_input = args
                        except:
                            action_input = args
                        response_content = f"THOUGHT: Using {action} tool.\nACTION: {action}\nACTION INPUT: {action_input}"
                    else:
                        action_input = str(args)
                        response_content = f"THOUGHT: Using {action} tool.\nACTION: {action}\nACTION INPUT: {action_input}"
                else:
                    response_content = f"[tool_calls]: {tool_calls}"

            # Fallback to other sources
            elif hasattr(response, 'additional_kwargs') and response.additional_kwargs:
                # Ollama sometimes puts response in additional_kwargs
                response_content = f"[additional_kwargs]: {response.additional_kwargs}"
            elif hasattr(response, 'response_metadata') and response.response_metadata:
                response_content = f"[response_metadata]: {response.response_metadata}"
            else:
                response_content = str(response)

            # Ensure we log something even if empty
            if not response_content:
                response_content = f"[EMPTY RESPONSE] type={type(response).__name__}, repr={repr(response)[:500]}"

            messages = [LogMessage(role="ASSISTANT", content=response_content)]
            token_estimate = self._estimate_tokens(response_content)

            entry = LogEntry(
                call_id=self._current_call_id or str(uuid.uuid4()),
                timestamp=timestamp,
                entry_type="RESPONSE",
                model=model_name,
                user_id=self.user_id,
                messages=messages,
                token_estimate=token_estimate,
                duration_ms=duration_ms
            )

            formatted = self._format_entry(entry)

            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(formatted)

            logger.debug(f"[LLMInterceptor] Logged response for call {self._current_call_id[:8] if self._current_call_id else 'unknown'}")
        except Exception as e:
            logger.error(f"[LLMInterceptor] Failed to log response: {e}")
        finally:
            # Reset call tracking
            self._current_call_id = None
            self._call_start_time = None

    async def ainvoke(self, prompt, **kwargs):
        """Async invoke with prompt and response logging."""
        self._log_request(prompt)
        response = await self.llm.ainvoke(prompt, **kwargs)
        self._log_response(response)
        return response

    def invoke(self, prompt, **kwargs):
        """Sync invoke with prompt and response logging."""
        self._log_request(prompt)
        response = self.llm.invoke(prompt, **kwargs)
        self._log_response(response)
        return response

    async def astream(self, prompt, **kwargs):
        """Async stream with prompt logging (response logged on completion)."""
        call_id = self._log_request(prompt)

        async def stream_with_logging():
            chunks = []
            async for chunk in self.llm.astream(prompt, **kwargs):
                chunks.append(chunk)
                yield chunk
            # Log aggregated response after streaming completes
            if chunks:
                full_content = ''.join(
                    c.content if hasattr(c, 'content') else str(c)
                    for c in chunks
                )
                # Create a mock response object for logging
                class MockResponse:
                    content = full_content
                self._current_call_id = call_id
                self._log_response(MockResponse())

        return stream_with_logging()

    def stream(self, prompt, **kwargs):
        """Sync stream with prompt logging (response logged on completion)."""
        call_id = self._log_request(prompt)

        def stream_with_logging():
            chunks = []
            for chunk in self.llm.stream(prompt, **kwargs):
                chunks.append(chunk)
                yield chunk
            # Log aggregated response after streaming completes
            if chunks:
                full_content = ''.join(
                    c.content if hasattr(c, 'content') else str(c)
                    for c in chunks
                )
                class MockResponse:
                    content = full_content
                self._current_call_id = call_id
                self._log_response(MockResponse())

        return stream_with_logging()

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped LLM."""
        return getattr(self.llm, name)
