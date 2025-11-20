"""
LLM Input Formatter

Utilities for extracting and formatting complete LLM input (messages, config, history)
for comprehensive prompt logging.
"""

from typing import Any, Dict, List, Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage


class LLMInputFormatter:
    """Formats LLM input for human-readable logging."""

    @staticmethod
    def format_messages(messages: List[BaseMessage]) -> str:
        """
        Format LangChain messages into human-readable text.

        Args:
            messages: List of LangChain messages

        Returns:
            Formatted string showing all messages
        """
        if not messages:
            return "[No messages]"

        lines = []
        for i, msg in enumerate(messages, 1):
            msg_type = type(msg).__name__
            content = msg.content if hasattr(msg, 'content') else str(msg)

            lines.append(f"--- Message {i} ({msg_type}) ---")
            lines.append(content)
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def extract_llm_config(llm: Any) -> Dict[str, Any]:
        """
        Extract LLM configuration (model name, temperature, etc.).

        Args:
            llm: LangChain LLM instance

        Returns:
            Dictionary of configuration parameters
        """
        config = {}

        # Extract common attributes
        if hasattr(llm, 'model'):
            config['model_name'] = llm.model
        elif hasattr(llm, 'model_name'):
            config['model_name'] = llm.model_name

        if hasattr(llm, 'temperature'):
            config['temperature'] = llm.temperature

        if hasattr(llm, 'max_tokens'):
            config['max_tokens'] = llm.max_tokens
        elif hasattr(llm, 'max_output_tokens'):
            config['max_tokens'] = llm.max_output_tokens

        if hasattr(llm, 'timeout'):
            config['timeout'] = llm.timeout

        if hasattr(llm, 'base_url'):
            config['base_url'] = llm.base_url

        # Add class name for reference
        config['llm_class'] = type(llm).__name__

        return config

    @staticmethod
    def get_system_messages(llm: Any) -> List[str]:
        """
        Extract system messages from LLM configuration.

        Args:
            llm: LangChain LLM instance

        Returns:
            List of system message contents
        """
        system_messages = []

        # Check various possible locations for system prompts
        if hasattr(llm, 'system_message'):
            system_messages.append(str(llm.system_message))

        if hasattr(llm, 'system_prompt'):
            system_messages.append(str(llm.system_prompt))

        if hasattr(llm, 'default_system_message'):
            system_messages.append(str(llm.default_system_message))

        return system_messages

    @staticmethod
    def format_complete_llm_input(
        messages: List[BaseMessage],
        llm: Any,
        prompt_description: str = ""
    ) -> str:
        """
        Format complete LLM input including system messages, history, current prompt, and config.

        Args:
            messages: List of messages sent to LLM
            llm: LangChain LLM instance
            prompt_description: Optional description of this prompt

        Returns:
            Formatted string with all LLM input details
        """
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("COMPLETE LLM INPUT CAPTURE".center(80))
        lines.append("=" * 80)
        lines.append("")

        if prompt_description:
            lines.append(f"Description: {prompt_description}")
            lines.append("")

        # Section 1: LLM Configuration
        lines.append("-" * 80)
        lines.append("SECTION 1: LLM CONFIGURATION")
        lines.append("-" * 80)
        config = LLMInputFormatter.extract_llm_config(llm)
        for key, value in config.items():
            lines.append(f"{key}: {value}")
        lines.append("")

        # Section 2: System Messages (if any)
        lines.append("-" * 80)
        lines.append("SECTION 2: SYSTEM MESSAGES")
        lines.append("-" * 80)
        system_messages = LLMInputFormatter.get_system_messages(llm)
        if system_messages:
            for i, sys_msg in enumerate(system_messages, 1):
                lines.append(f"System Message {i}:")
                lines.append(sys_msg)
                lines.append("")
        else:
            # Check messages list for SystemMessage instances
            system_msgs_from_list = [msg for msg in messages if isinstance(msg, SystemMessage)]
            if system_msgs_from_list:
                for i, msg in enumerate(system_msgs_from_list, 1):
                    lines.append(f"System Message {i}:")
                    lines.append(msg.content)
                    lines.append("")
            else:
                lines.append("[No system messages found]")
                lines.append("")

        # Section 3: Message History
        lines.append("-" * 80)
        lines.append("SECTION 3: CONVERSATION HISTORY")
        lines.append("-" * 80)

        # Separate messages by type
        history_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]

        if len(history_messages) > 1:
            lines.append(f"Total messages in history: {len(history_messages)}")
            lines.append("")
            for i, msg in enumerate(history_messages[:-1], 1):  # All except last
                msg_type = type(msg).__name__
                lines.append(f"--- History Message {i} ({msg_type}) ---")
                lines.append(msg.content)
                lines.append("")
        else:
            lines.append("[No conversation history - first message]")
            lines.append("")

        # Section 4: Current Prompt
        lines.append("-" * 80)
        lines.append("SECTION 4: CURRENT PROMPT (SENT TO LLM NOW)")
        lines.append("-" * 80)
        if history_messages:
            current_msg = history_messages[-1]
            lines.append(f"Message Type: {type(current_msg).__name__}")
            lines.append("")
            lines.append(current_msg.content)
        else:
            lines.append("[No current message]")
        lines.append("")

        # Section 5: Message Structure Details
        lines.append("-" * 80)
        lines.append("SECTION 5: MESSAGE STRUCTURE DETAILS")
        lines.append("-" * 80)
        lines.append(f"Total messages sent to LLM: {len(messages)}")
        lines.append("")
        lines.append("Message breakdown:")
        for i, msg in enumerate(messages, 1):
            msg_type = type(msg).__name__
            content_length = len(msg.content) if hasattr(msg, 'content') else 0
            lines.append(f"  {i}. {msg_type} - {content_length} characters")
        lines.append("")

        # Footer
        lines.append("=" * 80)
        lines.append("END OF LLM INPUT CAPTURE".center(80))
        lines.append("=" * 80)

        return "\n".join(lines)


# Convenience function
def format_llm_input(
    messages: List[BaseMessage],
    llm: Any,
    description: str = ""
) -> str:
    """
    Convenience function to format complete LLM input.

    Args:
        messages: List of messages sent to LLM
        llm: LangChain LLM instance
        description: Optional description

    Returns:
        Formatted string
    """
    return LLMInputFormatter.format_complete_llm_input(messages, llm, description)
