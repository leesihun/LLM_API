"""
Simple Chat Agent
Handles basic conversational interactions without tools
"""
from typing import List, Dict, Optional, Any

from backend.agents.base_agent import Agent


class ChatAgent(Agent):
    """
    Simple conversational agent
    Passes messages directly to LLM with system prompt
    """

    def run(
        self,
        user_input: str,
        conversation_history: List[Dict[str, str]],
        attached_files: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Run chat agent

        Args:
            user_input: User's message
            conversation_history: Full conversation history
            attached_files: Optional list of file metadata

        Returns:
            Agent response
        """
        # Load system prompt
        system_prompt = self.load_prompt("agents/chat_system.txt")

        # Add attached files information if present
        if attached_files and len(attached_files) > 0:
            files_info = self.format_attached_files(attached_files)
            system_prompt += files_info

        # Build messages for LLM
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        messages.extend(conversation_history)

        # Add current user input
        messages.append({"role": "user", "content": user_input})

        # Call LLM
        response = self.call_llm(messages)

        return response
