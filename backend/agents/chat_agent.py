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
        # Load system prompt
        system_prompt = self.load_prompt("agents/chat_system.txt")

        # Build messages for LLM
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        messages.extend(conversation_history)

        # Add current user input
        messages.append({"role": "user", "content": user_input})

        # Call LLM
        response = self.call_llm(messages)

        return response
