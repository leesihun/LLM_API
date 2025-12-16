"""
Auto Agent - LLM-based routing
Automatically selects the best agent (chat, react, plan_execute) based on user input
"""
from typing import List, Dict, Optional, Any

from backend.agents.base_agent import Agent
from backend.agents.chat_agent import ChatAgent
from backend.agents.react_agent import ReActAgent
from backend.agents.plan_execute_agent import PlanExecuteAgent


class AutoAgent(Agent):
    """
    Auto-routing agent that uses LLM to decide which agent to use
    """

    def __init__(self, model: str = None, temperature: float = None):
        super().__init__(model, temperature)

        # Initialize all available agents
        self.agents = {
            "chat": ChatAgent(model, temperature),
            "react": ReActAgent(model, temperature),
            "plan_execute": PlanExecuteAgent(model, temperature)
        }

    def run(
        self,
        user_input: str,
        conversation_history: List[Dict[str, str]],
        attached_files: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Run auto agent with LLM-based routing

        Args:
            user_input: User's message
            conversation_history: Full conversation history
            attached_files: Optional list of file metadata

        Returns:
            Response from selected agent
        """
        # Step 1: Determine which agent to use
        selected_agent = self._select_agent(user_input, conversation_history)

        # Step 2: Run the selected agent
        agent_instance = self.agents.get(selected_agent)

        if not agent_instance:
            # Fallback to chat if invalid selection
            agent_instance = self.agents["chat"]

        # Propagate session_id to selected agent
        agent_instance.session_id = self.session_id

        # Execute the agent with attached files
        response = agent_instance.run(user_input, conversation_history, attached_files)

        return response

    def _select_agent(
        self,
        user_input: str,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Use LLM to select appropriate agent

        Args:
            user_input: User's message
            conversation_history: Conversation history

        Returns:
            Agent name ("chat", "react", or "plan_execute")
        """
        # Load routing prompt
        routing_prompt = self.load_prompt(
            "agents/auto_router.txt",
            query=user_input
        )

        # Ask LLM to choose
        messages = [{"role": "user", "content": routing_prompt}]

        # Use lower temperature for more deterministic routing
        response = self.call_llm(messages, temperature=0.1)

        # Parse response
        response_lower = response.strip().lower()

        if "plan_execute" in response_lower:
            return "plan_execute"
        elif "react" in response_lower:
            return "react"
        elif "chat" in response_lower:
            return "chat"
        else:
            # Default to chat if unclear
            return "chat"
