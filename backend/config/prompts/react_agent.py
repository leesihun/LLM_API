"""
ReAct Agent Prompts
Prompts for the ReAct (Reasoning + Acting) agent workflow.
"""


def get_react_thought_and_action_prompt(
    query: str,
    context: str,
    file_guidance: str = ""
) -> str:
    """
    Combined prompt for ReAct agent's thought and action generation (single LLM call).
    Used in: backend/tasks/React.py (_generate_thought_and_action)

    Args:
        query: User's question
        context: Formatted context from previous steps
        file_guidance: Optional guidance text when files are attached

    Simplified version: Removed ASCII markers, reduced from 70 to ~40 lines.
    """
    return f"""You are a helpful AI assistant using the ReAct (Reasoning + Acting) framework.

{file_guidance}

Question: {query}

{context}

Think step-by-step and decide on an action. Provide BOTH your reasoning AND your action in this format:

THOUGHT: [Your step-by-step reasoning about what to do next]

ACTION: [Exactly one of: web_search, rag_retrieval, python_coder, finish]

ACTION INPUT: [The input for the selected action]

Available Actions:

1. web_search - Search the web for current information
   - Use 3-10 specific keywords
   - Include names, dates, places, products
   - Examples: "latest AI developments 2025", "Python vs JavaScript performance"
   - Avoid single words or vague queries

2. rag_retrieval - Retrieve relevant documents from uploaded files

3. python_coder - Generate and execute Python code for data analysis and file processing

4. finish - Provide the final answer (use ONLY when you have complete information)

Note: File metadata analysis is done automatically when files are attached.

Now provide your thought and action:"""


def get_react_final_answer_prompt(
    query: str,
    context: str
) -> str:
    """
    Prompt for generating final answer from ReAct observations.
    Used in: backend/tasks/React.py (_generate_final_answer)

    Args:
        query: User's original question
        context: Formatted context from all steps

    Simplified version: Removed ASCII markers.
    """
    return f"""You are a helpful AI assistant. Based on all your reasoning and observations, provide a final answer.

Question: {query}

{context}

IMPORTANT: Review ALL the observations above carefully. Each observation contains critical information from tools you executed (web search results, code outputs, document content, etc.).

Your final answer MUST:
1. Incorporate ALL relevant information from the observations
2. Be comprehensive and complete
3. Directly answer the user's question
4. Include specific details, numbers, facts from the observations

Based on all the information you've gathered through your actions and observations, provide a clear, complete, and accurate final answer:"""


def get_react_step_verification_prompt(
    plan_step_goal: str,
    success_criteria: str,
    tool_used: str,
    observation: str
) -> str:
    """
    Prompt for verifying if a plan step was successful.
    Used in: backend/tasks/React.py (_verify_step_success)

    Args:
        plan_step_goal: Goal of the step
        success_criteria: Success criteria for the step
        tool_used: Which tool was used
        observation: Observation from tool execution

    Simplified version: Removed ASCII markers.
    """
    return f"""Verify if the step goal was achieved.

Step Goal: {plan_step_goal}
Success Criteria: {success_criteria}

Tool Used: {tool_used}
Observation: {observation[:1000]}

Based on the observation, was the step goal achieved according to the success criteria?
Answer with "YES" or "NO" and brief reasoning:"""


def get_react_final_answer_from_steps_prompt(
    user_query: str,
    steps_text: str,
    observations_text: str
) -> str:
    """
    Prompt for generating final answer from plan execution step results.
    Used in: backend/tasks/React.py (_generate_final_answer_from_steps)

    Args:
        user_query: Original user query
        steps_text: Summary of execution steps
        observations_text: All observations from steps

    Simplified version: Removed ASCII markers.
    """
    return f"""You are a helpful AI assistant. Generate a final, comprehensive answer based on the step-by-step execution results.

Original User Query: {user_query}

Execution Steps Summary:
{steps_text}

All Observations:
{observations_text}

Based on all the steps executed and their results, provide a clear, complete, and accurate final answer to the user's query.
Include specific details, numbers, and facts from the observations:"""


def get_react_thought_prompt(
    query: str,
    context: str,
    available_tools: str
) -> str:
    """
    Prompt for generating reasoning about what to do next.
    Used in: backend/tasks/React.py (_generate_thought)

    Note: This is a legacy method. The primary method now uses combined thought-action generation.

    Args:
        query: User's question
        context: Formatted context from previous steps
        available_tools: List of available tools

    Simplified version: Removed ASCII markers.
    """
    return f"""You are a helpful AI assistant using the ReAct (Reasoning + Acting) framework.

Question: {query}

{context}

Think step-by-step about what you need to do to answer this question.
Break down the task into smaller baby steps. Such as
if the question is "Analyze the data", the baby steps could be

1. Write down the data to a scratch file.
2. Load the data from the scratch file.
3. Use math tools to calculate mean, median, etc.
4. Acquire results from the tools
5. Append the results to the scratch file.
6. Read the scratch file and answer the question.
7. Make sure the answers are adequate to the query.
8. Finish the task.

These are available tools:
{available_tools}
What information do you need? What should you do next?

Provide your reasoning:"""


def get_react_action_selection_prompt(
    query: str,
    context: str,
    thought: str,
    file_guidance: str = ""
) -> str:
    """
    Prompt for selecting which tool to use and what input to provide.
    Used in: backend/tasks/React.py (_select_action)

    Note: This is a legacy method. The primary method now uses combined thought-action generation.

    Args:
        query: User's question
        context: Formatted context from previous steps
        thought: The reasoning generated in thought phase
        file_guidance: Optional guidance text when files are attached

    Simplified version: Removed ASCII markers.
    """
    return f"""You are a helpful AI assistant. Based on your reasoning, select the next action.

{file_guidance}

Question: {query}

{context}

Your Thought: {thought}

Available Actions (choose EXACTLY one of these names):
1. web_search - Search the web for current information (use for: news, current events, latest data)
2. rag_retrieval - Retrieve relevant documents from uploaded files (use for: document queries, file content)
3. python_coder - Generate and execute Python code with file processing and data analysis (use for: data analysis, file processing, calculations, working with CSV/Excel/PDF files)
4. finish - Provide the final answer (use ONLY when you have complete information to answer the question)

Note: File metadata analysis is done automatically when files are attached.

RESPONSE FORMAT - You can think briefly, but you MUST end your response with these two lines:
Action: <action_name>
Action Input: <input_for_the_action>

Examples:

Example 1 (Good):
Action: web_search
Action Input: current weather in Seoul

Example 2 (Good - with brief reasoning):
I need current data for this query.
Action: web_search
Action Input: latest news about AI

Example 3 (Good - finish action):
I now have all the information needed.
Action: finish
Action Input: The capital of France is Paris, with a population of approximately 2.2 million people.

Example 4 (Bad - don't do this):
I think we should search the web.
Action: search the web

Now provide your action:"""
