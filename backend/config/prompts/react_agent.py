"""
ReAct Agent Prompts
Prompts for the ReAct (Reasoning + Acting) agent workflow.

Version: 2.0.0 - Modernized for Anthropic/Claude Code style
Changes: Specific roles, markdown structure, thinking triggers
"""


def get_react_thought_and_action_prompt(
    query: str,
    context: str,
    file_guidance: str = ""
) -> str:
    """
    Combined thought and action generation for ReAct reasoning.
    Optimized for single-call efficiency with clear tool selection.

    Args:
        query: User's question
        context: Formatted context from previous steps
        file_guidance: Optional guidance text when files are attached

    Returns:
        Combined thought and action prompt
    """
    file_section = f"\n{file_guidance}\n" if file_guidance else ""

    return f"""You are a ReAct reasoning specialist with expertise in tool-augmented problem solving.
{file_section}
## Query
{query}

## Context
{context}

## Your Task
Reason about the next action, then select exactly one tool.

## Available Tools
- **web_search** - Current information, news, real-time data
- **rag_retrieval** - Query uploaded documents
- **python_coder** - Data analysis, file processing, calculations
- **vision_analyzer** - Image analysis, OCR, visual Q&A (only when images are attached)
- **finish** - Provide final answer (only when complete)

## Response Format
THOUGHT: [Your reasoning about what to do next]

ACTION: [Tool name]

ACTION INPUT: [Input for the selected tool]

Select the single most appropriate tool based on: (1) whether you need current/real-time data, (2) whether files are attached, (3) what type of processing is required."""


def get_react_final_answer_prompt(
    query: str,
    context: str
) -> str:
    """
    Generate final answer from all ReAct observations.
    Emphasizes comprehensiveness and specificity.

    Args:
        query: User's original question
        context: Formatted context from all steps

    Returns:
        Final answer generation prompt
    """
    return f"""You are a synthesis specialist. Review all gathered information and provide a complete answer.

## Original Query
{query}

## All Observations
{context}

## Your Task
Synthesize the observations above into a clear, complete answer. Include specific details, numbers, and facts from the observations. Ensure your answer directly addresses the query.

Present information clearly by: (1) directly answering the question first, (2) supporting with specific facts and numbers, (3) organizing details logically."""


def get_react_step_verification_prompt(
    plan_step_goal: str,
    success_criteria: str,
    tool_used: str,
    observation: str
) -> str:
    """
    Verify if a plan step achieved its goal.
    Requires clear YES/NO decision with reasoning.

    Args:
        plan_step_goal: Goal of the step
        success_criteria: Success criteria for the step
        tool_used: Which tool was used
        observation: Observation from tool execution

    Returns:
        Step verification prompt
    """
    return f"""You are a verification specialist. Evaluate whether the step goal was achieved.

## Step Goal
{plan_step_goal}

## Success Criteria
{success_criteria}

## Tool Used
{tool_used}

## Observation
{observation[:100000]}

## Your Task
Compare the observation against the success criteria. Answer "YES" if criteria met, "NO" otherwise.

Provide brief reasoning for your decision.

Verify success by checking: (1) does the observation contain the expected data type, (2) are all required elements present, (3) are there any error indicators in the output."""


def get_react_final_answer_from_steps_prompt(
    user_query: str,
    steps_text: str,
    observations_text: str
) -> str:
    """
    Generate final answer from plan execution step results.
    Synthesizes multi-step workflow into coherent response.

    Args:
        user_query: Original user query
        steps_text: Summary of execution steps
        observations_text: All observations from steps

    Returns:
        Final answer from steps prompt
    """
    return f"""You are a synthesis specialist for multi-step workflows. Consolidate all execution results into a comprehensive answer.

## Original Query
{user_query}

## Execution Steps Summary
{steps_text}

## All Observations
{observations_text}

## Your Task
Synthesize all steps and observations into a clear, complete answer. Include specific details, numbers, and facts from the observations.

Connect the steps by: (1) identifying how each observation builds on previous ones, (2) noting which steps provide which pieces of the final answer, (3) ensuring no contradictions between steps."""


def get_react_thought_prompt(
    query: str,
    context: str,
    available_tools: str
) -> str:
    """
    Generate reasoning about what to do next.

    Note: This is a legacy method. The primary method now uses combined thought-action generation.

    Args:
        query: User's question
        context: Formatted context from previous steps
        available_tools: List of available tools

    Returns:
        Thought generation prompt
    """
    return f"""You are a ReAct reasoning specialist. Analyze the situation and determine the next step.

## Query
{query}

## Context
{context}

Think step-by-step about what you need to do to answer this question.
Break down the task into smaller steps. For example:
if the question is "Analyze the data", the steps could be:

1. Gather file metadata, ex. number of rows, number of columns, column names, column types, etc.
2. Figureout how to load the data, and organize it in a way that is easy to analyze.
3. Use python coder tools to calculate mean, median, etc.
4. Acquire results from the tools
5. Append the results to the scratch file.
6. Read the scratch file and answer the question.
7. Make sure the answers are adequate to the query.
8. Finish the task.

## Available Tools
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
    Select which tool to use and what input to provide.

    Note: This is a legacy method. The primary method now uses combined thought-action generation.

    Args:
        query: User's question
        context: Formatted context from previous steps
        thought: The reasoning generated in thought phase
        file_guidance: Optional guidance text when files are attached

    Returns:
        Action selection prompt
    """
    file_section = f"\n{file_guidance}\n" if file_guidance else ""

    return f"""You are a tool selection specialist. Based on your reasoning, select the next action.
{file_section}
## Query
{query}

## Context
{context}

## Your Thought
{thought}

## Available Tools
Choose exactly one:
1. **web_search** - Search the web for current information (use for: news, current events, latest data)
2. **rag_retrieval** - Retrieve relevant documents from uploaded files (use for: document queries, file content)
3. **python_coder** - Generate and execute Python code with file processing and data analysis (use for: data analysis, file processing, calculations, working with CSV/Excel/PDF files)
4. **vision_analyzer** - Analyze images using vision AI (use for: image description, visual questions, OCR, chart interpretation, comparing images)
5. **finish** - Provide the final answer (use ONLY when you have complete information to answer the question)

**Note:** File metadata analysis is done automatically when files are attached.

## Response Format
You can think briefly, but you MUST end your response with these two lines:
Action: <action_name>
Action Input: <input_for_the_action>

### Examples

**Example 1 (Good):**
Action: web_search
Action Input: current weather in Seoul

**Example 2 (Good - with brief reasoning):**
I need current data for this query.
Action: web_search
Action Input: latest news about AI

**Example 3 (Good - finish action):**
I now have all the information needed.
Action: finish
Action Input: The capital of France is Paris, with a population of approximately 2.2 million people.

**Example 4 (Bad - don't do this):**
I think we should search the web.
Action: search the web

Now provide your action:"""
