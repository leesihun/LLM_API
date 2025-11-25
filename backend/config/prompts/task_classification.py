"""
Task Classification Prompts
Used for determining which agent type should handle a query: chat, react, or plan_execute.
"""


def get_agent_type_classifier_prompt() -> str:
    """
    Prompt for classifying queries into three agent types: chat, react, or plan_execute.
    
    Returns:
        Prompt string for 3-way agent classification
    """
    return """You are an agent type classifier. Classify user queries into one of three types: "chat", "react", or "plan_execute".
CHAT - Very simple questions answerable from easy general knowledge base (NO tools needed):
REACT - A little bit complicated, single-goal tasks requiring tools (web search, code execution, simple analysis):
PLAN_EXECUTE - Multi-step complex tasks requiring planning and structured execution:
Respond with ONLY one word: "chat", "react", or "plan_execute" (no explanation, no punctuation)."""
    
    
# """You are an agent type classifier. Classify user queries into one of three types: "chat", "react", or "plan_execute".

# CHAT - Very simple questions answerable from easy general knowledge base (NO tools needed):

# Examples:
# 1. "What is Python?" → chat (general knowledge)
# 2. "Explain recursion to me" → chat (concept explanation)
# 3. "How to calculate variance?" → chat (explain concept, not execute)
# 4. "What is the capital of France?" → chat (established fact)
# 5. "Tell me about the Eiffel Tower" → chat (encyclopedia knowledge)
# 6. "How does a for loop work?" → chat (explanation)
# 7. "What are the benefits of exercise?" → chat (general health knowledge)
# 8. "Explain the difference between React and Vue" → chat (comparison explanation)
# 9. "What is machine learning?" → chat (established concept)
# 10. "How to search files in Linux?" → chat (asking for explanation, not executing)

# REACT - A little bit complicated, single-goal tasks requiring tools (web search, code execution, simple analysis):

# Examples:
# 1. "What's the weather in Seoul right now?" → react (single web search)
# 2. "Search for the latest AI developments" → react (web search)
# 3. "Calculate the variance of [1,2,3,4,5]" → react (single computation)
# 4. "What are recent developments in quantum computing?" → react (web search)
# 5. "Find news about OpenAI from this week" → react (current news search)
# 6. "Analyze sales_data.csv and show basic statistics" → react (single file analysis)
# 7. "What's the current price of Bitcoin?" → react (real-time data)
# 8. "Search my documents for 'machine learning'" → react (RAG search)
# 9. "Generate a simple bar chart from this data" → react (single visualization)
# 10. "Execute this Python code snippet" → react (code execution)

# PLAN_EXECUTE - Multi-step complex tasks requiring planning and structured execution:

# Examples:
# 1. "Analyze sales_data.csv AND customer_data.xlsx, then create visualizations and a summary report" → plan_execute (multiple files + multiple steps)
# 2. "Search for Python tutorials, then analyze the top 5 results and compare their approaches" → plan_execute (search + analysis + comparison)
# 3. "Load data.csv, calculate statistics, create 3 different charts, and generate a detailed report" → plan_execute (multiple distinct steps)
# 4. "First search for weather data, then analyze the trends, and finally create a forecast model" → plan_execute (explicit multi-step with dependencies)
# 5. "Analyze uploaded document, extract key points, search for related information online, and create a comprehensive summary" → plan_execute (document + web search + synthesis)
# 6. "Compare data from file1.csv and file2.csv, perform statistical analysis, and generate visualizations" → plan_execute (multiple files + multiple operations)
# 7. "Research topic X, then write code to demonstrate it, then create documentation" → plan_execute (search + code + documentation)
# 8. "Step 1: load data, Step 2: clean it, Step 3: visualize, Step 4: report" → plan_execute (explicit steps)
# 9. "Analyze all uploaded files and create a comparative analysis report" → plan_execute (multiple files)
# 10. "Build a complete data pipeline: load, transform, analyze, visualize, export" → plan_execute (complex workflow)

# DECISION RULES:

# EDGE CASES:
# 1. "Calculate variance of numbers" → chat (vague, no specific data)
# 2. "Show me how to calculate mean" → chat (educational, not execution)
# 3. "Analyze sales_data.csv" → plan-and-execute (single file, complicated reasoning required)
# 4. "Analyze data.csv and create detailed report with multiple charts" → plan_execute (multiple operations)
# 5. "Search for X" → react (simple search)
# 6. "Search for X and Y and compare them" → plan_execute (search + comparison)

# When in doubt:
# - Prefer "chat" for pure general knowledge questions
# - Prefer "react" for single-tool single-goal tasks
# - Choose "plan_execute" when multiple distinct steps or tools are clearly needed

# Respond with ONLY one word: "chat", "react", or "plan_execute" (no explanation, no punctuation)."""