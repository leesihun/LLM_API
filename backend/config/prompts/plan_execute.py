"""
Plan-Execute Agent Prompts
Prompts for planning and executing complex multi-step tasks.
"""

from typing import Optional

def get_execution_plan_prompt(
    query: str,
    conversation_history: str,
    available_tools: str,
    has_files: bool = False,
    file_info: Optional[str] = None  # NEW: File type information
) -> str:
    """
    Enhanced prompt for creating structured execution plans.
    
    Improvements:
    - Clearer tool selection guidance with file type awareness
    - Resolved synthesis step contradiction
    - ASCII-safe markers throughout
    - Explicit context vs goal distinction
    - Step dependency support
    - Better success criteria examples
    """
    
    # File-specific guidance with tool recommendations
    file_guidance = ""
    if has_files:
        file_guidance = """
IMPORTANT: Files are attached to this task.

FILE HANDLING STRATEGY:
1. Identify file types from the list below
2. Choose appropriate tools:
   - Structured data (CSV, Excel, JSON) → python_coder (for calculations, analysis)
   - Text documents (TXT, PDF, MD) → rag_retrieval (for search, extraction)
   - Mixed or unknown → python_coder first (can handle most formats)

3. ALWAYS start with file analysis/exploration before processing
"""
        if file_info:
            file_guidance += f"\nAttached Files:\n{file_info}\n"

    return f"""You are an AI planning expert. Create a DETAILED, executable plan for this multi-step task.

{file_guidance}

Conversation History:
{conversation_history}

Current User Query: {query}

Available Tools:
{available_tools}

================================================================================
PLANNING INSTRUCTIONS
================================================================================

Create a step-by-step execution plan with ONLY the work steps needed to complete the task.

For EACH step, provide these fields:

1. "step_num" (number): Sequential step number starting from 1
2. "goal" (string): WHAT to accomplish - the objective/outcome of this step
3. "primary_tools" (array): WHICH tool(s) to use - choose from valid tools below
4. "success_criteria" (string): HOW to verify success - concrete, measurable outcomes
5. "context" (string): HOW to execute - specific instructions, edge cases, techniques

FIELD DISTINCTION - Goal vs Context:
- Goal: The WHAT - "Calculate mean and median for all numeric columns"
- Context: The HOW - "Use pandas.describe() for quick stats. Handle missing values with dropna(). Print results with 2 decimal precision."

================================================================================
VALID TOOLS
================================================================================

[OK] web_search
   - When: Need current/real-time information, recent news, external data
   - Input: 3-10 specific keywords (names, dates, places, products)
   - Example: "latest AI developments 2025", "Bitcoin price January 2025"

[OK] rag_retrieval  
   - When: Search uploaded text documents (PDF, TXT, MD)
   - Input: Natural language query about document content
   - Example: "What does the contract say about termination?"

[OK] python_coder
   - When: Data analysis, file processing, calculations, visualizations
   - Handles: CSV, Excel, JSON, images, structured data
   - Input: Clear task description with expected output
   - Example: "Load sales.csv, calculate total revenue by region, create bar chart"

[OK] file_analyzer
   - When: Quick file metadata inspection (size, type, structure preview)
   - Input: Request for file information
   - Example: "Show me the structure of uploaded files"

[X] finish - DO NOT include as a tool in your plan (happens automatically)

================================================================================
STEP PLANNING RULES
================================================================================

[RULE 1] WORK STEPS ONLY
   - Include ONLY action steps (analysis, search, calculation, visualization)
   - Each step must produce tangible output (data, results, files, insights)

[RULE 2] SYNTHESIS STEP HANDLING
   - If final answer requires combining multiple results → ADD a synthesis step
   - If task is straightforward → last work step naturally completes the answer
   
   [OK] Multi-file comparison: 
        Step 1: Analyze file A
        Step 2: Analyze file B
        Step 3: Compare results from Step 1 and 2, generate summary report
   
   [OK] Single calculation:
        Step 1: Calculate variance of numbers (naturally complete)

[RULE 3] TOOL SELECTION
   - Use file type to guide tool choice (see FILE HANDLING STRATEGY above)
   - For files: python_coder > rag_retrieval (code is more powerful)
   - Start with exploration/analysis before complex operations

[RULE 4] STEP GRANULARITY
   - Break complex goals into 2-4 smaller steps
   - Each step should complete in one tool execution
   - Avoid overly granular steps (don't split "load file" and "show preview")


================================================================================
SUCCESS CRITERIA GUIDELINES
================================================================================

GOOD Success Criteria - Specific, Measurable, Verifiable:

[OK] "Data loaded successfully with 1000 rows, 5 columns displayed. Column names and types shown."
[OK] "Mean=45.6, Median=42.3, StdDev=12.1 calculated and printed with labels"
[OK] "Search returned 3-5 relevant articles from 2025 with titles and summaries"
[OK] "Bar chart saved to output/sales_by_region.png with proper axis labels and legend"
[OK] "Outliers identified: 8 values beyond 2 standard deviations, list printed"

BAD Success Criteria - Vague, Unmeasurable, Non-specific:

[X] "Data processed successfully" (what processing? what output?)
[X] "Analysis complete" (what analysis? what results?)
[X] "Information retrieved" (how much? what quality? what format?)
[X] "Code runs without errors" (but what should it produce?)
[X] "Step finished" (meaningless for verification)

FORMATTING NOTE: Use ASCII-safe markers only: [OK], [X], [!!!], [WARNING]
DO NOT use Unicode symbols: ✓, ✗, ⚠, etc.

================================================================================
RESPONSE FORMAT
================================================================================

You MUST respond with ONLY a JSON array of steps. No explanations, no markdown fences, JUST the array.

Required structure for each step:
{{
  "step_num": 1,
  "goal": "Clear description of what to accomplish",
  "primary_tools": ["tool_name"],
  "success_criteria": "Measurable success indicators",
  "context": "Specific execution instructions and techniques"
}}

================================================================================
EXAMPLE PLANS
================================================================================

EXAMPLE 1 - Multi-file analysis with synthesis:

[
  {{
    "step_num": 1,
    "goal": "Load and explore sales_2024.csv structure, identify key columns",
    "primary_tools": ["python_coder"],
    "success_criteria": "File loaded with row count, column names, data types, and first 5 rows displayed",
    "context": "Use pandas.read_csv(). Show df.shape, df.columns, df.dtypes, df.head(). Check for missing values with df.isnull().sum()."
  }},
  {{
    "step_num": 2,
    "goal": "Calculate total revenue by region and product category",
    "primary_tools": ["python_coder"],
    "success_criteria": "Aggregated revenue table printed with regions as rows, categories as columns, and total sums",
    "context": "Use pandas groupby() with multiple columns. Create pivot table for better readability. Sort by total revenue descending."
  }},
  {{
    "step_num": 3,
    "goal": "Create bar chart visualization of revenue by region",
    "primary_tools": ["python_coder"],
    "success_criteria": "Chart saved to output/revenue_by_region.png with clear labels, legend, and proper formatting",
    "context": "Use matplotlib.pyplot. Include title, axis labels, value labels on bars. Save with dpi=300 for quality."
  }},
  {{
    "step_num": 4,
    "goal": "Generate summary report combining all findings with key insights",
    "primary_tools": ["python_coder"],
    "success_criteria": "Formatted report with: total revenue, top 3 regions, top 3 products, growth trends, and recommendations printed",
    "context": "Synthesize results from steps 1-3. Use markdown-style formatting for readability. Include specific numbers and percentages."
  }}
]

EXAMPLE 2 - Simple single-file task (no synthesis needed):

[
  {{
    "step_num": 1,
    "goal": "Calculate mean, median, and standard deviation of the 'price' column in products.csv",
    "primary_tools": ["python_coder"],
    "success_criteria": "Three statistics printed with labels: Mean=$X.XX, Median=$Y.YY, StdDev=$Z.ZZ",
    "context": "Load CSV with pandas. Use df['price'].mean(), median(), std(). Format as currency with 2 decimals. Handle any missing values with dropna()."
  }}
]

EXAMPLE 3 - Web search + synthesis:

[
  {{
    "step_num": 1,
    "goal": "Search for latest developments in quantum computing from 2025",
    "primary_tools": ["web_search"],
    "success_criteria": "Retrieved 3-5 articles from 2025 with titles, sources, and key points",
    "context": "Use keywords: 'quantum computing breakthrough 2025', 'quantum computer advances January 2025'. Focus on recent technical developments, not general news."
  }},
  {{
    "step_num": 2,
    "goal": "Synthesize findings into structured summary with key breakthroughs and implications",
    "primary_tools": ["python_coder"],
    "success_criteria": "Formatted report with: 3-5 major breakthroughs listed, technical details for each, potential applications, and future outlook",
    "context": "Extract common themes from search results. Organize by: 1) Breakthrough name, 2) Technical details, 3) Implications, 4) Source. Use bullet points for clarity."
  }}
]

================================================================================

Now create a structured plan for the user's query.

Respond with ONLY the JSON array - no markdown fences, no explanations, no additional text:"""
