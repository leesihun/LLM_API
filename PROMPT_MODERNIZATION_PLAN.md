# LLM API System Prompt Modernization Plan

**Version:** 1.0.0
**Date:** 2025-11-27
**Author:** Claude Code Assistant
**Objective:** Modernize all system prompts based on Anthropic/Claude Code best practices

---

## Executive Summary

This plan outlines a comprehensive modernization of all system prompts in the LLM API project, adopting proven patterns from Anthropic's Claude Code and official prompt engineering guidelines. The modernization focuses on **clarity, directness, professional objectivity, and reduced verbosity** while maintaining technical accuracy.

### Key Principles from Anthropic/Claude Style Guide

1. **Conciseness over Verbosity**: More direct, less machine-like communication
2. **Professional Objectivity**: Prioritize truth over validation; respectful correction over false agreement
3. **Clarity First**: Explicit, specific instructions without ambiguity
4. **Structured Thinking**: "think" < "think hard" < "think harder" < "ultrathink" for deeper reasoning
5. **No Unnecessary Formatting**: Reserve markdown for code; use flowing prose for text
6. **Zero Flattery**: Skip positive adjectives; respond directly to questions
7. **No Emojis**: Professional, focused communication unless explicitly requested
8. **Fact-Based Progress**: Grounded, concise status reports

---

## Current State Analysis

### Strengths
- ✅ **Modular architecture** (v2.0.0) with separated prompt modules
- ✅ **Centralized registry** (`PromptRegistry`) with caching and validation
- ✅ **Base utilities** with standard markers and separators
- ✅ **Reusable rule blocks** (FILENAME_RULES, JSON_SAFETY_RULES, etc.)
- ✅ **Temporal context** integration for current time awareness

### Issues to Address
- ❌ **Verbose ASCII borders** (80-char `====` separators reduce readability)
- ❌ **Machine-like tone** ("You are a helpful AI assistant" is generic)
- ❌ **Excessive formatting** (overuse of borders, markers, and structural elements)
- ❌ **Unclear role definitions** (lacks specificity in domain expertise)
- ❌ **Mixed instruction styles** (some prompts imperative, others explanatory)
- ❌ **Redundant sections** (multiple "RULES" headers with overlapping content)

---

## Modernization Strategy

### Phase 1: Core Principles Adoption

#### 1.1 Communication Style Guidelines

**OLD PATTERN (Verbose, Machine-like):**
```
You are a helpful AI assistant using the ReAct (Reasoning + Acting) framework.

================================================================================
                                  TASK
================================================================================

Think step-by-step about what you need to do to answer this question.
```

**NEW PATTERN (Direct, Professional):**
```
You are a ReAct reasoning specialist with expertise in tool-augmented problem solving.

## Task

Analyze the question step-by-step to determine the most efficient approach.
```

**Key Changes:**
- Specific role ("ReAct reasoning specialist") vs generic ("helpful AI assistant")
- Markdown headers (`##`) replace ASCII borders
- Action-oriented language ("Analyze") vs passive ("Think about")
- Removed unnecessary framework explanation

---

#### 1.2 Instruction Clarity

**Anthropic Best Practice:**
- Use imperative verbs: "Analyze", "Generate", "Execute", "Verify"
- Be specific about expertise: "Python code generation expert", "Data analysis specialist"
- State expected behavior directly without hedging

**Implementation:**
```python
# BEFORE
def get_react_thought_and_action_prompt(query: str, context: str) -> str:
    return f"""You are a helpful AI assistant using the ReAct framework.

    Think step-by-step and decide on an action."""

# AFTER
def get_react_thought_and_action_prompt(query: str, context: str) -> str:
    return f"""You are a ReAct reasoning agent specializing in multi-step problem decomposition.

    Analyze the query and determine the single most effective next action."""
```

---

#### 1.3 Formatting Simplification

**OLD:** Heavy use of ASCII borders, markers, and separators
```
================================================================================
                               CONTEXT
================================================================================

[RULE] 1. EXACT FILENAMES
   - Copy EXACT filename from META DATA section
   - [X] NO generic names: 'data.json'
   - [OK] Example: filename = 'sales_report.json'

================================================================================
                            RESPONSE FORMAT
================================================================================
```

**NEW:** Clean markdown structure with minimal visual noise
```
## Context

### Filename Requirements
- Use exact filenames from metadata
- Avoid generic names like 'data.json'
- Example: `filename = 'sales_report.json'`

## Response Format
```

**Benefits:**
- 70% less visual clutter
- Easier to parse for both LLMs and humans
- More readable in logs and debugging

---

### Phase 2: Prompt-by-Prompt Modernization

#### 2.1 ReAct Agent Prompts ([react_agent.py](backend/config/prompts/react_agent.py))

**Target Files:**
- `get_react_thought_and_action_prompt()` - Combined thought/action generation
- `get_react_final_answer_prompt()` - Final answer synthesis
- `get_react_step_verification_prompt()` - Step success verification

**Modernization Approach:**

##### 2.1.1 Thought and Action Prompt

**Current Issues:**
- Generic "helpful AI assistant" role
- Verbose available actions list (5 actions with 2-3 line descriptions each)
- Excessive formatting with repeated line breaks

**Modernized Version:**
```python
def get_react_thought_and_action_prompt(
    query: str,
    context: str,
    file_guidance: str = ""
) -> str:
    """
    ReAct agent's combined thought and action generation.
    Optimized for single-call efficiency with clear tool selection.
    """
    return f"""You are a ReAct reasoning specialist with expertise in tool-augmented problem solving.

{file_guidance}

## Query
{query}

## Context
{context}

## Your Task
Reason step-by-step about the next action, then select exactly one tool.

## Available Tools
1. **web_search** - Current information, news, real-time data
2. **rag_retrieval** - Query uploaded documents
3. **python_coder** - Data analysis, file processing, calculations
4. **vision_analyzer** - Image analysis, OCR, visual Q&A
5. **finish** - Provide final answer (only when complete)

## Response Format
THOUGHT: [Your reasoning about what to do next]

ACTION: [Tool name]

ACTION INPUT: [Input for the selected tool]

Think hard about which tool best addresses the immediate need."""
```

**Key Improvements:**
- Specific role: "ReAct reasoning specialist with expertise in..."
- Tool list reduced from ~15 lines to 5 lines (bold names + concise descriptions)
- Clear hierarchy: Query → Context → Task → Tools → Format
- Removed ASCII borders (80-char `====`)
- Added "Think hard" trigger for deeper reasoning
- Removed redundant "Note:" section

---

##### 2.1.2 Final Answer Prompt

**Current Issues:**
- Repeated emphasis ("IMPORTANT", "MUST", all caps)
- Numbered lists for simple instructions
- Verbose instructions about observation review

**Modernized Version:**
```python
def get_react_final_answer_prompt(query: str, context: str) -> str:
    """
    Generate final answer from all ReAct observations.
    Emphasizes comprehensiveness and specificity.
    """
    return f"""You are a synthesis specialist. Review all gathered information and provide a complete answer.

## Original Query
{query}

## All Observations
{context}

## Your Task
Synthesize the observations above into a clear, complete answer. Include specific details, numbers, and facts from the observations. Ensure your answer directly addresses the query.

Think harder about how to present the information most effectively."""
```

**Key Improvements:**
- Specific role: "synthesis specialist"
- Removed ALL CAPS emphasis (trust LLM to follow instructions)
- Removed numbered "MUST" list (replaced with flowing prose)
- "Think harder" for more thoughtful synthesis
- Cleaner structure with markdown headers

---

##### 2.1.3 Step Verification Prompt

**Current Issues:**
- Too simple/minimal (could benefit from more structure)
- Lacks guidance on what makes a step successful

**Modernized Version:**
```python
def get_react_step_verification_prompt(
    plan_step_goal: str,
    success_criteria: str,
    tool_used: str,
    observation: str
) -> str:
    """
    Verify if a plan step achieved its goal.
    Requires clear YES/NO decision with reasoning.
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

Think hard about whether the success criteria are truly satisfied."""
```

**Key Improvements:**
- Specific role: "verification specialist"
- Clear structure with markdown headers
- Added "Think hard" for careful evaluation
- More explicit about comparison task

---

#### 2.2 Python Coder Tool Prompts

**Location:** `backend/config/prompts/python_coder.py` (needs to be created from orchestrator)

**Current State:** Prompts embedded in orchestrator code, making them hard to maintain

**Target Functions:**
- `get_code_generation_with_self_verification_prompt()` - Main code generation
- `get_output_adequacy_check_prompt()` - Output quality verification
- Code fixing/patching prompts (in `code_fixer.py`, `code_patcher.py`)

**Modernization Approach:**

##### 2.2.1 Code Generation with Self-Verification

**Key Principles:**
- Role: "Python code generation expert specializing in data analysis"
- Clear multi-phase structure: Context → Task → Requirements → Self-Check → Output
- Minimal formatting, maximum clarity
- Action-oriented language

**Modernized Structure:**
```python
def get_code_generation_with_self_verification_prompt(
    query: str,
    context: Optional[str],
    file_context: str,
    is_prestep: bool,
    has_json_files: bool,
    conversation_history: Optional[List[dict]] = None,
    plan_context: Optional[dict] = None,
    react_context: Optional[dict] = None
) -> str:
    """
    Generate Python code with built-in self-verification.
    Single LLM call returns: code + verification result + issues list.
    """

    # Build history section (if available)
    history_section = ""
    if conversation_history:
        history_section = _build_conversation_history_section(conversation_history)

    # Build context sections
    plan_section = _build_plan_context_section(plan_context) if plan_context else ""
    react_section = _build_react_context_section(react_context) if react_context else ""

    return f"""You are a Python code generation expert specializing in data analysis, file processing, and scientific computing.

{history_section}
{plan_section}
{react_section}

## Task
{query}

## File Context
{file_context}

{context if context else ""}

## Code Requirements
- Use exact filenames from File Context metadata
- No command-line arguments or user input
- Print results directly to stdout (no unnecessary file writes)
- Handle errors gracefully with try/except
{'- Use .get() for safe JSON key access' if has_json_files else ''}

## Your Task
1. Generate Python code that accomplishes the task
2. Self-verify the code meets all requirements
3. List any potential issues

## Output Format (JSON)
```json
{{
  "code": "# Python code here",
  "self_check_passed": true/false,
  "issues": ["issue 1 if any", "issue 2 if any"]
}}
```

Think harder about edge cases and data validation."""
```

**Key Improvements:**
- Specific expertise: "Python code generation expert specializing in..."
- Context sections built dynamically (history, plan, react)
- Requirements as bulleted list (not numbered, not ALL CAPS)
- Clear 3-step task breakdown
- JSON output format clearly defined
- "Think harder" for better code quality

---

##### 2.2.2 Output Adequacy Check

**Current Issues:**
- Likely embedded in orchestrator without dedicated prompt function
- Needs clear criteria for "adequate"

**Modernized Version:**
```python
def get_output_adequacy_check_prompt(
    query: str,
    code: str,
    output: str,
    context: Optional[str]
) -> str:
    """
    Verify if code output adequately answers the user's query.
    Returns: adequate (bool), reason (str), suggestion (str).
    """
    return f"""You are a code output evaluator specializing in data analysis quality assurance.

## Original Query
{query}

## Generated Code
```python
{code}
```

## Code Output
{output}

{f"## Additional Context\n{context}" if context else ""}

## Your Task
Evaluate whether the output adequately answers the query. Consider:
- Does it directly address the question?
- Are results complete and specific?
- Are calculations/analysis correct?
- Is output format appropriate?

## Response Format (JSON)
```json
{{
  "adequate": true/false,
  "reason": "Brief explanation of your decision",
  "suggestion": "How to improve if inadequate (empty if adequate)"
}}
```

Think hard about whether the user's question is truly answered."""
```

**Key Improvements:**
- Specific role: "code output evaluator specializing in QA"
- Clear evaluation criteria (bulleted, not numbered)
- Markdown code fences for readability
- JSON output with 3 fields
- "Think hard" for careful evaluation

---

#### 2.3 Plan-Execute Prompts ([plan_execute.py](backend/config/prompts/plan_execute.py))

**Target Function:**
- `get_execution_plan_prompt()` - Create structured multi-step plans

**Current Issues:**
- Excessive use of ASCII borders (section_border() creates 80-char `====`)
- Overuse of emoji-style markers ([OK], [ERROR], [!!!])
- Verbose "STEP FIELDS" explanation (could be more concise)
- "VALID TOOLS" section uses markers instead of simple bullets

**Modernized Version:**
```python
def get_execution_plan_prompt(
    query: str,
    conversation_history: str,
    available_tools: str,
    has_files: bool = False,
    file_info: Optional[str] = None,
    timezone: str = "UTC"
) -> str:
    """
    Generate structured execution plan for multi-step tasks.
    Returns JSON array of plan steps with goals, tools, and success criteria.
    """
    time_context = get_current_time_context(timezone)

    # File-specific guidance
    file_section = ""
    if has_files:
        file_section = f"""
## Attached Files
{file_info if file_info else 'Files are attached.'}

**File Strategy:** Identify file types → Use python_coder for structured data → Start with analysis step.
"""

    return f"""You are an AI planning expert specializing in task decomposition and workflow optimization.

{time_context}
{file_section}

## Conversation History
{conversation_history}

## Current Query
{query}

## Available Tools
{available_tools}

## Plan Structure
Each step must include:
1. **step_num** - Sequential number (1, 2, 3...)
2. **goal** - What to accomplish (objective/outcome)
3. **primary_tools** - Which tool(s) to use
4. **success_criteria** - How to verify success (measurable)
5. **context** - How to execute (specific instructions)

### Goal vs Context Example
- Goal: "Calculate mean and median for numeric columns"
- Context: "Use pandas.describe(). Handle missing values with dropna()."

## Tool Selection Guide
- **web_search** - Current information, news, external data (input: 3-10 keywords)
- **rag_retrieval** - Search uploaded documents (input: natural language query)
- **python_coder** - Data analysis, file processing, calculations, visualizations
- **file_analyzer** - Quick metadata inspection

**Note:** Do not include 'finish' step (happens automatically)

## Planning Rules
1. **Work steps only** - Each step must produce tangible output
2. **Synthesis last** - Add synthesis step only if combining multiple results
3. **Tool preference** - python_coder > rag_retrieval for data files
4. **Granularity** - Break complex goals into 2-4 completable steps

## Success Criteria Examples
**Good:**
- "Data loaded: 1000 rows, 5 columns. Column names and types shown."
- "Mean=45.6, Median=42.3 calculated with labels"
- "Chart saved to output/chart.png with axis labels and legend"

**Bad:**
- "Data processed successfully" (too vague)
- "Analysis complete" (no specifics)
- "Code runs without errors" (no output verification)

## Output Format
Respond with ONLY a JSON array. No markdown, no explanations.

```json
[
  {
    "step_num": 1,
    "goal": "Clear description",
    "primary_tools": ["tool_name"],
    "success_criteria": "Measurable indicators",
    "context": "Specific instructions"
  }
]
```

## Example
Query: "Analyze sales.csv and create revenue chart"

```json
[
  {
    "step_num": 1,
    "goal": "Load and explore sales.csv structure",
    "primary_tools": ["python_coder"],
    "success_criteria": "File loaded with row count, columns, types, first 5 rows displayed",
    "context": "Use pandas.read_csv(). Show df.shape, df.columns, df.head()."
  },
  {
    "step_num": 2,
    "goal": "Calculate revenue by region and create bar chart",
    "primary_tools": ["python_coder"],
    "success_criteria": "Chart saved to output/revenue.png with labels and legend",
    "context": "Use groupby() for aggregation. matplotlib for chart. Save with dpi=300."
  }
]
```

Think harder about step dependencies and optimal execution order."""
```

**Key Improvements:**
- Specific role: "AI planning expert specializing in task decomposition and workflow optimization"
- Removed ALL section_border() calls (80-char ASCII borders)
- Replaced [OK]/[ERROR] markers with markdown bold
- Simplified "Plan Structure" from verbose explanation to numbered list
- "Tool Selection Guide" uses simple markdown bold instead of markers
- "Planning Rules" uses numbered list without [RULE] prefix
- Success criteria examples use bold headers instead of markers
- Added "Think harder" for better planning
- Cleaner visual hierarchy with markdown headers only

---

#### 2.4 Task Classification Prompt ([task_classification.py](backend/config/prompts/task_classification.py))

**Current State:**
- Extremely minimal (18 lines total)
- Generic instructions without examples
- Lacks guidance on edge cases

**Current Issues:**
- Too simplistic for complex classification decisions
- No examples of each category
- Doesn't explain reasoning process

**Modernized Version:**
```python
def get_agent_type_classifier_prompt() -> str:
    """
    Classify queries into agent types: chat, react, or plan_execute.
    Provides examples and decision criteria for accurate classification.
    """
    return """You are a query classification specialist with expertise in intent recognition and workflow routing.

## Your Task
Classify the user's query into exactly one category:

### Categories

**chat** - Simple questions answerable from general knowledge, no tools needed
- Examples: "What is AI?", "Explain photosynthesis", "Define recursion"
- Criteria: Factual question, no computation, no external data

**react** - Single-goal tasks requiring 1-2 tools
- Examples: "Search for latest AI news", "Analyze this CSV file", "Calculate mean of these numbers"
- Criteria: One clear objective, straightforward tool usage

**plan_execute** - Multi-step complex tasks requiring planning
- Examples: "Analyze sales.csv, create visualizations, generate PowerPoint report"
- Criteria: Multiple distinct goals, workflow coordination needed

## Response
Respond with exactly one word: chat, react, or plan_execute

No explanation, no punctuation.

Think hard about the complexity and tool requirements."""
```

**Key Improvements:**
- Specific role: "query classification specialist with expertise in intent recognition"
- Clear category definitions with examples and criteria
- Markdown structure replaces simple paragraph format
- Added "Think hard" for better classification accuracy
- More detailed guidance reduces misclassification

---

#### 2.5 Base Prompt Utilities ([base.py](backend/config/prompts/base.py))

**Current Issues:**
- Heavy reliance on ASCII markers: [OK], [ERROR], [!!!], [RULE]
- 80-character section borders create visual clutter
- Emoji-style markers feel dated compared to modern Anthropic style

**Modernization Strategy:**

##### 2.5.1 Deprecate ASCII Markers

**BEFORE:**
```python
MARKER_OK = "[OK]"
MARKER_ERROR = "[X]"
MARKER_CRITICAL = "[!!!]"
MARKER_RULE = "[RULE]"
```

**AFTER (Markdown-based):**
```python
# Use markdown emphasis instead of markers
# [OK] → ✓ or **Good:**
# [X] → ✗ or **Bad:**
# [!!!] → **Critical:**
# [RULE] → Simple numbered lists

# DEPRECATED: Keep for backward compatibility during migration
MARKER_OK = "**Good:**"
MARKER_ERROR = "**Bad:**"
MARKER_CRITICAL = "**Critical:**"
MARKER_RULE = ""  # Use numbered lists instead
```

---

##### 2.5.2 Simplify Section Borders

**BEFORE:**
```python
def section_border(title: str = "", width: int = 80) -> str:
    """Create a standard section border with optional centered title."""
    border = "=" * width
    if title:
        return f"{border}\n{title.center(width)}\n{border}"
    return border
```

**AFTER:**
```python
def section_header(title: str, level: int = 2) -> str:
    """
    Create a markdown section header.

    Args:
        title: Section title
        level: Markdown header level (1-6, default 2 for ##)

    Returns:
        Markdown header string

    Example:
        section_header("Task", 2) -> "## Task"
    """
    prefix = "#" * min(max(level, 1), 6)
    return f"{prefix} {title}"

# DEPRECATED: Keep for backward compatibility
def section_border(title: str = "", width: int = 80) -> str:
    """DEPRECATED: Use section_header() instead."""
    if title:
        return section_header(title, level=2)
    return ""
```

---

##### 2.5.3 Modernize Reusable Rule Blocks

**BEFORE (Verbose with markers):**
```python
FILENAME_RULES = f"""
{MARKER_RULE} EXACT FILENAMES
   - Copy EXACT filename from META DATA section
   - {MARKER_ERROR} NO generic names: 'data.json', 'file.json', 'input.csv'
   - {MARKER_OK} Example: filename = 'sales_report_Q4_2024.json'
"""
```

**AFTER (Clean markdown):**
```python
FILENAME_RULES = """
### Filename Requirements
- Use exact filenames from metadata
- Avoid generic names: `data.json`, `file.json`, `input.csv`
- Example: `filename = 'sales_report_Q4_2024.json'`
"""
```

---

##### 2.5.4 New Utility Functions

Add modern helper functions for Anthropic-style prompts:

```python
def role_definition(
    title: str,
    expertise: Optional[str] = None,
    context: Optional[str] = None
) -> str:
    """
    Create a specific role definition following Anthropic best practices.

    Args:
        title: Professional role (e.g., "Python code generation expert")
        expertise: Optional specialization (e.g., "specializing in data analysis")
        context: Optional additional context

    Returns:
        Complete role definition string

    Example:
        role_definition(
            "Python code generation expert",
            "specializing in data analysis",
            "You work with pandas, numpy, and matplotlib."
        )
        → "You are a Python code generation expert specializing in data analysis.
           You work with pandas, numpy, and matplotlib."
    """
    parts = [f"You are a {title}"]

    if expertise:
        parts[0] += f" {expertise}"

    parts[0] += "."

    if context:
        parts.append(context)

    return " ".join(parts)


def thinking_trigger(depth: str = "hard") -> str:
    """
    Add a thinking depth trigger following Anthropic patterns.

    Args:
        depth: Thinking depth level
            - "normal" → "Think about..."
            - "hard" → "Think hard about..."
            - "harder" → "Think harder about..."
            - "ultra" → "Ultrathink about..."

    Returns:
        Thinking prompt string

    Examples:
        thinking_trigger("hard") → "Think hard about"
        thinking_trigger("ultra") → "Ultrathink about"
    """
    triggers = {
        "normal": "Think about",
        "hard": "Think hard about",
        "harder": "Think harder about",
        "ultra": "Ultrathink about"
    }
    return triggers.get(depth, "Think hard about")


def format_code_block(code: str, language: str = "python") -> str:
    """
    Format code with proper markdown fences.

    Args:
        code: Code content
        language: Syntax highlighting language

    Returns:
        Markdown code block
    """
    return f"```{language}\n{code}\n```"


def format_task_section(task: str, examples: Optional[List[str]] = None) -> str:
    """
    Format a task section with optional examples (Anthropic style).

    Args:
        task: Task description
        examples: Optional list of example strings

    Returns:
        Formatted task section with examples
    """
    lines = [f"## Your Task\n{task}"]

    if examples:
        lines.append("\n### Examples")
        for i, example in enumerate(examples, 1):
            lines.append(f"{i}. {example}")

    return "\n".join(lines)
```

---

### Phase 3: Migration Strategy

#### 3.1 Backward Compatibility Plan

**Critical:** Maintain 100% backward compatibility during migration.

**Strategy:**
1. Keep old functions with `DEPRECATED` markers
2. Create new functions with `_v2` suffix
3. Update callers incrementally
4. Remove deprecated functions in v3.0.0

**Example:**
```python
# OLD (deprecated but working)
def get_react_thought_and_action_prompt(query, context, file_guidance=""):
    """DEPRECATED: Use get_react_thought_and_action_prompt_v2()"""
    return get_react_thought_and_action_prompt_v2(query, context, file_guidance)

# NEW (modernized)
def get_react_thought_and_action_prompt_v2(query, context, file_guidance=""):
    """Modernized ReAct prompt following Anthropic best practices."""
    # ... new implementation
```

---

#### 3.2 Migration Order (Lowest Risk → Highest Impact)

**Week 1: Base Infrastructure**
1. Update `base.py` with new utilities (non-breaking)
2. Add deprecation warnings to old markers
3. Create new helper functions (role_definition, thinking_trigger, etc.)

**Week 2: Task Classification & Planning**
4. Modernize `task_classification.py` (low usage, easy to test)
5. Modernize `plan_execute.py` (moderate usage, clear boundaries)

**Week 3: Core Agent Prompts**
6. Modernize `react_agent.py` (high usage, incremental update)
   - Start with `get_react_final_answer_prompt()` (lowest risk)
   - Then `get_react_step_verification_prompt()`
   - Finally `get_react_thought_and_action_prompt()` (highest usage)

**Week 4: Python Coder Tool**
7. Extract prompts from `orchestrator.py` to new `python_coder.py` prompt module
8. Modernize code generation prompts
9. Modernize verification and fixing prompts

**Week 5: Testing & Refinement**
10. A/B testing: old vs new prompts on standard test suite
11. Measure: response quality, token usage, latency
12. Iterate based on results

**Week 6: Deprecation Cleanup (v3.0.0)**
13. Remove deprecated functions
14. Remove old ASCII markers
15. Update all documentation

---

#### 3.3 Testing Strategy

##### 3.3.1 Unit Tests
Create test suite for each modernized prompt:
```python
# tests/prompts/test_react_prompts_v2.py
import pytest
from backend.config.prompts.react_agent import (
    get_react_thought_and_action_prompt_v2,
    get_react_final_answer_prompt_v2
)

def test_react_thought_action_includes_role():
    prompt = get_react_thought_and_action_prompt_v2("test query", "")
    assert "ReAct reasoning specialist" in prompt
    assert "helpful AI assistant" not in prompt  # Old pattern

def test_react_no_ascii_borders():
    prompt = get_react_thought_and_action_prompt_v2("test", "")
    assert "=" * 80 not in prompt
    assert "## " in prompt  # Markdown headers

def test_react_includes_thinking_trigger():
    prompt = get_react_thought_and_action_prompt_v2("test", "")
    assert "Think hard" in prompt or "Think harder" in prompt
```

##### 3.3.2 Integration Tests
Test prompts in real agent workflows:
```python
# tests/integration/test_react_agent_with_new_prompts.py
import pytest
from backend.tasks.react import ReActAgent

@pytest.mark.asyncio
async def test_react_agent_with_modernized_prompts(mock_llm):
    """Test ReAct agent with new prompt style."""
    agent = ReActAgent(llm=mock_llm, use_v2_prompts=True)
    result = await agent.execute("What is AI?")

    # Verify prompt sent to LLM
    call_args = mock_llm.ainvoke.call_args
    prompt = call_args[0][0].content

    assert "ReAct reasoning specialist" in prompt
    assert "Think hard" in prompt
    assert "=" * 80 not in prompt
```

##### 3.3.3 A/B Testing Framework
Compare old vs new prompts on identical tasks:
```python
# scripts/ab_test_prompts.py
import asyncio
from backend.config.prompts.react_agent import (
    get_react_thought_and_action_prompt,  # OLD
    get_react_thought_and_action_prompt_v2  # NEW
)

async def ab_test_react_prompts(test_queries: List[str]):
    results = {"old": [], "new": []}

    for query in test_queries:
        # Test old prompt
        old_result = await agent.execute(query, use_v2_prompts=False)
        results["old"].append(measure_quality(old_result))

        # Test new prompt
        new_result = await agent.execute(query, use_v2_prompts=True)
        results["new"].append(measure_quality(new_result))

    # Compare metrics
    print_comparison(results)
```

**Metrics to Track:**
- Response quality (correctness, completeness)
- Token usage (input + output tokens)
- Latency (time to first token, total time)
- Tool selection accuracy (for ReAct prompts)
- Code generation success rate (for Python coder)

---

### Phase 4: Documentation Updates

#### 4.1 Update CLAUDE.md

Add new section documenting prompt modernization:

```markdown
## Prompt Engineering Guidelines (v2.0.0)

### Anthropic Style Principles
- **Direct, not verbose**: Skip fluff, get to the point
- **Specific roles**: "Python code generation expert" not "helpful AI assistant"
- **Markdown structure**: Clean headers, no ASCII borders
- **Thinking triggers**: "Think hard", "Think harder", "Ultrathink" for deeper reasoning
- **Zero flattery**: No "great question!" responses
- **Professional objectivity**: Truth over validation

### Prompt Structure Template
```python
def get_my_prompt(query: str) -> str:
    return f"""You are a [specific role with expertise].

## [Section Title]
[Content]

## Your Task
[Clear, action-oriented instructions]

[Thinking trigger if needed]"""
```

### Migration Status
- ✅ Task classification prompts (v2.0.0)
- ✅ Plan-Execute prompts (v2.0.0)
- ✅ ReAct agent prompts (v2.0.0)
- 🚧 Python coder prompts (in progress)
- ⏳ File analyzer prompts (planned)
```

---

#### 4.2 Update API Documentation

Document prompt changes in API docs:

```markdown
## Breaking Changes (v2.0.0)

### Prompt Modernization
All system prompts have been modernized following Anthropic/Claude Code best practices.

**User Impact:**
- More concise, direct responses from agents
- Reduced verbosity in logs and debugging output
- Improved token efficiency (10-20% reduction in prompt tokens)

**Developer Impact:**
- Old prompt functions deprecated (use `_v2` variants)
- ASCII markers ([OK], [X]) replaced with markdown
- Section borders (80-char `====`) replaced with markdown headers

**Migration Guide:**
See `PROMPT_MODERNIZATION_PLAN.md` for detailed migration instructions.
```

---

### Phase 5: Performance Optimization

#### 5.1 Token Reduction Analysis

**Expected Savings:**

| Component | Old Tokens | New Tokens | Savings |
|-----------|-----------|-----------|---------|
| ReAct thought/action | ~800 | ~600 | 25% |
| ReAct final answer | ~600 | ~450 | 25% |
| Plan-Execute planning | ~1200 | ~900 | 25% |
| Python code generation | ~1800 | ~1400 | 22% |
| **Total Average** | **~4400** | **~3350** | **24%** |

**Benefits:**
- 24% reduction in input tokens per workflow
- Faster LLM response times (less to parse)
- Lower API costs
- More context budget for user data

---

#### 5.2 Response Quality Improvements

**Expected Improvements:**
1. **Better tool selection** (ReAct agent)
   - Specific role → better understanding of tool capabilities
   - Clear examples → fewer selection errors

2. **Higher code quality** (Python coder)
   - Expert role → more sophisticated code patterns
   - Self-verification → fewer execution errors

3. **Better planning** (Plan-Execute)
   - "Think harder" → deeper task decomposition
   - Clear success criteria → more measurable steps

---

### Phase 6: Long-Term Maintenance

#### 6.1 Prompt Version Control

Track prompt changes with semantic versioning:

```python
# backend/config/prompts/react_agent.py
PROMPT_VERSION = "2.0.0"  # Major.Minor.Patch

def get_react_thought_and_action_prompt_v2(
    query: str,
    context: str,
    file_guidance: str = "",
    version: str = PROMPT_VERSION
) -> str:
    """
    ReAct thought and action generation prompt.

    Version: 2.0.0
    Changes: Modernized to Anthropic style (removed ASCII borders, added specific role)
    """
    # ... implementation
```

---

#### 6.2 Prompt Monitoring Dashboard

Create monitoring to track prompt performance:

```python
# backend/monitoring/prompt_metrics.py
class PromptMetrics:
    """Track prompt performance metrics."""

    @staticmethod
    def log_prompt_execution(
        prompt_name: str,
        version: str,
        tokens_input: int,
        tokens_output: int,
        latency_ms: float,
        success: bool
    ):
        """Log prompt execution metrics."""
        # ... implementation

    @staticmethod
    def get_prompt_performance(prompt_name: str, version: str) -> Dict:
        """Get aggregate performance metrics for a prompt."""
        return {
            "avg_tokens_input": ...,
            "avg_tokens_output": ...,
            "avg_latency_ms": ...,
            "success_rate": ...,
            "total_executions": ...
        }
```

---

#### 6.3 Continuous Improvement Process

**Quarterly Review:**
1. Analyze prompt metrics (token usage, success rate, latency)
2. Review Claude/Anthropic docs for new best practices
3. A/B test prompt variations
4. Update prompts based on findings

**Feedback Loop:**
- User feedback → Prompt improvement ideas
- Error analysis → Instruction refinement
- Success patterns → Reusable templates

---

## Implementation Checklist

### Week 1: Base Infrastructure ✅
- [ ] Update `base.py` with new utility functions
- [ ] Add deprecation warnings to old markers
- [ ] Create `role_definition()`, `thinking_trigger()` helpers
- [ ] Write unit tests for new utilities

### Week 2: Task Classification & Planning ✅
- [ ] Modernize `task_classification.py`
- [ ] Modernize `plan_execute.py`
- [ ] Add examples and thinking triggers
- [ ] Write integration tests

### Week 3: Core Agent Prompts ✅
- [ ] Modernize `get_react_final_answer_prompt()`
- [ ] Modernize `get_react_step_verification_prompt()`
- [ ] Modernize `get_react_thought_and_action_prompt()`
- [ ] Update ReActAgent to use new prompts

### Week 4: Python Coder Tool ✅
- [ ] Extract prompts to `python_coder.py` module
- [ ] Modernize code generation prompts
- [ ] Modernize verification prompts
- [ ] Update PythonCoderTool orchestrator

### Week 5: Testing & Refinement ✅
- [ ] Run A/B tests on standard test suite
- [ ] Measure token usage reduction
- [ ] Measure response quality improvements
- [ ] Iterate based on results

### Week 6: Cleanup & Documentation ✅
- [ ] Remove deprecated functions (v3.0.0)
- [ ] Update CLAUDE.md with new guidelines
- [ ] Update API documentation
- [ ] Create migration guide for external users

---

## Risk Assessment

### Low Risk ✅
- **Base utility updates**: Non-breaking, additive changes
- **Task classification**: Low usage, easy rollback
- **Documentation updates**: No code impact

### Medium Risk ⚠️
- **Plan-Execute prompts**: Moderate usage, affects multi-step workflows
- **ReAct final answer**: Affects response quality but easy to test

### High Risk 🔴
- **ReAct thought/action**: Core workflow, highest usage
- **Python coder generation**: Complex prompts, many edge cases

**Mitigation:**
- Incremental rollout with feature flags
- Extensive A/B testing before full deployment
- Quick rollback capability
- Gradual migration path (old → _v2 → default)

---

## Success Metrics

### Quantitative
- ✅ 20-30% reduction in prompt tokens
- ✅ Maintain or improve response quality (measured by test suite)
- ✅ Maintain or improve code execution success rate
- ✅ No increase in average latency

### Qualitative
- ✅ Cleaner, more readable prompt logs
- ✅ Easier prompt maintenance and updates
- ✅ Better alignment with Anthropic best practices
- ✅ More professional, direct agent responses

---

## Conclusion

This modernization plan brings the LLM API project's prompts in line with Anthropic/Claude Code best practices, emphasizing **clarity, directness, and professional objectivity**. The phased approach ensures backward compatibility while systematically improving prompt quality across all system components.

**Key Takeaways:**
1. **Specific roles** > Generic "helpful assistant"
2. **Markdown structure** > ASCII borders
3. **Direct instructions** > Verbose explanations
4. **Thinking triggers** > Passive prompts
5. **Professional objectivity** > False validation

**Next Steps:**
1. Review and approve this plan
2. Begin Week 1 implementation (base infrastructure)
3. Set up A/B testing framework
4. Track metrics throughout migration

---

**Document Version:** 1.0.0
**Last Updated:** 2025-11-27
**Status:** Ready for Review
