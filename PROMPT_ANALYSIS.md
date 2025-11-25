# Comprehensive Analysis: Prompt System Issues

**Date:** 2025-11-25
**Analyzed By:** Claude Code
**Scope:** Task Classification, ReAct Agent, Plan-Execute, Python Coder

---

## Executive Summary

After thorough analysis of the prompt architecture across all major system components, I've identified **critical issues and inconsistencies** that impact system performance, reliability, and maintainability. This document details:

- 6 major issue categories
- 15+ specific problems with code references
- Priority-ranked recommendations
- Architectural improvement suggestions

---

## Table of Contents

1. [Task Classification Prompt - Severe Oversimplification](#1-task-classification-prompt---severe-oversimplification)
2. [ReAct Agent Prompts - Inconsistent Messaging](#2-react-agent-prompts---inconsistent-messaging)
3. [Plan-Execute Prompts - Critical Structural Issues](#3-plan-execute-prompts---critical-structural-issues)
4. [Python Coder Prompts - The Deepest Issues](#4-python-coder-prompts---the-deepest-issues)
5. [Cross-System Issues](#5-cross-system-issues)
6. [Priority Recommendations](#6-priority-recommendations)
7. [Long-Term Architectural Recommendations](#7-long-term-architectural-recommendations)

---

## 1. TASK CLASSIFICATION PROMPT - SEVERE OVERSIMPLIFICATION

**Location:** `backend/config/prompts/task_classification.py:14-18`

### Current State (PROBLEMATIC):

```python
return """You are an agent type classifier. Classify user queries into one of three types: "chat", "react", or "plan_execute".
CHAT - Very simple questions answerable from easy general knowledge base (NO tools needed):
REACT - A little bit complicated, single-goal tasks requiring tools (web search, code execution, simple analysis):
PLAN_EXECUTE - Multi-step complex tasks requiring planning and structured execution:
Respond with ONLY one word: "chat", "react", or "plan_execute" (no explanation, no punctuation)."""
```

### Critical Problems:

1. **No examples provided** - The commented-out section (lines 21-79) contains 30+ valuable examples that are currently **DISABLED**
2. **No decision rules** - No guidance for edge cases
3. **Vague definitions** - "A little bit complicated" is subjective and unclear
4. **Missing context** - The classifier doesn't know when files are attached or what conversation history exists

### Impact:

- ❌ **Misclassification rate will be high** - Without examples, the LLM guesses
- ❌ **Inconsistent routing** - Same queries may route differently across sessions
- ❌ **Wasted resources** - Simple tasks routed to plan_execute, complex tasks to react

### Recommended Fix:

```python
def get_agent_type_classifier_prompt() -> str:
    return """You are an agent type classifier. Classify user queries into one of three types: "chat", "react", or "plan_execute".

CHAT - Very simple questions answerable from easy general knowledge base (NO tools needed):
Examples:
- "What is Python?" → chat (general knowledge)
- "Explain recursion to me" → chat (concept explanation)
- "What is the capital of France?" → chat (established fact)

REACT - Single-goal tasks requiring tools (web search, code execution, simple analysis):
Examples:
- "What's the weather in Seoul right now?" → react (single web search)
- "Calculate the variance of [1,2,3,4,5]" → react (single computation)
- "Analyze sales_data.csv and show basic statistics" → react (single file analysis)

PLAN_EXECUTE - Multi-step complex tasks requiring planning and structured execution:
Examples:
- "Analyze sales_data.csv AND customer_data.xlsx, then create visualizations and a summary report" → plan_execute (multiple files + multiple steps)
- "Load data.csv, calculate statistics, create 3 different charts, and generate a detailed report" → plan_execute (multiple distinct steps)

DECISION RULES:
- Prefer "chat" for pure knowledge questions
- Prefer "react" for single-tool single-goal tasks
- Choose "plan_execute" when multiple distinct steps or tools are clearly needed

Respond with ONLY one word: "chat", "react", or "plan_execute" (no explanation, no punctuation)."""
```

---

## 2. REACT AGENT PROMPTS - INCONSISTENT MESSAGING

**Location:** `backend/config/prompts/react_agent.py`

### Problem 2.1: File Handling Confusion

**Line 53:** "Note: File metadata analysis is done automatically when files are attached."

**Reality Check:** This is misleading. Looking at the code:
- File metadata is extracted in `orchestrator.py:590`
- But the ReAct agent still needs to decide WHETHER to use python_coder
- The prompt makes it sound like file analysis happens without agent action

### Problem 2.2: Tool Selection Guidance Mismatch

**Lines 40-51:** The prompt lists 4 tools but provides insufficient guidance:

```python
1. web_search - Search the web for current information
   - Use 3-10 specific keywords  # TOO DETAILED FOR THOUGHT-ACTION PHASE
   - Include names, dates, places, products
   - Examples: "latest AI developments 2025", "Python vs JavaScript performance"
   - Avoid single words or vague queries

2. rag_retrieval - Retrieve relevant documents from uploaded files
   # NO GUIDANCE - when to use vs python_coder?

3. python_coder - Generate and execute Python code for data analysis and file processing
   # NO GUIDANCE - what file types? what operations?

4. finish - Provide the final answer (use ONLY when you have complete information)
```

**The Issue:**
- web_search gets 4 bullet points of guidance
- rag_retrieval gets **ZERO** guidance
- python_coder gets **ZERO** guidance about capabilities
- No distinction between when to use rag_retrieval vs python_coder for file analysis

### Problem 2.3: Final Answer Prompt - Weak Instruction

**Lines 78-85:**

```python
IMPORTANT: Review ALL the observations above carefully. Each observation contains critical information...

Your final answer MUST:
1. Incorporate ALL relevant information from the observations
2. Be comprehensive and complete
3. Directly answer the user's question
4. Include specific details, numbers, facts from the observations
```

**The Issue:** This is generic and non-actionable. The LLM doesn't know:
- How to synthesize contradictory observations
- How to prioritize multiple observations
- Whether to include technical details or summarize
- Format expectations (bullet points? paragraphs? tables?)

---

## 3. PLAN-EXECUTE PROMPTS - CRITICAL STRUCTURAL ISSUES

**Location:** `backend/config/prompts/plan_execute.py`

### Problem 3.1: Contradictory Instructions

**Lines 59-64:**

```python
IMPORTANT RULES:
- Only include ACTUAL WORK steps (file analysis, data processing, searches, etc.)
- Do NOT include a separate "synthesize results" or "generate final answer" step - this happens automatically
- Each step should produce concrete output/data
- The final work step should naturally lead to a complete answer
```

**The Contradiction:**
- Says "Do NOT include synthesize results step"
- But then says "final work step should naturally lead to complete answer"
- **How can a work step (e.g., "calculate variance") naturally lead to a complete answer without synthesis?**

**Impact:** Plans end with raw data outputs, not complete answers. The ReAct agent then struggles to close the loop.

### Problem 3.2: Success Criteria Examples - Misaligned Format

**Lines 66-79:** The "GOOD vs BAD" examples are helpful BUT:

```python
GOOD Success Criteria (Specific, Measurable):
✓ "Successfully loaded data with column names, types, and shape displayed"
✓ "Mean and median calculated for all numeric columns with values printed"
```

**The Issue:** These use Unicode checkmarks (✓, ✗) which:
1. Don't render consistently across all LLM contexts
2. Are inconsistent with the ASCII-safe policy in python_coder prompts
3. May be parsed incorrectly by JSON extractors

### Problem 3.3: Example Response - No Context Field Clarity

**Lines 81-97:**

```python
Example response format:
[
  {
    "step_num": 1,
    "goal": "Load and explore the uploaded JSON file structure...",
    "primary_tools": ["python_coder"],
    "success_criteria": "Successfully loaded data with structure information...",
    "context": "Use json.load() to read JSON. Show df.head()..."  # PRESENT
  },
  {
    "step_num": 2,
    "goal": "Calculate mean and median of all numeric columns...",
    "primary_tools": ["python_coder"],
    "success_criteria": "Mean and median values calculated...",
    "context": "Use numpy statistical functions"  # PRESENT
  }
]
```

**But the schema at lines 48-55 says:**

```python
{
  "step_num": 1,
  "goal": "Clear, detailed description of what to accomplish",
  "primary_tools": ["tool_name"],
  "success_criteria": "How to verify success",
  "context": "Additional context or specific instructions for this step"  # REQUIRED
}
```

**The Problem:** The prompt **requires** a `context` field but doesn't explain:
- What goes in context vs goal?
- Is context optional or required?
- How detailed should context be?

---

## 4. PYTHON CODER PROMPTS - THE DEEPEST ISSUES

**Location:** `backend/config/prompts/python_coder/`

### Problem 4.1: Prompt Assembly Order - HISTORIES → INPUT → PLANS → REACTS

**Location:** `backend/config/prompts/python_coder/__init__.py:86-115`

**Current Order:**

```python
# 1. Conversation History (if provided)
if conversation_history and len(conversation_history) > 0:
    prompt_parts.append(get_conversation_history_section(conversation_history))

# 2. Base Generation Prompt (user's query)
prompt_parts.append(get_base_generation_prompt(query, context))

# 3. Plan Context (if from Plan-Execute workflow)
if plan_context:
    prompt_parts.append(get_plan_section(plan_context))

# 4. ReAct Context (if from ReAct iterations)
if react_context:
    prompt_parts.append(get_react_section(react_context))

# 5. Task Guidance (based on query type)
prompt_parts.append(get_task_guidance(query))

# 6. File Context (if files provided)
if file_context:
    prompt_parts.append(get_file_context_section(file_context))

# 7. Rules Section
prompt_parts.append(get_rules_section(has_json_files))

# 8. Checklists
prompt_parts.append(get_checklists_section())
```

### Critical Issues with This Order:

#### Issue 4.1a: User Query Buried in Context

- Conversation history comes FIRST (potentially thousands of tokens)
- User's actual query is sandwiched between history and plan context
- **LLMs have recency bias** - they pay more attention to beginning and end
- The most important information (user's question) is in the MIDDLE

#### Issue 4.1b: File Context Too Late

- File metadata appears at position #6
- But the task guidance (#5) may reference files
- The rules (#7) reference "META DATA section above" but it hasn't appeared yet!

#### Issue 4.1c: Critical Information Fragmentation

```
HISTORIES (1000+ tokens)
↓
INPUT (the actual question - buried)
↓
PLANS (200+ tokens)
↓
REACTS (500+ tokens with failed code)
↓
TASK GUIDANCE (references files)
↓
FILE CONTEXT (finally appears)  ← Too late!
↓
RULES (says "use META DATA section above") ← References #6
↓
CHECKLISTS
```

**Recommended Order:**

```python
# 1. USER'S ACTUAL QUESTION (highest priority)
prompt_parts.append(get_base_generation_prompt(query, context))

# 2. FILE CONTEXT (immediate context for the question)
if file_context:
    prompt_parts.append(get_file_context_section(file_context))

# 3. TASK GUIDANCE (now can reference files correctly)
prompt_parts.append(get_task_guidance(query))

# 4. PLAN CONTEXT (strategic direction)
if plan_context:
    prompt_parts.append(get_plan_section(plan_context))

# 5. REACT CONTEXT (failed attempts - learn from mistakes)
if react_context:
    prompt_parts.append(get_react_section(react_context))

# 6. CONVERSATION HISTORY (background context)
if conversation_history and len(conversation_history) > 0:
    prompt_parts.append(get_conversation_history_section(conversation_history))

# 7. RULES (now can correctly reference file context above)
prompt_parts.append(get_rules_section(has_json_files))

# 8. CHECKLISTS (final validation)
prompt_parts.append(get_checklists_section())
```

### Problem 4.2: Prestep vs Normal Mode - MASSIVE Prompt Divergence

**Prestep Prompt:** `generation.py:108-197`
- 90 lines of detailed, specific guidance
- Focused on "FAST PRE-ANALYSIS MODE"
- Heavy emphasis on avoiding sys.argv
- JSON-specific templates
- Clear workflow instructions

**Normal Prompt:** Assembled from 8 different sections
- No cohesive narrative
- File context and rules are separated
- Task guidance is generic
- No mode-specific optimizations

**The Problem:** These are essentially **two different prompting strategies** for the same tool. This creates:
- Inconsistent code quality between prestep and normal modes
- Difficult to maintain (changes must be duplicated)
- Different failure modes (prestep handles JSON better, normal handles plans better)

### Problem 4.3: Rules Section - Duplicate and Weak Guidance

**Location:** `templates.py:26-64`

```python
[RULE 1] EXACT FILENAMES
   - Copy EXACT filename from META DATA section above  ← References section that may not exist yet!
   - [X] NO generic names: 'data.json', 'file.json', 'input.csv'
   - [OK] Example: filename = 'sales_report_Q4_2024.json'

[RULE 2] NO COMMAND-LINE ARGS / USER INPUT
   - Code runs via subprocess WITHOUT arguments
   - [X] NO sys.argv, NO input(), NO argparse
   - [OK] All filenames must be HARDCODED

[RULE 3] USE ACCESS PATTERNS
   - Copy access patterns from META DATA section  ← Again, references section position
   - [X] DON'T guess keys or field names
   - [OK] Use .get() for safe dict access
```

**Issues:**
1. **Positional Dependencies:** Rules reference "META DATA section above" but that section's position varies
2. **Duplicate Guidance:** JSON safety rules appear in both RULE 3 and RULE 4
3. **Weak Enforcement:** Uses markers [X] and [OK] but doesn't explain consequences
4. **No Priority:** All rules seem equal weight, but filename correctness is WAY more critical than .get() usage

### Problem 4.4: Self-Verification Prompt - Overwhelming Checklist

**Location:** `verification.py:114-179`

**The 4-Step Checklist:**

```python
[STEP 1] Task Validation
   ? Question: Does my code directly answer the task described in the prompt above?
   >> Task Reference: {task_ref}
   >> Check: Code produces the requested output (not just partial answer)
   X Reject if: Code does something different or only partially addresses the question

[STEP 2] Filename Validation
   ? Question: Are ALL filenames HARDCODED and EXACT?
   >> Search for: Filenames from file list above (exact string match)
   X Reject if: ANY of these appear:
      - Generic names: 'data.json', 'file.json', 'input.csv', 'data.csv'
      - sys.argv (ANY use, including sys.argv[1], len(sys.argv))
      - input() function (user input)
      - argparse module

[STEP 3] Safety Validation (JSON)
   ? Question: Does the code use safe patterns?
   >> Check JSON access uses:
      - .get() for dict access (NOT data['key'])
      - isinstance() to check data type
      - try/except for json.JSONDecodeError
   X Reject if: Direct dict access data['key'] without .get()

[STEP 4] Template Validation (JSON)
   ? Question: Did I copy the template or write from scratch?
   >> Look for: Complete template structure from file section
   X Reject if: Manually wrote JSON loading instead of using template
```

**Problems:**
1. **Too Many Checks:** LLM must process 4 distinct validation steps WHILE generating code
2. **Conflicting Goals:** "Generate code" vs "Verify code" are different cognitive modes
3. **False Strictness:** Step 3 rejects `data['key']` even when appropriate (e.g., after validation)
4. **Template Dependency:** Step 4 assumes a template exists, but templates aren't always provided

**Impact:** The LLM either:
- Spends too many tokens on verification, reducing code quality
- Ignores verification to focus on code, defeating the purpose
- Gets confused and produces malformed JSON responses

### Problem 4.5: Conversation History Truncation - Information Loss

**Location:** `templates.py:240-246`

```python
def get_conversation_history_section(conversation_history: List[Dict]) -> str:
    for idx, turn in enumerate(conversation_history, 1):
        # ...
        # Truncate very long content
        if len(content) > 500:
            lines.append(f"{content[:500]}...")
            lines.append(f"[Content truncated - {len(content)} chars total]")
        else:
            lines.append(content)
```

**The Problem:** 500 characters is **extremely short** for conversation content:
- A typical Python code response is 1000-3000+ characters
- Previous analysis results may be 2000+ characters
- Truncating at 500 chars loses critical context like:
  - Variable names from previous code
  - DataFrame column names
  - Calculation results
  - Error messages

**Example Impact:**

```
Turn 1 (User): "Analyze sales_data.csv and calculate total revenue by region"
Turn 1 (AI): [2000 char response with code and results]
               → Truncated to first 500 chars
               → Missing: actual revenue calculations, region totals, summary statistics

Turn 2 (User): "Now create a bar chart showing those regional totals"
Turn 2 (AI): [Tries to generate chart code but doesn't know the actual values or variable names]
               → FAILURE
```

### Problem 4.6: ReAct Context - Code Overload

**Location:** `templates.py:166-177`

```python
# Generated Code (if failed)
if 'code' in iteration:
    lines.append("")
    lines.append("Generated Code:")
    lines.append("```python")
    # Show first 30 lines of code
    code_lines = iteration['code'].split('\n')
    for line in code_lines[:30]:
        lines.append(line)
    if len(code_lines) > 30:
        lines.append(f"... [{len(code_lines) - 30} more lines]")
    lines.append("```")
```

**The Problem:** Including 30 lines of FAILED code in each iteration creates:
1. **Token Waste:** Failed code is rarely helpful beyond understanding the error
2. **Cognitive Load:** LLM must parse multiple failed attempts
3. **Pattern Reinforcement:** Seeing failed code may bias the LLM to repeat patterns

**Better Approach:** Include only:
- Error message (full)
- Error location (line number + surrounding 3-5 lines)
- Error classification (syntax, logic, runtime)
- What was attempted (high-level summary, not full code)

---

## 5. CROSS-SYSTEM ISSUES

### Issue 5.1: Emoji Policy Inconsistency

**Python Coder Prompts:** (CORRECT)
- `generation.py:128`: "IMPORTANT: Do NOT use Unicode emojis in your response. Use ASCII-safe markers like [OK], [X], [WARNING], [!!!] instead."
- `verification.py:32`: Same warning
- `fixing.py:31`: Same warning

**Plan-Execute Prompts:** (INCONSISTENT)
- `plan_execute.py:66-79`: Uses Unicode ✓ and ✗ in examples

**ReAct Prompts:** (NO POLICY)
- No mention of emoji restrictions

**Task Classification:** (NO POLICY)
- No mention of emoji restrictions

**Impact:** LLM responses may include emojis that break parsing logic or display incorrectly.

### Issue 5.2: Tool Name Inconsistencies

**Backend Code:**
- `models.py`: Defines `ToolName` enum with values: `web_search`, `rag_retrieval`, `python_coder`, `file_analyzer`, `finish`

**ReAct Prompts:**
- `react_agent.py:40-51`: Lists only 4 tools (missing `file_analyzer`)

**Plan-Execute Prompts:**
- `plan_execute.py:57`: "Valid tool names ONLY: web_search, rag_retrieval, python_coder" (missing `file_analyzer` AND `finish`)

**Impact:** Plans may include invalid tool names, causing execution failures.

### Issue 5.3: No Unified Prompt Registry Usage

**Current State:**
- Task classification: Uses inline prompt (no registry)
- ReAct: Uses registry via `prompts.get_react_thought_and_action_prompt()`
- Plan-Execute: Uses registry via `prompts.get_execution_plan_prompt()`
- Python Coder: Uses registry via `prompts.python_coder.get_python_code_generation_prompt()`

**The Problem:** Some prompts are registered, some aren't. This makes:
- Difficult to audit all prompts
- Impossible to use centralized caching
- Hard to validate prompt parameters
- No visibility into prompt usage patterns

---

## 6. PRIORITY RECOMMENDATIONS

### 🔴 **CRITICAL (Fix Immediately)**

| Priority | Issue | File | Lines | Impact |
|----------|-------|------|-------|--------|
| 1 | Enable Task Classification Examples | `task_classification.py` | 21-79 | HIGH - Poor routing decisions |
| 2 | Fix Python Coder Prompt Order | `python_coder/__init__.py` | 86-115 | HIGH - User query buried |
| 3 | Fix Plan-Execute Contradiction | `plan_execute.py` | 59-64 | HIGH - Incomplete answers |
| 4 | Increase Conversation History Truncation | `templates.py` | 240-246 | HIGH - Context loss |

### 🟡 **HIGH (Fix Soon)**

| Priority | Issue | File | Lines | Impact |
|----------|-------|------|-------|--------|
| 5 | Add Tool Guidance to ReAct | `react_agent.py` | 40-51 | MEDIUM - Tool selection confusion |
| 6 | Reduce Self-Verification Complexity | `verification.py` | 114-179 | MEDIUM - Poor code quality |
| 7 | Reduce ReAct Code in Context | `templates.py` | 166-177 | MEDIUM - Token waste |
| 8 | Unify Emoji Policy | All prompts | Various | MEDIUM - Parsing errors |

### 🟢 **MEDIUM (Improve When Possible)**

| Priority | Issue | File | Lines | Impact |
|----------|-------|------|-------|--------|
| 9 | Strengthen Final Answer Prompt | `react_agent.py` | 78-85 | LOW - Generic responses |
| 10 | Fix Tool Name Lists | `plan_execute.py`, `react_agent.py` | 57, 40-51 | LOW - Validation errors |
| 11 | Register All Prompts | `task_classification.py` | N/A | LOW - Maintainability |
| 12 | Add File Handling Decision Tree | `react_agent.py` | 40-51 | LOW - Clarity |

---

## 7. LONG-TERM ARCHITECTURAL RECOMMENDATIONS

### 1. Unified Prompting Strategy

**Goal:** Create a base template that all prompts extend

**Benefits:**
- Standardize section headers, markers, formatting
- Single source of truth for tool lists, rules, policies
- Easier to maintain and update

**Implementation:**
```python
class BasePromptTemplate:
    """Base template for all system prompts."""

    def __init__(self):
        self.sections = []
        self.tools = self._load_valid_tools()
        self.emoji_policy = "[OK], [X], [WARNING], [!!!]"

    def add_section(self, name: str, content: str, priority: int = 5):
        """Add a section with priority-based ordering."""
        self.sections.append({
            "name": name,
            "content": content,
            "priority": priority
        })

    def render(self) -> str:
        """Render prompt with sections sorted by priority."""
        sorted_sections = sorted(self.sections, key=lambda x: x["priority"])
        return "\n\n".join([s["content"] for s in sorted_sections])
```

### 2. Dynamic Prompt Composition

**Goal:** Build prompts based on runtime context

**Benefits:**
- Only include relevant sections (don't show plan context if not in plan mode)
- Adaptive truncation based on token budget
- Reduced prompt size and cost

**Implementation:**
```python
class DynamicPromptBuilder:
    """Builds prompts dynamically based on context."""

    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.template = BasePromptTemplate()

    def build(self, context: dict) -> str:
        """Build prompt based on context."""
        # Add core sections (always included)
        self._add_core_sections(context)

        # Add conditional sections (based on context)
        if context.get("has_files"):
            self._add_file_sections(context)

        if context.get("plan_mode"):
            self._add_plan_sections(context)

        if context.get("conversation_history"):
            self._add_history_sections(context, budget_remaining)

        # Render and enforce token limit
        return self._render_with_budget()

    def _render_with_budget(self) -> str:
        """Render prompt and truncate if exceeds budget."""
        full_prompt = self.template.render()
        tokens = self._count_tokens(full_prompt)

        if tokens > self.max_tokens:
            # Adaptive truncation: remove lowest priority sections first
            return self._truncate_by_priority(full_prompt)

        return full_prompt
```

### 3. Prompt Testing Framework

**Goal:** Ensure prompts work as intended

**Benefits:**
- Catch breaking changes early
- Validate prompt composition
- Test edge cases

**Implementation:**
```python
class PromptTestSuite:
    """Test suite for prompt validation."""

    def test_task_classification_with_examples(self):
        """Test that all examples in prompt are correctly classified."""
        prompt = get_agent_type_classifier_prompt()

        # Extract examples from prompt
        examples = self._extract_examples(prompt)

        # Test each example
        for example, expected_type in examples:
            result = self._classify(example)
            assert result == expected_type, \
                f"Expected {expected_type}, got {result} for: {example}"

    def test_python_coder_section_order(self):
        """Test that file context appears before rules."""
        prompt = get_python_code_generation_prompt(
            query="Test",
            context=None,
            file_context="FILE METADATA"
        )

        file_pos = prompt.find("FILE METADATA")
        rules_pos = prompt.find("META DATA section above")

        assert file_pos < rules_pos, \
            "File context must appear before rules that reference it"

    def test_prompt_token_budget(self):
        """Test that prompts don't exceed token budget."""
        contexts = self._generate_test_contexts()

        for context in contexts:
            prompt = build_prompt(context)
            tokens = count_tokens(prompt)
            assert tokens <= MAX_TOKENS, \
                f"Prompt exceeds budget: {tokens} > {MAX_TOKENS}"
```

### 4. Prompt Versioning

**Goal:** Version prompts separately from code

**Benefits:**
- A/B test prompt variations
- Roll back prompts independently
- Track prompt performance over time

**Implementation:**
```python
# prompts/v1/task_classification.py
def get_agent_type_classifier_prompt_v1() -> str:
    """Version 1: Simple classifier (current)."""
    return "..."

# prompts/v2/task_classification.py
def get_agent_type_classifier_prompt_v2() -> str:
    """Version 2: Classifier with examples."""
    return "..."

# config/prompt_versions.py
PROMPT_VERSIONS = {
    "task_classification": {
        "current": "v2",
        "available": ["v1", "v2"],
        "rollout": {
            "v1": 0.1,  # 10% of traffic
            "v2": 0.9   # 90% of traffic
        }
    }
}

# Usage
def get_task_classification_prompt(version: str = "current") -> str:
    """Get task classification prompt with version support."""
    if version == "current":
        version = PROMPT_VERSIONS["task_classification"]["current"]

    if version == "v1":
        return get_agent_type_classifier_prompt_v1()
    elif version == "v2":
        return get_agent_type_classifier_prompt_v2()
    else:
        raise ValueError(f"Unknown version: {version}")
```

---

## Appendix A: File Reference Map

| Component | Primary Files | Lines of Interest |
|-----------|--------------|-------------------|
| Task Classification | `backend/config/prompts/task_classification.py` | 14-18 (active), 21-79 (disabled) |
| ReAct Agent | `backend/config/prompts/react_agent.py` | 7-56 (thought-action), 78-86 (final answer) |
| Plan-Execute | `backend/config/prompts/plan_execute.py` | 9-99 (planning prompt) |
| Python Coder | `backend/config/prompts/python_coder/__init__.py` | 86-115 (assembly), 184-230 (self-verification) |
| Python Coder Templates | `backend/config/prompts/python_coder/templates.py` | 9-64 (file/rules), 206-255 (history) |
| Python Coder Generation | `backend/config/prompts/python_coder/generation.py` | 108-197 (prestep), 200-232 (checklists) |
| Python Coder Verification | `backend/config/prompts/python_coder/verification.py` | 9-111 (verification), 114-179 (self-check) |

---

## Appendix B: Quick Fix Checklist

Use this checklist to track fixes:

- [ ] **Task Classification:** Uncomment examples (lines 21-79)
- [ ] **Python Coder:** Reorder prompt sections (user query first)
- [ ] **Plan-Execute:** Clarify synthesis step policy
- [ ] **Conversation History:** Increase truncation limit to 2000 chars
- [ ] **ReAct Tools:** Add guidance for rag_retrieval vs python_coder
- [ ] **Self-Verification:** Simplify 4-step checklist to 2 steps
- [ ] **ReAct Context:** Reduce failed code to error summary only
- [ ] **Emoji Policy:** Apply ASCII-safe markers to all prompts
- [ ] **Tool Names:** Add file_analyzer to all tool lists
- [ ] **Prompt Registry:** Register task classification prompt

---

**End of Report**
