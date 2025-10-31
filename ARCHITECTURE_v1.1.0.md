# Architecture Changes - v1.1.0

## Overview
This document explains the major restructuring of the Plan-Execute and ReAct integration implemented in v1.1.0.

## Problem Statement

### Previous Issues
1. **Unstructured Planning**: Plan-Execute generated plans as simple text without structure
2. **Disconnected Execution**: ReAct ran independently without following plan steps explicitly
3. **No Tool Fallback**: If a tool failed, the agent had no mechanism to try alternative tools
4. **Weak Step-Goal Mapping**: Plan steps weren't mapped 1:1 to execution steps

### Example of Old Behavior
```
Plan: "1. Analyze file, 2. Search web, 3. Answer"
ReAct: [Runs free-form for 10 iterations, might or might not follow plan]
```

## New Architecture

### 1. Structured Planning (Plan_execute.py)

Plans are now JSON-structured with explicit details:

```python
[
  {
    "step_num": 1,
    "goal": "Load and analyze the uploaded CSV file",
    "primary_tools": ["python_coder"],
    "fallback_tools": ["python_code", "rag_retrieval"],
    "success_criteria": "Data successfully loaded with statistics",
    "context": "Use pandas to read and describe the data"
  },
  {
    "step_num": 2,
    "goal": "Calculate statistical metrics",
    "primary_tools": ["python_code"],
    "fallback_tools": ["python_coder"],
    "success_criteria": "Mean, median, std displayed",
    "context": "Use numpy or pandas for calculations"
  }
]
```

### 2. Guided Execution (React.py)

ReAct now has two modes:
- **Free Mode** (`execute()`): Original behavior for simple queries
- **Guided Mode** (`execute_with_plan()`): Step-by-step execution for complex queries

#### Guided Mode Flow

```
For each plan step:
  1. Try primary_tools[0]
     - Generate tool input based on step goal
     - Execute tool
     - Verify success against success_criteria
     - If success → next step
  
  2. If failed, try primary_tools[1]
     - Repeat verification
     - If success → next step
  
  3. If all primary tools failed, try fallback_tools
     - Same process
  
  4. If all tools exhausted → mark step as failed, continue to next step
  
After all steps:
  - Synthesize final answer from all step results
```

### 3. Tool Fallback Mechanism

Each step automatically tries multiple tools:

```
Step 1: "Analyze CSV file"
  Attempt 1: python_coder → Success ✓
  
Step 2: "Search for current weather"
  Attempt 1: rag_retrieval → Failed (no relevant docs)
  Attempt 2: web_search → Success ✓
```

### 4. Step Verification

After each tool execution, LLM verifies if the step goal was achieved:

```python
async def _verify_step_success():
    # Check observation against success_criteria
    # Use heuristics for obvious cases
    # Use LLM for complex verification
    return True/False
```

## Data Flow

### Old Architecture
```
User Query
  ↓
Plan-Execute creates text plan
  ↓
ReAct reads plan but runs freely
  ↓
Answer (plan may or may not be followed)
```

### New Architecture
```
User Query
  ↓
Plan-Execute creates structured plan [Step 1, Step 2, ...]
  ↓
ReAct Guided Mode:
  ├─ Execute Step 1
  │  ├─ Try primary_tools
  │  ├─ Try fallback_tools
  │  └─ Verify success → StepResult
  ├─ Execute Step 2
  │  └─ (same process)
  └─ ...
  ↓
Generate final answer from all StepResults
```

## Key Benefits

### 1. Explicit Step-by-Step Execution
- Each plan step is executed as a discrete unit
- Clear goal and success criteria per step
- Steps are independent and traceable

### 2. Automatic Tool Fallback
- If primary tool fails, alternatives are tried automatically
- No manual intervention needed
- Robust handling of tool failures

### 3. Better Debugging & Monitoring
- Each step has detailed execution metadata
- Success/failure tracked per step
- Clear visibility into which tools were used

### 4. Separation of Concerns
- **Planning**: Strategic thinking about what needs to be done
- **Execution**: Tactical execution of each step
- **Verification**: Checking if goals are met

### 5. Partial Success Handling
- If Step 2 fails, Steps 3-5 still execute
- Final answer synthesizes from all available results
- Graceful degradation instead of total failure

## Example Execution

### User Query
"Analyze the uploaded sales.csv file and tell me the average revenue"

### Phase 1: Planning
```json
[
  {
    "step_num": 1,
    "goal": "Load sales.csv and display basic info",
    "primary_tools": ["python_coder"],
    "fallback_tools": ["python_code"],
    "success_criteria": "CSV loaded, columns and shape displayed"
  },
  {
    "step_num": 2,
    "goal": "Calculate average revenue",
    "primary_tools": ["python_code"],
    "fallback_tools": ["python_coder"],
    "success_criteria": "Average revenue number displayed"
  }
]
```

### Phase 2: Execution

**Step 1**
- Attempt 1: python_coder
  - Generates pandas code to load CSV
  - Executes successfully
  - Verification: ✓ SUCCESS
  - Observation: "Loaded 1000 rows, 5 columns..."

**Step 2**  
- Attempt 1: python_code
  - Calculates mean of 'revenue' column
  - Executes successfully
  - Verification: ✓ SUCCESS
  - Observation: "Average revenue: $45,230"

### Phase 3: Final Answer
"Based on the analysis of sales.csv:
- The file contains 1000 rows with 5 columns
- The average revenue is $45,230"

## File Changes Summary

### backend/models/schemas.py
- Added `PlanStep` schema
- Added `StepResult` schema

### backend/tasks/Plan_execute.py
- Restructured `_create_execution_plan()` to return `List[PlanStep]`
- Updated `execute()` to call `react_agent.execute_with_plan()`
- Added `_build_metadata_from_steps()` for detailed metadata

### backend/tasks/React.py
- Added `execute_with_plan()` - main guided mode entry point
- Added `_execute_step()` - execute single step with fallback
- Added `_execute_tool_for_step()` - run specific tool for step
- Added `_generate_action_input_for_step()` - create tool input
- Added `_verify_step_success()` - verify goal achievement
- Added `_generate_final_answer_from_steps()` - synthesize final answer

## Migration Notes

### Backward Compatibility
- Original ReAct `execute()` method remains unchanged
- Free-mode execution still available for simple queries
- Only Plan-Execute workflow uses new guided mode

### When Each Mode is Used
- **Free Mode**: Direct ReAct calls, simple queries
- **Guided Mode**: Plan-Execute calls with structured plans

## Testing Recommendations

1. Test with file uploads (CSV, Excel, JSON)
2. Test with multi-step queries requiring different tools
3. Test tool failure scenarios (ensure fallback works)
4. Test partial failures (some steps succeed, others fail)
5. Verify metadata includes step-by-step details

## Future Enhancements

Potential improvements for future versions:

1. **Dynamic Plan Adjustment**: Allow replanning if multiple steps fail
2. **Parallel Step Execution**: Execute independent steps in parallel
3. **Step Dependencies**: Define which steps depend on others
4. **Tool Learning**: Track which tools work best for which goals
5. **User Feedback Loop**: Allow users to approve/modify plans before execution

---

**Version**: 1.1.0  
**Date**: 2025-10-31  
**Authors**: Implementation based on user requirements for better Plan-Execute and ReAct integration

