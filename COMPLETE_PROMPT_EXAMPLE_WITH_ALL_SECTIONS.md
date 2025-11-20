# Complete Prompt Example - All Sections Filled

## Test Scenario

**User's Journey:**
1. User uploads `sales_Q4_2024.json`
2. User asks: "Analyze the sales data and create a visualization"
3. System uses Plan-Execute workflow
4. Step 3 uses ReAct agent
5. ReAct calls Python Coder tool (fails once, retries)

---

## Complete Output (What LLM Actually Sees)

```
================================================================================
                               PAST HISTORIES
================================================================================

=== Turn 1 (User) ===
Time: 2025-11-20 14:30:12
Analyze the sales data and create a visualization

=== Turn 2 (AI) ===
Time: 2025-11-20 14:30:15
I'll analyze the sales data from your file. Let me examine the structure first.

File received: sales_Q4_2024.json
Structure analysis:
- File type: JSON (list of 450 records)
- Fields: date, product, amount, region
- Date range: 2024-10-01 to 2024-12-31
- Regions: North, South, East, West

=== Turn 3 (AI) ===
Time: 2025-11-20 14:30:45
Analysis complete! Here are the key findings:

Total revenue: $1,245,890
Number of transactions: 450

Revenue by region:
- North America: $560,200 (45%)
- South America: $310,500 (25%)
- East Coast: $230,100 (18%)
- West Coast: $145,090 (12%)

Top 3 products:
1. Widget Pro: $450,000
2. Gadget X: $320,500
3. Device Plus: $185,200

=== Turn 4 (User) ===
Time: 2025-11-20 14:32:10
Now create a bar chart showing revenue by region and save it as 'revenue_by_region.png'

================================================================================

================================================================================
                            MY ORIGINAL INPUT PROMPT
================================================================================

Now create a bar chart showing revenue by region and save it as 'revenue_by_region.png'

================================================================================

================================================================================
                                    PLANS
================================================================================

[Plan-Execute Workflow - Step 3 of 4]

Full Plan:
  Step 1: Load and validate sales data file [OK] COMPLETED
    Success criteria: File loads without errors, structure validated
    Primary tools: file_analyzer, python_coder

  Step 2: Analyze sales data and calculate metrics [OK] COMPLETED
    Success criteria: Revenue totals, regional breakdown, top products identified
    Primary tools: python_coder

  Step 3: Create visualization (bar chart) <- CURRENT STEP
    Success criteria: PNG file created with chart, all 4 regions displayed
    Primary tools: python_coder

  Step 4: Verify output and summarize results
    Success criteria: File exists, image displays correctly
    Primary tools: file_analyzer

Previous Steps Results:
  Step 1 -> File loaded successfully: 450 records, 4 fields (date, product, amount, region)
  Step 2 -> Calculated totals: $1.25M total, North: $560k, South: $310k, East: $230k, West: $145k

================================================================================

================================================================================
                                   REACTS
================================================================================

[ReAct Iteration History]

=== Iteration 1 ===
Thought: I need to create a bar chart visualization showing revenue by region. I'll use matplotlib and load the sales data from the JSON file.
Action: python_coder
Tool Input: {
  "query": "Create a bar chart showing revenue by region from sales data",
  "files": ["sales_Q4_2024.json"]
}

Generated Code:
```python
import json
import matplotlib.pyplot as plt

# Load data
with open('data.json', 'r') as f:
    data = json.load(f)

# Calculate revenue by region
regions = {}
for record in data:
    region = record['region']
    amount = record['amount']
    if region in regions:
        regions[region] += amount
    else:
        regions[region] = amount

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(regions.keys(), regions.values())
plt.title('Q4 2024 Revenue by Region')
plt.xlabel('Region')
plt.ylabel('Revenue ($)')
plt.savefig('revenue_by_region.png')
print("Chart saved as revenue_by_region.png")
```

Observation: [ERROR] FileNotFoundError: [Errno 2] No such file or directory: 'data.json'
Error Reason: Code used generic filename 'data.json' instead of exact filename 'sales_Q4_2024.json' from file list

=== Iteration 2 (CURRENT) ===
Awaiting code generation...

================================================================================

================================================================================
                        FINAL TASK FOR LLM AT THIS STAGE
================================================================================

[TASK TYPE] Visualization/Plotting

Current Context:
- This is Plan Step 3: Create bar chart visualization
- ReAct Iteration 2: Retry after FileNotFoundError
- Data already analyzed in Turn 3: 4 regions with revenue amounts known

Your Task NOW:
Generate Python code to create a bar chart with:
  1. X-axis: 4 regions (North, South, East, West)
  2. Y-axis: Revenue amounts ($560k, $310k, $230k, $145k)
  3. Title: "Q4 2024 Revenue by Region"
  4. Save as: 'revenue_by_region.png'

Important Context from Previous Failure:
  [!!!] Iteration 1 FAILED because code used 'data.json' instead of 'sales_Q4_2024.json'
  [!!!] You MUST use the EXACT filename from META DATA section below

Workflow:
  1. Import matplotlib: import matplotlib.pyplot as plt
  2. Load file data using EXACT filename (use access patterns from METADATA below)
  3. Extract revenue by region data
  4. Create bar chart with labels
  5. Save: plt.savefig('revenue_by_region.png')
  6. Print confirmation

================================================================================

================================================================================
                          META DATA (AVAILABLE FILES)
================================================================================


[!!!] CRITICAL - EXACT FILENAMES REQUIRED [!!!]
ALL files are in the current working directory.
YOU MUST use the EXACT filenames shown below - NO generic names like 'file.json' or 'data.csv'!

Available files (USE THESE EXACT NAMES):

1. "sales_Q4_2024.json" - JSON (145KB)
   Structure: list (450 items)
   Each item has keys: date, product, amount, region

   ======================================================================
   [>>>] COPY-PASTE READY: Access Patterns (Pre-Validated)
   ======================================================================
   # These patterns match your JSON structure - copy them directly:

   # Pattern 1: Load all records
   value1 = data
   print(f'Total records: {len(value1)}')

   # Pattern 2: Get first record
   value2 = data[0]
   print(f'First record: {value2}')

   # Pattern 3: Get amount from first record
   value3 = data[0]['amount']
   print(f'First amount: {value3}')

   # Pattern 4: Get region from first record
   value4 = data[0]['region']
   print(f'First region: {value4}')

   # Pattern 5: Iterate through all records
   for record in data:
       region = record.get('region', 'Unknown')
       amount = record.get('amount', 0)
       # Process each record...

   ----------------------------------------------------------------------
   Sample Data (first 2 items):
   [
     {
       "date": "2024-10-01",
       "product": "Widget Pro",
       "amount": 1200,
       "region": "North"
     },
     {
       "date": "2024-10-02",
       "product": "Gadget X",
       "amount": 850,
       "region": "South"
     }
   ]

   [NOTE] Data already analyzed in Turn 3 - revenue by region:
          North: $560,200 | South: $310,500 | East: $230,100 | West: $145,090

================================================================================

================================================================================
                                     RULES
================================================================================

[RULE 1] EXACT FILENAMES
   - Copy EXACT filename from META DATA section above
   - [X] NO generic names: 'data.json', 'file.json', 'input.csv'
   - [OK] MUST USE: 'sales_Q4_2024.json'

   [!!!] CRITICAL: Iteration 1 failed because code used 'data.json'
   [!!!] This attempt MUST use 'sales_Q4_2024.json' exactly

[RULE 2] NO COMMAND-LINE ARGS / USER INPUT
   - Code runs via subprocess WITHOUT arguments
   - [X] NO sys.argv, NO input(), NO argparse
   - [OK] All filenames must be HARDCODED

[RULE 3] USE ACCESS PATTERNS
   - Copy access patterns from META DATA section
   - [X] DON'T guess keys or field names
   - [OK] Use .get() for safe dict access

   [CONFIRMED] Keys from data: date, product, amount, region

[RULE 4] JSON SAFETY
   - Use .get() for dict access: record.get('amount', 0)
   - Check type: isinstance(data, list) - this file is a LIST of dicts
   - Add error handling: try/except json.JSONDecodeError

================================================================================

================================================================================
                                   CHECKLISTS
================================================================================

[1] Task Completion
    ? Does code create a bar chart with 4 regions?
    ? Does code save as 'revenue_by_region.png'?
    ? Does code calculate revenue by region from the data?
    ? Does code have title "Q4 2024 Revenue by Region"?

[2] Filename Validation - CRITICAL (Previous failure here!)
    ? Using 'sales_Q4_2024.json' (exact from META DATA)?
    ? NOT using 'data.json' (this caused Iteration 1 to fail)?
    ? NOT using any other generic name?
    ? NO sys.argv, input(), or argparse?

    [!!!] CHECK TWICE: Previous iteration failed at this exact point!
    [!!!] Search your code for: 'sales_Q4_2024.json' ← Should appear
    [!!!] Search your code for: 'data.json' ← Should NOT appear

[3] Safety & Error Handling
    ? try/except for file operations?
    ? try/except for FileNotFoundError specifically?
    ? .get() for dict access when accessing 'amount' and 'region'?
    ? Check: isinstance(data, list)?

[4] Access Patterns
    ? Using iteration pattern from META DATA section?
    ? Accessing: record.get('amount') and record.get('region')?
    ? NOT making up field names?

[5] Context Awareness (from PAST HISTORIES & PLANS)
    ? Aware of previous failure (Iteration 1 used wrong filename)?
    ? Can use revenue totals from Turn 3 if needed? (optional)
    ? Producing output that satisfies Plan Step 3 success criteria?

[6] Previous Error Avoidance
    ? Reviewed the failed code from Iteration 1?
    ? Identified the specific error: FileNotFoundError on 'data.json'?
    ? Ensured this attempt uses correct filename?

================================================================================

Generate ONLY executable Python code (no markdown, no explanations):
```

---

## 실제 코드 예시 (with all context)

```python
from backend.config.prompts.python_coder import get_python_code_generation_prompt

# Prepare context data
conversation_history = [
    {
        "role": "user",
        "content": "Analyze the sales data and create a visualization",
        "timestamp": "2025-11-20 14:30:12"
    },
    {
        "role": "assistant",
        "content": """I'll analyze the sales data from your file. Let me examine the structure first.

File received: sales_Q4_2024.json
Structure analysis:
- File type: JSON (list of 450 records)
- Fields: date, product, amount, region
- Date range: 2024-10-01 to 2024-12-31
- Regions: North, South, East, West""",
        "timestamp": "2025-11-20 14:30:15"
    },
    {
        "role": "assistant",
        "content": """Analysis complete! Here are the key findings:

Total revenue: $1,245,890
Number of transactions: 450

Revenue by region:
- North America: $560,200 (45%)
- South America: $310,500 (25%)
- East Coast: $230,100 (18%)
- West Coast: $145,090 (12%)

Top 3 products:
1. Widget Pro: $450,000
2. Gadget X: $320,500
3. Device Plus: $185,200""",
        "timestamp": "2025-11-20 14:30:45"
    },
    {
        "role": "user",
        "content": "Now create a bar chart showing revenue by region and save it as 'revenue_by_region.png'",
        "timestamp": "2025-11-20 14:32:10"
    }
]

plan_context = {
    "current_step": 3,
    "total_steps": 4,
    "plan": [
        {
            "step_number": 1,
            "goal": "Load and validate sales data file",
            "status": "completed",
            "success_criteria": "File loads without errors, structure validated",
            "primary_tools": ["file_analyzer", "python_coder"]
        },
        {
            "step_number": 2,
            "goal": "Analyze sales data and calculate metrics",
            "status": "completed",
            "success_criteria": "Revenue totals, regional breakdown, top products identified",
            "primary_tools": ["python_coder"]
        },
        {
            "step_number": 3,
            "goal": "Create visualization (bar chart)",
            "status": "current",
            "success_criteria": "PNG file created with chart, all 4 regions displayed",
            "primary_tools": ["python_coder"]
        },
        {
            "step_number": 4,
            "goal": "Verify output and summarize results",
            "status": "pending",
            "success_criteria": "File exists, image displays correctly",
            "primary_tools": ["file_analyzer"]
        }
    ],
    "previous_results": [
        {
            "step_number": 1,
            "summary": "File loaded successfully: 450 records, 4 fields (date, product, amount, region)"
        },
        {
            "step_number": 2,
            "summary": "Calculated totals: $1.25M total, North: $560k, South: $310k, East: $230k, West: $145k"
        }
    ]
}

react_context = {
    "iteration": 2,
    "history": [
        {
            "thought": "I need to create a bar chart visualization showing revenue by region. I'll use matplotlib and load the sales data from the JSON file.",
            "action": "python_coder",
            "tool_input": {
                "query": "Create a bar chart showing revenue by region from sales data",
                "files": ["sales_Q4_2024.json"]
            },
            "code": """import json
import matplotlib.pyplot as plt

# Load data
with open('data.json', 'r') as f:
    data = json.load(f)

# Calculate revenue by region
regions = {}
for record in data:
    region = record['region']
    amount = record['amount']
    if region in regions:
        regions[region] += amount
    else:
        regions[region] = amount

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(regions.keys(), regions.values())
plt.title('Q4 2024 Revenue by Region')
plt.xlabel('Region')
plt.ylabel('Revenue ($)')
plt.savefig('revenue_by_region.png')
print("Chart saved as revenue_by_region.png")""",
            "observation": "FileNotFoundError: [Errno 2] No such file or directory: 'data.json'",
            "error_reason": "Code used generic filename 'data.json' instead of exact filename 'sales_Q4_2024.json' from file list",
            "status": "error"
        }
    ]
}

file_context = """

[!!!] CRITICAL - EXACT FILENAMES REQUIRED [!!!]
ALL files are in the current working directory.
YOU MUST use the EXACT filenames shown below - NO generic names!

Available files (USE THESE EXACT NAMES):

1. "sales_Q4_2024.json" - JSON (145KB)
   Structure: list (450 items)
   Each item has keys: date, product, amount, region

   ======================================================================
   [>>>] COPY-PASTE READY: Access Patterns (Pre-Validated)
   ======================================================================
   # Pattern 1: Load all records
   value1 = data

   # Pattern 2: Iterate through records
   for record in data:
       region = record.get('region', 'Unknown')
       amount = record.get('amount', 0)

   ----------------------------------------------------------------------
   Sample Data:
   [
     {"date": "2024-10-01", "product": "Widget Pro", "amount": 1200, "region": "North"},
     {"date": "2024-10-02", "product": "Gadget X", "amount": 850, "region": "South"}
   ]
"""

# Generate complete prompt
prompt = get_python_code_generation_prompt(
    query="Now create a bar chart showing revenue by region and save it as 'revenue_by_region.png'",
    context=None,
    file_context=file_context,
    has_json_files=True,
    conversation_history=conversation_history,
    plan_context=plan_context,
    react_context=react_context
)

print(prompt)
```

---

## 핵심 포인트

### 1. MY ORIGINAL INPUT PROMPT
- **실제 사용자가 물어본 것**: "Now create a bar chart..."
- Turn 4의 user input과 동일

### 2. PAST HISTORIES
- 전체 대화 기록 (Turn 1~4)
- 이전 분석 결과 포함

### 3. PLANS
- Plan-Execute의 4단계 중 3단계 진행 중
- 이전 단계 결과 포함

### 4. REACTS - 가장 중요!
- **Iteration 1의 실패한 코드가 그대로 보임**
- **에러 메시지**: `FileNotFoundError: 'data.json'`
- **실패 이유**: `'data.json'` 대신 `'sales_Q4_2024.json'` 사용해야 함

### 5. FINAL TASK
- 현재 상황 요약
- 이전 실패 강조
- 구체적인 할 일

### 6. RULES
- 이전 실패 기록 포함
- "Iteration 1 failed because..."

### 7. CHECKLISTS
- 이전 에러 회피 체크리스트 추가
- "Previous Error Avoidance" 섹션

---

## 이 구조의 장점

1. **LLM이 같은 실수 반복 안 함**
   - 실패한 코드를 직접 보여줌
   - 왜 실패했는지 명확히 설명

2. **문맥 유지**
   - 전체 대화 흐름 파악
   - 이전 분석 결과 재사용 가능

3. **계획 추적**
   - 전체 계획 중 어디인지 알 수 있음
   - 다음 단계 예측 가능

4. **반복 학습**
   - ReAct 기록으로 시행착오 확인
   - 점진적 개선

---

**Implementation Date**: 2025-11-20
**Version**: 4.0 (Complete Context Integration)
**Status**: ✅ Ready for Testing
