# Python Code Generator Tool Implementation Plan

## Overview
Implement a secure, AI-driven Python code generator as a **tool** (not API) with iterative **Verification → Modification** loops. The system will integrate into existing agent workflows (ReAct & Plan-and-Execute) and use subprocess-based isolated execution.

## Architecture Components

### 1. **Python Coder Tool** (`backend/tools/python_coder_tool.py`) ⭐ MAIN ENTRY POINT
- Primary interface for agents to generate, verify, and execute Python code
- Single async method: `execute_code_task(query: str, context: Optional[str]) -> Dict[str, Any]`
- Orchestrates the full workflow internally
- Returns structured output for agents to use

### 2. **Python Code Executor** (`backend/tools/python_executor_engine.py`)
- Low-level subprocess execution engine
- Use `subprocess.run()` with timeout for process isolation
- Temporary execution directory: `data/code_execution/<execution_id>/` (auto-cleaned after execution)
- **Hardcoded security restrictions** (not configurable):
  - No network access (blocks socket imports)
  - File system limited to execution directory only
  - Resource limits from config (timeout, memory)
  - **SAFE_PACKAGES** constant: 40+ whitelisted packages (see Section 9)
  - **BLOCKED_IMPORTS** constant: socket, subprocess, eval, exec, pickle, etc.
  - **SUPPORTED_FILE_TYPES** constant: All allowed file extensions
- Output capture: JSON-formatted stdout/stderr
- File handling: Copies input files to execution directory before running code

### 3. **File Input Handler** (`backend/tools/python_coder_tool.py` - internal)
- Prepares files for code execution
- Uses **SUPPORTED_FILE_TYPES** constant for validation (see Section 9)
- **Supported formats** (15+ types):
  - **PDF** (.pdf) - Extract text/tables using PyPDF2/pdfplumber
  - **Text files** (.txt, .md, .log, .rtf) - Direct read with encoding detection
  - **Office formats**:
    - Excel (.xlsx, .xls, .xlsm) - Load with pandas/openpyxl
    - Word (.docx, .doc) - Extract text with python-docx
  - **Data formats**:
    - CSV/TSV (.csv, .tsv) - Load with pandas
    - JSON (.json) - Load with json module
    - XML (.xml) - Parse with lxml/beautifulsoup4
    - YAML (.yaml, .yml) - Load with PyYAML
  - **Big data formats**:
    - Parquet (.parquet) - Load with pandas/pyarrow
    - HDF5 (.h5, .hdf5) - Load with h5py/pandas
    - NetCDF (.nc) - Scientific data with netCDF4/xarray
    - Feather (.feather) - Fast dataframe with pandas/pyarrow
  - **Images** (.png, .jpg, .gif, .bmp, .tiff) - Load with Pillow/imageio
  - **Compressed** (.zip, .tar, .gz, .bz2) - Extract with zipfile/tarfile
  - **Data files** (.dat) - Binary or text parsing
- **Actions**:
  - Validate file existence, size, type
  - Detect file encoding (chardet for text files)
  - Extract compressed archives automatically
  - Copy files to execution directory
  - Generate file path mapping for code (e.g., "input.xlsx" → "input.xlsx" in exec dir)
  - Provide rich metadata to code generator:
    - CSV/Excel: columns, dtypes, row count, sample data
    - PDF: page count, has_tables, text preview
    - Images: dimensions, format, EXIF data
    - JSON/XML: structure, keys, item count
    - Compressed: contents list
- Returns: `(file_paths: Dict[str, str], file_metadata: Dict[str, Any])`

### 4. **Code Generation Component** (`backend/tools/python_coder_tool.py` - internal)
- LLM-based code generator with structured prompts
- Inputs: User query, context, available packages, input files metadata
- Outputs: Python code + metadata (imports, expected outputs)
- Template-based generation for common tasks
- **File-aware generation**: Automatically includes appropriate file loading code based on file types

### 5. **Code Verification Component** (`backend/tools/python_coder_tool.py` - internal)
- **Static analysis checks**:
  - AST parsing for forbidden operations (exec, eval, file I/O outside execution directory)
  - Import validation using **SAFE_PACKAGES** and **BLOCKED_IMPORTS** constants
  - Syntax verification
- **LLM-based semantic checks**:
  - Does code match user intent?
  - Are there potential runtime errors?
  - Is output format parseable?
- Returns: `(approved: bool, issues: List[str])`

### 6. **Code Modification Component** (`backend/tools/python_coder_tool.py` - internal)
- Receives: Current code + verification issues + user query
- LLM analyzes and fixes issues OR overrides false positives
- **Actions**:
  - Fix legitimate issues (rewrite problematic sections)
  - Override false positives (mark as safe with explanation)
  - Improve code quality (add error handling, better output format)
- Returns: `(modified_code: str, changes: List[str])`

### 7. **Iterative Workflow** (inside `python_coder_tool.py`)
```
Prepare Input Files (if any)
    ↓
Generate Code (with file paths)
    ↓
[Verify → Modify] × N times (max 3 iterations)
    ↓
Execute Python Code
    ↓
Parse Output
    ↓
Return Result
```
- **Iteration loop**: `while not verified and iterations < max_iterations:`
  - Verify current code
  - If issues found → Modify code
  - Increment iteration counter
- After max iterations, either execute (if safe enough) or fail with detailed error
- Each iteration is logged for debugging

### 8. **Configuration** (`backend/config/settings.py`)
```python
# Python Code Execution settings - Simple and clean
PYTHON_CODE_ENABLED: bool = True
PYTHON_CODE_TIMEOUT: int = 30  # seconds
PYTHON_CODE_MAX_MEMORY: int = 512  # MB (future: cgroups)
PYTHON_CODE_EXECUTION_DIR: str = "./data/code_execution"
PYTHON_CODE_MAX_ITERATIONS: int = 3  # Max verification-modification loops
PYTHON_CODE_ALLOW_PARTIAL_EXECUTION: bool = False  # Execute even if minor issues remain
PYTHON_CODE_MAX_FILE_SIZE: int = 50  # MB - Maximum input file size

# Note: Safe packages, supported file types, and blocked imports are
# hardcoded in python_executor_engine.py and python_coder_tool.py
# This keeps configuration clean and prevents misconfiguration
```

### 9. **Hardcoded Security & Support Lists** (`backend/tools/python_executor_engine.py`)
These are **not configurable** to ensure consistent security:

```python
# In python_executor_engine.py

SAFE_PACKAGES = [
    # Standard library
    "json", "csv", "math", "datetime", "collections", "itertools", "re",
    "os", "sys", "zipfile", "tarfile", "gzip", "bz2", "statistics",

    # Data science core
    "numpy", "pandas", "scipy", "sklearn", "scikit-learn", "sympy", "statsmodels",

    # Visualization
    "matplotlib", "seaborn", "plotly",

    # Office formats
    "openpyxl", "xlrd", "xlwt", "python-docx",

    # PDF
    "PyPDF2", "pdfplumber", "pypdf",

    # Big data formats
    "pyarrow", "h5py", "tables", "netCDF4", "xarray",

    # Text/Web
    "lxml", "beautifulsoup4", "chardet",

    # Image
    "Pillow", "imageio",

    # Data validation
    "jsonschema", "cerberus",

    # NLP
    "nltk", "textblob",
]

BLOCKED_IMPORTS = [
    "socket", "subprocess", "os.system", "eval", "exec",
    "__import__", "importlib", "shutil.rmtree", "pickle",
]

SUPPORTED_FILE_TYPES = [
    ".txt", ".md", ".log", ".rtf",  # Text
    ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml",  # Data
    ".xlsx", ".xls", ".xlsm", ".docx", ".doc",  # Office
    ".pdf",  # PDF
    ".dat", ".h5", ".hdf5", ".nc", ".parquet", ".feather",  # Scientific
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".svg",  # Images
    ".zip", ".tar", ".gz", ".bz2", ".7z",  # Compressed
]
```

## Workflow Details

### Phase 0: File Preparation (if input files provided)
```python
# User provides file paths via query context
file_paths = extract_file_paths(query, context)

# Validate files
validated_files = []
for file_path in file_paths:
    if not validate_file(file_path):  # Check size, type, existence
        return error_response(f"Invalid file: {file_path}")
    validated_files.append(file_path)

# Extract metadata for code generation
file_metadata = {}
for file_path in validated_files:
    metadata = extract_metadata(file_path)  # Size, columns (CSV/Excel), pages (PDF), etc.
    file_metadata[file_path] = metadata

# Copy files to execution directory (done later in Phase 3)
```

### Phase 1: Code Generation
```python
code = generate_code(query, context, file_paths=validated_files, file_metadata=file_metadata)
# Returns: Python code with docstring + file loading code
# Example: If CSV provided, includes "df = pd.read_csv('input.csv')"
```

### Phase 2: Iterative Verification & Modification (Max 3 rounds)
```python
for iteration in range(max_iterations):
    # Step 1: Verify
    verified, issues = verify_code(code, query)

    if verified:
        break  # Code is approved

    # Step 2: Modify
    code, changes = modify_code(code, issues, query)
    log_modification(iteration, changes)

    if iteration == max_iterations - 1:
        # Final attempt failed
        if allow_partial_execution and only_minor_issues(issues):
            break  # Execute anyway
        else:
            return error_response(issues)
```

**Iteration Examples**:
- **Iteration 1**: Generate → Verify (syntax error) → Fix syntax
- **Iteration 2**: Verify (unsafe import) → Replace with safe alternative
- **Iteration 3**: Verify (false positive) → Override with explanation

### Phase 3: Execution
```python
execution_id = uuid.uuid4().hex
execution_dir = f"data/code_execution/{execution_id}"
os.makedirs(execution_dir)

# Copy input files to execution directory
file_mapping = {}  # Original path → execution path
for original_path in validated_files:
    filename = os.path.basename(original_path)
    target_path = os.path.join(execution_dir, filename)
    shutil.copy2(original_path, target_path)
    file_mapping[original_path] = filename  # Code uses just filename

# Write code
with open(f"{execution_dir}/script.py", "w") as f:
    f.write(code)

# Execute
result = subprocess.run(
    ["python", f"{execution_dir}/script.py"],
    timeout=30,
    capture_output=True,
    cwd=execution_dir  # Code accesses files by filename only
)

# Cleanup
shutil.rmtree(execution_dir)
```

### Phase 4: Output Parsing & Response
```python
output = parse_output(result.stdout, result.stderr)
response = format_result(output, code, modifications)
return response
```

## Tool Integration

### Add to ReAct Agent (`backend/core/react_agent.py`)
1. Add to `ToolName` enum: `PYTHON_CODER = "python_coder"`
2. Add execution case in `_execute_action()`:
```python
elif action == ToolName.PYTHON_CODER:
    from backend.tools.python_coder_tool import python_coder_tool
    result = await python_coder_tool.execute_code_task(action_input, context)
    return result
```
3. Update `_select_action()` prompt to include Python code generator tool description

### Add to Plan-and-Execute Agent (`backend/core/agent_graph.py`)
1. Create `python_coder_node(state: AgentState)` node function
2. Add to workflow in `create_agent_graph()`: `workflow.add_node("python_coder", python_coder_node)`
3. Update `tool_selection_node()` keyword detection:
```python
python_code_keywords = ["write code", "generate code", "implement", "calculate", "compute", "run python", "create script"]
if any(keyword in plan_lower or keyword in query_lower for keyword in python_code_keywords):
    tools_used.append("python_coder")
```

### Update Smart Agent Router (`backend/tasks/smart_agent_task.py`)
- Add detection for code generation queries
- Route to appropriate agent (ReAct for exploratory, Plan-and-Execute for comprehensive)

## File Handling Details

### Supported File Types & Processing

| File Type | Extensions | Processing Library | Use Case |
|-----------|-----------|-------------------|----------|
| **PDF** | .pdf | PyPDF2, pdfplumber | Text extraction, table extraction |
| **Text** | .txt, .md, .log, .rtf | Built-in open() | Read plain text, markdown, logs |
| **CSV/TSV** | .csv, .tsv | pandas, csv | Data analysis, statistics |
| **Excel** | .xlsx, .xls, .xlsm | pandas, openpyxl, xlrd | Data analysis, multi-sheet processing |
| **Word** | .docx, .doc | python-docx | Document text extraction |
| **JSON** | .json | json, pandas | Data parsing, API responses |
| **XML** | .xml | lxml, beautifulsoup4 | Structured data parsing |
| **YAML** | .yaml, .yml | PyYAML | Configuration files |
| **Parquet** | .parquet | pandas, pyarrow | Big data, columnar storage |
| **HDF5** | .h5, .hdf5 | h5py, pandas | Scientific data, large arrays |
| **NetCDF** | .nc | netCDF4, xarray | Climate/scientific data |
| **Feather** | .feather | pandas, pyarrow | Fast dataframe storage |
| **Images** | .png, .jpg, .gif, .bmp, .tiff | Pillow, imageio | Image analysis, EXIF extraction |
| **Compressed** | .zip, .tar, .gz, .bz2 | zipfile, tarfile, gzip | Archive extraction |
| **Data files** | .dat | Custom parsers | Scientific data, binary formats |

### File Metadata Extraction Examples

```python
# CSV/Excel files
metadata = {
    "type": "csv",
    "size_mb": 2.5,
    "columns": ["name", "age", "salary", "department"],
    "row_count": 1000,
    "sample_data": df.head(3).to_dict()
}

# PDF files
metadata = {
    "type": "pdf",
    "size_mb": 1.2,
    "page_count": 15,
    "has_tables": True,
    "text_preview": "First 200 chars..."
}

# JSON files
metadata = {
    "type": "json",
    "size_mb": 0.5,
    "structure": "array of objects",
    "keys": ["id", "name", "timestamp"],
    "item_count": 500
}
```

### Code Generation with Files

The code generator automatically creates appropriate file loading code:

```python
# Example 1: CSV analysis request
User query: "Calculate average salary by department from the uploaded CSV"
Files: ["data/employees.csv"]

Generated code:
"""
import pandas as pd

# Load data
df = pd.read_csv('employees.csv')

# Calculate average salary by department
result = df.groupby('department')['salary'].mean()

# Output results
print(result.to_json())
"""

# Example 2: PDF text extraction
User query: "Extract all email addresses from this PDF"
Files: ["documents/contract.pdf"]

Generated code:
"""
import PyPDF2
import re

# Load PDF
with open('contract.pdf', 'rb') as f:
    reader = PyPDF2.PdfReader(f)
    text = ''
    for page in reader.pages:
        text += page.extract_text()

# Extract emails
email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
emails = re.findall(email_pattern, text)

# Output results
print(json.dumps({'emails': emails}))
"""

# Example 3: Multi-file processing
User query: "Compare sales data from Q1 and Q2 Excel files"
Files: ["sales_q1.xlsx", "sales_q2.xlsx"]

Generated code:
"""
import pandas as pd

# Load both files
q1 = pd.read_excel('sales_q1.xlsx')
q2 = pd.read_excel('sales_q2.xlsx')

# Calculate totals
q1_total = q1['amount'].sum()
q2_total = q2['amount'].sum()
growth = ((q2_total - q1_total) / q1_total) * 100

# Output comparison
result = {
    'q1_total': q1_total,
    'q2_total': q2_total,
    'growth_percent': growth
}
print(json.dumps(result))
"""
```

## Tool Output Format

```python
{
    "success": bool,
    "code": str,                    # Final executed code
    "output": str,                  # Execution output (stdout)
    "error": Optional[str],         # Execution errors (stderr)
    "iterations": int,              # How many verify-modify loops
    "modifications": List[str],     # What was changed in each iteration
    "execution_time": float,        # Seconds
    "input_files": List[str],       # Files used in execution
    "file_metadata": Dict[str, Any],# File information
    "verification_history": [       # Full audit trail
        {"iteration": 1, "issues": [...], "action": "modified"},
        {"iteration": 2, "issues": [], "action": "approved"}
    ]
}
```

## Security Measures

1. **Process Isolation**: Subprocess with no shell=True
2. **Resource Limits**: Timeout + memory constraints
3. **Filesystem Isolation**: Restricted to temporary execution directory
4. **Import Whitelist**: Only pre-approved packages
5. **Static + Dynamic Analysis**: Multi-layer security checks
6. **Auto-cleanup**: Delete execution directory after completion
7. **Iteration Limits**: Prevent infinite verification loops
8. **Modification Tracking**: Full audit trail

## Error Handling

- **File validation errors**:
  - File not found: Return clear error with file path
  - Unsupported file type: List supported formats
  - File too large: Return size limit (50MB default)
  - File access denied: Return permission error
- **Generation failure**: Return error explaining what went wrong
- **Verification failure after max iterations**: Return detailed issues list
- **Execution timeout**: Kill process, return timeout error
- **Execution crash**: Return stderr + traceback
- **Parse failure**: Return raw output + parsing error
- **File processing errors**: Handle corrupt files, encoding issues gracefully

## File Structure
```
backend/
├── tools/
│   ├── python_coder_tool.py           # Main tool (generation, verification, modification)
│   └── python_executor_engine.py      # Subprocess execution engine
├── core/
│   ├── react_agent.py                 # Add Python coder tool integration
│   └── agent_graph.py                 # Add Python coder tool node
└── config/
    └── settings.py                    # Add Python code execution configuration
```

## Testing Strategy

1. **Unit tests**: Individual components (generate, verify, modify, execute)
2. **Integration tests**: Full workflow with various query types
3. **File handling tests**:
   - Test each supported file type (PDF, CSV, Excel, JSON, etc.)
   - Test multi-file scenarios
   - Test file size limits
   - Test invalid/corrupt files
   - Test file encoding issues (UTF-8, Latin-1, etc.)
4. **Iteration tests**: Multi-round verification-modification cycles
5. **Security tests**: Forbidden operations (should fail safely)
6. **Edge cases**: Syntax errors, timeouts, false positives

## Implementation Steps

### Phase 1: Core Infrastructure
1. Create `python_executor_engine.py` with subprocess isolation
2. Add configuration settings to `settings.py` with all safe packages
3. Implement security checks (AST parsing, import validation)

### Phase 2: File Handling System
4. Implement file handling utilities:
   - File validation (type, size, existence)
   - Encoding detection with chardet
   - Compressed archive extraction
   - Metadata extraction for each file type:
     - CSV/Excel: columns, dtypes, sample data
     - PDF: page count, text preview
     - Images: dimensions, EXIF
     - JSON/XML: structure analysis
     - HDF5/Parquet: dataset info
5. Create file copying and path mapping system

### Phase 3: Code Generation
6. Implement `python_coder_tool.py` with generation/verification/modification:
   - Add file-aware code generation templates
   - Include file metadata in prompts
   - Generate appropriate file loading code for each type
   - Implement verification component
   - Implement modification component with iteration loop

### Phase 4: Package Installation
7. Install required packages (organized by category):

   **Core data science:**
   ```bash
   pip install numpy pandas scipy scikit-learn matplotlib seaborn plotly sympy statsmodels
   ```

   **Office file formats:**
   ```bash
   pip install openpyxl xlrd xlwt python-docx
   ```

   **PDF processing:**
   ```bash
   pip install PyPDF2 pdfplumber pypdf
   ```

   **Data formats:**
   ```bash
   pip install pyarrow h5py tables netCDF4 PyYAML
   ```

   **Text/Web processing:**
   ```bash
   pip install lxml beautifulsoup4 chardet
   ```

   **Image processing:**
   ```bash
   pip install Pillow imageio
   ```

   **NLP (optional):**
   ```bash
   pip install nltk textblob
   ```

   **Data validation:**
   ```bash
   pip install jsonschema cerberus
   ```

### Phase 5: Agent Integration
8. Integrate into ReAct agent (add PYTHON_CODER tool)
9. Integrate into Plan-and-Execute agent (add python_coder_node)
10. Update Smart Agent Router with code generation query detection

### Phase 6: Testing
11. Write unit tests for each component
12. Write file handling tests for all supported formats:
    - PDF, CSV, Excel, JSON, XML, YAML
    - Parquet, HDF5, NetCDF, Feather
    - Images, Word docs, compressed files
13. Write integration tests for full workflow
14. Write security tests (forbidden operations)
15. Test with real queries and various file combinations

### Phase 7: Deployment
16. Update documentation with supported libraries and file types
17. Create example notebooks/scripts for common use cases
18. Deploy and monitor initial usage

## Example Usage

**Agent ReAct Workflow**:
```
User: "Calculate the Fibonacci sequence up to 100 using Python"
↓
ReAct Agent selects PYTHON_CODER tool
↓
Python Coder Tool:
  - Generate: Creates recursive Fibonacci code
  - Verify: Detects potential stack overflow issue
  - Modify: Converts to iterative approach
  - Verify: Approved
  - Execute: Runs Python code in isolated process
  - Parse: Extracts results
↓
Agent receives: {"success": true, "output": "[1, 1, 2, 3, 5, 8, 13, ...]", ...}
↓
Agent formats response: "Here's the Fibonacci sequence up to 100: ..."
```

**Plan-and-Execute Workflow with Files**:
```
User: "Analyze this CSV data and plot the results"
Context: file_paths=["uploads/user123/sales_data.csv"]
↓
Planning Node: Identifies need for code execution with file input
↓
Tool Selection: Selects python_coder
↓
Python Coder Node:
  - Validates sales_data.csv (size, type, columns)
  - Extracts metadata: {columns: ["date", "amount", "product"], rows: 500}
  - Generates pandas analysis + matplotlib plot code
  - Executes code with file copied to execution directory
  - Returns analysis results
↓
Reasoning Node: Explains the analysis results with insights
```

**Example Usage Scenarios**:

1. **CSV Data Analysis**:
```
User: "Calculate total sales by product category from the uploaded CSV"
Files: ["sales_2024.csv"]
→ Tool generates pandas groupby code
→ Returns: {"Electronics": 125000, "Clothing": 89000, "Food": 67000}
```

2. **PDF Text Extraction**:
```
User: "Extract all phone numbers from this contract PDF"
Files: ["contract_v2.pdf"]
→ Tool generates PyPDF2 extraction + regex pattern matching
→ Returns: {"phone_numbers": ["555-1234", "555-5678", "555-9012"]}
```

3. **Excel Multi-Sheet Analysis**:
```
User: "Compare revenue across all sheets in this workbook"
Files: ["quarterly_report.xlsx"]
→ Tool generates openpyxl/pandas code to iterate sheets
→ Returns: {"Q1": 50000, "Q2": 62000, "Q3": 58000, "Q4": 71000}
```

4. **Multi-File Data Merging**:
```
User: "Merge customer data from these three CSV files and find duplicates"
Files: ["customers_2022.csv", "customers_2023.csv", "customers_2024.csv"]
→ Tool generates pandas concat + duplicate detection code
→ Returns: {"total_customers": 5000, "duplicates": 234, "unique": 4766}
```

5. **JSON API Response Analysis**:
```
User: "Parse this JSON file and calculate average response time"
Files: ["api_logs.json"]
→ Tool generates json parsing + statistical analysis
→ Returns: {"avg_response_ms": 156, "max": 3421, "min": 12, "count": 10000}
```

6. **Word Document Text Extraction**:
```
User: "Extract all headings and their content from this Word document"
Files: ["report.docx"]
→ Tool generates python-docx code to iterate paragraphs and styles
→ Returns: {"headings": [{"level": 1, "text": "Introduction", "content": "..."}]}
```

7. **XML Data Parsing**:
```
User: "Parse this XML configuration file and extract all database connection strings"
Files: ["config.xml"]
→ Tool generates lxml parsing code with XPath queries
→ Returns: {"connections": ["server=db1", "server=db2", "server=db3"]}
```

8. **Parquet Big Data Analysis**:
```
User: "Analyze user behavior patterns from this Parquet file"
Files: ["user_events.parquet"]
→ Tool generates pandas code to load Parquet and compute statistics
→ Returns: {"total_events": 5000000, "unique_users": 125000, "top_events": [...]}
```

9. **Image EXIF Extraction**:
```
User: "Extract location and timestamp from these photos"
Files: ["photo1.jpg", "photo2.jpg", "photo3.jpg"]
→ Tool generates Pillow code to read EXIF data
→ Returns: [{"file": "photo1.jpg", "lat": 37.7749, "lon": -122.4194, "date": "2024-01-15"}]
```

10. **Compressed Archive Analysis**:
```
User: "List all CSV files in this ZIP archive and count total rows"
Files: ["data_backup.zip"]
→ Tool generates zipfile extraction + pandas analysis
→ Returns: {"files": ["data1.csv", "data2.csv"], "total_rows": 15000}
```

11. **HDF5 Scientific Data**:
```
User: "Extract temperature measurements from this HDF5 file"
Files: ["climate_data.h5"]
→ Tool generates h5py code to read datasets and attributes
→ Returns: {"temperatures": [array], "units": "Celsius", "time_range": "2020-2023"}
```

12. **Text Encoding Detection**:
```
User: "Read this file with unknown encoding and count word frequency"
Files: ["foreign_text.txt"]
→ Tool generates chardet detection + text analysis
→ Returns: {"encoding": "ISO-8859-1", "word_count": 5000, "top_words": [...]}
```

13. **YAML Configuration Parsing**:
```
User: "Extract all environment variables from this YAML config"
Files: ["docker-compose.yml"]
→ Tool generates PyYAML parsing code
→ Returns: {"env_vars": ["DB_HOST", "API_KEY", "PORT"], "services": 3}
```

14. **Statistical Analysis with Statsmodels**:
```
User: "Perform regression analysis on this sales data"
Files: ["sales_history.csv"]
→ Tool generates pandas + statsmodels OLS regression
→ Returns: {"r_squared": 0.87, "p_value": 0.001, "coefficients": {...}}
```

15. **NLP Text Analysis**:
```
User: "Analyze sentiment and extract key phrases from customer reviews"
Files: ["reviews.csv"]
→ Tool generates textblob/nltk code for sentiment + keyword extraction
→ Returns: {"avg_sentiment": 0.65, "positive": 450, "negative": 50, "keywords": [...]}
```

## Library Dependencies Summary

### Complete Requirements File
Create `requirements_python_coder.txt`:

```txt
# Core data science
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0
sympy>=1.12
statsmodels>=0.14.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0

# Office file formats
openpyxl>=3.1.0          # Excel .xlsx
xlrd>=2.0.1              # Excel .xls (read)
xlwt>=1.3.0              # Excel .xls (write)
python-docx>=0.8.11      # Word documents

# PDF processing
PyPDF2>=3.0.0
pdfplumber>=0.9.0
pypdf>=3.9.0

# Big data formats
pyarrow>=12.0.0          # Parquet, Feather
h5py>=3.8.0              # HDF5
tables>=3.8.0            # PyTables (HDF5)
netCDF4>=1.6.0           # NetCDF scientific data
xarray>=2023.1.0         # Multi-dimensional arrays (works with NetCDF)

# Data formats
PyYAML>=6.0              # YAML
lxml>=4.9.0              # XML parsing
beautifulsoup4>=4.12.0   # HTML/XML parsing
jsonschema>=4.17.0       # JSON validation
cerberus>=1.3.5          # Data validation

# Text processing
chardet>=5.1.0           # Encoding detection

# Image processing
Pillow>=9.5.0            # PIL - Image manipulation
imageio>=2.28.0          # Image I/O

# NLP (optional - can be large)
nltk>=3.8.0              # Natural Language Toolkit
textblob>=0.17.0         # Simple NLP

# Utilities
python-dateutil>=2.8.2   # Date parsing
```

### Library Categories & Use Cases

| Category | Libraries | Total Size | Priority | Use Cases |
|----------|-----------|-----------|----------|-----------|
| **Essential** | numpy, pandas, matplotlib | ~100MB | High | 90% of data tasks |
| **Office** | openpyxl, xlrd, python-docx | ~5MB | High | Business files |
| **PDF** | PyPDF2, pdfplumber | ~10MB | High | Document processing |
| **Data formats** | pyarrow, h5py, PyYAML, lxml | ~50MB | Medium | Big data, config files |
| **Scientific** | scipy, statsmodels, netCDF4 | ~80MB | Medium | Statistical analysis |
| **Visualization** | seaborn, plotly | ~30MB | Medium | Advanced plots |
| **Images** | Pillow, imageio | ~15MB | Medium | Image analysis |
| **Text/NLP** | chardet, beautifulsoup4, nltk | ~20MB | Low | Text processing |
| **Validation** | jsonschema, cerberus | ~1MB | Low | Data validation |

**Total estimated size: ~300-400MB**

### Optional Libraries (Consider Adding)

```txt
# Database support
sqlalchemy>=2.0.0        # SQL toolkit
psycopg2-binary>=2.9.0   # PostgreSQL
pymongo>=4.3.0           # MongoDB

# Additional visualization
bokeh>=3.1.0             # Interactive plots
altair>=5.0.0            # Declarative visualization

# Time series
prophet>=1.1.0           # Time series forecasting
tslearn>=0.5.0           # Time series ML

# Advanced ML (if needed)
xgboost>=1.7.0           # Gradient boosting
lightgbm>=3.3.0          # Light GBM

# Geospatial
geopandas>=0.12.0        # Geographic data
shapely>=2.0.0           # Geometric operations

# Network analysis
networkx>=3.1            # Graph theory

# Audio processing
librosa>=0.10.0          # Audio analysis
soundfile>=0.12.0        # Audio I/O
```

## Future Enhancements

### Core Features
1. **Memory limits**: Use cgroups on Linux for true memory isolation
2. **Package installation**: Allow dynamic pip install during execution
3. **Multi-file execution**: Support importing custom modules
4. **Interactive mode**: REPL-style code execution
5. **Code caching**: Reuse generated code for similar queries
6. **Human-in-the-loop**: Ask for approval before executing potentially risky code

### File Handling Enhancements
7. **Image processing**: Support image analysis with PIL/OpenCV/scikit-image
   - Extract EXIF data
   - Perform image transformations
   - Detect objects/faces (if ML models available)
8. **Advanced PDF**: Table extraction, OCR for scanned documents
9. **Database files**: SQLite, HDF5, Parquet file support
10. **Compressed files**: Automatic extraction of .zip, .tar.gz, .7z files
11. **XML/HTML parsing**: BeautifulSoup for web scraping data files
12. **Large file streaming**: Process files larger than memory using chunking
13. **File output generation**: Allow code to create output files (plots, reports, processed data)
   - Save matplotlib plots as PNG/PDF
   - Generate Excel reports with formatting
   - Create summary CSV files
14. **Binary data formats**: Support for scientific data formats (NetCDF, FITS, etc.)
