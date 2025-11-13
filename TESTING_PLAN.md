# Manual Testing Plan - Backend Refactoring v2.0.0

**Purpose:** Verify all refactored workflows function correctly
**Date:** November 13, 2025
**Branch:** `claude/refactor-backend-comprehensive-011CV5JXQqhtcBzTtvABSemt`

---

## Prerequisites

### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start Ollama
ollama serve

# Pull required models
ollama pull gemma3:12b
ollama pull gpt-oss:20b
ollama pull bge-m3:latest
```

### 2. Start Backend Server
```bash
python run_backend.py
# Server should start at http://0.0.0.0:1007
```

### 3. Verify Health
```bash
curl http://localhost:1007/health
# Expected: {"status": "healthy", "ollama": "connected"}
```

---

## Test Suite

### Test 1: Simple Chat (Non-Agentic)

**Objective:** Verify simple chat responses work without triggering agentic workflow

**Endpoint:** `POST /api/chat`

**Request:**
```json
{
  "message": "Hello, how are you?",
  "session_id": "test-session-1",
  "user_id": "test-user"
}
```

**Expected Behavior:**
- ✅ Task classifier identifies as "chat" (not "agentic")
- ✅ Response generated directly without ReAct agent
- ✅ Response time: 2-5 seconds
- ✅ Response contains friendly greeting

**Success Criteria:**
- Response received within 5 seconds
- No tool execution in logs
- Response is coherent and relevant

**Verification:**
```bash
curl -X POST http://localhost:1007/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, how are you?",
    "session_id": "test-session-1",
    "user_id": "test-user"
  }'
```

---

### Test 2: Web Search (Agentic)

**Objective:** Verify agentic workflow with web search tool

**Endpoint:** `POST /api/chat`

**Request:**
```json
{
  "message": "What's the latest news about artificial intelligence in 2025?",
  "session_id": "test-session-2",
  "user_id": "test-user"
}
```

**Expected Behavior:**
- ✅ Task classifier identifies as "agentic"
- ✅ ReAct agent initialized
- ✅ Web search tool selected and executed
- ✅ Search results retrieved from Tavily
- ✅ Final answer synthesized from search results
- ✅ Response time: 15-30 seconds

**Success Criteria:**
- Agentic workflow triggered
- Web search executed successfully
- Response contains recent information
- Response cites sources or URLs

**Verification:**
```bash
curl -X POST http://localhost:1007/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the latest news about AI?",
    "session_id": "test-session-2",
    "user_id": "test-user"
  }'
```

**Check Logs:**
```bash
tail -f data/logs/app.log | grep -i "web_search"
```

---

### Test 3: Python Code Generation (File Upload)

**Objective:** Verify Python code generation and execution with file upload

**Preparation:**
```bash
# Create sample CSV file
cat > /tmp/sample_data.csv << EOF
name,age,city
Alice,30,New York
Bob,25,San Francisco
Charlie,35,Los Angeles
EOF
```

**Step 1: Upload File**
```bash
curl -X POST http://localhost:1007/api/upload \
  -F "files=@/tmp/sample_data.csv" \
  -H "user-id: test-user" \
  > upload_response.json

# Extract file path from response
FILE_PATH=$(jq -r '.file_paths[0]' upload_response.json)
```

**Step 2: Request Analysis**
```json
{
  "message": "Analyze this CSV file. Show me the average age and count by city.",
  "session_id": "test-session-3",
  "user_id": "test-user",
  "file_paths": ["<FILE_PATH from upload>"]
}
```

**Expected Behavior:**
- ✅ File uploaded successfully
- ✅ Task classifier identifies as "agentic"
- ✅ ReAct agent detects file attachment
- ✅ Python coder tool selected
- ✅ File metadata extracted (columns, dtypes, preview)
- ✅ Python code generated with pandas
- ✅ Code executed successfully
- ✅ Results returned in final answer
- ✅ Response time: 20-40 seconds

**Success Criteria:**
- Code generates without errors
- Execution completes successfully
- Results are accurate (avg age ≈ 30, 3 cities)
- Final answer includes both statistics

**Verification:**
```bash
# Check execution directory
ls -la data/scratch/test-session-3/

# Check logs
tail -f data/logs/app.log | grep -i "python_coder"
```

---

### Test 4: File Analysis (No Code Execution)

**Objective:** Verify file metadata extraction without code generation

**Preparation:**
```bash
# Create sample JSON file
cat > /tmp/config.json << EOF
{
  "app_name": "LLM_API",
  "version": "2.0.0",
  "features": ["chat", "search", "code_gen"],
  "config": {
    "max_tokens": 4096,
    "temperature": 0.3
  }
}
EOF
```

**Step 1: Upload File**
```bash
curl -X POST http://localhost:1007/api/upload \
  -F "files=@/tmp/config.json" \
  -H "user-id: test-user" \
  > upload_response.json

FILE_PATH=$(jq -r '.file_paths[0]' upload_response.json)
```

**Step 2: Request Analysis**
```json
{
  "message": "What's in this JSON file?",
  "session_id": "test-session-4",
  "user_id": "test-user",
  "file_paths": ["<FILE_PATH>"]
}
```

**Expected Behavior:**
- ✅ File analyzer extracts metadata
- ✅ JSON structure analyzed
- ✅ Key-value pairs identified
- ✅ Response describes file contents
- ✅ Response time: 5-10 seconds

**Success Criteria:**
- File metadata extracted correctly
- Response mentions "LLM_API", "version 2.0.0"
- Response lists features
- No code execution triggered

---

### Test 5: RAG Retrieval (Document Search)

**Objective:** Verify RAG retrieval from document store

**Prerequisites:**
- Document store must exist in `data/faiss_index/`
- If not, upload documents first

**Request:**
```json
{
  "message": "Search for information about ReAct agent implementation",
  "session_id": "test-session-5",
  "user_id": "test-user"
}
```

**Expected Behavior:**
- ✅ Task classifier identifies as "agentic"
- ✅ ReAct agent initialized
- ✅ RAG retrieval tool selected
- ✅ Documents retrieved from FAISS
- ✅ Relevant passages returned
- ✅ Final answer synthesized from documents
- ✅ Response time: 10-20 seconds

**Success Criteria:**
- RAG tool executes successfully
- Relevant documents retrieved
- Response contains information from documents
- Sources cited (if available)

**Verification:**
```bash
# Check if FAISS index exists
ls -la data/faiss_index/

# Check logs
tail -f data/logs/app.log | grep -i "rag"
```

---

### Test 6: Plan-Execute Task (Complex Multi-Step)

**Objective:** Verify Plan-Execute workflow for complex tasks

**Request:**
```json
{
  "message": "Create a comprehensive report on climate change with these sections: 1) Current state, 2) Main causes, 3) Projected impacts. Use web search for current data.",
  "session_id": "test-session-6",
  "user_id": "test-user"
}
```

**Expected Behavior:**
- ✅ Task classifier identifies as "agentic"
- ✅ Smart agent detects complex multi-step task
- ✅ Planner creates structured plan with 3+ steps
- ✅ Executor runs each step sequentially
- ✅ Web search executed for each section
- ✅ Results synthesized into final report
- ✅ Response time: 45-90 seconds

**Success Criteria:**
- Plan created with clear steps
- Each step executed successfully
- Multiple web searches performed
- Final report has all 3 sections
- Report is well-structured and coherent

**Verification:**
```bash
# Check logs for plan creation
tail -f data/logs/app.log | grep -i "plan"

# Check logs for multiple tool executions
tail -f data/logs/app.log | grep -i "web_search"
```

---

### Test 7: Error Handling (Invalid File)

**Objective:** Verify graceful error handling

**Request:**
```json
{
  "message": "Analyze this file",
  "session_id": "test-session-7",
  "user_id": "test-user",
  "file_paths": ["/invalid/path/file.csv"]
}
```

**Expected Behavior:**
- ✅ System detects invalid file path
- ✅ Error message returned to user
- ✅ No server crash
- ✅ Graceful degradation

**Success Criteria:**
- Error response with clear message
- Server remains running
- Session not corrupted

---

### Test 8: Conversation History

**Objective:** Verify conversation persistence and retrieval

**Step 1: Send Multiple Messages**
```bash
# Message 1
curl -X POST http://localhost:1007/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, my name is Alice",
    "session_id": "test-session-8",
    "user_id": "test-user"
  }'

# Message 2
curl -X POST http://localhost:1007/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is my name?",
    "session_id": "test-session-8",
    "user_id": "test-user"
  }'
```

**Step 2: Retrieve History**
```bash
curl http://localhost:1007/api/conversations?user_id=test-user
```

**Expected Behavior:**
- ✅ Messages stored in conversation store
- ✅ Session maintains context
- ✅ Second response mentions "Alice"
- ✅ Conversation history retrievable

**Success Criteria:**
- Context preserved across messages
- History API returns all messages
- Responses show contextual awareness

---

### Test 9: Performance Benchmark

**Objective:** Measure response times for various workflows

**Metrics to Collect:**
- Simple chat response time
- Agentic workflow response time
- Code generation time
- File upload + analysis time
- Plan-Execute workflow time

**Benchmark Script:**
```bash
#!/bin/bash

echo "=== Performance Benchmark ==="

# Test 1: Simple Chat
echo "Test 1: Simple Chat"
time curl -X POST http://localhost:1007/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "session_id": "bench-1", "user_id": "bench-user"}' \
  -s -o /dev/null

# Test 2: Web Search
echo "Test 2: Web Search"
time curl -X POST http://localhost:1007/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Latest AI news", "session_id": "bench-2", "user_id": "bench-user"}' \
  -s -o /dev/null

# Add more benchmarks as needed
```

**Target Metrics:**
- Simple chat: < 5s
- Web search: < 30s
- Code generation: < 40s
- Plan-Execute: < 90s

---

## Test Results Template

### Test Execution Log

| Test # | Test Name | Status | Time | Notes |
|--------|-----------|--------|------|-------|
| 1 | Simple Chat | ⏳ | - | - |
| 2 | Web Search | ⏳ | - | - |
| 3 | Python Code Gen | ⏳ | - | - |
| 4 | File Analysis | ⏳ | - | - |
| 5 | RAG Retrieval | ⏳ | - | - |
| 6 | Plan-Execute | ⏳ | - | - |
| 7 | Error Handling | ⏳ | - | - |
| 8 | Conversation History | ⏳ | - | - |
| 9 | Performance Benchmark | ⏳ | - | - |

**Legend:**
- ⏳ Not started
- 🟡 In progress
- ✅ Passed
- ❌ Failed
- ⚠️ Partial success

---

## Issues Discovered

### During Testing

| Issue # | Test | Severity | Description | Status |
|---------|------|----------|-------------|--------|
| - | - | - | - | - |

**Severity Levels:**
- 🔴 Critical: Blocking issue, must fix before merge
- 🟡 High: Important issue, should fix before merge
- 🟢 Medium: Minor issue, can fix later
- 🔵 Low: Enhancement, future improvement

---

## Sign-Off

### Testing Completed By:
- **Name:** _______________
- **Date:** _______________
- **Environment:** _______________

### Approval:
- [ ] All critical tests passed
- [ ] No blocking issues found
- [ ] Performance meets targets
- [ ] Ready for production deployment

---

## Appendix: Troubleshooting

### Common Issues

**Issue 1: Ollama Connection Failed**
```bash
# Check if Ollama is running
curl http://127.0.0.1:11434/api/tags

# Start Ollama if not running
ollama serve
```

**Issue 2: Module Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Issue 3: File Upload Fails**
```bash
# Check permissions on data/uploads/
ls -la data/uploads/
chmod 755 data/uploads/
```

**Issue 4: Code Execution Timeout**
```bash
# Check settings.py
# Increase python_code_timeout if needed
```

**Issue 5: Empty Responses**
```bash
# Check logs
tail -f data/logs/app.log

# Verify task classification
# Check if "agentic" vs "chat" correctly identified
```

---

**Last Updated:** November 13, 2025
**Version:** 2.0.0
**Status:** Ready for Execution
