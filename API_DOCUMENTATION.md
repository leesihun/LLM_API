# API Documentation

**Complete API Reference for Agentic AI Backend**

Version: 1.0.0
Base URL: `http://localhost:8000`

---

## Table of Contents

- [Overview](#overview)
- [Authentication](#authentication)
- [OpenAI-Compatible Endpoints](#openai-compatible-endpoints)
- [File Management](#file-management)
- [Health & Status](#health--status)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [SDK Examples](#sdk-examples)
- [Webhooks](#webhooks)

---

## Overview

This API provides OpenAI-compatible endpoints with enhanced agentic capabilities, including:
- Multi-agent reasoning (ReAct and Plan-and-Execute)
- 8+ integrated tools (web search, RAG, math, etc.)
- Conversation memory
- Document upload and retrieval (RAG)
- JWT authentication

### API Characteristics

- **Protocol**: HTTP/1.1, REST
- **Authentication**: JWT Bearer tokens
- **Content-Type**: `application/json` (default), `multipart/form-data` (file upload)
- **Character Encoding**: UTF-8
- **Response Format**: JSON
- **OpenAI Compatibility**: Drop-in replacement for OpenAI API

### Base URLs

| Environment | URL |
|------------|-----|
| Development | `http://localhost:8000` |
| Production | `https://your-domain.com` |

---

## Authentication

All endpoints (except `/` and `/health`) require JWT authentication.

### Authentication Flow

```
1. Login with credentials → Receive JWT token
2. Include token in Authorization header for all requests
3. Token expires after 24 hours (configurable)
4. Refresh by logging in again
```

### Login

Authenticate user and receive JWT access token.

**Endpoint**: `POST /api/auth/login`

**Request Headers**:
```
Content-Type: application/json
```

**Request Body**:
```json
{
  "username": "string",
  "password": "string"
}
```

**Parameters**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `username` | string | Yes | User's username |
| `password` | string | Yes | User's password (plain text, transmitted over HTTPS) |

**Response** (200 OK):
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "username": "admin",
    "role": "admin"
  }
}
```

**Response Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `access_token` | string | JWT token for authentication |
| `token_type` | string | Always "bearer" |
| `user.username` | string | Username |
| `user.role` | string | User role ("admin" or "user") |

**Error Responses**:

| Status Code | Description | Response Body |
|------------|-------------|---------------|
| 401 | Invalid credentials | `{"detail": "Incorrect username or password"}` |
| 422 | Validation error | `{"detail": [...]}` |
| 500 | Server error | `{"error": "Internal server error", "detail": "..."}` |

**Example Request (cURL)**:
```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "administrator"
  }'
```

**Example Request (Python)**:
```python
import requests

response = requests.post(
    "http://localhost:8000/api/auth/login",
    json={
        "username": "admin",
        "password": "administrator"
    }
)

data = response.json()
token = data["access_token"]
print(f"Token: {token}")
```

**Example Request (JavaScript)**:
```javascript
const response = await fetch('http://localhost:8000/api/auth/login', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    username: 'admin',
    password: 'administrator'
  })
});

const data = await response.json();
const token = data.access_token;
console.log('Token:', token);
```

**Default Credentials**:

| Username | Password | Role |
|----------|----------|------|
| `admin` | `administrator` | admin |
| `guest` | `guest_test1` | user |

---

### Get Current User

Retrieve information about the authenticated user.

**Endpoint**: `GET /api/auth/me`

**Request Headers**:
```
Authorization: Bearer <jwt_token>
```

**Response** (200 OK):
```json
{
  "username": "admin",
  "role": "admin"
}
```

**Error Responses**:

| Status Code | Description |
|------------|-------------|
| 401 | Missing or invalid token |
| 403 | Token expired |

**Example Request (cURL)**:
```bash
curl -X GET http://localhost:8000/api/auth/me \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

**Example Request (Python)**:
```python
headers = {"Authorization": f"Bearer {token}"}
response = requests.get("http://localhost:8000/api/auth/me", headers=headers)
user = response.json()
print(f"Logged in as: {user['username']}")
```

---

## OpenAI-Compatible Endpoints

These endpoints follow the OpenAI API specification for drop-in compatibility.

### List Models

List all available LLM models.

**Endpoint**: `GET /v1/models`

**Request Headers**:
```
Authorization: Bearer <jwt_token>
```

**Response** (200 OK):
```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt-oss:20b",
      "object": "model",
      "created": 1730000000,
      "owned_by": "ollama"
    }
  ]
}
```

**Response Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `object` | string | Always "list" |
| `data` | array | Array of model objects |
| `data[].id` | string | Model identifier |
| `data[].object` | string | Always "model" |
| `data[].created` | integer | Unix timestamp |
| `data[].owned_by` | string | Model provider (e.g., "ollama") |

**Example Request (cURL)**:
```bash
curl -X GET http://localhost:8000/v1/models \
  -H "Authorization: Bearer <token>"
```

**Example Request (Python)**:
```python
headers = {"Authorization": f"Bearer {token}"}
response = requests.get("http://localhost:8000/v1/models", headers=headers)
models = response.json()
print(f"Available models: {[m['id'] for m in models['data']]}")
```

**Example Request (OpenAI SDK)**:
```python
from openai import OpenAI

client = OpenAI(
    api_key=token,
    base_url="http://localhost:8000/v1"
)

models = client.models.list()
print([model.id for model in models.data])
```

---

### Chat Completions

Create a chat completion with optional agentic capabilities.

**Endpoint**: `POST /v1/chat/completions`

**Request Headers**:
```
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

**Request Body**:
```json
{
  "model": "gpt-oss:20b",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What is the capital of France?"
    }
  ],
  "session_id": "session-abc123",
  "temperature": 0.7,
  "max_tokens": 1000,
  "stream": false,
  "agent_type": "auto"
}
```

**Parameters**:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | Yes | - | Model ID from `/v1/models` |
| `messages` | array | Yes | - | Array of message objects |
| `messages[].role` | string | Yes | - | One of: "system", "user", "assistant" |
| `messages[].content` | string | Yes | - | Message content |
| `session_id` | string | No | null | Session ID for conversation continuity |
| `temperature` | float | No | 0.7 | Sampling temperature (0.0-1.0) |
| `max_tokens` | integer | No | null | Maximum tokens to generate |
| `stream` | boolean | No | false | Stream response (not yet implemented) |
| `agent_type` | string | No | "auto" | Agent selection: "auto", "react", "plan_execute" |

**Message Roles**:

| Role | Description | Usage |
|------|-------------|-------|
| `system` | System instructions | Sets behavior/personality |
| `user` | User messages | Questions and requests |
| `assistant` | AI responses | Previous assistant replies (for context) |

**Agent Types**:

| Type | Description | Best For |
|------|-------------|----------|
| `auto` | Automatic agent selection | Default, recommended |
| `react` | ReAct agent (iterative reasoning) | Exploratory queries, sequential reasoning |
| `plan_execute` | Plan-and-Execute agent (LangGraph) | Complex batch queries, parallel tools |

**Response** (200 OK):
```json
{
  "id": "chatcmpl-a1b2c3d4",
  "object": "chat.completion",
  "created": 1730000000,
  "model": "gpt-oss:20b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris. It has been the capital since 987 AD and is known for landmarks like the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  },
  "x_session_id": "session-abc123"
}
```

**Response Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique completion ID |
| `object` | string | Always "chat.completion" |
| `created` | integer | Unix timestamp |
| `model` | string | Model used |
| `choices` | array | Array of completion choices (usually 1) |
| `choices[].index` | integer | Choice index (0-based) |
| `choices[].message` | object | Generated message |
| `choices[].message.role` | string | Always "assistant" |
| `choices[].message.content` | string | Generated response |
| `choices[].finish_reason` | string | Stop reason: "stop", "length", "error" |
| `usage` | object | Token usage (always 0 for Ollama) |
| `x_session_id` | string | Session ID (custom field) |

**Error Responses**:

| Status Code | Description |
|------------|-------------|
| 400 | Invalid request (e.g., empty messages) |
| 401 | Unauthorized (missing/invalid token) |
| 500 | Server error (e.g., Ollama unavailable) |

---

#### Chat Completion Examples

##### Example 1: Simple Chat

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss:20b",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```

**Python**:
```python
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    },
    json={
        "model": "gpt-oss:20b",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ]
    }
)

data = response.json()
assistant_message = data["choices"][0]["message"]["content"]
print(assistant_message)
```

**OpenAI SDK**:
```python
from openai import OpenAI

client = OpenAI(
    api_key=token,
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model="gpt-oss:20b",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

print(response.choices[0].message.content)
```

##### Example 2: Agentic Query with Web Search

```json
{
  "model": "gpt-oss:20b",
  "messages": [
    {
      "role": "user",
      "content": "Search for the latest news about AI and summarize the top 3 findings"
    }
  ],
  "agent_type": "auto"
}
```

**Behavior**:
1. System detects "search" keyword → Routes to agentic task
2. Smart agent router selects appropriate agent (likely ReAct)
3. ReAct agent executes web search tool
4. LLM summarizes results
5. Returns comprehensive response

##### Example 3: RAG Query with Document Retrieval

```json
{
  "model": "gpt-oss:20b",
  "messages": [
    {
      "role": "user",
      "content": "Based on the uploaded contract document, what are the termination conditions?"
    }
  ],
  "agent_type": "auto"
}
```

**Behavior**:
1. System detects "document" keyword → Routes to agentic task
2. Agent selects RAG tool
3. Retrieves relevant document chunks
4. LLM answers based on retrieved context

##### Example 4: Multi-Tool Query

```json
{
  "model": "gpt-oss:20b",
  "messages": [
    {
      "role": "user",
      "content": "Search for weather in Seoul AND analyze the uploaded temperature data"
    }
  ],
  "agent_type": "plan_execute"
}
```

**Behavior**:
1. System detects "search" and "analyze" → Routes to agentic task
2. Smart agent selects Plan-and-Execute (multiple tools detected)
3. Agent executes web search and data analysis in parallel
4. LLM synthesizes results from both tools

##### Example 5: Conversation with Memory

**First message**:
```json
{
  "model": "gpt-oss:20b",
  "messages": [
    {"role": "user", "content": "My name is Alice and I live in Paris"}
  ]
}
```

**Response includes** `x_session_id: "session-xyz123"`

**Follow-up message** (using session_id):
```json
{
  "model": "gpt-oss:20b",
  "messages": [
    {"role": "user", "content": "Where do I live?"}
  ],
  "session_id": "session-xyz123"
}
```

**Expected Response**: "You mentioned that you live in Paris."

##### Example 6: Math Calculation

```json
{
  "model": "gpt-oss:20b",
  "messages": [
    {
      "role": "user",
      "content": "Calculate the derivative of x^3 + 2x^2 - 5x + 7"
    }
  ]
}
```

**Behavior**:
1. ReAct agent detects math query
2. Executes math_calculator tool with SymPy
3. Returns: "3x^2 + 4x - 5"

##### Example 7: Python Code Execution

```json
{
  "model": "gpt-oss:20b",
  "messages": [
    {
      "role": "user",
      "content": "Write Python code to calculate factorial of 10 and execute it"
    }
  ]
}
```

**Behavior**:
1. Agent generates Python code
2. Executes python_executor tool
3. Returns result: "3628800"

##### Example 8: Force Specific Agent

```json
{
  "model": "gpt-oss:20b",
  "messages": [
    {"role": "user", "content": "Search for AI news"}
  ],
  "agent_type": "react"
}
```

This forces use of ReAct agent instead of auto-selection.

---

### Streaming (Not Yet Implemented)

Streaming responses are planned but not yet implemented. Setting `"stream": true` will currently return a complete response.

**Planned Behavior**:
```json
{
  "model": "gpt-oss:20b",
  "messages": [...],
  "stream": true
}
```

**Planned Response Format** (Server-Sent Events):
```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1730000000,"model":"gpt-oss:20b","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1730000000,"model":"gpt-oss:20b","choices":[{"index":0,"delta":{"content":"The"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1730000000,"model":"gpt-oss:20b","choices":[{"index":0,"delta":{"content":" capital"},"finish_reason":null}]}

...

data: [DONE]
```

---

## File Management

These endpoints manage document uploads for RAG (Retrieval-Augmented Generation).

### Upload Document

Upload a document for RAG indexing. Documents are user-isolated.

**Endpoint**: `POST /api/files/upload`

**Request Headers**:
```
Authorization: Bearer <jwt_token>
Content-Type: multipart/form-data
```

**Request Body** (multipart/form-data):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Document file (PDF, DOCX, TXT, JSON) |

**Supported Formats**:
- PDF (.pdf)
- Word Document (.docx)
- Text File (.txt)
- JSON (.json)

**File Size Limit**: No explicit limit (recommended: <10MB)

**Response** (200 OK):
```json
{
  "success": true,
  "file_id": "a1b2c3d4",
  "doc_id": "d8a4c890f6ee07d71386dbe1934de12e",
  "filename": "contract.pdf",
  "size": 102400,
  "message": "File uploaded and indexed successfully"
}
```

**Response Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Upload success status |
| `file_id` | string | Short file identifier (8 chars) |
| `doc_id` | string | Vector DB document ID (32 chars) |
| `filename` | string | Original filename |
| `size` | integer | File size in bytes |
| `message` | string | Success message |

**Error Responses**:

| Status Code | Description |
|------------|-------------|
| 400 | Invalid file format |
| 401 | Unauthorized |
| 413 | File too large |
| 500 | Server error (e.g., vector DB failure) |

**Example Request (cURL)**:
```bash
curl -X POST http://localhost:8000/api/files/upload \
  -H "Authorization: Bearer <token>" \
  -F "file=@contract.pdf"
```

**Example Request (Python)**:
```python
files = {"file": open("contract.pdf", "rb")}
headers = {"Authorization": f"Bearer {token}"}

response = requests.post(
    "http://localhost:8000/api/files/upload",
    headers=headers,
    files=files
)

data = response.json()
print(f"Uploaded file ID: {data['file_id']}")
print(f"Document ID: {data['doc_id']}")
```

**Example Request (JavaScript)**:
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:8000/api/files/upload', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`
  },
  body: formData
});

const data = await response.json();
console.log('File ID:', data.file_id);
```

**Processing Pipeline**:
1. File saved to `data/uploads/{username}/{file_id}_{filename}`
2. Document loaded and parsed
3. Text split into chunks (1000 chars, 200 overlap)
4. Embeddings generated (all-MiniLM-L6-v2)
5. Stored in vector DB (FAISS/Chroma) with metadata

**Metadata Stored**:
```json
{
  "user_id": "admin",
  "doc_id": "d8a4c890...",
  "filename": "contract.pdf",
  "file_path": "data/uploads/admin/a1b2c3d4_contract.pdf",
  "upload_time": "2025-10-23T12:00:00Z"
}
```

---

### List Documents

List all uploaded documents for the authenticated user with pagination.

**Endpoint**: `GET /api/files/documents`

**Request Headers**:
```
Authorization: Bearer <jwt_token>
```

**Query Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `page` | integer | No | 1 | Page number (1-based) |
| `page_size` | integer | No | 20 | Items per page (max: 100) |

**Response** (200 OK):
```json
{
  "documents": [
    {
      "file_id": "a1b2c3d4",
      "filename": "contract.pdf",
      "full_path": "a1b2c3d4_contract.pdf",
      "size": 102400,
      "created": 1730000000
    },
    {
      "file_id": "e5f6g7h8",
      "filename": "report.docx",
      "full_path": "e5f6g7h8_report.docx",
      "size": 51200,
      "created": 1729999000
    }
  ],
  "total": 2,
  "page": 1,
  "page_size": 20,
  "total_pages": 1
}
```

**Response Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `documents` | array | Array of document objects |
| `documents[].file_id` | string | Short file identifier |
| `documents[].filename` | string | Original filename |
| `documents[].full_path` | string | Stored filename with ID |
| `documents[].size` | integer | File size in bytes |
| `documents[].created` | float | Unix timestamp |
| `total` | integer | Total number of documents |
| `page` | integer | Current page number |
| `page_size` | integer | Items per page |
| `total_pages` | integer | Total number of pages |

**Error Responses**:

| Status Code | Description |
|------------|-------------|
| 401 | Unauthorized |
| 422 | Invalid query parameters |

**Example Request (cURL)**:
```bash
curl -X GET "http://localhost:8000/api/files/documents?page=1&page_size=10" \
  -H "Authorization: Bearer <token>"
```

**Example Request (Python)**:
```python
headers = {"Authorization": f"Bearer {token}"}
params = {"page": 1, "page_size": 10}

response = requests.get(
    "http://localhost:8000/api/files/documents",
    headers=headers,
    params=params
)

data = response.json()
print(f"Total documents: {data['total']}")
for doc in data['documents']:
    print(f"- {doc['filename']} ({doc['size']} bytes)")
```

**Sorting**:
- Documents are sorted by creation time (newest first)

**Pagination Example**:
```
Total documents: 45
Page size: 20

Page 1: Documents 1-20
Page 2: Documents 21-40
Page 3: Documents 41-45 (5 documents)
```

---

### Delete Document

Delete an uploaded document. User can only delete their own documents.

**Endpoint**: `DELETE /api/files/documents/{file_id}`

**Request Headers**:
```
Authorization: Bearer <jwt_token>
```

**Path Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file_id` | string | Yes | File identifier (8 chars) |

**Response** (200 OK):
```json
{
  "success": true,
  "message": "File deleted successfully"
}
```

**Error Responses**:

| Status Code | Description |
|------------|-------------|
| 401 | Unauthorized |
| 404 | File not found |
| 403 | File belongs to another user |

**Example Request (cURL)**:
```bash
curl -X DELETE http://localhost:8000/api/files/documents/a1b2c3d4 \
  -H "Authorization: Bearer <token>"
```

**Example Request (Python)**:
```python
file_id = "a1b2c3d4"
headers = {"Authorization": f"Bearer {token}"}

response = requests.delete(
    f"http://localhost:8000/api/files/documents/{file_id}",
    headers=headers
)

data = response.json()
print(data['message'])
```

**Note**: This only deletes the file from disk. Vector DB entries remain (future enhancement: cascade delete).

---

## Health & Status

### Root Endpoint

Get API information.

**Endpoint**: `GET /`

**Authentication**: None required

**Response** (200 OK):
```json
{
  "message": "HE Team LLM Assistant API",
  "version": "1.0.0",
  "status": "running"
}
```

**Example Request**:
```bash
curl http://localhost:8000/
```

---

### Health Check

Check API and Ollama connectivity status.

**Endpoint**: `GET /health`

**Authentication**: None required

**Response** (200 OK):
```json
{
  "status": "healthy",
  "ollama_host": "http://127.0.0.1:11434",
  "model": "gpt-oss:20b"
}
```

**Response Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | "healthy" or "unhealthy" |
| `ollama_host` | string | Ollama service URL |
| `model` | string | Configured model name |

**Error Response** (503 Service Unavailable):
```json
{
  "status": "unhealthy",
  "error": "Ollama service unavailable",
  "ollama_host": "http://127.0.0.1:11434"
}
```

**Example Request**:
```bash
curl http://localhost:8000/health
```

**Use Cases**:
- Kubernetes liveness probes
- Monitoring systems
- Load balancer health checks

---

## Error Handling

### Error Response Format

All errors follow a consistent JSON format:

```json
{
  "error": "Error category",
  "detail": "Detailed error message",
  "status_code": 400
}
```

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Successful request |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Missing or invalid authentication |
| 403 | Forbidden | Valid auth but insufficient permissions |
| 404 | Not Found | Resource not found |
| 413 | Payload Too Large | File too large |
| 422 | Unprocessable Entity | Validation error |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server-side error |
| 503 | Service Unavailable | Service temporarily unavailable (e.g., Ollama down) |

### Common Error Scenarios

#### 1. Authentication Errors

**401 Unauthorized** - Missing token:
```json
{
  "detail": "Not authenticated"
}
```

**401 Unauthorized** - Invalid credentials:
```json
{
  "detail": "Incorrect username or password"
}
```

**403 Forbidden** - Token expired:
```json
{
  "detail": "Token has expired"
}
```

#### 2. Validation Errors

**422 Unprocessable Entity** - Missing required field:
```json
{
  "detail": [
    {
      "loc": ["body", "messages"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**422 Unprocessable Entity** - Invalid enum value:
```json
{
  "detail": [
    {
      "loc": ["body", "agent_type"],
      "msg": "value is not a valid enumeration member; permitted: 'auto', 'react', 'plan_execute'",
      "type": "type_error.enum"
    }
  ]
}
```

#### 3. Server Errors

**500 Internal Server Error** - Ollama unavailable:
```json
{
  "error": "Internal server error",
  "detail": "Error generating response: Connection refused to Ollama at http://127.0.0.1:11434"
}
```

**500 Internal Server Error** - Tool execution failure:
```json
{
  "error": "Internal server error",
  "detail": "Error uploading file: Failed to index document"
}
```

### Error Handling Best Practices

**Python Example**:
```python
import requests

try:
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        headers={"Authorization": f"Bearer {token}"},
        json={"model": "gpt-oss:20b", "messages": [...]},
        timeout=60
    )
    response.raise_for_status()  # Raises HTTPError for 4xx/5xx
    data = response.json()
    print(data["choices"][0]["message"]["content"])

except requests.exceptions.HTTPError as e:
    if e.response.status_code == 401:
        print("Authentication failed. Please login again.")
    elif e.response.status_code == 500:
        print(f"Server error: {e.response.json()['detail']}")
    else:
        print(f"HTTP error: {e}")

except requests.exceptions.Timeout:
    print("Request timed out. Try again.")

except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
```

**JavaScript Example**:
```javascript
try {
  const response = await fetch('http://localhost:8000/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: 'gpt-oss:20b',
      messages: [...]
    })
  });

  if (!response.ok) {
    const error = await response.json();
    if (response.status === 401) {
      console.error('Authentication failed');
    } else {
      console.error('Error:', error.detail);
    }
    throw new Error(`HTTP ${response.status}`);
  }

  const data = await response.json();
  console.log(data.choices[0].message.content);

} catch (error) {
  console.error('Request failed:', error);
}
```

---

## Rate Limiting

**Current Status**: Rate limiting is not yet implemented.

**Planned Implementation**: Token bucket algorithm with per-user limits.

**Planned Limits**:
- 100 requests/minute per user
- 1000 requests/day per user
- Admin role: unlimited

**Planned Response Headers**:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1730001234
```

**Planned Error Response** (429 Too Many Requests):
```json
{
  "error": "Rate limit exceeded",
  "detail": "You have exceeded the rate limit of 100 requests per minute",
  "retry_after": 60
}
```

---

## SDK Examples

### Python (OpenAI SDK)

Install:
```bash
pip install openai
```

**Usage**:
```python
from openai import OpenAI

# Initialize client
client = OpenAI(
    api_key="your-jwt-token",
    base_url="http://localhost:8000/v1"
)

# List models
models = client.models.list()
print([model.id for model in models.data])

# Chat completion
response = client.chat.completions.create(
    model="gpt-oss:20b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ],
    temperature=0.7
)

print(response.choices[0].message.content)

# With conversation memory
session_id = None
messages = []

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    messages.append({"role": "user", "content": user_input})

    # Include session_id in extra_body (custom parameter)
    extra = {"session_id": session_id} if session_id else {}

    response = client.chat.completions.create(
        model="gpt-oss:20b",
        messages=messages,
        extra_body=extra
    )

    assistant_message = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_message})

    # Extract session_id from response (custom field)
    if hasattr(response, 'x_session_id'):
        session_id = response.x_session_id

    print(f"Assistant: {assistant_message}")
```

### Python (Requests)

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# Login
def login(username, password):
    response = requests.post(
        f"{BASE_URL}/api/auth/login",
        json={"username": username, "password": password}
    )
    return response.json()["access_token"]

# Chat
def chat(token, message, session_id=None):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-oss:20b",
        "messages": [{"role": "user", "content": message}]
    }

    if session_id:
        data["session_id"] = session_id

    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers=headers,
        json=data
    )

    result = response.json()
    return result["choices"][0]["message"]["content"], result.get("x_session_id")

# Upload file
def upload_file(token, file_path):
    headers = {"Authorization": f"Bearer {token}"}
    files = {"file": open(file_path, "rb")}

    response = requests.post(
        f"{BASE_URL}/api/files/upload",
        headers=headers,
        files=files
    )

    return response.json()

# Example usage
token = login("admin", "administrator")
print(f"Logged in, token: {token[:]}...")

# Simple chat
response, _ = chat(token, "What is the capital of France?")
print(f"Response: {response}")

# Upload document
result = upload_file(token, "contract.pdf")
print(f"Uploaded: {result['filename']}")

# Query document
response, _ = chat(token, "What does the contract say about termination?")
print(f"Response: {response}")
```

### JavaScript (Fetch API)

```javascript
const BASE_URL = 'http://localhost:8000';

// Login
async function login(username, password) {
  const response = await fetch(`${BASE_URL}/api/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password })
  });
  const data = await response.json();
  return data.access_token;
}

// Chat
async function chat(token, message, sessionId = null) {
  const body = {
    model: 'gpt-oss:20b',
    messages: [{ role: 'user', content: message }]
  };

  if (sessionId) {
    body.session_id = sessionId;
  }

  const response = await fetch(`${BASE_URL}/v1/chat/completions`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(body)
  });

  const data = await response.json();
  return {
    message: data.choices[0].message.content,
    sessionId: data.x_session_id
  };
}

// Upload file
async function uploadFile(token, file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${BASE_URL}/api/files/upload`, {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${token}` },
    body: formData
  });

  return await response.json();
}

// Example usage
(async () => {
  const token = await login('admin', 'administrator');
  console.log('Logged in');

  const result = await chat(token, 'What is 2+2?');
  console.log('Response:', result.message);

  // With file input element
  const fileInput = document.getElementById('fileInput');
  const uploadResult = await uploadFile(token, fileInput.files[0]);
  console.log('Uploaded:', uploadResult.filename);
})();
```

### cURL Scripts

**login.sh**:
```bash
#!/bin/bash

TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"administrator"}' \
  | jq -r '.access_token')

echo "Token: $TOKEN"
echo $TOKEN > token.txt
```

**chat.sh**:
```bash
#!/bin/bash

TOKEN=$(cat token.txt)

curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"gpt-oss:20b\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"$1\"}
    ]
  }" | jq -r '.choices[0].message.content'
```

**Usage**:
```bash
./login.sh
./chat.sh "What is the capital of France?"
```

---

## Webhooks

**Current Status**: Webhooks are not yet implemented.

**Planned Features**:
- Event notifications (chat completion, file upload, etc.)
- Custom webhook URLs per user
- Retry logic with exponential backoff
- HMAC signature verification

**Planned Events**:
- `chat.completion.created`
- `file.uploaded`
- `file.deleted`
- `agent.execution.started`
- `agent.execution.completed`

**Planned Webhook Payload**:
```json
{
  "event": "chat.completion.created",
  "timestamp": 1730000000,
  "data": {
    "id": "chatcmpl-abc123",
    "user_id": "admin",
    "session_id": "session-xyz",
    "model": "gpt-oss:20b",
    "message": {
      "role": "assistant",
      "content": "..."
    }
  }
}
```

---

## Additional Resources

- **API Explorer**: http://localhost:8000/docs (Swagger UI)
- **ReDoc**: http://localhost:8000/redoc (Alternative documentation)
- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **README**: [README.md](README.md)
- **Development Guide**: [CLAUDE.md](CLAUDE.md)

---

## API Changelog

### Version 1.0.0 (2025-10-23)

**Initial Release**:
- OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/models`)
- JWT authentication (`/api/auth/login`, `/api/auth/me`)
- File management (`/api/files/upload`, `/api/files/documents`, `/api/files/documents/{file_id}`)
- Health checks (`/`, `/health`)
- Multi-agent system (ReAct, Plan-and-Execute)
- 8+ integrated tools
- Conversation memory
- User isolation

**Known Limitations**:
- No streaming support (planned)
- No rate limiting (planned)
- No webhooks (planned)
- Vector DB does not cascade delete on file deletion

---

## Support & Contact

- **GitHub Issues**: [Report bugs or request features]
- **API Documentation**: http://localhost:8000/docs
- **Email**: support@example.com

---

**API Version**: 1.0.0
**Last Updated**: 2025-10-23
**Maintained by**: HE Team
