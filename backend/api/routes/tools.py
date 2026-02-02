"""
Tools API endpoints
Provides API access to web search, Python execution, and RAG tools
"""
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from pydantic import BaseModel

from backend.utils.auth import get_optional_user, get_current_user
from backend.core.llm_backend import llm_backend
from tools.web_search import WebSearchTool
from tools.python_coder import PythonCoderTool
from tools.rag import RAGTool
import config


router = APIRouter(prefix="/api/tools", tags=["tools"])


# ============================================================================
# Request/Response Schemas
# ============================================================================

class ToolContext(BaseModel):
    """Context information passed to tools"""
    chat_history: Optional[List[Dict[str, str]]] = None
    react_scratchpad: Optional[str] = None
    current_thought: Optional[str] = None
    current_action: Optional[str] = None
    user_query: Optional[str] = None
    plan: Optional[Dict[str, Any]] = None
    plan_history: Optional[List[Dict[str, Any]]] = None
    session_id: Optional[str] = None
    username: Optional[str] = None


class ToolResponse(BaseModel):
    """Standard tool response format"""
    success: bool
    answer: str  # Human-readable answer for agent observation
    data: Dict[str, Any]  # Structured data
    metadata: Dict[str, Any]  # Execution metadata
    error: Optional[str] = None


class WebSearchRequest(BaseModel):
    """Web search request"""
    query: str
    max_results: Optional[int] = None
    context: Optional[ToolContext] = None


class PythonCoderRequest(BaseModel):
    """Python code execution request"""
    code: str
    session_id: str
    timeout: Optional[int] = None
    context: Optional[ToolContext] = None


class RAGQueryRequest(BaseModel):
    """RAG retrieval request"""
    query: str
    collection_name: str
    max_results: Optional[int] = None
    context: Optional[ToolContext] = None


class RAGCollectionRequest(BaseModel):
    """RAG collection management"""
    collection_name: str


class RAGUploadRequest(BaseModel):
    """RAG document upload request"""
    collection_name: str


# ============================================================================
# Helper Functions
# ============================================================================

def load_prompt(prompt_name: str, **kwargs) -> str:
    """Load and format prompt template"""
    prompt_path = config.PROMPTS_DIR / "tools" / prompt_name

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")

    with open(prompt_path, 'r', encoding='utf-8') as f:
        template = f.read()

    return template.format(**kwargs)


def format_context(context: Optional[ToolContext]) -> str:
    """Format context for LLM prompts"""
    if not context:
        return "No additional context provided."

    parts = []

    if context.user_query:
        parts.append(f"User Query: {context.user_query}")

    if context.chat_history:
        history_str = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in context.chat_history[-5:]  # Last 5 messages
        ])
        parts.append(f"Recent Chat History:\n{history_str}")

    if context.current_thought:
        parts.append(f"Current Thought: {context.current_thought}")

    if context.react_scratchpad:
        parts.append(f"ReAct History:\n{context.react_scratchpad[-500:]}")  # Last 500 chars

    if context.plan:
        parts.append(f"Current Plan: {context.plan}")

    return "\n\n".join(parts) if parts else "No context provided."


# ============================================================================
# Tool Endpoints
# ============================================================================

@router.get("/list")
def list_tools(current_user: Optional[dict] = Depends(get_optional_user)):
    """
    List all available tools

    Returns:
        List of tool metadata
    """
    tools = [
        {
            "name": "websearch",
            "description": "Search the web for current information",
            "enabled": True
        },
        {
            "name": "python_coder",
            "description": "Execute Python code in sandboxed environment",
            "enabled": True
        },
        {
            "name": "rag",
            "description": "Retrieve information from document collections",
            "enabled": True
        }
    ]

    return {"tools": tools}


@router.post("/websearch", response_model=ToolResponse)
async def websearch(
    request: WebSearchRequest,
    current_user: Optional[dict] = Depends(get_optional_user)
):
    """
    Pure web search - no LLM processing
    Returns raw Tavily results for agent to interpret

    Args:
        request: Search request with query
        current_user: Authenticated user (optional)

    Returns:
        Raw search results in data field, no answer field
    """
    print("\n" + "=" * 80)
    print("[TOOLS API] /api/tools/websearch endpoint called")
    print("=" * 80)
    print(f"User: {current_user['username'] if current_user else 'guest'}")
    print(f"Query: {request.query}")
    print(f"Max results: {request.max_results or 'default'}")

    start_time = time.time()

    try:
        # Perform search using tool (no LLM calls)
        tool = WebSearchTool()
        search_result = tool.search(
            query=request.query,
            max_results=request.max_results
        )

        if not search_result["success"]:
            return ToolResponse(
                success=False,
                answer="",
                data={},
                metadata={"execution_time": time.time() - start_time},
                error=search_result.get("error", "Unknown error")
            )

        execution_time = time.time() - start_time

        # Return raw results without any LLM processing
        return ToolResponse(
            success=True,
            answer="",  # No answer - agent will interpret raw data
            data={
                "query": request.query,
                "results": search_result["results"],
                "num_results": search_result["num_results"]
            },
            metadata={
                "execution_time": execution_time
            }
        )

    except Exception as e:
        return ToolResponse(
            success=False,
            answer="",
            data={},
            metadata={"execution_time": time.time() - start_time},
            error=str(e)
        )


@router.post("/python_coder", response_model=ToolResponse)
async def python_coder(
    request: PythonCoderRequest,
    current_user: Optional[dict] = Depends(get_optional_user)
):
    """
    Execute Python code in sandboxed environment
    
    NEVER raises HTTP exceptions - always returns ToolResponse with error details
    Even on failure, stdout/stderr are included in the response

    Args:
        request: Code execution request
        current_user: Authenticated user (optional)

    Returns:
        Execution results with stdout/stderr always included
    """
    print("\n" + "=" * 80)
    print("[TOOLS API] /api/tools/python_coder endpoint called")
    print("=" * 80)
    print(f"User: {current_user['username'] if current_user else 'guest'}")
    print(f"Session ID: {request.session_id}")
    print(f"Code length: {len(request.code)} chars")
    print(f"Timeout: {request.timeout or 'default'}s")
    print(f"Context provided: {bool(request.context)}")
    
    start_time = time.time()
    tool = None
    result = None

    try:
        # Initialize tool with session ID (factory automatically selects backend)
        print(f"\n[TOOLS API] Initializing PythonCoderTool...")
        print(f"[TOOLS API] Mode: {config.PYTHON_EXECUTOR_MODE}")
        tool = PythonCoderTool(session_id=request.session_id)
        print(f"[TOOLS API] [OK] Tool initialized ({type(tool).__name__})")

    except Exception as e:
        # Tool initialization failed - return immediately
        error_msg = f"Failed to initialize Python executor: {str(e)}"
        print(f"[TOOLS API] ERROR: {error_msg}")
        execution_time = time.time() - start_time
        
        return ToolResponse(
            success=False,
            answer=f"Tool initialization error: {str(e)}",
            data={
                "stdout": "",
                "stderr": error_msg,
                "files": {},
                "workspace": "",
                "returncode": -1
            },
            metadata={"execution_time": execution_time},
            error=str(e)
        )

    try:
        # Execute code - this should handle all execution errors internally
        print(f"\n[TOOLS API] Calling tool.execute()...")
        result = tool.execute(
            code=request.code,
            timeout=request.timeout,
            context=request.context.dict() if request.context else None
        )
        print(f"[TOOLS API] [OK] Execution completed: {'SUCCESS' if result['success'] else 'FAILED'}")

        # Format human-readable answer
        if result["success"]:
            # Success - show output and files
            answer = "Code executed successfully."
            if result['stdout']:
                answer += f"\n\nOutput:\n{result['stdout']}"
            else:
                answer += "\n\n(No output)"
                
            if result['files']:
                file_list = ", ".join(result['files'].keys())
                answer += f"\n\nFiles in workspace: {file_list}"
        else:
            # Failure - show both stdout and stderr
            answer = "Code execution failed."
            
            if result['stdout']:
                answer += f"\n\nOutput (before error):\n{result['stdout']}"
                
            if result['stderr']:
                answer += f"\n\nError:\n{result['stderr']}"
            else:
                answer += "\n\n(No error message)"

        execution_time = time.time() - start_time

        return ToolResponse(
            success=result["success"],
            answer=answer,
            data={
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "files": result["files"],
                "workspace": result["workspace"],
                "returncode": result["returncode"]
            },
            metadata={
                "execution_time": execution_time,
                "code_execution_time": result["execution_time"]
            },
            error=result.get("error")
        )

    except Exception as e:
        # Unexpected error during execution or response formatting
        # This should rarely happen since tool.execute() handles its own errors
        error_msg = f"Unexpected error in Python executor: {str(e)}"
        print(f"[TOOLS API] CRITICAL ERROR: {error_msg}")
        execution_time = time.time() - start_time
        
        # Try to extract any partial results from the tool if available
        stdout = ""
        stderr = error_msg
        files = {}
        workspace = str(tool.workspace) if tool else ""
        
        # If we got a partial result before the error, include it
        if result and isinstance(result, dict):
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "") + f"\n\nAdditional error: {error_msg}"
            files = result.get("files", {})
            workspace = result.get("workspace", workspace)
        
        return ToolResponse(
            success=False,
            answer=f"Unexpected execution error: {str(e)}",
            data={
                "stdout": stdout,
                "stderr": stderr,
                "files": files,
                "workspace": workspace,
                "returncode": -1
            },
            metadata={"execution_time": execution_time},
            error=str(e)
        )


@router.get("/python_coder/files/{session_id}")
async def list_python_files(
    session_id: str,
    current_user: Optional[dict] = Depends(get_optional_user)
):
    """
    List files in Python coder workspace
    
    Returns success response even on errors (no HTTP exceptions)
    """
    try:
        tool = PythonCoderTool(session_id=session_id)
        files = tool.list_files()
        return {"success": True, "files": files, "error": None}
    except Exception as e:
        print(f"[TOOLS API] ERROR listing Python files: {str(e)}")
        return {"success": False, "files": [], "error": str(e)}


@router.get("/python_coder/files/{session_id}/{filename}")
async def read_python_file(
    session_id: str,
    filename: str,
    current_user: Optional[dict] = Depends(get_optional_user)
):
    """
    Read a file from Python coder workspace
    
    Returns success response even on errors (no HTTP exceptions)
    """
    try:
        tool = PythonCoderTool(session_id=session_id)
        content = tool.read_file(filename)

        if content is None:
            return {
                "success": False,
                "filename": filename,
                "content": "",
                "error": f"File '{filename}' not found in workspace"
            }

        return {
            "success": True,
            "filename": filename,
            "content": content,
            "error": None
        }
    except Exception as e:
        print(f"[TOOLS API] ERROR reading Python file '{filename}': {str(e)}")
        return {
            "success": False,
            "filename": filename,
            "content": "",
            "error": str(e)
        }


@router.post("/rag/collections", response_model=ToolResponse)
async def create_rag_collection(
    request: RAGCollectionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create a new RAG collection (requires authentication)"""
    username = current_user["username"]

    try:
        tool = RAGTool(username=username)
        result = tool.create_collection(request.collection_name)

        if result["success"]:
            answer = f"Collection '{request.collection_name}' created successfully."
        else:
            answer = f"Failed to create collection: {result.get('error', 'Unknown error')}"

        return ToolResponse(
            success=result["success"],
            answer=answer,
            data=result,
            metadata={}
        )
    except Exception as e:
        return ToolResponse(
            success=False,
            answer=f"Error creating collection: {str(e)}",
            data={},
            metadata={},
            error=str(e)
        )


@router.get("/rag/collections")
async def list_rag_collections(
    current_user: dict = Depends(get_current_user)
):
    """List all RAG collections for user (requires authentication)"""
    username = current_user["username"]

    try:
        tool = RAGTool(username=username)
        result = tool.list_collections()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/rag/collections/{collection_name}")
async def delete_rag_collection(
    collection_name: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a RAG collection (requires authentication)"""
    username = current_user["username"]

    try:
        tool = RAGTool(username=username)
        result = tool.delete_collection(collection_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/upload")
async def upload_to_rag(
    collection_name: str = Form(...),
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload document to RAG collection (requires authentication)

    Supports text files (txt, md, json, csv) and binary files (pdf, docx).
    Binary files are saved to temp location for processing.
    """
    import tempfile
    import os

    username = current_user["username"]

    print(f"\n[RAG UPLOAD] Uploading file: {file.filename}")
    print(f"[RAG UPLOAD] Collection: {collection_name}")
    print(f"[RAG UPLOAD] User: {username}")

    try:
        tool = RAGTool(username=username)

        # Read file content
        content = await file.read()

        # Get file extension
        file_ext = Path(file.filename).suffix.lower()

        # Binary formats that need file-based processing
        binary_formats = ['.pdf', '.docx']

        if file_ext in binary_formats:
            # Save to temp file for binary processing
            print(f"[RAG UPLOAD] Binary format detected ({file_ext}), saving to temp file")

            # Create temp file with proper extension
            with tempfile.NamedTemporaryFile(
                mode='wb',
                suffix=file_ext,
                delete=False
            ) as tmp_file:
                tmp_file.write(content)
                tmp_path = tmp_file.name

            print(f"[RAG UPLOAD] Temp file created: {tmp_path}")

            try:
                # Upload from file path (RAGTool will read the binary file)
                result = tool.upload_document(
                    collection_name=collection_name,
                    document_path=tmp_path,
                    document_content=None,  # Let tool read from file
                    document_name=file.filename  # Use original filename
                )
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                    print(f"[RAG UPLOAD] Temp file cleaned up")
                except Exception as cleanup_error:
                    print(f"[RAG UPLOAD] Warning: Failed to clean up temp file: {cleanup_error}")
        else:
            # Text-based formats can be passed directly
            print(f"[RAG UPLOAD] Text format detected ({file_ext}), decoding as UTF-8")
            try:
                content_str = content.decode('utf-8')
            except UnicodeDecodeError:
                # Try with latin-1 as fallback
                content_str = content.decode('latin-1')

            # Upload with content string
            result = tool.upload_document(
                collection_name=collection_name,
                document_path=file.filename,
                document_content=content_str
            )

        print(f"[RAG UPLOAD] Upload result: {result}")
        return result
    except Exception as e:
        print(f"[RAG UPLOAD] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag/collections/{collection_name}/documents")
async def list_rag_documents(
    collection_name: str,
    current_user: dict = Depends(get_current_user)
):
    """List all documents in a RAG collection (requires authentication)"""
    username = current_user["username"]

    try:
        tool = RAGTool(username=username)
        result = tool.list_documents(collection_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/rag/collections/{collection_name}/documents/{document_id}")
async def delete_rag_document(
    collection_name: str,
    document_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a specific document from a RAG collection (requires authentication)"""
    username = current_user["username"]

    try:
        tool = RAGTool(username=username)
        result = tool.delete_document(collection_name, document_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/query", response_model=ToolResponse)
async def query_rag(
    request: RAGQueryRequest,
    current_user: Optional[dict] = Depends(get_optional_user)
):
    """Query RAG collection with LLM-enhanced retrieval and synthesis"""
    username = current_user["username"] if current_user else "guest"
    
    print("\n" + "=" * 80)
    print("[TOOLS API] /api/tools/rag/query endpoint called")
    print("=" * 80)
    print(f"User: {username}")
    print(f"Collection: {request.collection_name}")
    print(f"Query: {request.query}")
    print(f"Max results: {request.max_results or 'default'}")
    print(f"Context provided: {bool(request.context)}")
    
    start_time = time.time()

    try:
        # Step 1: Optimize query using LLM
        print(f"\n[TOOLS API] Step 1: Formatting context...")
        context_str = format_context(request.context)
        print(f"[TOOLS API] Context formatted ({len(context_str)} chars)")

        print(f"\n[TOOLS API] Step 2: Loading query optimization prompt...")
        query_prompt = load_prompt(
            "rag_query.txt",
            user_query=request.query,
            context=context_str
        )
        print(f"[TOOLS API] Prompt loaded ({len(query_prompt)} chars)")

        print(f"\n[TOOLS API] Step 3: Calling LLM for query optimization...")
        messages = [{"role": "user", "content": query_prompt}]
        optimized_query = llm_backend.chat(
            messages,
            config.TOOL_MODELS.get("rag", config.OLLAMA_MODEL),
            0.3
        ).strip()
        print(f"[TOOLS API] [OK] Optimized query: '{optimized_query}'")

        # Step 2: Retrieve documents
        print(f"\n[TOOLS API] Step 4: Initializing RAGTool...")
        tool = RAGTool(username=username)
        print(f"[TOOLS API] Calling tool.retrieve()...")
        retrieval_result = tool.retrieve(
            collection_name=request.collection_name,
            query=optimized_query,
            max_results=request.max_results
        )
        print(f"[TOOLS API] [OK] Retrieval completed: {retrieval_result.get('num_results', 0)} documents")

        if not retrieval_result["success"]:
            return ToolResponse(
                success=False,
                answer="RAG retrieval failed",
                data={},
                metadata={"execution_time": time.time() - start_time},
                error=retrieval_result.get("error", "Unknown error")
            )

        # Step 3: Format documents for LLM
        docs_formatted = "\n\n".join([
            f"Document: {doc['document']}\nChunk {doc['chunk_index']} (Score: {doc['score']:.2f}):\n{doc['chunk']}"
            for doc in retrieval_result["documents"]
        ])

        # Step 4: Synthesize answer
        synthesis_prompt = load_prompt(
            "rag_synthesize.txt",
            user_query=request.query,
            documents=docs_formatted,
            context=context_str
        )

        messages = [{"role": "user", "content": synthesis_prompt}]
        answer = llm_backend.chat(
            messages,
            config.TOOL_MODELS.get("rag", config.OLLAMA_MODEL),
            config.TOOL_PARAMETERS.get("rag", {}).get("temperature", 0.5)
        )

        execution_time = time.time() - start_time

        return ToolResponse(
            success=True,
            answer=answer,
            data={
                "optimized_query": optimized_query,
                "documents": retrieval_result["documents"],
                "num_results": retrieval_result["num_results"]
            },
            metadata={
                "execution_time": execution_time,
                "collection": request.collection_name
            }
        )

    except Exception as e:
        return ToolResponse(
            success=False,
            answer=f"RAG query error: {str(e)}",
            data={},
            metadata={"execution_time": time.time() - start_time},
            error=str(e)
        )
