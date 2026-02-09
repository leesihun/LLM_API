"""
Dedicated Websearch API Server
Runs as a standalone service, deployable on a remote machine.
Only serves the websearch endpoint - no python_coder or RAG.
"""
import uvicorn
import sys
import io
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Set UTF-8 encoding for Windows console to handle emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from tools.web_search import WebSearchTool


# ============================================================================
# Request/Response Schemas (inline to minimize dependencies)
# ============================================================================

class WebSearchRequest(BaseModel):
    query: str
    max_results: Optional[int] = None

class ToolResponse(BaseModel):
    success: bool
    answer: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    error: Optional[str] = None


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="LLM Websearch API",
    description="Dedicated websearch server (Tavily)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "websearch_api"}


@app.post("/api/tools/websearch", response_model=ToolResponse)
async def websearch(request: WebSearchRequest):
    """
    Pure web search - no LLM processing.
    Returns raw Tavily results for agent to interpret.
    """
    print("\n" + "=" * 80)
    print("[WEBSEARCH SERVER] /api/tools/websearch endpoint called")
    print("=" * 80)
    print(f"Query: {request.query}")
    print(f"Max results: {request.max_results or 'default'}")

    start_time = time.time()

    try:
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

        return ToolResponse(
            success=True,
            answer="",
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


def main():
    """Start the websearch server"""
    port = config.WEBSEARCH_SERVER_PORT
    print("=" * 70)
    print("LLM Websearch API Server")
    print("=" * 70)
    print(f"Host: 0.0.0.0")
    print(f"Port: {port}")
    print("=" * 70)
    print()
    print("Available Tools:")
    print("  - Web Search (Tavily)")
    print()
    print(f"Health Check: http://localhost:{port}/health")
    print()
    print("Starting server...")
    print("=" * 70)
    print()

    uvicorn.run(
        "websearch_server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level=config.LOG_LEVEL.lower(),
        access_log=True,
        workers=2,
    )


if __name__ == "__main__":
    main()
