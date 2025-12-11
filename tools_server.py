"""
Tools API Server
Runs on a separate port (1006) to avoid deadlock when main server calls tools
"""
import uvicorn
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from backend.api.routes import tools

# Create FastAPI app
app = FastAPI(
    title="LLM Tools API",
    description="Tools API for websearch, python_coder, and RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include tools router
app.include_router(tools.router)

# Health check
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "tools_api"}


def main():
    """Start the tools server"""
    print("=" * 70)
    print("LLM Tools API Server")
    print("=" * 70)
    print(f"Host: {config.SERVER_HOST}")
    print(f"Port: {config.TOOLS_PORT}")
    print("=" * 70)
    print()
    print("Available Tools:")
    print("  - Web Search (Tavily)")
    print("  - Python Code Executor")
    print("  - RAG (Retrieval Augmented Generation)")
    print()
    print(f"Health Check: http://localhost:{config.TOOLS_PORT}/health")
    print()
    print("Starting server...")
    print("=" * 70)
    print()

    uvicorn.run(
        app,
        host=config.SERVER_HOST,
        port=config.TOOLS_PORT,
        reload=False,
        log_level=config.LOG_LEVEL.lower()
    )


if __name__ == "__main__":
    main()
