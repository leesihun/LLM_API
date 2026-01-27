"""
Tools API Server
Runs on a separate port (1006) to avoid deadlock when main server calls tools
"""
import uvicorn
import sys
import io
import atexit
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Set UTF-8 encoding for Windows console to handle emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

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


@app.on_event("startup")
async def startup_event():
    """Run on server startup"""
    if config.PRELOAD_MODEL_ON_STARTUP:
        print("\n" + "=" * 70)
        print("MODEL PRELOADING (Tools Server)")
        print("=" * 70)

        from backend.core.llm_backend import llm_backend

        # Check if backend is available
        if not llm_backend.is_available():
            print("[Startup] LLM backend not available - skipping model preload")
            print("=" * 70 + "\n")
            return

        # Get the actual backend instance (unwrap the interceptor)
        backend = llm_backend.backend

        # Check if it's an Ollama backend (or AutoLLMBackend with Ollama active)
        from backend.core.llm_backend import OllamaBackend, AutoLLMBackend

        ollama_backend = None
        if isinstance(backend, OllamaBackend):
            ollama_backend = backend
        elif isinstance(backend, AutoLLMBackend):
            # For AutoLLMBackend, get the active backend
            active = backend._get_backend()
            if isinstance(active, OllamaBackend):
                ollama_backend = active

        if ollama_backend:
            # Preload the default model
            success = ollama_backend.preload_model(
                model=config.OLLAMA_MODEL,
                keep_alive=config.PRELOAD_KEEP_ALIVE
            )

            if success:
                print(f"[Startup] Model '{config.OLLAMA_MODEL}' is now loaded in GPU memory")
                print(f"[Startup] Keep-alive setting: {config.PRELOAD_KEEP_ALIVE}")
            else:
                print(f"[Startup] Failed to preload model '{config.OLLAMA_MODEL}'")
        else:
            print(f"[Startup] Model preloading only supported for Ollama backend")
            print(f"[Startup] Current backend: {backend.__class__.__name__}")

        print("=" * 70 + "\n")

    # Start OpenCode headless server for autonomous operation
    if config.PYTHON_EXECUTOR_MODE == "opencode":
        from tools.python_coder.opencode_server import start_opencode_server, stop_opencode_server

        print("\n" + "=" * 70)
        print("OPENCODE SERVER INITIALIZATION")
        print("=" * 70)
        print("[TOOLS SERVER] Starting OpenCode headless server...")

        try:
            start_opencode_server()
            print("[TOOLS SERVER] OpenCode server started successfully")
            print("[TOOLS SERVER] Using 'build' agent for autonomous operation")

            # Register shutdown handler
            atexit.register(stop_opencode_server)
            print("[TOOLS SERVER] Shutdown handler registered")
        except Exception as e:
            print(f"[TOOLS SERVER] Failed to start OpenCode server: {e}")
            print("[TOOLS SERVER] Python coder tool will not work in opencode mode")

        print("=" * 70 + "\n")


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

    # When using workers, uvicorn needs an import string, not the app object
    # Format: "module:variable" where the app is defined
    uvicorn.run(
        "tools_server:app",  # Changed from 'app' to import string
        host=config.SERVER_HOST,
        port=config.TOOLS_PORT,
        reload=False,
        log_level=config.LOG_LEVEL.lower(),
        access_log=True,
        workers=config.TOOLS_SERVER_WORKERS
    )


if __name__ == "__main__":
    main()
