"""
Main Server Launcher
Starts the FastAPI backend server
"""
import uvicorn
import sys
import io
from pathlib import Path

# Set UTF-8 encoding for Windows console to handle emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import config


def main():
    """Start the server"""
    print("=" * 70)
    print("LLM API - Backend Server")
    print("=" * 70)
    print(f"Host: {config.SERVER_HOST}")
    print(f"Port: {config.SERVER_PORT}")
    print(f"LLM Backend: {config.LLM_BACKEND}")
    print(f"Ollama: {config.OLLAMA_HOST}")
    print(f"Llama.cpp: {config.LLAMACPP_HOST}")
    print(f"Default Model: {config.OLLAMA_MODEL}")
    print("=" * 70)
    print()
    print("API Documentation:")
    print(f"  - Swagger UI: http://localhost:{config.SERVER_PORT}/docs")
    print(f"  - ReDoc: http://localhost:{config.SERVER_PORT}/redoc")
    print(f"  - Health Check: http://localhost:{config.SERVER_PORT}/health")
    print()
    print("Starting server...")
    print("=" * 70)
    print()

    # DEV_MODE enables auto-reload but requires single worker
    workers = 1 if config.DEV_MODE else config.SERVER_WORKERS

    uvicorn.run(
        "backend.api.app:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        reload=config.DEV_MODE,
        log_level=config.LOG_LEVEL.lower(),
        workers=workers
    )


if __name__ == "__main__":
    main()