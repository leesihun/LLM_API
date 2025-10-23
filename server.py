"""
Main Server Launcher
Starts the FastAPI backend server
"""

import uvicorn
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.config.settings import settings


def main():
    """Start the server"""
    print("=" * 70)
    print("HE Team LLM Assistant - Backend Server")
    print("=" * 70)
    print(f"Host: {settings.server_host}")
    print(f"Port: {settings.server_port}")
    print(f"Ollama: {settings.ollama_host}")
    print(f"Model: {settings.ollama_model}")
    print("=" * 70)
    print()
    print("API Documentation:")
    print(f"  - Swagger UI: http://localhost:{settings.server_port}/docs")
    print(f"  - ReDoc: http://localhost:{settings.server_port}/redoc")
    print()
    print("Starting server...")
    print("=" * 70)
    print()

    uvicorn.run(
        "backend.api.app:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=True,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main()