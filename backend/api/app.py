"""
Main FastAPI application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config
from backend.api.routes import auth, models, admin, chat, sessions, tools

# Create FastAPI app
app = FastAPI(
    title="LLM API",
    description="OpenAI-compatible LLM API with Ollama and llama.cpp support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/")
def root():
    """API health check"""
    return {
        "status": "online",
        "service": "LLM API",
        "version": "1.0.0",
        "backends": {
            "ollama": config.OLLAMA_HOST,
            "llamacpp": config.LLAMACPP_HOST,
            "active": config.LLM_BACKEND
        }
    }


@app.get("/health")
def health():
    """Health check endpoint"""
    from backend.core.llm_backend import llm_backend

    llm_status = "available" if llm_backend.is_available() else "unavailable"

    return {
        "status": "healthy",
        "llm_backend": llm_status,
        "database": "connected"
    }


# Include routers
app.include_router(auth.router)  # /api/auth/*
app.include_router(models.router)  # /v1/models
app.include_router(admin.router)  # /api/admin/*
app.include_router(chat.router)  # /v1/chat/completions
app.include_router(sessions.router)  # /api/chat/sessions, /api/chat/history
app.include_router(tools.router)  # /api/tools/*


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle unexpected errors gracefully"""
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": "internal_error"
            }
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        log_level=config.LOG_LEVEL.lower()
    )
