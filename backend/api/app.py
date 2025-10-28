"""
FastAPI Application
Main server entry point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from pathlib import Path

from backend.config.settings import settings
from backend.api.routes import auth_router, openai_router, files_router, admin_router, tools_router, chat_router


# ============================================================================
# Logging Configuration
# ============================================================================

def setup_logging():
    """Configure application logging with proper Unicode handling for Windows"""
    import sys

    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create console handler with UTF-8 encoding for Windows compatibility
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.log_level))
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )

    # Force UTF-8 encoding on the stream to handle Unicode characters
    if hasattr(console_handler.stream, 'reconfigure'):
        # Python 3.7+ on Windows
        console_handler.stream.reconfigure(encoding='utf-8', errors='replace')

    # Create file handler with UTF-8 encoding
    file_handler = logging.FileHandler(settings.log_file, encoding='utf-8')
    file_handler.setLevel(getattr(logging, settings.log_level))
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[file_handler, console_handler]
    )


setup_logging()
logger = logging.getLogger(__name__)


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="HE Team LLM Assistant API",
    description="Agentic AI backend with LangGraph, RAG, and Web Search",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# ============================================================================
# CORS Configuration
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Include Routers
# ============================================================================

app.include_router(auth_router)
app.include_router(openai_router)
app.include_router(files_router)
app.include_router(admin_router)
app.include_router(tools_router)
app.include_router(chat_router)


# ============================================================================
# Health Check Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "HE Team LLM Assistant API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ollama_host": settings.ollama_host,
        "model": settings.ollama_model
    }


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    import httpx

    logger.info("Starting HE Team LLM Assistant API...")
    logger.info(f"Ollama Host: {settings.ollama_host}")
    logger.info(f"Ollama Model: {settings.ollama_model}")
    logger.info(f"Server: {settings.server_host}:{settings.server_port}")

    # Test Ollama connection
    try:
        logger.info("Testing Ollama connection...")
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{settings.ollama_host}/api/tags", timeout=10.0)
            if response.status_code == 200:
                logger.info("✓ Ollama connection successful!")
                models = response.json().get("models", [])
                logger.info(f"✓ Available models: {[m.get('name') for m in models]}")
            else:
                logger.error(f"✗ Ollama returned status {response.status_code}")
    except Exception as e:
        logger.error("=" * 80)
        logger.error("✗ OLLAMA CONNECTION FAILED!")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Ollama Host: {settings.ollama_host}")
        logger.error("Please check:")
        logger.error("  1. Is Ollama running? (ollama serve)")
        logger.error("  2. Is it accessible at the configured host?")
        logger.error("  3. Try: curl http://127.0.0.1:11434/api/tags")
        logger.error("=" * 80)

    logger.info("Application started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("Shutting down HE Team LLM Assistant API...")
