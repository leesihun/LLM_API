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
from backend.api.routes import create_routes


# ============================================================================
# Logging Configuration
# ============================================================================

def setup_logging():
    """Configure application logging with readable, consistent formatting."""
    import sys

    class ReadabilityFilter(logging.Filter):
        """Reduce noisy logs: drop banner/separator lines and collapse long/multiline messages."""
        SEPARATOR_CHARS = set("=~^!-#>X")

        def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
            try:
                if isinstance(record.msg, str):
                    msg = record.msg.strip()
                    # Skip empty or banner-only lines (e.g., "====...", "-----...")
                    if msg and len(msg) >= 10 and all(ch in self.SEPARATOR_CHARS for ch in set(msg)):
                        return False
                    # Collapse multiline logs to first line with an indicator
                    if "\n" in msg:
                        first_line = msg.splitlines()[0].strip()
                        msg = f"{first_line} [...]"
                    # Truncate overly long messages
                    if len(msg) > 400:
                        msg = msg[:400] + "..."
                    record.msg = msg
            except Exception:
                # Never block logging if the filter has an issue
                pass
            return True

    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_format = '%(asctime)s %(levelname)s %(name)s:%(funcName)s:%(lineno)d - %(message)s'

    # Create console handler with UTF-8 encoding for Windows compatibility
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.log_level))
    console_handler.setFormatter(logging.Formatter(log_format))
    console_handler.addFilter(ReadabilityFilter())

    # Force UTF-8 encoding on the stream to handle Unicode characters
    if hasattr(console_handler.stream, 'reconfigure'):
        console_handler.stream.reconfigure(encoding='utf-8', errors='replace')

    # Create file handler with UTF-8 encoding
    file_handler = logging.FileHandler(settings.log_file, encoding='utf-8')
    file_handler.setLevel(getattr(logging, settings.log_level))
    file_handler.setFormatter(logging.Formatter(log_format))
    file_handler.addFilter(ReadabilityFilter())

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format=log_format,
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

for router in create_routes():
    app.include_router(router)


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

    # Preload model into VRAM for faster first request
    try:
        logger.info(f"Preloading model '{settings.ollama_model}' into VRAM...")
        from backend.utils.llm_factory import LLMFactory

        llm = LLMFactory.create_llm()
        # Send a minimal warmup request to load model
        await llm.ainvoke("Hello")
        logger.info(f"✓ Model '{settings.ollama_model}' preloaded successfully!")
    except Exception as e:
        logger.warning(f"⚠ Model preload failed (non-critical): {e}")
        logger.warning("First API request may be slower while model loads")

    logger.info("Application started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("Shutting down HE Team LLM Assistant API...")
