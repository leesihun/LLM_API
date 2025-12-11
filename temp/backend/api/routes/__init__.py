"""
Routes Module
Aggregates all API route modules and provides a unified interface for route registration

This module exports individual routers and provides a create_routes() function
for easy integration with the FastAPI application.
"""

from typing import List
from fastapi import APIRouter

from .auth import auth_router
from .chat import chat_router, openai_router
from .admin import admin_router
from .files import files_router
from .tools import tools_router


def create_routes() -> List[APIRouter]:
    """
    Create and return all API routers for the application

    Returns:
        List of APIRouter instances to be registered with FastAPI app

    Usage:
        from backend.api.routes import create_routes
        for router in create_routes():
            app.include_router(router)
    """
    return [
        auth_router,
        openai_router,
        chat_router,
        admin_router,
        files_router,
        tools_router
    ]


# Export individual routers for backward compatibility
__all__ = [
    "create_routes",
    "auth_router",
    "chat_router",
    "openai_router",
    "admin_router",
    "files_router",
    "tools_router"
]
