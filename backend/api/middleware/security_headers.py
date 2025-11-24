"""
Security Headers Middleware
Adds security-related HTTP headers to all responses
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from typing import Callable

from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to all HTTP responses

    Security headers added:
        - X-Content-Type-Options: nosniff
            Prevents MIME type sniffing
        - X-Frame-Options: DENY
            Prevents clickjacking by disabling iframe embedding
        - X-XSS-Protection: 1; mode=block
            Enables browser XSS filtering (legacy browsers)
        - Strict-Transport-Security: max-age=31536000; includeSubDomains
            Enforces HTTPS for 1 year
        - Content-Security-Policy: default-src 'self'
            Restricts resource loading to same origin
        - Referrer-Policy: strict-origin-when-cross-origin
            Controls referrer information

    Usage:
        app = FastAPI()
        app.add_middleware(SecurityHeadersMiddleware)
    """

    def __init__(
        self,
        app,
        enable_hsts: bool = True,
        enable_csp: bool = False,  # CSP can break functionality, off by default
        hsts_max_age: int = 31536000,  # 1 year
        hsts_include_subdomains: bool = True,
        csp_directives: str = "default-src 'self'",
    ):
        """
        Initialize security headers middleware

        Args:
            app: FastAPI/Starlette application
            enable_hsts: Enable Strict-Transport-Security header
            enable_csp: Enable Content-Security-Policy header
            hsts_max_age: HSTS max-age in seconds (default: 1 year)
            hsts_include_subdomains: Include subdomains in HSTS
            csp_directives: CSP policy directives
        """
        super().__init__(app)
        self.enable_hsts = enable_hsts
        self.enable_csp = enable_csp
        self.hsts_max_age = hsts_max_age
        self.hsts_include_subdomains = hsts_include_subdomains
        self.csp_directives = csp_directives

        logger.info("SecurityHeadersMiddleware initialized")
        logger.info(f"  HSTS: {'enabled' if enable_hsts else 'disabled'}")
        logger.info(f"  CSP: {'enabled' if enable_csp else 'disabled'}")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and add security headers to response

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain

        Returns:
            HTTP response with security headers
        """
        # Process request
        response = await call_next(request)

        # Add security headers
        self._add_security_headers(response)

        return response

    def _add_security_headers(self, response: Response) -> None:
        """
        Add security headers to response

        Args:
            response: HTTP response to modify
        """
        # X-Content-Type-Options: Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # X-Frame-Options: Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # X-XSS-Protection: Enable XSS filter (legacy browsers)
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Referrer-Policy: Control referrer information
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Permissions-Policy: Restrict browser features
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )

        # Strict-Transport-Security: Enforce HTTPS
        if self.enable_hsts:
            hsts_value = f"max-age={self.hsts_max_age}"
            if self.hsts_include_subdomains:
                hsts_value += "; includeSubDomains"
            response.headers["Strict-Transport-Security"] = hsts_value

        # Content-Security-Policy: Restrict resource loading
        if self.enable_csp:
            response.headers["Content-Security-Policy"] = self.csp_directives


async def add_security_headers_middleware(request: Request, call_next: Callable) -> Response:
    """
    Alternative function-based middleware for adding security headers

    This is a simpler alternative to the class-based middleware above.
    Use this if you prefer function-based middleware.

    Usage:
        app = FastAPI()
        app.middleware("http")(add_security_headers_middleware)
    """
    # Process request
    response = await call_next(request)

    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

    return response
