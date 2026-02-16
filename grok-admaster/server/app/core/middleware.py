"""
Custom middleware for Grok-AdMaster API.

Provides:
- Correlation ID injection for request tracing
- Request/response logging
- Performance monitoring
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import time
import uuid
from app.core.logging_config import set_correlation_id, get_logger

logger = get_logger(__name__)


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Inject correlation ID into each request for tracing."""

    async def dispatch(self, request: Request, call_next):
        """Process request and inject correlation ID.

        The correlation ID is:
        1. Taken from X-Request-ID header if present
        2. Generated as UUID if not present
        3. Added to response headers
        4. Available in all logs via correlation_id context
        """
        # Get or generate correlation ID
        correlation_id = request.headers.get("X-Request-ID")
        if not correlation_id:
            correlation_id = str(uuid.uuid4())

        # Set in logging context
        set_correlation_id(correlation_id)

        # Add to request state for access in endpoints
        request.state.correlation_id = correlation_id

        # Log request
        start_time = time.time()
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else None,
            },
        )

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Add correlation ID to response headers
            response.headers["X-Request-ID"] = correlation_id

            # Log response
            logger.info(
                f"Request completed: {request.method} {request.url.path} - {response.status_code} ({duration_ms:.2f}ms)",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                },
            )

            return response

        except Exception as e:
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Log error
            logger.error(
                f"Request failed: {request.method} {request.url.path} - {str(e)} ({duration_ms:.2f}ms)",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": duration_ms,
                    "error": str(e),
                },
                exc_info=True,
            )

            # Re-raise to let FastAPI handle the error
            raise


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        """Add security headers to response."""
        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response
