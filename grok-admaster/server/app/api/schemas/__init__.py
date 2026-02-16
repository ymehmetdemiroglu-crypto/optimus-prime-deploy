"""
API Schemas

Standardized request/response schemas for the API.
"""

from app.api.schemas.responses import (
    SuccessResponse,
    ErrorResponse,
    PaginatedResponse,
    success_response,
    error_response,
    paginated_response,
)

__all__ = [
    "SuccessResponse",
    "ErrorResponse",
    "PaginatedResponse",
    "success_response",
    "error_response",
    "paginated_response",
]
