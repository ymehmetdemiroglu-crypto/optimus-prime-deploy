"""
Standardized API Response Schemas

This module provides consistent response envelopes for all API endpoints.

Response Format:
- Success: {"success": true, "data": {...}, "message": "..."}
- Error: {"success": false, "error": {...}, "message": "..."}
- Paginated: {"success": true, "data": [...], "pagination": {...}, "message": "..."}

Benefits:
- Consistent frontend parsing
- Better error handling
- Built-in pagination support
- Type safety with Pydantic
"""

from typing import TypeVar, Generic, Optional, Any, List, Dict
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


T = TypeVar("T")


class ErrorDetail(BaseModel):
    """
    Error detail information.

    Provides structured error information for clients.
    """

    code: str = Field(..., description="Error code (e.g., 'VALIDATION_ERROR', 'NOT_FOUND')")
    message: str = Field(..., description="Human-readable error message")
    field: Optional[str] = Field(None, description="Field that caused the error (for validation errors)")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error context")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid email format",
                "field": "email",
                "details": {"pattern": "^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$"}
            }
        }
    )


class SuccessResponse(BaseModel, Generic[T]):
    """
    Standard success response envelope.

    Usage:
        @app.get("/items/{id}", response_model=SuccessResponse[ItemSchema])
        async def get_item(id: int):
            item = await get_item_from_db(id)
            return success_response(data=item, message="Item retrieved successfully")
    """

    success: bool = Field(default=True, description="Always true for success responses")
    data: T = Field(..., description="Response data")
    message: Optional[str] = Field(None, description="Human-readable success message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp (UTC)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "data": {"id": 1, "name": "Example Item"},
                "message": "Item retrieved successfully",
                "timestamp": "2026-02-16T10:30:00Z"
            }
        }
    )


class ErrorResponse(BaseModel):
    """
    Standard error response envelope.

    Usage:
        @app.exception_handler(ValidationException)
        async def validation_exception_handler(request, exc):
            return JSONResponse(
                status_code=400,
                content=error_response(
                    error=ErrorDetail(
                        code="VALIDATION_ERROR",
                        message=str(exc),
                        field=exc.field
                    )
                ).dict()
            )
    """

    success: bool = Field(default=False, description="Always false for error responses")
    error: ErrorDetail = Field(..., description="Error details")
    message: str = Field(..., description="Human-readable error message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp (UTC)")
    request_id: Optional[str] = Field(None, description="Request correlation ID for debugging")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": False,
                "error": {
                    "code": "NOT_FOUND",
                    "message": "Item with ID 123 not found"
                },
                "message": "Item not found",
                "timestamp": "2026-02-16T10:30:00Z",
                "request_id": "550e8400-e29b-41d4-a716-446655440000"
            }
        }
    )


class PaginationMeta(BaseModel):
    """
    Pagination metadata.

    Provides information about the current page, total records, etc.
    """

    total: int = Field(..., description="Total number of records", ge=0)
    count: int = Field(..., description="Number of records in current page", ge=0)
    per_page: int = Field(..., description="Records per page", ge=1)
    current_page: int = Field(..., description="Current page number", ge=1)
    total_pages: int = Field(..., description="Total number of pages", ge=0)
    has_next: bool = Field(..., description="Whether there is a next page")
    has_prev: bool = Field(..., description="Whether there is a previous page")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total": 150,
                "count": 20,
                "per_page": 20,
                "current_page": 3,
                "total_pages": 8,
                "has_next": True,
                "has_prev": True
            }
        }
    )


class PaginatedResponse(BaseModel, Generic[T]):
    """
    Standard paginated response envelope.

    Usage:
        @app.get("/items", response_model=PaginatedResponse[ItemSchema])
        async def list_items(skip: int = 0, limit: int = 20):
            items = await get_items_from_db(skip=skip, limit=limit)
            total = await count_items()
            return paginated_response(
                data=items,
                total=total,
                skip=skip,
                limit=limit
            )
    """

    success: bool = Field(default=True, description="Always true for success responses")
    data: List[T] = Field(..., description="Array of items")
    pagination: PaginationMeta = Field(..., description="Pagination metadata")
    message: Optional[str] = Field(None, description="Human-readable success message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp (UTC)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "data": [
                    {"id": 1, "name": "Item 1"},
                    {"id": 2, "name": "Item 2"}
                ],
                "pagination": {
                    "total": 150,
                    "count": 2,
                    "per_page": 20,
                    "current_page": 1,
                    "total_pages": 8,
                    "has_next": True,
                    "has_prev": False
                },
                "message": "Items retrieved successfully",
                "timestamp": "2026-02-16T10:30:00Z"
            }
        }
    )


# Helper functions for creating responses


def success_response(
    data: Any,
    message: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a success response.

    Args:
        data: Response data (any JSON-serializable object)
        message: Optional success message

    Returns:
        dict: Standardized success response

    Example:
        return success_response(
            data={"id": 1, "name": "Test"},
            message="Item created successfully"
        )
    """
    return {
        "success": True,
        "data": data,
        "message": message,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


def error_response(
    error: ErrorDetail,
    message: Optional[str] = None,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create an error response.

    Args:
        error: Error detail object
        message: Optional error message (defaults to error.message)
        request_id: Optional request correlation ID

    Returns:
        dict: Standardized error response

    Example:
        return JSONResponse(
            status_code=404,
            content=error_response(
                error=ErrorDetail(
                    code="NOT_FOUND",
                    message="Item not found"
                ),
                message="The requested item does not exist"
            )
        )
    """
    return {
        "success": False,
        "error": error.model_dump(),
        "message": message or error.message,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "request_id": request_id
    }


def paginated_response(
    data: List[Any],
    total: int,
    skip: int = 0,
    limit: int = 100,
    message: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a paginated response.

    Args:
        data: List of items for current page
        total: Total number of items across all pages
        skip: Number of items skipped (offset)
        limit: Number of items per page
        message: Optional success message

    Returns:
        dict: Standardized paginated response

    Example:
        items = await db.query(Item).offset(skip).limit(limit).all()
        total = await db.query(Item).count()

        return paginated_response(
            data=items,
            total=total,
            skip=skip,
            limit=limit,
            message="Items retrieved successfully"
        )
    """
    current_page = (skip // limit) + 1
    total_pages = (total + limit - 1) // limit  # Ceiling division

    return {
        "success": True,
        "data": data,
        "pagination": {
            "total": total,
            "count": len(data),
            "per_page": limit,
            "current_page": current_page,
            "total_pages": total_pages,
            "has_next": skip + limit < total,
            "has_prev": skip > 0
        },
        "message": message,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


# Common error responses for reuse

class CommonErrors:
    """
    Common error responses for reuse across endpoints.
    """

    @staticmethod
    def not_found(resource: str, identifier: Any) -> ErrorDetail:
        """Resource not found error"""
        return ErrorDetail(
            code="NOT_FOUND",
            message=f"{resource} with identifier '{identifier}' not found"
        )

    @staticmethod
    def validation_error(message: str, field: Optional[str] = None) -> ErrorDetail:
        """Validation error"""
        return ErrorDetail(
            code="VALIDATION_ERROR",
            message=message,
            field=field
        )

    @staticmethod
    def unauthorized(message: str = "Authentication required") -> ErrorDetail:
        """Unauthorized error"""
        return ErrorDetail(
            code="UNAUTHORIZED",
            message=message
        )

    @staticmethod
    def forbidden(message: str = "Insufficient permissions") -> ErrorDetail:
        """Forbidden error"""
        return ErrorDetail(
            code="FORBIDDEN",
            message=message
        )

    @staticmethod
    def conflict(message: str, details: Optional[Dict[str, Any]] = None) -> ErrorDetail:
        """Conflict error (e.g., duplicate entry)"""
        return ErrorDetail(
            code="CONFLICT",
            message=message,
            details=details
        )

    @staticmethod
    def internal_error(message: str = "An internal error occurred") -> ErrorDetail:
        """Internal server error"""
        return ErrorDetail(
            code="INTERNAL_ERROR",
            message=message
        )
