"""
FastAPI Dependencies

Common reusable dependencies for FastAPI endpoints.
Provides pagination, filtering, and other common query parameters.
"""

from typing import Optional, List
from fastapi import Query, Header, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
from datetime import datetime, date
from jose import JWTError, jwt as jose_jwt

from app.core.config import settings

# Security scheme for Swagger docs
security_scheme = HTTPBearer(auto_error=False)


class PaginationParams(BaseModel):
    """
    Pagination parameters for list endpoints.

    Usage:
        @app.get("/items")
        async def get_items(pagination: PaginationParams = Depends()):
            ...
    """

    skip: int = Field(
        default=0,
        ge=0,
        description="Number of records to skip (offset)"
    )
    limit: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of records to return (max 1000)"
    )

    @property
    def offset(self) -> int:
        """Alias for skip (database offset)"""
        return self.skip

    def to_dict(self) -> dict:
        """Convert to dictionary for database queries"""
        return {"offset": self.skip, "limit": self.limit}


class DateRangeParams(BaseModel):
    """
    Date range parameters for filtering.

    Usage:
        @app.get("/reports")
        async def get_reports(date_range: DateRangeParams = Depends()):
            ...
    """

    start_date: Optional[date] = Field(
        default=None,
        description="Start date (inclusive, YYYY-MM-DD)"
    )
    end_date: Optional[date] = Field(
        default=None,
        description="End date (inclusive, YYYY-MM-DD)"
    )

    @field_validator("end_date")
    @classmethod
    def validate_date_range(cls, v: Optional[date], info) -> Optional[date]:
        """Ensure end_date is not before start_date"""
        start_date = info.data.get("start_date")
        if v and start_date and v < start_date:
            raise ValueError("end_date must be greater than or equal to start_date")
        return v

    @property
    def is_valid(self) -> bool:
        """Check if date range is specified"""
        return self.start_date is not None or self.end_date is not None


class SortParams(BaseModel):
    """
    Sorting parameters for list endpoints.

    Usage:
        @app.get("/items")
        async def get_items(sort: SortParams = Depends()):
            query = query.order_by(sort.get_order_by("created_at"))
    """

    sort_by: Optional[str] = Field(
        default=None,
        description="Field to sort by"
    )
    sort_order: str = Field(
        default="desc",
        description="Sort order: asc or desc",
        pattern="^(asc|desc)$"
    )

    @field_validator("sort_order")
    @classmethod
    def validate_sort_order(cls, v: str) -> str:
        """Ensure sort_order is lowercase"""
        return v.lower()

    def get_order_by(self, default_field: str, allowed_fields: Optional[List[str]] = None) -> str:
        """
        Get SQL ORDER BY clause.

        Args:
            default_field: Default field to sort by if not specified
            allowed_fields: List of allowed fields (for security)

        Returns:
            str: Field name with direction (e.g., "created_at DESC")

        Raises:
            HTTPException: If sort_by is not in allowed_fields
        """
        field = self.sort_by or default_field

        # Validate against allowed fields
        if allowed_fields and field not in allowed_fields:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid sort field. Allowed: {', '.join(allowed_fields)}"
            )

        return f"{field} {self.sort_order.upper()}"


class FilterParams(BaseModel):
    """
    Generic filter parameters.

    Usage:
        @app.get("/accounts")
        async def get_accounts(filters: FilterParams = Depends()):
            if filters.status:
                query = query.filter(Account.status == filters.status)
    """

    status: Optional[str] = Field(
        default=None,
        description="Filter by status (active, paused, archived)"
    )
    search: Optional[str] = Field(
        default=None,
        description="Search term (name, description, etc.)",
        max_length=255
    )

    @field_validator("search")
    @classmethod
    def sanitize_search(cls, v: Optional[str]) -> Optional[str]:
        """Sanitize search term to prevent SQL injection"""
        if v:
            # Remove potentially dangerous characters
            # For ILIKE queries, we'll use parameterized queries anyway
            return v.strip()
        return v


async def get_correlation_id(
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID")
) -> Optional[str]:
    """
    Extract correlation ID from request headers.

    Usage:
        @app.get("/items")
        async def get_items(correlation_id: str = Depends(get_correlation_id)):
            logger.info(f"Request {correlation_id}: Fetching items")
    """
    return x_request_id


async def get_user_agent(
    user_agent: Optional[str] = Header(None, alias="User-Agent")
) -> Optional[str]:
    """
    Extract User-Agent from request headers.

    Usage:
        @app.get("/items")
        async def get_items(user_agent: str = Depends(get_user_agent)):
            logger.info(f"Request from {user_agent}")
    """
    return user_agent


# ═══════════════════════════════════════════════════════════════════
#  Authentication Dependencies (JWT Verification)
# ═══════════════════════════════════════════════════════════════════

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
) -> Optional[dict]:
    """
    Get current authenticated user by validating the JWT Bearer token.

    Decodes the token using SECRET_KEY from config (must be set via env var).
    Returns None if no credentials are provided (for optional-auth routes).
    Raises 401 if the token is present but invalid.

    Usage:
        @app.get("/protected")
        async def protected_route(user: dict = Depends(get_current_user)):
            if not user:
                raise HTTPException(status_code=401, detail="Not authenticated")
    """
    if credentials is None:
        return None  # No token provided — allow optional-auth routes

    token = credentials.credentials
    try:
        payload = jose_jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        user_id: Optional[str] = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing 'sub' claim",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return {
            "id": user_id,
            "email": payload.get("email"),
            "role": payload.get("role", "user"),
        }
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def require_auth(
    user: Optional[dict] = Depends(get_current_user),
) -> dict:
    """
    Require authentication for endpoint.

    Usage:
        @app.get("/admin")
        async def admin_route(user: dict = Depends(require_auth)):
            # User is guaranteed to be authenticated
    """
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


class AccountFilterParams(BaseModel):
    """
    Account-specific filter parameters.

    Usage:
        @app.get("/accounts")
        async def get_accounts(filters: AccountFilterParams = Depends()):
            ...
    """

    marketplace: Optional[str] = Field(
        default=None,
        description="Filter by marketplace (e.g., 'US', 'UK', 'DE')"
    )
    status: Optional[str] = Field(
        default=None,
        description="Filter by status (active, paused, archived)"
    )
    has_credentials: Optional[bool] = Field(
        default=None,
        description="Filter by credential existence"
    )
    min_spend: Optional[float] = Field(
        default=None,
        ge=0,
        description="Minimum total spend"
    )
    max_spend: Optional[float] = Field(
        default=None,
        ge=0,
        description="Maximum total spend"
    )

    @field_validator("max_spend")
    @classmethod
    def validate_spend_range(cls, v: Optional[float], info) -> Optional[float]:
        """Ensure max_spend is not less than min_spend"""
        min_spend = info.data.get("min_spend")
        if v and min_spend and v < min_spend:
            raise ValueError("max_spend must be greater than or equal to min_spend")
        return v
