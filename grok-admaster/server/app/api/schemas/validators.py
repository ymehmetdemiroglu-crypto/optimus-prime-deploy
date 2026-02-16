"""
Comprehensive Pydantic Validators

Reusable validators for common fields across schemas.

Usage:
    from app.api.schemas.validators import validate_email, validate_phone

    class UserSchema(BaseModel):
        email: str
        phone: Optional[str] = None

        _validate_email = field_validator("email")(validate_email)
        _validate_phone = field_validator("phone")(validate_phone)
"""

import re
from typing import Optional, Any
from pydantic import field_validator, ValidationError
from datetime import date, datetime


# Email validation

EMAIL_REGEX = re.compile(
    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
)


def validate_email(v: Optional[str]) -> Optional[str]:
    """
    Validate email address format.

    Checks:
    - Not empty if provided
    - Matches standard email pattern
    - Has valid domain extension

    Raises:
        ValueError: If email format is invalid
    """
    if v is None:
        return v

    v = v.strip().lower()

    if not v:
        raise ValueError("Email cannot be empty")

    if not EMAIL_REGEX.match(v):
        raise ValueError("Invalid email format")

    return v


# Phone validation

PHONE_REGEX = re.compile(
    r'^\+?[1-9]\d{1,14}$'  # E.164 format
)


def validate_phone(v: Optional[str]) -> Optional[str]:
    """
    Validate phone number format (E.164).

    Accepts:
    - International format: +1234567890
    - Up to 15 digits
    - Optional + prefix

    Raises:
        ValueError: If phone format is invalid
    """
    if v is None:
        return v

    # Remove spaces, hyphens, parentheses
    v = re.sub(r'[\s\-\(\)]', '', v.strip())

    if not v:
        raise ValueError("Phone number cannot be empty")

    if not PHONE_REGEX.match(v):
        raise ValueError(
            "Invalid phone format. Use international format: +1234567890"
        )

    return v


# URL validation

URL_REGEX = re.compile(
    r'^https?://'  # http:// or https://
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
    r'localhost|'  # localhost
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
    r'(?::\d+)?'  # optional port
    r'(?:/?|[/?]\S+)$',
    re.IGNORECASE
)


def validate_url(v: Optional[str]) -> Optional[str]:
    """
    Validate URL format.

    Checks:
    - Starts with http:// or https://
    - Has valid domain or IP
    - Optional port and path

    Raises:
        ValueError: If URL format is invalid
    """
    if v is None:
        return v

    v = v.strip()

    if not v:
        raise ValueError("URL cannot be empty")

    if not URL_REGEX.match(v):
        raise ValueError("Invalid URL format. Must start with http:// or https://")

    return v


# Amazon-specific validators

ASIN_REGEX = re.compile(r'^[A-Z0-9]{10}$')


def validate_asin(v: Optional[str]) -> Optional[str]:
    """
    Validate Amazon ASIN format.

    ASIN format:
    - Exactly 10 characters
    - Uppercase letters and digits only

    Raises:
        ValueError: If ASIN format is invalid
    """
    if v is None:
        return v

    v = v.strip().upper()

    if not v:
        raise ValueError("ASIN cannot be empty")

    if not ASIN_REGEX.match(v):
        raise ValueError(
            "Invalid ASIN format. Must be 10 uppercase alphanumeric characters"
        )

    return v


MARKETPLACE_CODES = {
    'US', 'CA', 'MX',  # North America
    'UK', 'DE', 'FR', 'IT', 'ES', 'NL', 'SE', 'PL', 'TR',  # Europe
    'JP', 'AU', 'SG', 'IN', 'AE', 'SA',  # Asia Pacific & Middle East
    'BR',  # South America
}


def validate_marketplace(v: Optional[str]) -> Optional[str]:
    """
    Validate Amazon marketplace code.

    Valid codes:
    - North America: US, CA, MX
    - Europe: UK, DE, FR, IT, ES, NL, SE, PL, TR
    - Asia Pacific: JP, AU, SG, IN, AE, SA
    - South America: BR

    Raises:
        ValueError: If marketplace code is invalid
    """
    if v is None:
        return v

    v = v.strip().upper()

    if not v:
        raise ValueError("Marketplace code cannot be empty")

    if v not in MARKETPLACE_CODES:
        raise ValueError(
            f"Invalid marketplace code. Must be one of: {', '.join(sorted(MARKETPLACE_CODES))}"
        )

    return v


# Currency validation

CURRENCY_CODES = {
    'USD', 'CAD', 'MXN',  # North America
    'GBP', 'EUR',  # Europe
    'JPY', 'AUD', 'SGD', 'INR', 'AED', 'SAR',  # Asia Pacific & Middle East
    'BRL',  # South America
}


def validate_currency(v: Optional[str]) -> Optional[str]:
    """
    Validate currency code (ISO 4217).

    Common codes:
    - USD, CAD, MXN (North America)
    - GBP, EUR (Europe)
    - JPY, AUD, SGD, INR, AED, SAR (Asia Pacific)
    - BRL (South America)

    Raises:
        ValueError: If currency code is invalid
    """
    if v is None:
        return v

    v = v.strip().upper()

    if not v:
        raise ValueError("Currency code cannot be empty")

    if v not in CURRENCY_CODES:
        raise ValueError(
            f"Invalid currency code. Must be one of: {', '.join(sorted(CURRENCY_CODES))}"
        )

    return v


# Numeric validators

def validate_positive(v: Optional[float]) -> Optional[float]:
    """
    Validate that number is positive (> 0).

    Raises:
        ValueError: If number is not positive
    """
    if v is None:
        return v

    if v <= 0:
        raise ValueError("Value must be positive (greater than 0)")

    return v


def validate_non_negative(v: Optional[float]) -> Optional[float]:
    """
    Validate that number is non-negative (>= 0).

    Raises:
        ValueError: If number is negative
    """
    if v is None:
        return v

    if v < 0:
        raise ValueError("Value must be non-negative (0 or greater)")

    return v


def validate_percentage(v: Optional[float]) -> Optional[float]:
    """
    Validate that number is a valid percentage (0-100).

    Raises:
        ValueError: If not between 0 and 100
    """
    if v is None:
        return v

    if not 0 <= v <= 100:
        raise ValueError("Percentage must be between 0 and 100")

    return v


# String validators

def validate_non_empty_string(v: Optional[str]) -> Optional[str]:
    """
    Validate that string is not empty after stripping whitespace.

    Raises:
        ValueError: If string is empty or only whitespace
    """
    if v is None:
        return v

    v = v.strip()

    if not v:
        raise ValueError("String cannot be empty")

    return v


def validate_alphanumeric(v: Optional[str]) -> Optional[str]:
    """
    Validate that string contains only alphanumeric characters.

    Raises:
        ValueError: If string contains non-alphanumeric characters
    """
    if v is None:
        return v

    v = v.strip()

    if not v:
        raise ValueError("String cannot be empty")

    if not v.isalnum():
        raise ValueError("String must contain only alphanumeric characters")

    return v


def validate_slug(v: Optional[str]) -> Optional[str]:
    """
    Validate URL slug format.

    Slug format:
    - Lowercase letters, numbers, hyphens only
    - Cannot start or end with hyphen
    - No consecutive hyphens

    Raises:
        ValueError: If slug format is invalid
    """
    if v is None:
        return v

    v = v.strip().lower()

    if not v:
        raise ValueError("Slug cannot be empty")

    if not re.match(r'^[a-z0-9]+(?:-[a-z0-9]+)*$', v):
        raise ValueError(
            "Invalid slug format. Use lowercase letters, numbers, and hyphens only"
        )

    return v


# Date validators

def validate_future_date(v: Optional[date]) -> Optional[date]:
    """
    Validate that date is in the future.

    Raises:
        ValueError: If date is not in the future
    """
    if v is None:
        return v

    if v <= date.today():
        raise ValueError("Date must be in the future")

    return v


def validate_past_date(v: Optional[date]) -> Optional[date]:
    """
    Validate that date is in the past.

    Raises:
        ValueError: If date is not in the past
    """
    if v is None:
        return v

    if v >= date.today():
        raise ValueError("Date must be in the past")

    return v


def validate_date_range(
    start_date: Optional[date],
    end_date: Optional[date]
) -> tuple[Optional[date], Optional[date]]:
    """
    Validate that end_date is after start_date.

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        tuple: (start_date, end_date)

    Raises:
        ValueError: If end_date is before start_date
    """
    if start_date and end_date and end_date < start_date:
        raise ValueError("End date must be greater than or equal to start date")

    return start_date, end_date


# Password validators

def validate_password_strength(v: Optional[str]) -> Optional[str]:
    """
    Validate password strength.

    Requirements:
    - At least 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character

    Raises:
        ValueError: If password doesn't meet requirements
    """
    if v is None:
        return v

    if len(v) < 8:
        raise ValueError("Password must be at least 8 characters long")

    if not re.search(r'[A-Z]', v):
        raise ValueError("Password must contain at least one uppercase letter")

    if not re.search(r'[a-z]', v):
        raise ValueError("Password must contain at least one lowercase letter")

    if not re.search(r'\d', v):
        raise ValueError("Password must contain at least one digit")

    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
        raise ValueError("Password must contain at least one special character")

    return v


# JSON validators

def validate_json_dict(v: Any) -> dict:
    """
    Validate that value is a dictionary.

    Raises:
        ValueError: If value is not a dict
    """
    if not isinstance(v, dict):
        raise ValueError("Value must be a JSON object (dictionary)")

    return v


def validate_json_array(v: Any) -> list:
    """
    Validate that value is a list.

    Raises:
        ValueError: If value is not a list
    """
    if not isinstance(v, list):
        raise ValueError("Value must be a JSON array (list)")

    return v
