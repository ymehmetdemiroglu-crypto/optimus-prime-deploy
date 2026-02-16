"""
OpenAPI Examples

Provides comprehensive examples for API documentation (Swagger UI).

Usage:
    from app.api.schemas.examples import AccountExamples

    @app.post("/accounts", responses=AccountExamples.create_responses)
    async def create_account(account: AccountCreate):
        ...
"""

from typing import Dict, Any


class AccountExamples:
    """
    OpenAPI examples for Account endpoints.
    """

    # Request examples
    create_request = {
        "example": {
            "name": "My Amazon Seller Account",
            "marketplace": "US",
            "seller_id": "A2EXAMPLE123",
            "status": "active"
        }
    }

    update_request = {
        "example": {
            "name": "Updated Account Name",
            "status": "paused"
        }
    }

    # Response examples
    single_response = {
        "200": {
            "description": "Account retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "data": {
                            "id": 1,
                            "name": "My Amazon Seller Account",
                            "marketplace": "US",
                            "seller_id": "A2EXAMPLE123",
                            "status": "active",
                            "created_at": "2026-02-15T10:30:00Z",
                            "updated_at": "2026-02-15T10:30:00Z"
                        },
                        "message": "Account retrieved successfully",
                        "timestamp": "2026-02-16T10:30:00Z"
                    }
                }
            }
        },
        "404": {
            "description": "Account not found",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": {
                            "code": "NOT_FOUND",
                            "message": "Account with ID 999 not found"
                        },
                        "message": "Account not found",
                        "timestamp": "2026-02-16T10:30:00Z",
                        "request_id": "550e8400-e29b-41d4-a716-446655440000"
                    }
                }
            }
        }
    }

    list_response = {
        "200": {
            "description": "Accounts retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "data": [
                            {
                                "id": 1,
                                "name": "Account 1",
                                "marketplace": "US",
                                "status": "active"
                            },
                            {
                                "id": 2,
                                "name": "Account 2",
                                "marketplace": "UK",
                                "status": "paused"
                            }
                        ],
                        "pagination": {
                            "total": 25,
                            "count": 2,
                            "per_page": 20,
                            "current_page": 1,
                            "total_pages": 2,
                            "has_next": True,
                            "has_prev": False
                        },
                        "message": "Accounts retrieved successfully",
                        "timestamp": "2026-02-16T10:30:00Z"
                    }
                }
            }
        }
    }

    create_responses = {
        "201": {
            "description": "Account created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "data": {
                            "id": 1,
                            "name": "My Amazon Seller Account",
                            "marketplace": "US",
                            "seller_id": "A2EXAMPLE123",
                            "status": "active",
                            "created_at": "2026-02-16T10:30:00Z",
                            "updated_at": "2026-02-16T10:30:00Z"
                        },
                        "message": "Account created successfully",
                        "timestamp": "2026-02-16T10:30:00Z"
                    }
                }
            }
        },
        "400": {
            "description": "Validation error",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": "Invalid marketplace code. Must be one of: US, UK, DE, FR...",
                            "field": "marketplace"
                        },
                        "message": "Validation failed",
                        "timestamp": "2026-02-16T10:30:00Z"
                    }
                }
            }
        },
        "409": {
            "description": "Conflict (duplicate account)",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": {
                            "code": "CONFLICT",
                            "message": "Account with seller_id 'A2EXAMPLE123' already exists"
                        },
                        "message": "Duplicate account",
                        "timestamp": "2026-02-16T10:30:00Z"
                    }
                }
            }
        }
    }


class CredentialExamples:
    """
    OpenAPI examples for Credential endpoints.
    """

    create_request = {
        "example": {
            "account_id": 1,
            "client_id": "amzn1.application-oa2-client.abc123",
            "client_secret": "amzn1.application-oa2-client-secret.xyz789",
            "refresh_token": "Atzr|IwEBIJ..."
        }
    }

    single_response = {
        "200": {
            "description": "Credentials retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "data": {
                            "id": 1,
                            "account_id": 1,
                            "marketplace_id": "ATVPDKIKX0DER",
                            "region": "us-east-1",
                            "client_id": "amzn1.application-oa2-client.***123",
                            "refresh_token": "***REDACTED***",
                            "client_secret": "***REDACTED***",
                            "created_at": "2026-02-15T10:30:00Z",
                            "updated_at": "2026-02-15T10:30:00Z"
                        },
                        "message": "Credentials retrieved successfully (sensitive data redacted)",
                        "timestamp": "2026-02-16T10:30:00Z"
                    }
                }
            }
        }
    }


class CampaignExamples:
    """
    OpenAPI examples for Campaign endpoints.
    """

    create_request = {
        "example": {
            "account_id": 1,
            "name": "Q1 Product Launch Campaign",
            "campaign_type": "SPONSORED_PRODUCTS",
            "targeting_type": "AUTO",
            "daily_budget": 50.00,
            "status": "ENABLED"
        }
    }

    performance_response = {
        "200": {
            "description": "Campaign performance retrieved",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "data": {
                            "campaign_id": 123456789,
                            "name": "Q1 Product Launch Campaign",
                            "metrics": {
                                "impressions": 45230,
                                "clicks": 892,
                                "ctr": 1.97,
                                "spend": 234.56,
                                "sales": 1847.32,
                                "acos": 12.7,
                                "roas": 7.87,
                                "conversions": 23
                            },
                            "date_range": {
                                "start": "2026-02-01",
                                "end": "2026-02-15"
                            }
                        },
                        "message": "Performance data retrieved successfully",
                        "timestamp": "2026-02-16T10:30:00Z"
                    }
                }
            }
        }
    }


class DashboardExamples:
    """
    OpenAPI examples for Dashboard endpoints.
    """

    summary_response = {
        "200": {
            "description": "Dashboard summary",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "data": {
                            "total_accounts": 5,
                            "active_campaigns": 23,
                            "total_spend_30d": 12847.32,
                            "total_sales_30d": 87234.56,
                            "average_acos": 14.7,
                            "average_roas": 6.8,
                            "top_campaigns": [
                                {
                                    "id": 123,
                                    "name": "Best Seller Campaign",
                                    "sales": 15234.00,
                                    "acos": 11.2
                                }
                            ],
                            "alerts": [
                                {
                                    "type": "HIGH_ACOS",
                                    "campaign_id": 456,
                                    "message": "Campaign ACOS above 20%",
                                    "severity": "warning"
                                }
                            ]
                        },
                        "message": "Dashboard summary generated",
                        "timestamp": "2026-02-16T10:30:00Z"
                    }
                }
            }
        }
    }


class ErrorExamples:
    """
    Common error response examples.
    """

    validation_error = {
        "400": {
            "description": "Validation error",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": "Invalid email format",
                            "field": "email"
                        },
                        "message": "Validation failed",
                        "timestamp": "2026-02-16T10:30:00Z",
                        "request_id": "550e8400-e29b-41d4-a716-446655440000"
                    }
                }
            }
        }
    }

    unauthorized = {
        "401": {
            "description": "Unauthorized",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": {
                            "code": "UNAUTHORIZED",
                            "message": "Authentication required"
                        },
                        "message": "Unauthorized",
                        "timestamp": "2026-02-16T10:30:00Z"
                    }
                }
            }
        }
    }

    forbidden = {
        "403": {
            "description": "Forbidden",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": {
                            "code": "FORBIDDEN",
                            "message": "Insufficient permissions to access this resource"
                        },
                        "message": "Forbidden",
                        "timestamp": "2026-02-16T10:30:00Z"
                    }
                }
            }
        }
    }

    not_found = {
        "404": {
            "description": "Resource not found",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": {
                            "code": "NOT_FOUND",
                            "message": "Resource with ID 123 not found"
                        },
                        "message": "Not found",
                        "timestamp": "2026-02-16T10:30:00Z"
                    }
                }
            }
        }
    }

    internal_error = {
        "500": {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": {
                            "code": "INTERNAL_ERROR",
                            "message": "An unexpected error occurred"
                        },
                        "message": "Internal server error",
                        "timestamp": "2026-02-16T10:30:00Z",
                        "request_id": "550e8400-e29b-41d4-a716-446655440000"
                    }
                }
            }
        }
    }


def merge_responses(*response_dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple response dictionaries for OpenAPI docs.

    Usage:
        responses = merge_responses(
            AccountExamples.single_response,
            ErrorExamples.not_found,
            ErrorExamples.internal_error
        )

        @app.get("/accounts/{id}", responses=responses)
        async def get_account(id: int):
            ...
    """
    merged = {}
    for response_dict in response_dicts:
        merged.update(response_dict)
    return merged
