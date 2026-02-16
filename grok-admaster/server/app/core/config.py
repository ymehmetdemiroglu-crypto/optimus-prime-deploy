from pydantic_settings import BaseSettings
from typing import Optional
import os


def _parse_cors_origins(v: str) -> list[str]:
    """Parse CORS origins from comma-separated string with validation."""
    if not v or not v.strip():
        return ["http://localhost:5173", "http://127.0.0.1:5173"]

    origins = [o.strip() for o in v.split(",") if o.strip()]

    # Validate origins - reject wildcards in production
    for origin in origins:
        if "*" in origin:
            raise ValueError(
                f"Wildcard CORS origin '{origin}' is not allowed. "
                "Use explicit origins for security."
            )
        if not origin.startswith(("http://", "https://")):
            raise ValueError(
                f"Invalid CORS origin '{origin}'. Must start with http:// or https://"
            )

    return origins


class Settings(BaseSettings):
    PROJECT_NAME: str = "Optimus Pryme"
    API_V1_STR: str = "/api/v1"
    ENV: str = "development"

    # Database
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str  # No default - must be set via environment
    POSTGRES_DB: str = "optimus_pryme"
    POSTGRES_PORT: int = 5432
    
    # Direct URL override
    DATABASE_URL: Optional[str] = None

    # CORS (comma-separated origins)
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Computed Database URL
    @property
    def ASYNC_DATABASE_URL(self) -> str:
        if self.DATABASE_URL:
            if self.DATABASE_URL.startswith("postgresql://"):
                return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
            return self.DATABASE_URL
            
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def CORS_ORIGINS_LIST(self) -> list[str]:
        return _parse_cors_origins(os.getenv("CORS_ORIGINS", self.CORS_ORIGINS))

    # JWT - SECRET_KEY must be set via environment variable
    SECRET_KEY: str  # No default - REQUIRED for security
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Seed
    SEED_DB: bool = False

    # AI & Intelligence APIs
    OPENROUTER_API_KEY: Optional[str] = None
    
    # DataForSEO (Competitor Intelligence)
    DATAFORSEO_LOGIN: Optional[str] = None
    DATAFORSEO_PASSWORD: Optional[str] = None
    
    # RapidAPI (Fallback)
    RAPIDAPI_KEY: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


def validate_production_settings(settings: Settings) -> None:
    """Validate critical security settings in all environments."""
    errors = []

    # Always validate SECRET_KEY
    if not settings.SECRET_KEY:
        errors.append("SECRET_KEY is required. Set it via environment variable.")
    elif len(settings.SECRET_KEY) < 32:
        errors.append("SECRET_KEY must be at least 32 characters long for security.")

    # Always validate POSTGRES_PASSWORD
    if not settings.POSTGRES_PASSWORD:
        errors.append("POSTGRES_PASSWORD is required. Set it via environment variable.")
    elif settings.POSTGRES_PASSWORD == "password":
        errors.append("POSTGRES_PASSWORD cannot be 'password'. Use a strong password.")

    # Additional production-only checks
    is_production = (getattr(settings, "ENV", "development") or "").lower() == "production"
    if is_production:
        if not settings.OPENROUTER_API_KEY:
            errors.append("OPENROUTER_API_KEY should be set in production for AI features.")

    if errors:
        error_msg = "Configuration validation failed:\n  - " + "\n  - ".join(errors)
        raise ValueError(error_msg)


settings = Settings()
validate_production_settings(settings)
