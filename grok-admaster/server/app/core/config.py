from pydantic_settings import BaseSettings
from typing import Optional
import os


def _parse_cors_origins(v: str) -> list[str]:
    if not v or not v.strip():
        return ["http://localhost:5173", "http://127.0.0.1:5173"]
    return [o.strip() for o in v.split(",") if o.strip()]


class Settings(BaseSettings):
    PROJECT_NAME: str = "Optimus Pryme"
    API_V1_STR: str = "/api/v1"
    ENV: str = "development"

    # Database
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "password"
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

    # JWT
    SECRET_KEY: str = "CHANGE_THIS_IN_PRODUCTION_SECRET_KEY"
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
    """Fail fast if production env is missing required secrets."""
    if (getattr(settings, "ENV", "development") or "").lower() != "production":
        return
    if not settings.SECRET_KEY or settings.SECRET_KEY == "CHANGE_THIS_IN_PRODUCTION_SECRET_KEY":
        raise ValueError("SECRET_KEY must be set in production")
    if not settings.POSTGRES_PASSWORD or settings.POSTGRES_PASSWORD == "password":
        raise ValueError("POSTGRES_PASSWORD must be set to a non-default value in production")


settings = Settings()
validate_production_settings(settings)
