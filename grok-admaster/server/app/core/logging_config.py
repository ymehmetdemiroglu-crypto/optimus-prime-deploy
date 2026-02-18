"""
Structured logging configuration for Optimus Prime.

Provides:
- JSON-formatted logs for production
- Human-readable logs for development
- Correlation IDs for request tracing
- Consistent log formatting across all modules
"""

import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict
from contextvars import ContextVar

# Context variable for correlation ID (thread-safe)
correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging in production."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add correlation ID if present
        corr_id = correlation_id.get()
        if corr_id:
            log_data["correlation_id"] = corr_id

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


class ColoredConsoleFormatter(logging.Formatter):
    """Human-readable colored formatter for development."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        color = self.COLORS.get(record.levelname, self.RESET)

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")

        # Build log message
        parts = [
            f"{color}{timestamp}{self.RESET}",
            f"{color}{self.BOLD}{record.levelname:8}{self.RESET}",
            f"{record.name}:{record.lineno}",
            f"{record.getMessage()}",
        ]

        # Add correlation ID if present
        corr_id = correlation_id.get()
        if corr_id:
            parts.insert(3, f"[{corr_id[:8]}]")

        log_line = " | ".join(parts)

        # Add exception info if present
        if record.exc_info:
            log_line += "\n" + self.formatException(record.exc_info)

        return log_line


def setup_logging(env: str = "development", level: str = "INFO") -> None:
    """Configure logging for the application.

    Args:
        env: Environment (development/production)
        level: Minimum log level (DEBUG/INFO/WARNING/ERROR)
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Use JSON formatter in production, colored formatter in development
    if env.lower() == "production":
        formatter = StructuredFormatter()
    else:
        formatter = ColoredConsoleFormatter()

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Configure third-party loggers to be less verbose
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Log initialization
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: env={env}, level={level}")


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds extra fields to all log records."""

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Add extra fields to log record."""
        # Get extra fields from kwargs or default to empty dict
        extra_fields = kwargs.get("extra", {})

        # Add correlation ID if present
        corr_id = correlation_id.get()
        if corr_id:
            extra_fields["correlation_id"] = corr_id

        # Wrap extra fields
        kwargs["extra"] = {"extra_fields": extra_fields}

        return msg, kwargs


def get_logger(name: str) -> LoggerAdapter:
    """Get a logger with automatic correlation ID support.

    Args:
        name: Logger name (usually __name__)

    Returns:
        LoggerAdapter with extra field support
    """
    logger = logging.getLogger(name)
    return LoggerAdapter(logger, {})


def set_correlation_id(corr_id: str) -> None:
    """Set correlation ID for current context.

    This should be called at the start of each request.

    Args:
        corr_id: Correlation ID (request ID, trace ID, etc.)
    """
    correlation_id.set(corr_id)


def get_correlation_id() -> str:
    """Get current correlation ID.

    Returns:
        Current correlation ID or empty string
    """
    return correlation_id.get()
