"""
Logging configuration for the application.

Purpose:
- Centralized logging setup
- Structured logging for analysis
- Multiple output formats (console, file, JSON)
- Token usage tracking through logs
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime


class TokenUsageFilter(logging.Filter):
    """
    Custom filter to track token usage in logs.
    
    Adds token usage information to log records.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add token usage info if available."""
        if not hasattr(record, 'tokens'):
            record.tokens = 0
        return True


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    Outputs logs as JSON for easy parsing and analysis.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add token usage if available
        if hasattr(record, 'tokens'):
            log_data['tokens'] = record.tokens
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


def setup_logging(
    level: str = 'INFO',
    log_file: Optional[str] = None,
    json_format: bool = False,
    log_dir: str = 'data/logs'
) -> None:
    """
    Configure logging for the application.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional log file name (created in log_dir)
        json_format: Whether to use JSON formatting
        log_dir: Directory for log files
    
    Usage:
        # Basic setup
        setup_logging(level='INFO')
        
        # With file logging
        setup_logging(level='DEBUG', log_file='app.log')
        
        # With JSON formatting
        setup_logging(level='INFO', json_format=True, log_file='app.json')
    """
    # Create log directory if needed
    if log_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        log_file_path = log_path / log_file
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Set formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(formatter)
    console_handler.addFilter(TokenUsageFilter())
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        file_handler.addFilter(TokenUsageFilter())
        root_logger.addHandler(file_handler)
    
    # Log configuration
    root_logger.info(
        f"Logging configured: level={level}, file={log_file}, json={json_format}"
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Logger instance
    
    Usage:
        logger = get_logger(__name__)
        logger.info("Message")
    """
    return logging.getLogger(name)


# Convenience function for logging with token usage
def log_with_tokens(logger: logging.Logger, level: str, 
                   message: str, tokens: int) -> None:
    """
    Log a message with token usage information.
    
    Args:
        logger: Logger instance
        level: Log level ('debug', 'info', 'warning', 'error')
        message: Log message
        tokens: Number of tokens used
    
    Usage:
        log_with_tokens(logger, 'info', 'Processed input', tokens=150)
    """
    log_method = getattr(logger, level.lower())
    log_method(message, extra={'tokens': tokens})
