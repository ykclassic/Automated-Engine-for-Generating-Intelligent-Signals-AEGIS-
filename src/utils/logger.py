"""
AEGIS Logger Module
Structured JSON logging for production monitoring
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import traceback


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, "extra"):
            log_obj.update(record.extra)
        
        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = traceback.format_exception(*record.exc_info)
        
        return json.dumps(log_obj)


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Setup structured logger with console and optional file output
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for logging
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers = []  # Clear existing handlers
    
    # Console handler with JSON formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)
    
    return logger


class LoggerMixin:
    """Mixin to add structured logging to classes"""
    
    def __init__(self):
        self._logger: Optional[logging.Logger] = None
    
    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._logger = setup_logger(self.__class__.__name__)
        return self._logger
    
    def log_event(
        self,
        event: str,
        level: str = "info",
        **kwargs
    ) -> None:
        """
        Log structured event with additional context
        
        Args:
            event: Event description
            level: Log level
            **kwargs: Additional context fields
        """
        log_method = getattr(self.logger, level.lower())
        
        # Create log record with extra fields
        extra = {"event": event, "context": kwargs}
        
        # Use adapter to pass extra fields
        logger_adapter = logging.LoggerAdapter(self.logger, {"extra": extra})
        log_method(event)
    
    def log_error(
        self,
        error: Exception,
        context: Optional[Dict] = None
    ) -> None:
        """
        Log error with full context
        
        Args:
            error: Exception instance
            context: Additional context
        """
        error_context = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }
        self.logger.error(
            f"Error occurred: {str(error)}",
            extra={"error_context": error_context},
            exc_info=True
        )
