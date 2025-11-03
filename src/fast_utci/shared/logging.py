"""
Unified logging configuration for fast-utci.

Provides a simple logging setup that ensures consistent logging across all modules.
"""

import logging
import sys
from typing import Optional


def setup_logging(level: int = logging.INFO, format_string: Optional[str] = None) -> None:
    """
    Configure logging for fast-utci package.
    
    Args:
        level: Logging level (default: INFO)
        format_string: Optional custom format string. If None, uses default format.
    
    Default format: '%(levelname)s:%(name)s:%(message)s'
    """
    if format_string is None:
        format_string = '%(levelname)s:%(name)s:%(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_string,
        stream=sys.stderr,
        force=True  # Override any existing configuration
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the given module name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

