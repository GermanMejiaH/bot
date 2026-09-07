"""Centralized Loguru logger configuration for DTA."""

import sys
from pathlib import Path

from loguru import logger


def setup_logger(log_level: str = "INFO", log_file: str | Path = "logs/dta.log") -> None:
    """Configure loguru handlers for stdout and file output."""
    logger.remove()

    # Terminal handler
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    # File handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_path),
        rotation="10 MB",
        retention="7 days",
        level=log_level,
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
    )



# Export standard logger
__all__ = ["logger", "setup_logger"]
