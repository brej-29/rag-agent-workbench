import logging
from typing import Optional


def configure_logging(log_level: str) -> None:
    """Configure root logging for the application.

    This keeps configuration minimal while ensuring consistent formatting.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module-level logger."""
    return logging.getLogger(name or "app")