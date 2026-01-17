import os

from app.core.logging import get_logger

logger = get_logger(__name__)


def get_port(default: int = 7860) -> int:
    """Return the port the application should bind to.

    - Uses the PORT environment variable when set (e.g. on Hugging Face Spaces).
    - Falls back to the provided default (7860 for Spaces compatibility).
    - Logs a message indicating the chosen port and whether we appear to be
      running inside a Hugging Face Spaces environment.
    """
    raw = os.getenv("PORT")
    try:
        port = int(raw) if raw else default
    except (TypeError, ValueError):
        port = default

    # Heuristic to detect HF Spaces: SPACE_ID or SPACE_REPO_ID are usually set.
    hf_spaces_mode = bool(os.getenv("SPACE_ID") or os.getenv("SPACE_REPO_ID"))

    logger.info(
        "Starting on port=%d hf_spaces_mode=%s",
        port,
        hf_spaces_mode,
    )

    return port