import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables from a local .env file if present
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App
    APP_NAME: str = Field(default="rag-agent-workbench")
    APP_VERSION: str = Field(default="0.1.0")

    # Pinecone
    PINECONE_API_KEY: str = Field(..., description="Pinecone API key")
    PINECONE_INDEX_NAME: str = Field(
        ..., description="Name of the Pinecone index (used for configuration checks)"
    )
    PINECONE_HOST: str = Field(
        ..., description="Pinecone index host URL for data-plane operations"
    )
    PINECONE_NAMESPACE: str = Field(
        default="dev", description="Default Pinecone namespace"
    )
    PINECONE_TEXT_FIELD: str = Field(
        default="chunk_text",
        description=(
            "Text field name used by the Pinecone integrated embedding index. "
            "For example, set to 'content' if your index field_map uses that name."
        ),
    )

    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Application log level")

    # HTTP client defaults
    HTTP_TIMEOUT_SECONDS: float = Field(
        default=10.0, description="Default timeout for outbound HTTP requests"
    )
    HTTP_MAX_RETRIES: int = Field(
        default=3, description="Max retries for outbound HTTP requests"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()  # type: ignore[call-arg]


def get_env_bool(name: str, default: bool = False) -> bool:
    """Utility to parse boolean flags from environment variables."""
    raw: Optional[str] = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}