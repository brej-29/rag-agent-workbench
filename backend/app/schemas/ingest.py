from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ArxivIngestRequest(BaseModel):
    query: str = Field(..., description="Search query for arXiv")
    max_docs: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Maximum number of documents to fetch (capped at 20)",
    )
    namespace: Optional[str] = Field(
        default=None, description="Target Pinecone namespace (defaults to env)"
    )
    category: Optional[str] = Field(
        default=None,
        description="Optional category label for ingested papers",
    )


class OpenAlexIngestRequest(BaseModel):
    query: str = Field(..., description="Search query for OpenAlex works")
    max_docs: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Maximum number of documents to fetch (capped at 20)",
    )
    namespace: Optional[str] = Field(
        default=None, description="Target Pinecone namespace (defaults to env)"
    )
    mailto: str = Field(
        ...,
        description="Contact email passed to OpenAlex via the mailto query parameter",
    )


class WikiIngestRequest(BaseModel):
    titles: List[str] = Field(
        ...,
        description="List of Wikipedia page titles (first 20 will be used)",
    )
    namespace: Optional[str] = Field(
        default=None, description="Target Pinecone namespace (defaults to env)"
    )


class IngestResponse(BaseModel):
    namespace: str
    source: str
    ingested_documents: int
    ingested_chunks: int
    skipped_documents: int
    details: Optional[Dict[str, Any]] = None