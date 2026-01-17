from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class UploadTextRequest(BaseModel):
    title: str = Field(..., description="Document title")
    source: str = Field(
        default="manual",
        description="Source label for the document (e.g. manual, docling)",
    )
    text: str = Field(..., description="Full text content of the document")
    namespace: Optional[str] = Field(
        default=None, description="Target Pinecone namespace (defaults to env)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata fields to store alongside the document",
    )


class UploadTextResponse(BaseModel):
    namespace: str
    source: str
    ingested_documents: int
    ingested_chunks: int


class NamespaceStat(BaseModel):
    vector_count: int


class DocumentsStatsResponse(BaseModel):
    namespaces: Dict[str, NamespaceStat]