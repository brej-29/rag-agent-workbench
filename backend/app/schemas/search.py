from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., description="User query text")
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of results to return",
    )
    namespace: Optional[str] = Field(
        default=None, description="Target Pinecone namespace (defaults to env)"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filters passed directly to Pinecone search",
    )


class SearchHit(BaseModel):
    id: str
    score: float
    fields: Dict[str, Any]


class SearchResponse(BaseModel):
    namespace: str
    query: str
    top_k: int
    hits: List[SearchHit]