from datetime import date
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from app.schemas.chat import Profile


class ParseRequest(BaseModel):
    kind: Literal["policy", "standard"]
    title: str = Field("", max_length=300)
    text: str = Field(..., min_length=1, max_length=1_000_000)
    standard: Optional[Literal["iso-27001", "iso-42001"]] = Field(None, description="Override detection.")


class GapCheckRequest(BaseModel):
    policy_text: str = Field(..., min_length=1, max_length=200_000)
    articles: Optional[List[str]] = Field(None, max_length=12, description="Article keys or annex sections (nis2-ir/anx-1#11).")
    profile: Optional[Profile] = None
    as_of: Optional[date] = None
