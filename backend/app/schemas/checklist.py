from datetime import date
from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.chat import Profile


class ChecklistRequest(BaseModel):
    profile: Optional[Profile] = Field(default=None, description="Known facts; anything left out is 'not sure'.")
    as_of: Optional[date] = Field(default=None, description="Date the checklist is computed for (default today).")
