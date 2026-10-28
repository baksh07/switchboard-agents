from typing import Literal

from pydantic import BaseModel, Field


class LeadScore(BaseModel):
    name: str | None = Field(description="Full name of the person if given")
    email: str | None = Field(description="Their email address if given")
    title: str | None = Field(description="job title if given")
    company_name:str | None = Field(description="comapny name if given")
    company_domain: str | None = Field(description="comapnu web domain usually the part of a work email after @ Null fr free email providers.")
    fit_score: int = Field(ge=0, le=100, description="How well the lead matches our real customers, 0-100")
    intent: Literal['high', 'medium', 'low'] = Field(description="How ready they seem to buy")
    reasons: list[str] = Field(description="1-4 short reasons for the score")
    action:Literal["pursue", "reject"] = Field(description="pursue = worth a sales follow-up, reject= not a fit")


  