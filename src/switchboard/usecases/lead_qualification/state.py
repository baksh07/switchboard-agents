"""
Data types for the lead_qualification use case
"""


from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

class LeadScore(BaseModel):
    """Facts extracted from an inbound lead, plus a first-pass qualification verdict."""

    name: str | None = Field(description="Full name of the person, if given.")
    email: str | None = Field(description="Their email address, if given.")
    title: str | None = Field(description="Job title, if given.")
    company_name: str | None = Field(description="Company name, if given.")
    company_domain: str | None = Field(
        description="Company web domain, usually the part of a work email after '@'. Null for free email providers."
    )
    fit_score: int = Field(ge=0, le=100, description="How well the lead matches our ideal customer, 0-100.")
    intent: Literal["high", "medium", "low"] = Field(description="How ready they seem to buy.")
    reasons: list[str] = Field(description="1-4 short reasons for the score.")
    action: Literal["pursue", "reject"] = Field(description="pursue = worth a sales follow-up, reject = not a fit.")



class LeadState(TypedDict, total = False):
    """
    Everything the graph knows about one lead as it moves through the nodes.
    """
    lead_text: str
    score: dict
    messages: Annotated[list[AnyMessage], add_messages]
    approved: bool
    write_rounds: int
    summary: str