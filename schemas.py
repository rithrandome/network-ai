"""Data models shared across network-ai's pipeline steps."""

from pydantic import BaseModel, Field


class Role(BaseModel):
    title: str | None = None
    company: str | None = None
    start: str | None = None
    end: str | None = None  # None means current


class Education(BaseModel):
    institution: str | None = None
    degree: str | None = None


class Candidate(BaseModel):
    """One possible match for the person being looked up.

    Resolution returns several of these because same-name matches are normal,
    even with a company in the query - the user picks the right one.
    """

    name: str
    url: str
    headline: str | None = None
    location: str | None = None
    roles: list[Role] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    profile_text: str = Field(default="", description="Full profile page as markdown - input to the brief")

    @property
    def current_role(self) -> Role | None:
        return self.roles[0] if self.roles else None

