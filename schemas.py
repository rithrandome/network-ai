"""Data models shared across network-ai's pipeline steps."""

from pydantic import BaseModel, Field


class Role(BaseModel):
    title: str | None = None
    company: str | None = None
    start: str | None = None
    end: str | None = None  # None means current


class Candidate(BaseModel):
    """One possible match for the person being looked up.

    Resolution returns several of these because same-name matches are normal,
    even with a company in the query - the user picks the right one. Education
    isn't a field: the SDK drops it from the entity, but it's in profile_text.
    """

    name: str
    url: str
    headline: str | None = None
    location: str | None = None
    roles: list[Role] = Field(default_factory=list)
    profile_text: str = Field(default="", description="Full profile page as markdown - input to the brief")

    @property
    def current_role(self) -> Role | None:
        return self.roles[0] if self.roles else None


class Brief(BaseModel):
    """What you need to know about the person before reaching out."""

    summary: str = Field(description="A short, plain-language summary of who the person is")
    facts: list[str] = Field(description="Two specific, interesting facts taken from the profile")
    icebreakers: list[str] = Field(
        description="Two conversation openers that reference something specific in the profile"
    )
