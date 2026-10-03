"""Data models shared across network-ai's pipeline steps."""

from typing import Literal

from pydantic import BaseModel, Field

from knobs import THEIR_ROLE


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


# --- You ---------------------------------------------------------------------

class Entry(BaseModel):
    """One job or project - the unit the user deepens over time.

    The upload fills title/org/dates/bullets; `story` is the walkthrough the
    resume has no room for (the problem, the decisions, the result). It's
    optional and asked for when a draft is about to use this entry.
    """

    kind: Literal["job", "project"] = "job"
    title: str = ""
    org: str = Field(default="", description="Company for a job; course, client or 'Personal' for a project")
    location: str = ""
    start: str = ""
    end: str = ""
    bullets: list[str] = Field(default_factory=list)
    unverified: list[str] = Field(default_factory=list,
                                  description="Bullets the reader couldn't find in the source - not offered until kept")
    story: str = ""

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.org}:{self.title}".lower()

    @property
    def label(self) -> str:
        if self.kind == "job":
            return " at ".join(p for p in (self.title, self.org) if p) or "Untitled role"
        return f"{self.title} ({self.org})" if self.org else (self.title or "Untitled project")

    @property
    def offered(self) -> list[str]:
        return [b for b in self.bullets if b not in self.unverified]


class Education(BaseModel):
    school: str = ""
    degree: str = Field(default="", description="e.g. B.Math, MS")
    field: str = Field(default="", description="e.g. Computer Science")
    start: str = ""
    end: str = ""


class Me(BaseModel):
    """The sender, built from an uploaded resume or LinkedIn PDF and deepened
    over time. Shared by every instance."""

    name: str = ""
    headline: str = ""
    location: str = ""
    summary: str = ""
    entries: list[Entry] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list,
                                  description="Extra achievements typed by hand, one per item")
    resume_text: str = Field(default="", description="Text of the last upload - the source bullets are checked against")
    source: str = ""

    @property
    def first_name(self) -> str:
        return self.name.split()[0] if self.name.strip() else ""

    def credibility_items(self) -> list[str]:
        """Everything that can be picked as credibility: checked bullets, then hand-typed highlights."""
        items = [b for entry in self.entries for b in entry.offered] + self.highlights
        return list(dict.fromkeys(i for i in items if i.strip()))

    def entry_for(self, item: str) -> Entry | None:
        return next((e for e in self.entries if item in e.bullets), None)


# --- The role ------------------------------------------------------------------

class JobPosting(BaseModel):
    """A job description, pasted or fetched from a link, reduced to what a
    message can use."""

    title: str | None = Field(default=None, description="Job title exactly as posted")
    company: str | None = Field(default=None, description="Hiring company")
    job_id: str | None = Field(default=None, description="Requisition / job ID if the posting shows one")
    location: str | None = None
    requirements: list[str] = Field(default_factory=list,
                                    description="Up to five of the most important requirements, short phrases")


class JobSource(BaseModel):
    """What the user gave us, kept so a JD is never fetched or parsed twice."""

    input: str
    url: str | None = None
    text: str = ""
    posting: JobPosting


# --- Suggestions (one model call after the goal) -------------------------------

# Constrained decoding enforces this list - the model can only pick a role we
# can count on later, never invent a category.
TheirRole = Literal[tuple(THEIR_ROLE.options)]  # type: ignore[valid-type]


class Suggestions(BaseModel):
    role_guess: TheirRole = Field(description="Which of the fixed categories this person is to the sender")
    role_reason: str = Field(description="One short phrase quoting the profile evidence for role_guess")
    common_ground: list[str] = Field(
        description="Things the sender and this person genuinely share (school, employer, city, community), "
                    "each a short phrase. Empty if there is nothing real.")
    hooks: list[str] = Field(description="Up to three specific things about this person or their work to reference")
    ask_wording: list[str] = Field(description="Two or three concrete asks that fit the goal, each a short phrase")


# --- The instance --------------------------------------------------------------

class IngredientState(BaseModel):
    items: list[str] = Field(default_factory=list)
    level: str = "off"


class FactUse(BaseModel):
    fact_id: str = Field(description="ID of a fact from <facts>, e.g. F1")
    level: Literal["hint", "mention", "lead"]


class DraftOutput(BaseModel):
    """What the model returns. `used` makes the draft checkable."""

    subject: str | None = Field(default=None, description="Subject line, only for channels that have one")
    body: str = Field(description="The message itself")
    used: list[FactUse] = Field(description="Every fact used in the body and the level it was used at")


class DraftVersion(BaseModel):
    """One generated draft, with the knobs that produced it."""

    subject: str | None = None
    body: str
    used: list[FactUse] = Field(default_factory=list)
    settings: dict[str, str]
    ingredients: dict[str, IngredientState]
    refine: str = ""
    problems: list[str] = Field(default_factory=list, description="Checks still failing after retries")
    attempts: int = 1
    model: str = ""
    created_at: str


class Instance(BaseModel):
    """One person you're reaching out to - saved until deleted, and the
    playground every regenerate builds on."""

    id: str
    created_at: str
    updated_at: str
    person: Candidate
    brief: Brief | None = None
    settings: dict[str, str] = Field(default_factory=dict)
    ingredients: dict[str, IngredientState] = Field(default_factory=dict)
    job: JobSource | None = None
    suggestions: Suggestions | None = None
    drafts: list[DraftVersion] = Field(default_factory=list)

    @property
    def title(self) -> str:
        goal = self.settings.get("goal")
        return f"{self.person.name} ({goal.lower()})" if goal else self.person.name
