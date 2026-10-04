"""Data models shared across network-ai's pipeline steps."""

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from knobs import THEIR_ROLE

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def parse_month_year(text: str) -> tuple[int, int] | None:
    """"Mar 2027" -> (2027, 3); "2027" -> (2027, 12) (counts the whole year); else None."""
    year = re.search(r"\b(19|20)\d{2}\b", text)
    if not year:
        return None
    month = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", text.lower())
    return int(year.group()), MONTHS[month.group(1)] if month else 12


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


def looped(text: str, run: int = 8) -> bool:
    """True if any run of words repeats - the mark of a model stuck in a loop."""
    words = text.lower().split()
    runs = [tuple(words[i:i + run]) for i in range(len(words) - run + 1)]
    return len(runs) != len(set(runs))


class Capped(BaseModel):
    """A model the model fills in, with every list capped by max_length.

    A small model can fall into a loop ("They have experience in the field
    of...") and repeat an item until it runs out of tokens, cutting the JSON
    off mid-string. The cap is in the JSON schema, so constrained decoding
    (Ollama's grammar) closes the list once it's full. Providers that don't
    enforce maxItems are trimmed here before validation instead of failing.
    A string that loops until its own length cap (see llm.py) is dropped.
    """

    @model_validator(mode="before")
    @classmethod
    def _trim_lists(cls, data):
        if not isinstance(data, dict):
            return data
        for name, field in cls.model_fields.items():
            cap = next((m.max_length for m in field.metadata if getattr(m, "max_length", None)), None)
            value = data.get(name)
            if isinstance(value, list):
                value = [v for v in value if not (isinstance(v, str) and looped(v))]
                data = {**data, name: value[:cap] if cap else value}
        return data


class Brief(Capped):
    """What you need to know about the person before reaching out."""

    summary: str = Field(description="A short, plain-language summary of who the person is")
    facts: list[str] = Field(max_length=3, description="Two specific, interesting facts taken from the profile")
    icebreakers: list[str] = Field(
        max_length=3,
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
    """One degree. `degree` is the line as written - "B.Tech in Instrumentation
    and Control Engineering, Minor in Computer Applications" - not split into
    degree/major/minor, which resumes write too many ways to split reliably."""

    school: str = ""
    degree: str = Field(default="", description="The degree line as written, majors and minors included")
    grade: str = Field(default="", description="GPA, CGPA or percentage as written, e.g. GPA: 4.0")
    start: str = ""
    end: str = ""

    @model_validator(mode="before")
    @classmethod
    def _fold_field(cls, data):
        """Profiles saved before degree took the whole line had a separate field."""
        if isinstance(data, dict) and data.get("field"):
            data = dict(data)
            data["degree"] = ", ".join(p for p in (data.get("degree"), data.pop("field")) if p)
        return data


class Certification(BaseModel):
    name: str = Field(default="", description="Certification name, e.g. AWS Certified Machine Learning - Specialty")
    issuer: str = Field(default="", description="Who issued it, e.g. Amazon Web Services")
    issued: str = Field(default="", description="When it was earned, as written, e.g. Mar 2024")
    expires: str = Field(default="", description="When it expires, as written; empty if it doesn't")

    @model_validator(mode="before")
    @classmethod
    def _date_to_issued(cls, data):
        if isinstance(data, dict) and "date" in data:
            data = dict(data)
            data.setdefault("issued", data.pop("date"))
        return data

    @property
    def expired(self) -> bool:
        """True only when the expiry date is readable and in the past."""
        found = parse_month_year(self.expires)
        return bool(found) and found < (date.today().year, date.today().month)

    @property
    def label(self) -> str:
        return f"{self.name} ({self.issuer})" if self.issuer and self.issuer.lower() not in self.name.lower() \
            else self.name


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
    certifications: list[Certification] = Field(default_factory=list,
                                                description="From the file, plus any added by hand")
    highlights: list[str] = Field(default_factory=list,
                                  description="Extra achievements typed by hand, one per item")
    resume_text: str = Field(default="", description="Text of the last upload - the source bullets are checked against")
    source: str = ""

    @property
    def first_name(self) -> str:
        return self.name.split()[0] if self.name.strip() else ""

    def credibility_items(self) -> list[str]:
        """Everything that can be picked as credibility: checked bullets, certifications,
        then hand-typed highlights."""
        items = ([b for entry in self.entries for b in entry.offered]
                 + [f"Certified: {c.label}" for c in self.certifications if c.name and not c.expired]
                 + self.highlights)
        return list(dict.fromkeys(i for i in items if i.strip()))

    def entry_for(self, item: str) -> Entry | None:
        return next((e for e in self.entries if item in e.bullets), None)


# --- The role ------------------------------------------------------------------

class JobPosting(Capped):
    """A job description, pasted or fetched from a link, reduced to what a
    message can use."""

    title: str | None = Field(default=None, description="Job title exactly as posted")
    company: str | None = Field(default=None, description="Hiring company")
    job_id: str | None = Field(default=None, description="Requisition / job ID if the posting shows one")
    location: str | None = None
    requirements: list[str] = Field(default_factory=list, max_length=8,
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


class Suggestions(Capped):
    role_guess: TheirRole = Field(description="Which of the fixed categories this person is to the sender")
    role_reason: str = Field(description="One short phrase quoting the profile evidence for role_guess")
    common_ground: list[str] = Field(
        max_length=5,
        description="Things the sender and this person genuinely share (school, employer, city, community), "
                    "each a short phrase. Empty if there is nothing real.")
    hooks: list[str] = Field(max_length=3, description="Up to three specific things about this person or their work to reference")
    ask_wording: list[str] = Field(max_length=3, description="Two or three concrete asks that fit the goal, each a short phrase")


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
