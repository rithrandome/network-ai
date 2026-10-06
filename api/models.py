"""Request and response models - the API's contract.

The pipeline's own Pydantic models (schemas.py) are used as they are wherever
they fit: Candidate, Instance, Me. What's here is only what the HTTP layer adds -
request bodies, and views that bundle a model with what the UI derives from it
(pickable options, warnings, the story to ask for) so the front end never
recomputes a business rule.
"""

from typing import Literal

from pydantic import BaseModel, Field

from schemas import Candidate, Instance, Me

ExaMode = Literal["live", "offline", "mock"]
Level = Literal["off", "hint", "mention", "lead"]


# --- Search ---------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(min_length=1, description='Free text, e.g. "Jane Doe, Northwind Analytics, Austin"')
    mode: ExaMode | None = Field(default=None, description="Overrides EXA_MODE; mock when no Exa key is set")


# --- Instances -----------------------------------------------------------------

class CreateInstance(BaseModel):
    person: Candidate


class InstanceSummary(BaseModel):
    """One row of the sidebar list."""

    id: str
    title: str
    updated_at: str
    drafts: int


class StoryPrompt(BaseModel):
    """A picked credibility item comes from this entry and it has no story yet."""

    entry_key: str
    label: str


class InstanceView(BaseModel):
    instance: Instance
    options: dict[str, list[str]] = Field(description="Pickable items per ingredient")
    suggested: list[str] = Field(description="Items that are model suggestions (shown with ✦)")
    warnings: list[str] = Field(description="Knob combinations worth questioning - never enforced")
    story_prompt: StoryPrompt | None = None
    notice: str | None = Field(default=None, description="A step that partly failed but didn't block")


class SettingsPatch(BaseModel):
    """Only the fields sent are changed. A goal change also moves ingredient
    levels and the channel that were still at the old goal's defaults."""

    goal: str | None = None
    their_role: str | None = None
    status: str | None = None
    channel: str | None = None
    tone: str | None = None


class IngredientPatch(BaseModel):
    items: list[str] | None = None
    level: Level | None = Field(default=None, description="A new lead moves the old lead to mention")


class JobRequest(BaseModel):
    input: str = Field(min_length=1, description="A job link, or the pasted description")
    mode: ExaMode | None = None


class DraftRequest(BaseModel):
    refine: str = Field(default="", description='e.g. "shorter, and less formal"; empty for a fresh draft')
    base: int | None = Field(default=None, description="Version index to refine; defaults to the latest")


# --- Profile ----------------------------------------------------------------------

class ProfileView(BaseModel):
    me: Me
    strength: float = Field(ge=0, le=1)
    hint: str = Field(description="The one next thing worth adding (markdown)")


class StoryUpdate(BaseModel):
    entry_key: str
    story: str


# --- App info ----------------------------------------------------------------------

class ModelInfo(BaseModel):
    role: Literal["extract", "compose"]
    label: str
    model: str
    local: bool
    vendor: str
    used_for: str


class AppSettings(BaseModel):
    models: list[ModelInfo]
    search_mode: ExaMode = Field(description="What a search uses when the request doesn't say")
    has_exa_key: bool


class SettingDef(BaseModel):
    key: str
    label: str
    options: list[str]
    custom: bool = Field(description="Takes a typed value besides the options")


class IngredientDef(BaseModel):
    key: str
    label: str
    help: str
    placeholder: str


class ChannelDef(BaseModel):
    name: str
    max_chars: int
    has_subject: bool


class GoalRuleDef(BaseModel):
    show_status: bool
    channel: str
    levels: dict[str, Level]


class KnobRegistry(BaseModel):
    settings: list[SettingDef]
    ingredients: list[IngredientDef]
    levels: list[Level]
    channels: list[ChannelDef]
    goal_rules: dict[str, GoalRuleDef]
    role_statuses: list[str] = Field(description="Statuses where the job field applies")
