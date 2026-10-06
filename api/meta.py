"""App-wide routes: which models run, the knob registry, people search."""

import os

from fastapi import APIRouter

from api.errors import upstream
from api.models import (AppSettings, ChannelDef, ExaMode, GoalRuleDef, IngredientDef, KnobRegistry, ModelInfo,
                        SearchRequest, SettingDef)
from knobs import CHANNELS, CUSTOM_OK, GOAL_RULES, INGREDIENTS, LEVELS, ROLE_STATUSES, SETTINGS, default_levels
from llm import describe, is_local, vendor
from pipeline import find_candidates
from schemas import Candidate

router = APIRouter(tags=["app"])

USED_FOR = {
    "extract": "your resume, job postings, the person's profile, knob suggestions",
    "compose": "every draft, including retries",
}


def search_mode(requested: ExaMode | None = None) -> ExaMode | None:
    """The request's mode, else mock when there's no key (as the Streamlit app
    does), else None so EXA_MODE decides."""
    if requested:
        return requested
    return None if os.environ.get("EXA_API_KEY") else "mock"


@router.get("/settings")
def get_settings() -> AppSettings:
    models = [ModelInfo(role=role, label=label, model=describe(role), local=is_local(role), vendor=vendor(role),
                        used_for=USED_FOR[role])
              for role, label in (("extract", "Reading"), ("compose", "Writing"))]
    return AppSettings(models=models, search_mode=search_mode() or os.environ.get("EXA_MODE", "live"),
                       has_exa_key=bool(os.environ.get("EXA_API_KEY")))


@router.get("/knobs")
def get_knobs() -> KnobRegistry:
    return KnobRegistry(
        settings=[SettingDef(key=s.key, label=s.label, options=s.options, custom=s.key in CUSTOM_OK)
                  for s in SETTINGS],
        ingredients=[IngredientDef(key=i.key, label=i.label, help=i.help, placeholder=i.placeholder)
                     for i in INGREDIENTS],
        levels=LEVELS,
        channels=[ChannelDef(name=c.name, max_chars=c.max_chars, has_subject=c.has_subject)
                  for c in CHANNELS.values()],
        goal_rules={goal: GoalRuleDef(show_status=rule.show_status, channel=rule.channel,
                                      levels=default_levels(goal))
                    for goal, rule in GOAL_RULES.items()},
        role_statuses=sorted(ROLE_STATUSES),
    )


@router.post("/search")
def search_people(request: SearchRequest) -> list[Candidate]:
    with upstream("Search"):
        return find_candidates(request.query, mode=search_mode(request.mode))
