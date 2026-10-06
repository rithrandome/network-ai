"""
network-ai pipeline: describe a person -> pick the right candidate -> brief.

  find_candidates(query)
      free-text description -> Exa people search -> candidate profiles (the
      user picks one; same-name matches are normal)
  write_brief(candidate)
      full profile markdown -> chat model -> Brief (summary, facts,
      icebreakers), schema-constrained so the UI can render each part
  seed_knobs(instance, me) / change_goal(instance, goal)
      goal rules + suggestions -> starting knob values, never overwriting
      anything the user already set
  set_setting / set_level / set_job, ingredient_options, knob_warnings,
  story_entry
      the rules behind each knob edit, shared by the Streamlit widgets and
      the API so neither front end carries its own copy

The model comes from llm.py: each role (extract, compose) picks its own
provider.

Run:  EXA_MODE=mock uv run python pipeline.py
"""

from langchain_core.prompts import PromptTemplate

from knobs import CHANNELS, DEFAULT_SETTINGS, INGREDIENT_KEYS, LEVELS, default_levels, rule_for, warnings_for
from llm import structured
from profile_match import rank_credibility
from schemas import Brief, Candidate, Entry, IngredientState, Instance, JobPosting, JobSource, Me
from sources.exa_client import find_person

BRIEF_TEMPLATE = """
Below is a person's professional profile. Produce:
1. A short, plain-language summary of who they are
2. Two specific, interesting facts about them
3. Two conversation openers that reference something specific in their profile

Use only the information in the profile - do not invent facts that aren't there.

Profile:
{profile}
"""


def find_candidates(query: str, mode: str | None = None) -> list[Candidate]:
    """query is free text, e.g. "Ashutosh Mishra, Tampa, Capgemini, Citi".
    mode overrides EXA_MODE: live | offline | mock."""
    return find_person(query, mode=mode)


def write_brief(candidate: Candidate) -> Brief:
    prompt = PromptTemplate(input_variables=["profile"], template=BRIEF_TEMPLATE)
    # Constrained decoding (see llm.py): format instructions in the prompt
    # aren't enough for a 1B model - it echoes the schema's shape instead of
    # filling it in.
    chain = prompt | structured("extract", Brief)
    return chain.invoke({"profile": candidate.profile_text})


def role_item(posting: JobPosting) -> str:
    """The role as one ingredient item: "GenAI Developer at Capgemini (job ID R-48213)"."""
    text = " at ".join(part for part in (posting.title, posting.company) if part) or "the role"
    return f"{text} (job ID {posting.job_id})" if posting.job_id else text


def seed_knobs(instance: Instance, me: Me) -> Instance:
    """Fill empty knobs from the goal's rules and the suggestions.

    Only empty values are filled, so re-running after a refresh keeps every
    choice the user made. Open suggestions (common ground, hooks, asks) are
    not selected here - the user opts in.
    """
    settings = instance.settings
    for key, value in DEFAULT_SETTINGS.items():
        settings.setdefault(key, value)
    if not settings.get("their_role") and instance.suggestions:
        settings["their_role"] = instance.suggestions.role_guess
    if not settings.get("channel"):
        settings["channel"] = rule_for(settings.get("goal", "")).channel

    levels = default_levels(settings.get("goal", ""))
    for key in INGREDIENT_KEYS:
        instance.ingredients.setdefault(key, IngredientState(level=levels[key]))

    # Your own most relevant achievement and the JD aren't guesses - start them selected.
    credibility = instance.ingredients["credibility"]
    ranked = rank_credibility(me, instance)
    if not credibility.items and ranked:
        credibility.items = [ranked[0]]
    role = instance.ingredients["role"]
    if not role.items and instance.job:
        role.items = [role_item(instance.job.posting)]
    return instance


def change_goal(instance: Instance, goal: str) -> Instance:
    """Switch goals. A level or channel the user changed stays put; one still at
    the old goal's default moves to the new goal's default."""
    old_goal = instance.settings.get("goal", "")
    old, new = default_levels(old_goal), default_levels(goal)
    if instance.settings.get("channel") == rule_for(old_goal).channel:
        instance.settings["channel"] = rule_for(goal).channel
    moved = []
    for key, state in instance.ingredients.items():
        if state.level == old.get(key):
            state.level = new[key]
            moved.append(key)
    # One lead only: a lead the user chose beats one the new defaults brought in.
    leads = [k for k, s in instance.ingredients.items() if s.level == "lead"]
    if len(leads) > 1:
        for key in leads:
            if key in moved:
                instance.ingredients[key].level = "mention"
    instance.settings["goal"] = goal
    return instance


# --- Knob edits ------------------------------------------------------------------

def set_setting(instance: Instance, key: str, value: str) -> Instance:
    if key == "goal":
        return change_goal(instance, value)
    instance.settings[key] = value
    return instance


def set_level(instance: Instance, key: str, level: str) -> Instance:
    """Only one ingredient leads: a new Lead moves the old one to Mention."""
    if level not in LEVELS:
        raise ValueError(f"Unknown level {level!r} - use one of {', '.join(LEVELS)}")
    if level == "lead":
        for other, state in instance.ingredients.items():
            if other != key and state.level == "lead":
                state.level = "mention"
    instance.ingredients.setdefault(key, IngredientState()).level = level
    return instance


def set_job(instance: Instance, job: JobSource) -> Instance:
    """A new job replaces the old one's item in the role ingredient and keeps any the user typed."""
    old = role_item(instance.job.posting) if instance.job else None
    instance.job = job
    role = instance.ingredients.get("role")
    if role is not None:
        role.items = [role_item(job.posting), *(i for i in role.items if i != old)]
    return instance


def ingredient_options(instance: Instance, me: Me) -> tuple[dict[str, list[str]], set[str]]:
    """Pickable items per ingredient, and which of them are model suggestions."""
    s, brief = instance.suggestions, instance.brief
    suggested = {
        "ask": s.ask_wording if s else [],
        "common_ground": s.common_ground if s else [],
        # The suggestion filter often leaves no hooks; the brief's facts are
        # grounded in the profile too.
        "hook": [*(s.hooks if s else []), *(brief.facts if brief else [])],
    }
    options = {**suggested,
               "credibility": rank_credibility(me, instance),
               "role": [role_item(instance.job.posting)] if instance.job else []}
    return options, {item for items in suggested.values() for item in items}


def knob_warnings(instance: Instance) -> list[str]:
    """Combinations worth questioning - shown, never enforced."""
    notes = warnings_for(instance.settings)
    channel = instance.settings.get("channel")
    if channel == "LinkedIn note":
        loud = [k for k, s in instance.ingredients.items() if s.level in ("mention", "lead") and s.items]
        if len(loud) > 2:
            notes.append(f"A {CHANNELS[channel].max_chars}-character note fits about two things at "
                         f"Mention or Lead - you have {len(loud)}.")
    return notes


def story_entry(instance: Instance, me: Me) -> Entry | None:
    """The entry worth asking a story for right now: a picked credibility item
    comes from it and it has none. One at a time."""
    credibility = instance.ingredients.get("credibility")
    if not credibility or credibility.level == "off":
        return None
    return next((e for e in map(me.entry_for, credibility.items) if e and not e.story.strip()), None)


if __name__ == "__main__":
    candidates = find_candidates("Jane Doe, Northwind Analytics")
    chosen = candidates[0]
    print(f"Chose: {chosen.name} - {chosen.headline}\n")
    print(write_brief(chosen).model_dump_json(indent=2))
