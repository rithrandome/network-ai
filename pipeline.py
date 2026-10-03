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

The model comes from llm.py: local Ollama by default, Claude with
LLM_PROVIDER=anthropic.

Run:  EXA_MODE=mock uv run python pipeline.py
"""

from langchain_core.prompts import PromptTemplate

from knobs import DEFAULT_SETTINGS, INGREDIENT_KEYS, default_levels, rule_for
from llm import structured
from schemas import Brief, Candidate, IngredientState, Instance, JobPosting, Me
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

    # Your own highlights and the JD aren't guesses - start them selected.
    credibility = instance.ingredients["credibility"]
    if not credibility.items and me.highlights:
        credibility.items = [me.highlights[0]]
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


if __name__ == "__main__":
    candidates = find_candidates("Jane Doe, Northwind Analytics")
    chosen = candidates[0]
    print(f"Chose: {chosen.name} - {chosen.headline}\n")
    print(write_brief(chosen).model_dump_json(indent=2))
