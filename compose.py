"""Instance knobs -> a checked draft.

  plan(instance, me)      knobs -> MessagePlan: each selected item becomes a
                          numbered fact at its ingredient's level; Off items
                          are kept aside for the leak check, never rendered
  compose(instance, me)   SYSTEM_PROMPT + render(plan) -> model -> DraftOutput
                          -> validators -> on problems, the draft and the list
                          of problems go back as the next turn (at most 2
                          retries). A draft is never truncated to fit: a cut
                          sentence reads worse than an over-long one, so what
                          still fails is shown to the user instead.

Run:  EXA_MODE=mock uv run python compose.py
"""

import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import create_model

from knobs import CHANNELS, INGREDIENT_KEYS, ROLE_STATUSES
from llm import describe, structured
from prompts import STATUS_PHRASES, SYSTEM_PROMPT, Fact, MessagePlan, render
from schemas import DraftOutput, DraftVersion, Entry, FactUse, Instance, Me
from store import now
from validators import validate

MAX_RETRIES = 2
MAX_BACKGROUND_CHARS = 600  # enough for the gist; a small model copies long text

SUBJECT_LINE = re.compile(r"^\s*subject:\s*(.+?)\s*\n+", re.IGNORECASE)


def _recipient(instance: Instance) -> str:
    person, role = instance.person, instance.person.current_role
    parts = [person.name]
    if role and (role.title or role.company):
        parts.append(" at ".join(p for p in (role.title, role.company) if p))
    elif person.headline:
        parts.append(person.headline)
    return " - ".join(parts)


def _single_lead(levels: dict[str, str]) -> dict[str, str]:
    """Only one ingredient leads; the UI enforces this, the plan makes sure."""
    leads = [key for key in INGREDIENT_KEYS if levels.get(key) == "lead"]
    return {key: ("mention" if key in leads[1:] else level) for key, level in levels.items()}


def plan(instance: Instance, me: Me, refine: str = "", previous_draft: str = "") -> MessagePlan:
    settings = instance.settings
    levels = _single_lead({key: state.level for key, state in instance.ingredients.items()})
    status = settings.get("status", "")

    facts, off_items, storied = [], [], set()
    for key in INGREDIENT_KEYS:
        state = instance.ingredients.get(key)
        if not state:
            continue
        items = list(dict.fromkeys(i.strip() for i in state.items if i.strip()))
        if levels[key] == "off":
            off_items += items
            continue
        for n, text in enumerate(items):
            if key == "role" and status in ROLE_STATUSES:
                text = f"{text} - {STATUS_PHRASES[status]}"
            # A lead ingredient with several items leads on the first; the rest are mentioned.
            level = "mention" if levels[key] == "lead" and n > 0 else levels[key]
            fact = Fact(f"F{len(facts) + 1}", key, text, level)
            entry = me.entry_for(text) if key == "credibility" else None
            if entry:
                fact.source = entry.label
                if entry.key not in storied:  # two bullets from one job share one story
                    storied.add(entry.key)
                    fact.background = " ".join(entry.story.split())[:MAX_BACKGROUND_CHARS]
            facts.append(fact)

    # A role at Mention with nothing in it is as good as off.
    role_on = any(f.ingredient == "role" for f in facts)
    ask = instance.ingredients.get("ask")
    return MessagePlan(
        settings=settings,
        channel=CHANNELS[settings.get("channel") or "LinkedIn note"],
        recipient=_recipient(instance),
        sender_first_name=me.first_name,
        facts=facts,
        off_items=off_items,
        role_off=not role_on,
        open_ask=bool(ask and ask.level != "off" and not any(i.strip() for i in ask.items)),
        previous_draft=previous_draft,
        refine=refine.strip(),
    )


def output_schema(message_plan: MessagePlan) -> type[DraftOutput]:
    """DraftOutput narrowed to this plan, so constrained decoding enforces it:
    fact_id can only be one of the plan's IDs (gemma3:1b once wrote "])1"),
    and subject is required on channels that have one."""
    fields = {}
    if message_plan.channel.has_subject:
        fields["subject"] = (str, ...)
    ids = tuple(f.id for f in message_plan.facts)
    if ids:
        use = create_model("PlannedFactUse", __base__=FactUse, fact_id=(Literal[ids], ...))  # type: ignore[valid-type]
        fields["used"] = (list[use], ...)  # type: ignore[valid-type]
    return create_model("PlannedDraft", __base__=DraftOutput, **fields) if fields else DraftOutput


def _normalize(draft: DraftOutput) -> DraftOutput:
    """Move a "Subject: ..." line the model wrote into the body into its field."""
    match = SUBJECT_LINE.match(draft.body)
    if not match:
        return DraftOutput.model_validate(draft.model_dump())
    return DraftOutput(subject=draft.subject or match.group(1), body=draft.body[match.end():], used=draft.used)


def _feedback(problems: list[str]) -> str:
    return ("That draft has problems:\n" + "\n".join(f"- {p}" for p in problems)
            + "\nReturn the full revised draft, fixing every problem and keeping everything else.")


def compose(instance: Instance, me: Me, refine: str = "", base: DraftVersion | None = None) -> DraftVersion:
    """refine rewrites base (default: the latest draft) with the current knobs."""
    base = base or (instance.drafts[-1] if instance.drafts else None)
    previous = base.body if (refine and base) else ""
    message_plan = plan(instance, me, refine, previous)
    user_message = render(message_plan)
    # Invented-detail check: anything in the prompt is fair game, plus the sender's name.
    corpus = f"{user_message}\n{me.name}"

    model = structured("compose", output_schema(message_plan))
    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(user_message)]
    for attempt in range(1, MAX_RETRIES + 2):
        draft = _normalize(model.invoke(messages))
        problems = validate(draft, message_plan, corpus)
        if not problems or attempt > MAX_RETRIES:
            break
        messages += [AIMessage(draft.model_dump_json()), HumanMessage(_feedback(problems))]

    return DraftVersion(
        subject=draft.subject if message_plan.channel.has_subject else None,
        body=draft.body,
        used=draft.used,
        settings=dict(instance.settings),
        ingredients={k: v.model_copy(deep=True) for k, v in instance.ingredients.items()},
        refine=refine,
        problems=problems,
        attempts=attempt,
        model=describe(),
        created_at=now(),
    )


if __name__ == "__main__":
    from pipeline import find_candidates, seed_knobs
    from sources.job_posting import read_job
    from store import new_instance

    me = Me(name="Alex Tester", headline="ML engineer, ex-AWS", entries=[Entry(
        title="Associate Consultant", org="Contoso Cloud Services",
        bullets=["Built an AWS Bedrock text-to-SQL agent at ~90% accuracy"],
        story="Ops teams couldn't query AWS Config without learning its SQL dialect. I added a "
              "validator agent and a retry loop on execution errors, which got accuracy to ~90%.")])
    instance = new_instance(find_candidates("Jane Doe, Northwind Analytics")[0])
    instance.settings = {"goal": "Referral", "status": "Applied", "their_role": "Potential teammate"}
    instance.job = read_job("https://careers.northwind.example/jobs/NW-2207")
    seed_knobs(instance, me)
    print(render(plan(instance, me)), "\n---")
    draft = compose(instance, me)
    print(f"Subject: {draft.subject}\n{draft.body}\n\nused={draft.used}\n"
          f"attempts={draft.attempts} problems={draft.problems}")
