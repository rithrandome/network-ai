"""Matching the sender's profile against one reach-out, in code, with no model.

  rank_credibility(me, instance)  the sender's bullets, most relevant first:
                                  word overlap with the job's title and
                                  requirements (weighted double), then with the
                                  recipient's profile. Ties keep resume order.
  shared_ground(me, person)       employers, schools and city that appear in
                                  both profiles, phrased as common-ground items
  strength(me)                    how complete the profile is, and the one next
                                  thing worth adding

Exact matches beat a model here: a shared employer is either in both texts or
it isn't, and a 1B model asked to judge it stretches ("both work in tech").
These items are merged ahead of the model's suggestions.
"""

import re

from schemas import Candidate, Instance, Me
from validators import content_words


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _mentions(text: str, name: str) -> bool:
    name = _norm(name)
    return len(name) >= 3 and f" {name} " in f" {_norm(text)} "


def rank_credibility(me: Me, instance: Instance) -> list[str]:
    items = me.credibility_items()
    job_words: set[str] = set()
    if instance.job:
        p = instance.job.posting
        job_words = content_words(" ".join([p.title or "", *p.requirements]))
    person_words = content_words(instance.person.profile_text)

    def score(item: str) -> int:
        words = content_words(item)
        return 2 * len(words & job_words) + len(words & person_words)

    return sorted(items, key=score, reverse=True)  # sorted() is stable: ties keep resume order


def shared_ground(me: Me, person: Candidate) -> list[str]:
    profile = " ".join([person.profile_text, person.headline or "", person.location or "",
                        *(f"{r.title or ''} {r.company or ''}" for r in person.roles)])
    found = []
    for org in dict.fromkeys(e.org for e in me.entries if e.kind == "job" and e.org):
        if _mentions(profile, org):
            found.append(f"Both worked at {org}")
    for school in dict.fromkeys(s.school for s in me.education if s.school):
        if _mentions(profile, school):
            found.append(f"Both studied at {school}")
    city = me.location.split(",")[0].strip()
    if city and person.location and _mentions(person.location, city):
        found.append(f"Both based in {city}")
    return found


def strength(me: Me) -> tuple[float, str]:
    """0-1, and the single next step that would help drafts most. An upload
    alone lands around 60%. That's endowed progress: the meter starts partly
    filled, and the remaining part is stories, the context drafts need most."""
    score, hint = 0.0, ""
    if me.name and me.headline:
        score += 0.15
    if me.entries:
        score += 0.25
    if me.skills:
        score += 0.1
    if me.education:
        score += 0.1
    if me.entries:
        with_story = [e for e in me.entries if e.story.strip()]
        score += 0.4 * len(with_story) / len(me.entries)
        missing = next((e for e in me.entries if not e.story.strip()), None)
        if missing:
            hint = (f"Add the story behind **{missing.label}**: the problem, the decision you made, the "
                    f"result. Drafts that use it can then say why it matters, not just that it happened.")
    if not hint and not me.entries:
        hint = "Add a job or project - credibility items come from them."
    return min(score, 1.0), hint
