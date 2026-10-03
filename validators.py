"""Deterministic checks on a draft. Each returns a list of problems, worded as
instructions the model can act on - they go straight back as retry feedback.

Only what code can check reliably lives here. Whether a HINT was subtle enough
or the tone landed is a judgment call, left to the eval's LLM judge.
"""

import re

from prompts import MessagePlan
from schemas import DraftOutput

STOPWORDS = {"the", "and", "for", "with", "from", "that", "this", "their", "your", "both",
             "about", "into", "have", "work", "working", "experience", "team", "sender",
             "has", "are", "was", "who", "would", "could", "you", "our"}

GREETING = re.compile(r"^\s*(hi|hello|hey|dear)\b[^,\n]*[,\n]\s*", re.IGNORECASE)
# Durations ("a 15-minute chat") are an ask's shape, not an invented claim.
DURATION = re.compile(r"\b\d+[- ]?(minute|min)s?\b", re.IGNORECASE)
# Capitalized words the model may use without a source.
ALLOWED_CAPS = {"i", "linkedin", "thanks", "thank", "best", "regards", "cheers", "hi", "hello"}
APPLIED_WORDS = re.compile(r"\b(applied|applying|application)\b", re.IGNORECASE)


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3 and w not in STOPWORDS}


def grounded(item: str, text: str) -> bool:
    """Most of the item's content words (all of a two-word item) appear in text."""
    words = content_words(item)
    return bool(words) and len(words & content_words(text)) * 2 > len(words)


def first_sentence(body: str) -> str:
    rest = GREETING.sub("", body, count=1)
    return re.split(r"(?<=[.!?])\s", rest.strip(), maxsplit=1)[0]


def check_length(draft: DraftOutput, plan: MessagePlan) -> list[str]:
    ch, problems = plan.channel, []
    if len(draft.body) > ch.max_chars:
        problems.append(f"The body is {len(draft.body)} characters; the {ch.name} limit is {ch.max_chars}. "
                        f"Cut it to under {ch.max_chars} - drop HINT facts and filler first.")
    if ch.has_subject and not (draft.subject or "").strip():
        problems.append(f"A {ch.name} needs a subject line.")
    return problems


def check_fact_ids(draft: DraftOutput, plan: MessagePlan) -> list[str]:
    known = {f.id for f in plan.facts}
    bad = sorted({u.fact_id for u in draft.used} - known)
    return [f"`used` lists {', '.join(bad)}, which aren't in <facts>. Only list real fact IDs."] if bad else []


def distinctive(fact_text: str, plan: MessagePlan) -> set[str]:
    """A fact's words that aren't also in the recipient line - "Northwind" in a
    draft to someone at Northwind doesn't show the role was used."""
    words = content_words(fact_text)
    return (words - content_words(plan.recipient)) or words


def states(fact_text: str, text: str, plan: MessagePlan) -> bool:
    """Paraphrase is fine; two of the fact's distinctive words (or its only one) is enough."""
    words = distinctive(fact_text, plan)
    return len(words & content_words(text)) >= min(2, len(words))


def check_lead(draft: DraftOutput, plan: MessagePlan) -> list[str]:
    opening = first_sentence(draft.body)
    return [f"The first sentence must be built on {f.id} ({f.text!r}); it isn't."
            for f in plan.at("lead") if not states(f.text, opening, plan)]


def check_mentions(draft: DraftOutput, plan: MessagePlan) -> list[str]:
    claimed = {u.fact_id for u in draft.used}
    problems = []
    for f in plan.at("lead") + plan.at("mention"):
        if f.id not in claimed:
            problems.append(f"{f.id} is {f.level.upper()} but isn't in `used`.")
        elif not states(f.text, draft.body, plan):
            problems.append(f"{f.id} ({f.text!r}) must be stated in the body; it's missing.")
    return problems


def check_off_leak(draft: DraftOutput, plan: MessagePlan) -> list[str]:
    """Off items should never surface. Only their distinctive words count - a word
    that's also in a selected fact or the recipient line isn't a leak."""
    allowed = content_words(" ".join([plan.recipient, *(f.text for f in plan.facts)]))
    body = content_words(draft.body)
    problems = []
    for item in plan.off_items:
        leaked = (content_words(item) - allowed) & body
        if len(leaked) >= 2 or (leaked and len(content_words(item)) <= 2):
            problems.append(f"The body mentions {', '.join(sorted(leaked))}, which the sender turned off. Remove it.")
    if plan.role_off and APPLIED_WORDS.search(draft.body):
        problems.append("Don't mention applying or an application - the role is turned off.")
    return problems


def check_invented(draft: DraftOutput, plan: MessagePlan, corpus: str) -> list[str]:
    """Numbers and proper nouns in the draft must come from the prompt. Names are
    checked in the body only - subjects are often Title Cased."""
    known = corpus.lower()
    text = DURATION.sub("", f"{draft.subject or ''}\n{draft.body}")
    numbers = {n for n in re.findall(r"\d[\d,.%]*\d|\d", text) if n not in known}
    names = set()
    for sentence in re.split(r"(?<=[.!?\n])\s*", DURATION.sub("", draft.body)):
        for word in re.findall(r"[A-Za-z][\w+#-]*", sentence)[1:]:  # skip the sentence-initial word
            if word[0].isupper() and word.lower() not in ALLOWED_CAPS and word.lower() not in known:
                names.add(word)
    problems = []
    if numbers:
        problems.append(f"These numbers aren't in the facts: {', '.join(sorted(numbers))}. Remove them.")
    if names:
        problems.append(f"These names aren't in the facts: {', '.join(sorted(names))}. Remove them.")
    return problems


def validate(draft: DraftOutput, plan: MessagePlan, corpus: str) -> list[str]:
    return [*check_length(draft, plan), *check_fact_ids(draft, plan), *check_lead(draft, plan),
            *check_mentions(draft, plan), *check_off_leak(draft, plan), *check_invented(draft, plan, corpus)]
