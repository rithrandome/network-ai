"""Person resolution via Exa's search API.

Given a name plus whatever the user knows (company, location, a hint), returns
a short list of candidate profiles for the user to pick from.

Request shape: `category: "people"` with full page `text`. Exa's default
request is query + highlights with no category, but people search is Exa's
documented surface for retrieving people's profile documents, and it returns
the whole LinkedIn profile as markdown (headline, About, Experience) plus a
structured entity - so the profile itself is the input to the brief and no
paste/PDF step is required. Filtering lives in the natural-language query:
people search rejects date filters and excludeDomains.

Finding people who *match criteria* ("recruiters at Stripe") is a different
workflow - list-building - and belongs on Exa's Agent API, not here.

Modes (EXA_MODE in .env), so development doesn't spend search credit:
  live    - cache-first: a query already cached is replayed, otherwise Exa is
            called once and the raw response is cached. (default)
  offline - cache only; a query not in the cache raises instead of spending.
  mock    - a fixed fictional response from mock_data/ - no key, no network.

Run:  uv run python -m sources.exa_client "Jane Doe" --company Northwind
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

from schemas import Candidate, Education, Role

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".cache" / "exa"  # real people's data - gitignored
MOCK_PATH = ROOT / "mock_data" / "exa_people_search.json"
SEARCH_URL = "https://api.exa.ai/search"

# Product decision, not boilerplate: the user picks from a short list, and
# a longer one mostly adds same-name strangers.
NUM_CANDIDATES = 5


class ExaCacheMiss(RuntimeError):
    pass


def build_query(name: str, company: str = "", location: str = "", hint: str = "") -> str:
    return " ".join(part.strip() for part in (name, company, location, hint) if part and part.strip())


def _request_body(query: str) -> dict:
    return {
        "query": query,
        "category": "people",
        "numResults": NUM_CANDIDATES,
        "contents": {"text": True},
    }


def _cache_path(body: dict) -> Path:
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
    return CACHE_DIR / f"{digest}.json"


def _fetch(body: dict, mode: str) -> dict:
    if mode == "mock":
        return json.loads(MOCK_PATH.read_text())

    cache_path = _cache_path(body)
    if cache_path.exists():
        return json.loads(cache_path.read_text())["response"]
    if mode == "offline":
        raise ExaCacheMiss(f"EXA_MODE=offline and no cached response for {body['query']!r}")

    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        raise RuntimeError("EXA_API_KEY is not set - add it to .env, or set EXA_MODE=mock")

    response = requests.post(
        SEARCH_URL,
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        json=body,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"request": body, "response": data}, indent=2))
    return data


def _headline_from_text(text: str) -> str | None:
    # Profile markdown starts "# Name", then the headline as the next non-empty line.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 2 and lines[0].startswith("# ") and not lines[1].startswith("#"):
        return lines[1]
    return None


def _to_candidate(result: dict) -> Candidate:
    entities = result.get("entities") or []
    props = (entities[0].get("properties") or {}) if entities else {}
    text = result.get("text") or ""

    roles = [
        Role(
            title=work.get("title"),
            company=(work.get("company") or {}).get("name"),
            start=(work.get("dates") or {}).get("from"),
            end=(work.get("dates") or {}).get("to"),
        )
        for work in props.get("workHistory") or []
    ]
    education = [
        Education(institution=(edu.get("institution") or {}).get("name"), degree=edu.get("degree"))
        for edu in props.get("educationHistory") or []
    ]

    return Candidate(
        name=props.get("name") or result.get("title") or "Unknown",
        url=result.get("url", ""),
        headline=_headline_from_text(text),
        location=props.get("location"),
        roles=roles,
        education=education,
        profile_text=text,
    )


def find_person(name: str, company: str = "", location: str = "", hint: str = "",
                mode: str | None = None) -> list[Candidate]:
    """Return candidate profiles for the described person, best match first."""
    mode = mode or os.environ.get("EXA_MODE", "live")
    body = _request_body(build_query(name, company, location, hint))
    data = _fetch(body, mode)
    return [_to_candidate(result) for result in data.get("results", [])]


def _main() -> None:
    parser = argparse.ArgumentParser(description="Look up candidate profiles for a person via Exa.")
    parser.add_argument("name")
    parser.add_argument("--company", default="")
    parser.add_argument("--location", default="")
    parser.add_argument("--hint", default="")
    parser.add_argument("--mode", choices=["live", "offline", "mock"])
    args = parser.parse_args()

    candidates = find_person(args.name, args.company, args.location, args.hint, mode=args.mode)
    for rank, c in enumerate(candidates, start=1):
        role = c.current_role
        print(f"#{rank} {c.name} - {c.headline or '(no headline)'}")
        print(f"    {c.url}")
        print(f"    {c.location or '?'} | now: {role.title if role else '?'} @ {role.company if role else '?'}"
              f" | {len(c.roles)} roles, {len(c.education)} education, {len(c.profile_text)} chars of profile")


if __name__ == "__main__":
    _main()
