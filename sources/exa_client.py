"""Person resolution via Exa's search API (exa-py SDK).

The user types one free-text description - "Ashutosh Mishra, Tampa, Capgemini,
Citi" - and it goes to exa.search() as the query. Exa's search is semantic, so
there's no need to split it into name/company/location fields first; filtering
belongs in the natural-language query anyway (people search rejects date
filters and excludeDomains).

Request, and why each part deviates from Exa's recommended query + highlights:
  category="people"  - restricts results to people's profile documents. Without
                       it, a name query returns any page mentioning the name
                       (news, articles, other people's posts) and the picker
                       would show pages, not people.
  contents text      - the brief needs the whole profile (headline, About, full
                       Experience); highlights are query-relevant excerpts only.
                       Measured: text adds nothing to the $0.007 search cost.
  num_results        - left at the server default of 10.

Finding people who *match criteria* ("recruiters at Stripe") is list-building
and belongs on Exa's Agent API, not here.

SDK note: exa-py parses results into dataclasses, and its person entity keeps
only name, location and work history - the API's educationHistory is dropped.
Education is still in the profile text the brief reads.

Modes (EXA_MODE in .env), so development doesn't spend search credit:
  live    - cache-first: a query already cached is replayed, otherwise Exa is
            called once and the results are cached. (default)
  offline - cache only; a query not in the cache raises instead of spending.
  mock    - a fixed fictional response from mock_data/ - no key, no network.

Run:  uv run python -m sources.exa_client "Jane Doe, Northwind Analytics" --mode mock
"""

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from exa_py import Exa

from schemas import Candidate, Role

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".cache" / "exa"  # real people's data - gitignored
MOCK_PATH = ROOT / "mock_data" / "exa_people_search.json"


class ExaCacheMiss(RuntimeError):
    pass


def _search_kwargs(query: str) -> dict:
    return {
        "query": query,
        "type": "auto",
        "category": "people",
        "contents": {"text": True},
    }


def _cache_path(kwargs: dict) -> Path:
    digest = hashlib.sha256(json.dumps(kwargs, sort_keys=True).encode()).hexdigest()[:16]
    return CACHE_DIR / f"{digest}.json"


def _fetch(kwargs: dict, mode: str) -> list[dict]:
    """Search results as plain dicts (the SDK's Result dataclasses, serialized)."""
    if mode == "mock":
        return json.loads(MOCK_PATH.read_text())["results"]

    cache_path = _cache_path(kwargs)
    if cache_path.exists():
        return json.loads(cache_path.read_text())["results"]
    if mode == "offline":
        raise ExaCacheMiss(f"EXA_MODE=offline and no cached response for {kwargs['query']!r}")

    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        raise RuntimeError("EXA_API_KEY is not set - add it to .env, or set EXA_MODE=mock")

    response = Exa(api_key=api_key).search(**kwargs)
    # Cache the full results, not just the Candidate fields - a later change
    # to what we extract shouldn't need a paid re-search.
    results = [dataclasses.asdict(result) for result in response.results]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "request": kwargs,
        "cost_dollars": response.cost_dollars.total if response.cost_dollars else None,
        "results": results,
    }, indent=2))
    return results


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
            start=(work.get("dates") or {}).get("from_date"),
            end=(work.get("dates") or {}).get("to_date"),
        )
        for work in props.get("work_history") or []
    ]

    return Candidate(
        name=props.get("name") or result.get("title") or "Unknown",
        url=result.get("url", ""),
        headline=_headline_from_text(text),
        location=props.get("location"),
        roles=roles,
        profile_text=text,
    )


def find_person(query: str, mode: str | None = None) -> list[Candidate]:
    """Return candidate profiles for a free-text description, best match first."""
    mode = mode or os.environ.get("EXA_MODE", "live")
    results = _fetch(_search_kwargs(query.strip()), mode)
    return [_to_candidate(result) for result in results]


def _main() -> None:
    parser = argparse.ArgumentParser(description="Look up candidate profiles for a person via Exa.")
    parser.add_argument("query", help='e.g. "Ashutosh Mishra, Tampa, Capgemini, Citi"')
    parser.add_argument("--mode", choices=["live", "offline", "mock"])
    args = parser.parse_args()

    for rank, c in enumerate(find_person(args.query, mode=args.mode), start=1):
        role = c.current_role
        print(f"#{rank} {c.name} - {c.headline or '(no headline)'}")
        print(f"    {c.url}")
        print(f"    {c.location or '?'} | now: {role.title if role else '?'} @ {role.company if role else '?'}"
              f" | {len(c.roles)} roles, {len(c.profile_text)} chars of profile")


if __name__ == "__main__":
    _main()
