# network-ai

A networking assistant for job seekers. Describe someone you want to reach out
to in one line - a name plus whatever you know ("Ashutosh Mishra, Tampa,
Capgemini, Citi") - pick the right profile from the candidates, and get a
brief: who they are, what's interesting about them, and how to open the
conversation.

Early stage: resolution and the brief work end to end. Context capture (who
this person is to you, what this reach-out is for), per-channel outreach drafts
and a tracker come next.

## Pipeline

```
one free-text description
  --> Exa people search        -> up to 10 candidate profiles (full profile markdown
                                  + structured work history), via exa-py
  --> you pick the right one    (same-name matches are normal)
  --> local Ollama model        -> Brief {summary, facts, icebreakers},
                                  Pydantic-constrained
  --> Streamlit UI
```

## Why this, not that

**Exa people search, not a search engine + a profile scraper.** One call
returns the candidate profiles themselves - headline, About, full experience -
as markdown, plus a structured entity. That replaces two vendors (one to find
the URL, one to fetch the profile), and scraping LinkedIn directly is exactly
what got profile-scraping APIs shut down. Picking from candidates instead of
taking the first hit fixes the common-name problem.

**One search box, not name/company/location fields.** Exa's search is
semantic, so the description goes in as the query as-is; splitting it into
fields first would add nothing.

**People search with full text, not Exa's default query + highlights.**
`category="people"` limits results to profiles - without it a name query
returns any page mentioning the name. Full `text` because the brief needs the
whole profile, and highlights are only query-relevant excerpts; measured, text
adds nothing to the $0.007 search cost.

**Search, not Exa's Agent API.** Resolving a known person is one call, and the
user needs to see the candidates to pick the right one. An agent run is slower,
costs more, and picks for you. It fits a later "find people matching criteria"
feature (list-building), not this.

**The profile text, not the structured entity, feeds the brief.** The entity
is incomplete - exa-py keeps only name, location and work history, dropping the
education the API returns - so it's only used for the one-line candidate label.

**Cache-first Exa client.** Every response is cached by request, so the same
lookup never spends credit twice. `EXA_MODE=offline` guarantees no spend while
developing; `EXA_MODE=mock` uses a fictional fixture with no key at all.

**Local Ollama for generation.** No API cost; `num_ctx` is raised to 8192
because Ollama's default 2048-token window silently truncates a full profile.

## Run it

Requires [Ollama](https://ollama.com) running locally:

```bash
ollama pull gemma3:1b
uv sync
cp .env.example .env        # add EXA_API_KEY to look up real people

EXA_MODE=mock uv run python pipeline.py      # CLI, fictional data, no keys
uv run streamlit run app.py                  # app; live search if EXA_API_KEY is set, else mock
uv run python -m sources.exa_client "Name, company, city"   # candidate lookup only
```

## Files

- `pipeline.py` - `find_candidates()` and `write_brief()`
- `sources/exa_client.py` - Exa people search (exa-py), cache/offline/mock modes
- `schemas.py` - `Candidate`, `Role`, `Brief`
- `mock_data/exa_people_search.json` - fictional Exa response (two Jane Does)
- `app.py` - Streamlit front end
