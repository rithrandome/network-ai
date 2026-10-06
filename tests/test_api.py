"""API tests: mock search, a temporary database, and stubbed model calls, so
they run with no keys, no network and no Ollama. What's under test is the
HTTP layer and the knob rules it exposes, not the models.

Run:  uv run pytest
"""

import pytest
from fastapi.testclient import TestClient

import store
from api import instances, profile
from api.main import app
from schemas import Brief, DraftVersion, Entry, JobPosting, JobSource, Me, Suggestions


def fake_suggest(instance, me):
    return Suggestions(role_guess="Potential teammate", role_reason="Senior ML Engineer",
                       common_ground=["Both worked at Contoso"], hooks=["Her talk on RAG evals"],
                       ask_wording=["a 15-minute chat about the team"])


def fake_brief(person):
    return Brief(summary=f"{person.name} builds ML systems.", facts=["Leads the search team"], icebreakers=[])


def fake_compose(instance, me, refine="", base=None):
    return DraftVersion(body=f"Hi {instance.person.name.split()[0]}, ...", settings=dict(instance.settings),
                        ingredients={k: v.model_copy() for k, v in instance.ingredients.items()},
                        refine=refine, created_at=store.now())


def fake_read_job(value, mode=None):
    return JobSource(input=value, posting=JobPosting(title="ML Engineer", company="Northwind", job_id="NW-2207"))


def broken(*args, **kwargs):
    raise RuntimeError("model unavailable")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(instances, "suggest", fake_suggest)
    monkeypatch.setattr(instances, "write_brief", fake_brief)
    monkeypatch.setattr(instances, "compose", fake_compose)
    monkeypatch.setattr(instances, "read_job", fake_read_job)
    return TestClient(app)


def new_instance(client) -> dict:
    person = client.post("/api/search", json={"query": "Jane Doe, Northwind", "mode": "mock"}).json()[0]
    response = client.post("/api/instances", json={"person": person})
    assert response.status_code == 201
    return response.json()["instance"]


def ingredients(view: dict) -> dict:
    return view["instance"]["ingredients"]


# --- App info ------------------------------------------------------------------------

def test_knob_registry(client):
    knobs = client.get("/api/knobs").json()
    goal = next(s for s in knobs["settings"] if s["key"] == "goal")
    assert "Referral" in goal["options"] and goal["custom"]
    assert knobs["goal_rules"]["Referral"]["levels"]["role"] == "lead"
    assert {c["name"]: c["max_chars"] for c in knobs["channels"]}["LinkedIn note"] == 200


def test_settings_lists_both_roles(client):
    models = client.get("/api/settings").json()["models"]
    assert [m["role"] for m in models] == ["extract", "compose"]


def test_search_mock_returns_candidates(client):
    found = client.post("/api/search", json={"query": "Jane Doe", "mode": "mock"}).json()
    assert len(found) >= 2 and all(c["name"] and c["url"] for c in found)


def test_offline_cache_miss_is_409(client):
    response = client.post("/api/search", json={"query": "nobody-cached-zz9", "mode": "offline"})
    assert response.status_code == 409


# --- The new-reach-out flow --------------------------------------------------------

def test_full_flow(client):
    inst = new_instance(client)
    iid = inst["id"]

    view = client.patch(f"/api/instances/{iid}/settings", json={"goal": "Referral", "status": "Applied"}).json()
    assert view["instance"]["settings"]["goal"] == "Referral"

    view = client.put(f"/api/instances/{iid}/job", json={"input": "https://jobs.example/NW-2207"}).json()
    assert view["instance"]["job"]["posting"]["job_id"] == "NW-2207"

    view = client.post(f"/api/instances/{iid}/suggest").json()
    assert view["notice"] is None
    settings = view["instance"]["settings"]
    assert settings["channel"] == "Email"  # from the Referral rule
    assert settings["their_role"] == "Potential teammate"  # from the suggestion
    role = ingredients(view)["role"]
    assert role["level"] == "lead" and role["items"] == ["ML Engineer at Northwind (job ID NW-2207)"]
    assert "Her talk on RAG evals" in view["options"]["hook"]
    assert "Her talk on RAG evals" in view["suggested"]
    assert ingredients(view)["hook"]["items"] == []  # open suggestions start unselected

    # One lead only: leading with credibility moves the role to mention.
    view = client.patch(f"/api/instances/{iid}/ingredients/credibility", json={"level": "lead"}).json()
    assert ingredients(view)["credibility"]["level"] == "lead"
    assert ingredients(view)["role"]["level"] == "mention"

    view = client.post(f"/api/instances/{iid}/drafts", json={}).json()
    assert len(view["instance"]["drafts"]) == 1
    view = client.post(f"/api/instances/{iid}/drafts", json={"refine": "shorter", "base": 0}).json()
    assert view["instance"]["drafts"][-1]["refine"] == "shorter"

    listed = client.get("/api/instances").json()
    assert listed[0]["id"] == iid and listed[0]["drafts"] == 2 and "referral" in listed[0]["title"]

    assert client.delete(f"/api/instances/{iid}").status_code == 204
    assert client.get(f"/api/instances/{iid}").status_code == 404


def test_goal_change_keeps_what_the_user_changed(client):
    iid = new_instance(client)["id"]
    client.patch(f"/api/instances/{iid}/settings", json={"goal": "Referral"})
    client.post(f"/api/instances/{iid}/suggest")
    client.patch(f"/api/instances/{iid}/ingredients/ask", json={"level": "hint"})  # Referral default: mention

    view = client.patch(f"/api/instances/{iid}/settings", json={"goal": "Coffee chat"}).json()
    levels = {k: v["level"] for k, v in ingredients(view).items()}
    assert levels["ask"] == "hint"  # user's choice stays
    assert levels["credibility"] == "hint" and levels["role"] == "off"  # defaults move
    assert view["instance"]["settings"]["channel"] == "LinkedIn note"


def test_suggest_needs_a_goal(client):
    iid = new_instance(client)["id"]
    assert client.post(f"/api/instances/{iid}/suggest").status_code == 409


def test_failed_suggestions_still_seed_the_knobs(client, monkeypatch):
    monkeypatch.setattr(instances, "suggest", broken)
    iid = new_instance(client)["id"]
    client.patch(f"/api/instances/{iid}/settings", json={"goal": "Coffee chat"})
    view = client.post(f"/api/instances/{iid}/suggest").json()
    assert "suggestions" in view["notice"]
    assert ingredients(view)["ask"]["level"] == "mention"


def test_failed_draft_is_502(client, monkeypatch):
    monkeypatch.setattr(instances, "compose", broken)
    iid = new_instance(client)["id"]
    response = client.post(f"/api/instances/{iid}/drafts", json={})
    assert response.status_code == 502 and "model unavailable" in response.json()["detail"]


def test_bad_ingredient_and_level(client):
    iid = new_instance(client)["id"]
    assert client.patch(f"/api/instances/{iid}/ingredients/nope", json={"level": "off"}).status_code == 404
    assert client.patch(f"/api/instances/{iid}/ingredients/ask", json={"level": "loud"}).status_code == 422
    assert client.post(f"/api/instances/{iid}/drafts", json={"base": 3}).status_code == 404


# --- Profile ---------------------------------------------------------------------------

BULLET = "Built an AWS Bedrock text-to-SQL agent at ~90% accuracy"


def test_profile_edit_keeps_resume_text(client):
    store.save_me(Me(name="Alex Tester", resume_text="original file text"))
    view = client.put("/api/profile", json={"name": "Alex T.", "headline": "ML engineer"}).json()
    assert view["me"]["name"] == "Alex T."
    assert store.load_me().resume_text == "original file text"


def test_story_is_asked_for_just_in_time(client):
    store.save_me(Me(name="Alex Tester", entries=[Entry(title="Consultant", org="Contoso", bullets=[BULLET])]))
    iid = new_instance(client)["id"]
    client.patch(f"/api/instances/{iid}/settings", json={"goal": "Referral"})
    view = client.post(f"/api/instances/{iid}/suggest").json()
    assert ingredients(view)["credibility"]["items"] == [BULLET]  # top-ranked bullet preselected
    prompt = view["story_prompt"]
    assert prompt["label"] == "Consultant at Contoso"

    profile_view = client.patch("/api/profile/story", json={"entry_key": prompt["entry_key"],
                                                            "story": "The problem: ..."}).json()
    assert profile_view["strength"] > 0.5
    assert client.get(f"/api/instances/{iid}").json()["story_prompt"] is None


def test_unreadable_resume_is_422(client):
    response = client.post("/api/profile/resume", files={"file": ("resume.txt", b"too short", "text/plain")})
    assert response.status_code == 422


def test_failed_resume_read_is_502(client, monkeypatch):
    monkeypatch.setattr(profile, "build_profile", broken)
    response = client.post("/api/profile/sample")
    assert response.status_code == 502
