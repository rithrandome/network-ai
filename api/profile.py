"""Profile routes: the sender, built from an uploaded resume and deepened over time."""

from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from api.errors import upstream
from api.models import ProfileView, StoryUpdate
from profile_match import strength
from schemas import Me
from sources.resume import build_profile, read_text
from store import load_me, save_me

router = APIRouter(prefix="/profile", tags=["profile"])

SAMPLE = Path(__file__).resolve().parent.parent / "mock_data" / "resume.txt"


def profile_view(me: Me) -> ProfileView:
    score, hint = strength(me)
    return ProfileView(me=me, strength=score, hint=hint)


def _build(name: str, data: bytes, existing: Me | None) -> ProfileView:
    text = read_text(name, data)  # ResumeUnreadable -> 422
    with upstream("Reading the resume"):
        me = build_profile(text, existing=existing, source=name)
    save_me(me)
    return profile_view(me)


@router.get("")
def get_profile() -> ProfileView:
    return profile_view(load_me())


@router.put("")
def put_profile(me: Me) -> ProfileView:
    """Save edits. The uploaded file's text stays on the server when the
    client doesn't send it back - bullets are checked against it."""
    stored = load_me()
    if not me.resume_text:
        me = me.model_copy(update={"resume_text": stored.resume_text, "source": me.source or stored.source})
    save_me(me)
    return profile_view(me)


@router.post("/resume")
def upload_resume(file: UploadFile) -> ProfileView:
    """A first upload builds the profile; a later one replaces what came from
    the old file and keeps stories, kept bullets and hand-typed highlights."""
    stored = load_me()
    existing = stored if (stored.name or stored.entries) else None
    return _build(file.filename or "resume", file.file.read(), existing)


@router.post("/sample")
def use_sample() -> ProfileView:
    """The fictional sample resume, for trying the app. Replaces the profile."""
    return _build(SAMPLE.name, SAMPLE.read_bytes(), None)


@router.patch("/story")
def save_story(update: StoryUpdate) -> ProfileView:
    """The just-in-time ask: save the story behind one job or project."""
    me = load_me()
    entry = next((e for e in me.entries if e.key == update.entry_key), None)
    if not entry:
        raise HTTPException(404, f"No entry {update.entry_key!r}")
    entry.story = update.story.strip()
    save_me(me)
    return profile_view(me)
