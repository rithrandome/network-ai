"""The profile page (/profile): upload a resume or LinkedIn PDF, review what was read, deepen it.

Built for getting started fast and going deeper later:
  - Upload first, no form. One file gives a usable profile, and a usable
    profile gives a first draft.
  - The review screen flags only what needs a decision: bullets the reader
    couldn't find word for word in the file.
  - Depth is optional and comes later. Each job or project has a "story" box
    (the walkthrough a resume has no room for). The strength meter starts
    well above zero after an upload and points at the one next thing worth
    adding. The credibility knob asks for a story when a draft is about to use
    that entry.

state.me is the source of truth, as state.inst is for knobs: each widget is
set from it before rendering, writes back through on_change, and marks it
dirty so app.py saves it before the next render. Nothing here needs a Save
button.
"""

from pathlib import Path

import streamlit as st

from llm import describe, is_local, vendor
from profile_match import strength
from schemas import Certification, Education, Entry, Me
from sources.resume import ResumeUnreadable, build_profile, read_text
from store import load_me

SAMPLE = Path(__file__).parent / "mock_data" / "resume.txt"
STORY_PLACEHOLDER = """The problem: what was broken or missing, and for whom
What you built: the approach, in a sentence or two
The hard decision: what you chose over what, and why
The result: a number if you have one, or what changed"""

state = st.session_state


def _me() -> Me:
    if "me" not in state:
        state.me = load_me()
    return state.me


def _touch() -> None:
    state.me_dirty = True


def _bind(key: str, value, write) -> str:
    """Set a widget's value from the profile and return its key. `write` gets
    the widget's value when the user changes it."""
    state[key] = value
    state[f"{key}:write"] = write
    return key


def _changed(key: str) -> None:
    state[f"{key}:write"](state[key])
    _touch()


def _text(label: str, obj, attr: str, key: str, area: bool = False, **kw) -> None:
    _bind(key, getattr(obj, attr), lambda v: setattr(obj, attr, v.strip()))
    widget = st.text_area if area else st.text_input
    widget(label, key=key, on_change=_changed, args=(key,), **kw)


# --- Upload ----------------------------------------------------------------------------

def privacy_note() -> None:
    if is_local("extract"):
        st.caption(f"Your file is read by {describe('extract')} on this machine. Nothing leaves it.")
    else:
        st.caption(f"Your file is read by {describe('extract')}: its text is sent to {vendor('extract')}'s "
                   "API. The profile is stored only in this app's local database.")


def _build(name: str, data: bytes, existing: Me | None) -> None:
    try:
        text = read_text(name, data)
    except ResumeUnreadable as exc:
        st.error(str(exc))
        return
    with st.status("Reading your resume...", expanded=True) as status:
        try:
            me = build_profile(text, existing=existing, source=name, on_step=st.write)
        except Exception as exc:
            status.update(label="The model call failed", state="error")
            st.error(str(exc))
            return
        status.update(label="Done - check what was read below", state="complete")
    state.me = me
    state.me_rev = state.get("me_rev", 0) + 1  # fresh widget keys for the new entries
    _touch()
    st.rerun()


def upload_screen(mock: bool) -> None:
    st.subheader("Start with your resume")
    st.write("Upload a resume or your LinkedIn profile and it becomes your profile: jobs, projects and the "
             "bullets your drafts can draw credibility from. You can review everything before it's used.")
    file = st.file_uploader("Resume or LinkedIn PDF", type=["pdf", "docx", "txt", "md"],
                            help="LinkedIn: open your profile → Resources (or More) → Save to PDF.")
    privacy_note()
    if file and st.button("Build my profile", type="primary"):
        _build(file.name, file.getvalue(), None)
    cols = st.columns(2)
    if mock and cols[0].button("Use the sample resume", help="A fictional profile, for trying the app"):
        _build(SAMPLE.name, SAMPLE.read_bytes(), None)
    if cols[1].button("Fill it in by hand instead", type="tertiary"):
        state.me = Me(entries=[Entry()])
        state.me_rev = state.get("me_rev", 0) + 1
        _touch()
        st.rerun()


# --- Review and deepen ----------------------------------------------------------------

def _entry_editor(entry: Entry, me: Me, k: str) -> None:
    icon = "💼" if entry.kind == "job" else "🛠️"
    marks = (" · ✍️ story" if entry.story.strip() else "") + (" · ⚠️ check" if entry.unverified else "")
    with st.expander(f"{icon} {entry.label}{marks}"):
        a, b = st.columns(2)
        with a:
            _text("Title" if entry.kind == "job" else "Project", entry, "title", f"{k}:title")
            _text("Start", entry, "start", f"{k}:start")
        with b:
            _text("Company" if entry.kind == "job" else "Context (course, client, Personal)", entry, "org",
                  f"{k}:org")
            _text("End", entry, "end", f"{k}:end", placeholder="Present")

        def set_bullets(value: str) -> None:
            entry.bullets = [line.strip(" -•") for line in value.splitlines() if line.strip(" -•")]
            entry.unverified = [u for u in entry.unverified if u in entry.bullets]

        st.text_area("Bullets - one per line. Each one can be picked as credibility in a draft.",
                     key=_bind(f"{k}:bullets", "\n".join(entry.bullets), set_bullets),
                     on_change=_changed, args=(f"{k}:bullets",), height=120)

        if entry.unverified:
            st.warning("These weren't found word for word in your file, so they may be the reader's "
                       "paraphrase. They aren't offered to drafts until you keep them.", icon="⚠️")
            for i, bullet in enumerate(list(entry.unverified)):
                st.checkbox(f"Keep: {bullet}", key=_bind(f"{k}:keep{i}", False,
                                                         lambda v, b=bullet: v and entry.unverified.remove(b)),
                            on_change=_changed, args=(f"{k}:keep{i}",))

        _text("The story behind it (optional, the more the better)", entry, "story", f"{k}:story", area=True,
              height=140, placeholder=STORY_PLACEHOLDER,
              help="Never shown as-is. When a draft uses this entry, the model gets the story as background, "
                   "so it can say why the work mattered.")

        def remove() -> None:
            me.entries.remove(entry)
            state.me_rev = state.get("me_rev", 0) + 1
            _touch()

        st.button("Remove", key=f"{k}:remove", on_click=remove, type="tertiary")


def _table_editor(me: Me, attr: str, model: type, columns: dict[str, str], key: str) -> None:
    """An editable table over a list of models; rows without the first column are dropped."""
    current = getattr(me, attr)
    rows = [r.model_dump() for r in current] or [model().model_dump()]
    # data_editor keeps a diff in its state; the return value is the edited table.
    edited = st.data_editor(rows, key=key, num_rows="dynamic", width="stretch", column_config=columns)
    first = next(iter(columns))
    items = [model(**{c: (r.get(c) or "").strip() for c in model.model_fields}) for r in edited]
    items = [i for i in items if getattr(i, first)]
    if items != current:
        setattr(me, attr, items)
        _touch()


def review_screen(me: Me, mock: bool) -> None:
    k = f"me:{state.get('me_rev', 0)}"
    score, hint = strength(me)
    st.progress(score, text=f"Profile strength: {round(score * 100)}%")
    if hint:
        st.caption(hint)
    if me.source:
        st.caption(f"Read from {me.source}. Edits save as you go.")

    _text("Name", me, "name", f"{k}:name")
    _text("Headline", me, "headline", f"{k}:headline", placeholder="e.g. ML engineer, ex-AWS")
    _text("Location", me, "location", f"{k}:location")
    _text("Summary", me, "summary", f"{k}:summary", area=True, height=90)

    st.markdown("**Experience and projects**")
    if any(e.unverified for e in me.entries):
        st.caption("⚠️ marks entries with bullets to check.")
    for i, entry in enumerate(me.entries):
        _entry_editor(entry, me, f"{k}:e{i}")

    def add(kind: str) -> None:
        me.entries.append(Entry(kind=kind))
        state.me_rev = state.get("me_rev", 0) + 1
        _touch()

    a, b = st.columns(2)
    a.button("+ Job", on_click=add, args=("job",), width="stretch")
    b.button("+ Project", on_click=add, args=("project",), width="stretch")

    st.markdown("**Education**")
    _table_editor(me, "education", Education, {"school": "School", "degree": "Degree (as written, minors included)",
                                                "grade": "GPA / grade", "start": "Start", "end": "End"}, f"{k}:edu")

    st.markdown("**Certifications**")
    st.caption("Add any that aren't on your resume. Each one can be picked as credibility in a draft; "
               "expired ones aren't offered.")
    _table_editor(me, "certifications", Certification, {"name": "Certification", "issuer": "Issued by",
                                                        "issued": "Issued", "expires": "Expires"}, f"{k}:certs")

    st.markdown("**Skills**")
    st.multiselect("Skills", me.skills, key=_bind(f"{k}:skills", list(me.skills),
                                                  lambda v: setattr(me, "skills", list(v))),
                   on_change=_changed, args=(f"{k}:skills",), accept_new_options=True,
                   label_visibility="collapsed", placeholder="Add a skill")

    st.markdown("**Anything else worth bragging about**")
    st.text_area("Extra highlights - one per line", key=_bind(
        f"{k}:highlights", "\n".join(me.highlights),
        lambda v: setattr(me, "highlights", [h.strip(" -•") for h in v.splitlines() if h.strip(" -•")])),
        on_change=_changed, args=(f"{k}:highlights",), height=90, label_visibility="collapsed",
        placeholder="Things not on the resume, e.g. Spoke at PyCon 2025 on RAG evals")

    with st.expander("Upload a newer resume"):
        st.caption("Replaces what came from the old file. Your stories, kept bullets and extra highlights stay.")
        file = st.file_uploader("Newer resume", type=["pdf", "docx", "txt", "md"], key=f"{k}:reupload",
                                label_visibility="collapsed")
        privacy_note()
        if file and st.button("Update my profile"):
            _build(file.name, file.getvalue(), me)


def me_page(mock: bool) -> None:
    st.header("Profile")
    me = _me()
    if not (me.name or me.entries):
        upload_screen(mock)
    else:
        review_screen(me, mock)
