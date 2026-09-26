"""Streamlit front end for network-ai."""

import os

import streamlit as st

from pipeline import find_candidates, write_brief

st.set_page_config(page_title="network-ai", page_icon="🤝")
st.title("network-ai 🤝")
st.caption("Describe someone you want to reach out to, pick the right profile, get a brief.")

has_exa_key = bool(os.environ.get("EXA_API_KEY"))
use_mock = st.toggle("Use mock data (fictional profiles, no search credit)", value=not has_exa_key,
                     disabled=not has_exa_key)
if use_mock:
    st.caption("Searching the mock fixture." + ("" if has_exa_key else " Add EXA_API_KEY to .env for real search."))
else:
    st.caption("Live Exa people search - about $0.007 per new lookup; repeated lookups come from the local cache for free.")

with st.form("lookup"):
    query = st.text_input("Who are you looking for?", label_visibility="collapsed",
                          placeholder="Name plus anything you know - e.g. Ashutosh Mishra, Tampa, Capgemini, Citi")
    submitted = st.form_submit_button("Find", type="primary")

if submitted and query.strip():
    with st.spinner("Searching..."):
        try:
            st.session_state.candidates = find_candidates(query, mode="mock" if use_mock else None)
            st.session_state.pop("brief", None)
        except Exception as exc:
            st.error(f"Lookup failed: {exc}")

candidates = st.session_state.get("candidates")
if candidates == []:
    st.warning("No profiles found - try adding where they work or live.")
elif candidates:
    st.subheader("Which one?")
    labels = []
    for c in candidates:
        role = c.current_role
        now = f"{role.title} @ {role.company}" if role and role.company else (c.headline or "")
        labels.append(f"**{c.name}** - {now} · {c.location or 'location unknown'}")
    choice = st.radio("Candidates", range(len(candidates)), format_func=lambda i: labels[i], label_visibility="collapsed")
    chosen = candidates[choice]
    st.caption(chosen.url)

    if st.button("Write brief"):
        with st.spinner("Reading their profile..."):
            try:
                st.session_state.brief = (chosen.url, write_brief(chosen))
            except Exception as exc:
                st.error(f"Couldn't write a brief: {exc}")

    brief_for = st.session_state.get("brief")
    if brief_for and brief_for[0] == chosen.url:
        brief = brief_for[1]
        st.subheader("Summary")
        st.write(brief.summary)
        st.subheader("Interesting facts")
        for fact in brief.facts:
            st.write(f"- {fact}")
        st.subheader("Icebreakers")
        for opener in brief.icebreakers:
            st.write(f"- {opener}")
