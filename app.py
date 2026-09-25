"""Streamlit front end for network-ai."""

import streamlit as st

from pipeline import find_candidates, write_brief

st.set_page_config(page_title="network-ai", page_icon="🤝")
st.title("network-ai 🤝")
st.caption("Describe someone you want to reach out to, pick the right profile, get a brief.")

use_mock = st.toggle("Use mock data (no API key, no search credit)", value=True)

with st.form("lookup"):
    name = st.text_input("Full name")
    col1, col2 = st.columns(2)
    company = col1.text_input("Company", placeholder="optional")
    location = col2.text_input("Location", placeholder="optional")
    hint = st.text_input("Anything else you know", placeholder="optional - e.g. 'technical recruiter', 'Georgia Tech alum'")
    submitted = st.form_submit_button("Find", type="primary")

if submitted and name:
    with st.spinner(f"Looking up {name}..."):
        try:
            st.session_state.candidates = find_candidates(name, company, location, hint,
                                                          mode="mock" if use_mock else None)
            st.session_state.pop("brief", None)
        except Exception as exc:
            st.error(f"Lookup failed: {exc}")

candidates = st.session_state.get("candidates")
if candidates == []:
    st.warning("No profiles found - try adding a company or location.")
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
