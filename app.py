import streamlit as st

# ── Navigation ───────────────────────────────────────────────
pages = st.navigation([
    st.Page("housing_app.py", title="חלק א: מחירי דיור", icon="🏠", default=True),
    st.Page("heart_app.py", title="חלק ב: התקפי לב", icon="❤️"),
])
pages.run()
