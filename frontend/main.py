"""The one true entry point: `streamlit run main.py`.

BUG FOUND: before st.navigation(), Streamlit derived each sidebar nav
label from the literal command used to launch the app, so a lowercase
"streamlit run app.py" produced a lowercase "app" label even though the
file is App.py, purely because the filesystem is case-insensitive.
st.Page(..., title=...) sets the label explicitly in code instead, fixing
this independent of launch-command casing. The other page files are
unchanged; st.Page can reference an existing script directly.
"""
import streamlit as st

st.set_page_config(
    page_title="MedReport AI",
    page_icon=":stethoscope:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Marker App.py checks for, see its own top-of-file guard.
st.session_state["_launched_via_main"] = True

# BUG FOUND on first deployment to Linux: these paths must match the files'
# real names exactly.
pg = st.navigation([
    st.Page("app.py", title="Home", default=True),
    st.Page("pages/2_results.py", title="Results"),
    st.Page("pages/3_History.py", title="History"),
])
pg.run()
