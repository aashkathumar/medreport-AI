"""The one true entry point: `streamlit run main.py`.

BUG FOUND (recurring across this whole project): before st.navigation(),
Streamlit derived each sidebar nav label directly from the literal argv
string used to launch the app, "streamlit run app.py" (lowercase, typed
from habit/shell history) produced a lowercase "app" label even though the
actual file on disk is named App.py, since macOS's case-insensitive
filesystem resolves either spelling to the same file. That made the nav
label depend on which casing someone happened to type at the command line,
not on anything in the code, unfixable by renaming files, since renaming
doesn't change what gets typed next time.

st.Page(..., title=...) sets the label explicitly, in code, independent of
the launch command's casing entirely, this is the actual fix, not
another reminder to type the filename correctly. app.py/2_results.py/
3_History.py are unchanged otherwise; st.Page can reference an existing
script file directly, so none of their own content had to move.
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
