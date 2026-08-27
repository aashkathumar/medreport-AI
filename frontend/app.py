import os
import threading
import time
import uuid
from datetime import datetime
import streamlit as st
import requests
from streamlit.runtime.scriptrunner import add_script_run_ctx

# Set MEDREPORT_API_URL as an env var / Streamlit secret when deployed to the cloud.
API = os.environ.get("MEDREPORT_API_URL", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="MedReport AI",
    page_icon=":stethoscope:",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "user_id" not in st.session_state:
    st.session_state.user_id = "demo_user"
if "report" not in st.session_state:
    st.session_state.report = None
if "session_reports" not in st.session_state:
    # Reports generated during THIS browser session only -- nothing is
    # persisted server-side (see routes.py), so this list is the only place
    # "history" exists, and it's gone the moment the tab closes.
    st.session_state.session_reports = []
if "profile" not in st.session_state:
    st.session_state.profile = {}
if "upload_status" not in st.session_state:
    st.session_state.upload_status = None  # None | "running" | "done" | "error" | "manual_needed"
if "upload_error" not in st.session_state:
    st.session_state.upload_error = None

with st.sidebar:
    st.header("Your Profile")
    st.session_state.profile["age"] = st.number_input("Age", min_value=16, max_value=90, value=30, step=1)
    st.session_state.profile["sex"] = st.selectbox("Sex", ["prefer_not_to_say", "male", "female"])
    st.session_state.profile["diet_type"] = st.selectbox("Diet", ["omnivore", "vegetarian", "vegan"])
    st.caption("Your profile personalises the lifestyle recommendations.")
    st.divider()

    # No login, no directory to pick from -- just a label for this session's
    # reports. Gone (along with everything else in session_state) the moment
    # this tab closes. `key=` keeps it correctly sticky within the session.
    st.text_input("Your name (this session only)", key="user_id")

st.title("MedReport AI")
st.caption(
    "Upload your blood test or urinalysis report and receive plain-English "
    "explanations with personalised lifestyle guidance."
)

st.divider()
st.subheader("Upload Your Report")
st.write("Upload a PDF blood test or urinalysis report.")


def _run_upload(file_bytes, file_name, user_id, age, sex, diet_type):
    """Runs in a background thread instead of inline in the button handler.

    A background thread keeps running regardless of which page is currently
    rendered (they all share the same server-side session), so switching
    tabs no longer kills the request -- the result lands in session_state
    whenever it finishes, and any page can notice.
    """
    try:
        resp = requests.post(
            f"{API}/upload-pdf",
            files={"file": (file_name, file_bytes, "application/pdf")},
            params={"user_id": user_id, "age": age, "sex": sex, "diet_type": diet_type},
            timeout=600,
        )
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.report = data
            st.session_state.session_reports.insert(0, {
                "report_id": data.get("report_id") or str(uuid.uuid4()),
                "time_label": datetime.now().strftime("%d %B %Y, %H:%M"),
                "report_data": data,
            })
            st.session_state.upload_status = "done"
            # Fires the one-time auto-redirect below. Set here, not read here --
            # this runs in the background thread, the redirect itself has to
            # happen from the main script thread's next render.
            st.session_state.auto_redirect_pending = True
        elif resp.status_code == 422:
            st.session_state.upload_status = "manual_needed"
        else:
            st.session_state.upload_error = f"Error {resp.status_code}: {resp.text[:300]}"
            st.session_state.upload_status = "error"
    except requests.exceptions.ConnectionError:
        st.session_state.upload_error = "Cannot connect to backend. Make sure FastAPI is running on port 8000."
        st.session_state.upload_status = "error"
    except Exception as e:  # background thread -- never let this vanish silently
        st.session_state.upload_error = str(e)
        st.session_state.upload_status = "error"


status = st.session_state.upload_status

if status == "running":
    st.info(
        "Your report is being generated in the background -- this can take a few "
        "minutes for longer reports. Feel free to switch tabs; it keeps processing "
        "and will be ready here (and on the Results page) when done."
    )
    if st.button("Check progress"):
        st.rerun()
    time.sleep(2)
    st.rerun()
elif status == "done":
    if st.session_state.get("auto_redirect_pending"):
        # BUG FOUND: previously required a manual "View results" click every
        # time, and the raw uploader kept rendering underneath the "done"
        # banner regardless -- both the completed-report state and a fresh
        # upload form were visible at once, which read as broken. Redirect
        # happens automatically the FIRST time a report finishes; the flag is
        # cleared before switching so this fires exactly once, not every time
        # this page reruns while status stays "done".
        st.session_state.auto_redirect_pending = False
        st.switch_page("pages/2_results.py")
    st.success("Report explained!")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("View results", type="primary"):
            st.switch_page("pages/2_results.py")
    with col_b:
        if st.button("Upload another report"):
            st.session_state.upload_status = None
            st.session_state.upload_error = None
            st.rerun()
    st.divider()
elif status == "error":
    st.error(st.session_state.upload_error)
elif status == "manual_needed":
    st.warning("Could not extract results automatically. Please enter values manually below.")

uploaded = None
if status not in ("running", "done"):
    # BUG FOUND: this used to render unconditionally, so a completed report's
    # success banner and a live "pick a new file" uploader showed on screen
    # at the same time -- confusing, and redundant with "Upload another
    # report" above, which already exists specifically to get back here.
    uploaded = st.file_uploader(
        "Choose your report PDF", type=["pdf"],
        help="Your file is processed transiently and never stored raw.",
    )

if uploaded and status != "running":
    profile = st.session_state.get("profile", {})
    st.info(
        f"Profile: Age {profile.get('age', 30)}, "
        f"{profile.get('sex','unknown')}, {profile.get('diet_type','omnivore')} diet"
    )
    if st.button("Explain my results", type="primary"):
        st.session_state.upload_status = "running"
        st.session_state.upload_error = None
        worker = threading.Thread(
            target=_run_upload,
            args=(
                uploaded.getvalue(), uploaded.name,
                st.session_state.get("user_id", "demo_user"),
                profile.get("age", 30), profile.get("sex", "unknown"),
                profile.get("diet_type", "omnivore"),
            ),
            daemon=True,
        )
        # REQUIRED, not cosmetic. st.session_state resolves through the thread's
        # ScriptRunContext. A bare thread has none, so Streamlit logs
        # "missing ScriptRunContext!" and silently falls back to a PROCESS-WIDE
        # mock session state -- the worker's writes would land there while the
        # main thread keeps reading the real one, and the spinner never clears.
        add_script_run_ctx(worker)
        worker.start()
        st.rerun()

st.divider()
st.subheader("Manual Entry (fallback)")
st.caption("If your PDF did not parse correctly, enter values manually here.")

with st.expander("Enter test values manually"):
    col1, col2, col3 = st.columns(3)
    with col1:
        test_name = st.text_input("Test name (e.g. Haemoglobin)")
    with col2:
        test_value = st.number_input("Value", min_value=0.0, format="%.2f")
    with col3:
        test_unit = st.text_input("Unit (e.g. g/dL)")

    if st.button("Add test value"):
        if "manual_results" not in st.session_state:
            st.session_state.manual_results = []
        st.session_state.manual_results.append({
            "raw_name": test_name, "value": test_value, "unit": test_unit,
        })
        st.success(f"Added: {test_name} = {test_value} {test_unit}")

    if st.session_state.get("manual_results"):
        st.write("Entered values:", st.session_state.manual_results)
