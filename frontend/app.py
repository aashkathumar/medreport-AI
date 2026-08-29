import os
import threading
import time
import uuid
from datetime import datetime
import streamlit as st
import requests
from streamlit.runtime.scriptrunner import add_script_run_ctx
from theme import inject as inject_theme

# Set MEDREPORT_API_URL as an env var / Streamlit secret when deployed to the cloud.
API = os.environ.get("MEDREPORT_API_URL", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="MedReport AI",
    page_icon=":stethoscope:",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme()

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

# Original illustration (not traced from any reference image) in the same
# visual language as the moodboard the user shared -- a card, floating
# accent blobs, colour-coded rows -- but drawn around what this app
# actually shows a patient: a report with status-flagged result rows and a
# pulse-line motif, rather than a generic doctor/calendar scene.
_HERO_SVG = """
<svg viewBox="0 0 600 400" xmlns="http://www.w3.org/2000/svg" style="width:100%; height:auto; max-width:360px;">
  <circle cx="440" cy="110" r="90" fill="#FFD166" opacity="0.85"/>
  <circle cx="490" cy="260" r="115" fill="#FF7A59" opacity="0.8"/>
  <rect x="140" y="60" width="300" height="280" rx="22" fill="#FFFFFF" stroke="#E3E6FA" stroke-width="2"/>
  <rect x="168" y="90" width="244" height="30" rx="7" fill="#EEF0FE"/>
  <circle cx="186" cy="105" r="7" fill="#4F5FE0"/>
  <rect x="204" y="99" width="120" height="11" rx="5.5" fill="#C7CCF2"/>
  <g>
    <rect x="168" y="140" width="244" height="36" rx="9" fill="#F4F6FE"/>
    <circle cx="188" cy="158" r="8" fill="#2FBF71"/>
    <rect x="208" y="151" width="150" height="9" rx="4.5" fill="#333A66"/>
    <rect x="208" y="164" width="95" height="6" rx="3" fill="#A9AFDA"/>
  </g>
  <g>
    <rect x="168" y="188" width="244" height="36" rx="9" fill="#F4F6FE"/>
    <circle cx="188" cy="206" r="8" fill="#FFB020"/>
    <rect x="208" y="199" width="160" height="9" rx="4.5" fill="#333A66"/>
    <rect x="208" y="212" width="105" height="6" rx="3" fill="#A9AFDA"/>
  </g>
  <g>
    <rect x="168" y="236" width="244" height="36" rx="9" fill="#F4F6FE"/>
    <circle cx="188" cy="254" r="8" fill="#2FBF71"/>
    <rect x="208" y="247" width="135" height="9" rx="4.5" fill="#333A66"/>
    <rect x="208" y="260" width="85" height="6" rx="3" fill="#A9AFDA"/>
  </g>
  <polyline points="168,300 200,300 213,276 227,324 240,300 412,300"
    fill="none" stroke="#4F5FE0" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

st.markdown(
    '<div style="font-size:1.15rem; font-weight:700; color:#4F5FE0; '
    'letter-spacing:0.02em; margin-bottom:14px;">MedReport AI</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f"""
<div style="background: linear-gradient(135deg, #4F5FE0 0%, #6C63FF 100%);
    border-radius: 22px; padding: 40px 44px; margin-bottom: 28px;
    display: flex; align-items: center; gap: 32px; flex-wrap: wrap;">
  <div style="flex: 1 1 320px; min-width: 260px;">
    <div style="color: #FFFFFF; font-size: 2.1rem; font-weight: 700; line-height: 1.18; margin-bottom: 14px;">
      Understand Your Lab Results, Instantly
    </div>
    <div style="color: #E7E9FC; font-size: 1.02rem; line-height: 1.55;">
      Upload your blood test or urinalysis report and receive plain-English
      explanations with personalised lifestyle guidance -- grounded only in
      NHS UK and NIH MedlinePlus sources.
    </div>
  </div>
  <div style="flex: 1 1 280px; min-width: 240px; max-width: 360px;">
    {_HERO_SVG}
  </div>
</div>
""",
    unsafe_allow_html=True,
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
        "This may take a moment for larger reports -- please wait. Your report is "
        "being generated in the background, so feel free to switch tabs; it keeps "
        "processing and will be ready here (and on the Results page) when done."
    )
    # BUG FOUND: the page already polls itself every 2s below regardless --
    # a "Check progress" button here did nothing the auto-poll wasn't
    # already doing, and clicking it right as generation finished could
    # trigger the done-state auto-redirect while the user was mid-click,
    # producing a confusing double transition (redirected to Results, then
    # navigating back to this page showed "Report explained!" again).
    # Removed; the silent auto-poll is enough on its own.
    time.sleep(2)
    st.rerun()
elif status == "done":
    # CHANGED: used to auto-redirect to Results the FIRST time a report
    # finished, then fall back to this manual banner on every later visit --
    # two different experiences for the same "done" state, depending on
    # whether this was the first time it was seen. Now always shows this
    # banner and always requires the explicit click, so the behaviour is
    # identical regardless of when or how many times you land here.
    st.success("Report explained!")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("View results", type="primary"):
            st.switch_page("pages/2_Results.py")
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
    st.caption("This may take a moment for larger reports -- please wait.")
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

def _run_manual_explain(manual_results, user_id, age, sex, diet_type):
    """Mirrors _run_upload above -- same background-thread pattern, same
    session_state fields, so the Results page and status banner behave
    identically regardless of which path (PDF or manual) produced the report.
    """
    try:
        resp = requests.post(
            f"{API}/explain-manual",
            json={
                "results": manual_results,
                "user_id": user_id, "age": age, "sex": sex, "diet_type": diet_type,
            },
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
        else:
            st.session_state.upload_error = f"Error {resp.status_code}: {resp.text[:300]}"
            st.session_state.upload_status = "error"
    except requests.exceptions.ConnectionError:
        st.session_state.upload_error = "Cannot connect to backend. Make sure FastAPI is running on port 8000."
        st.session_state.upload_status = "error"
    except Exception as e:
        st.session_state.upload_error = str(e)
        st.session_state.upload_status = "error"


st.divider()
st.subheader("Manual Entry (fallback)")
st.caption("If your PDF did not parse correctly, enter values manually here.")

if "manual_results" not in st.session_state:
    st.session_state.manual_results = []

# BUG FOUND: this expander used to always default to collapsed
# (st.expander's implicit expanded=False) on every rerun, including the
# rerun that fires the instant "Explain my results" finishes. That made a
# just-submitted, still-populated manual_results list look like it had
# been wiped, when it was only hidden. Force it open whenever there's
# something in it, so submitted values stay visible instead of appearing
# to vanish behind the success banner.
with st.expander("Enter test values manually", expanded=bool(st.session_state.manual_results)):
    col1, col2, col3 = st.columns(3)
    with col1:
        test_name = st.text_input("Test name (e.g. Haemoglobin)")
    with col2:
        # BUG FOUND: st.number_input always pre-fills the field with "0.00".
        # Clicking into it (unlike triple-click or Cmd+A) just places the
        # cursor somewhere inside that text rather than selecting it, so
        # typing "14.5" merges with the leftover zeros instead of replacing
        # them (e.g. produced "0.00145"). A plain text field has nothing
        # pre-filled to collide with; the string is parsed to a float only
        # when "Add test value" is clicked.
        test_value_raw = st.text_input("Value (e.g. 14.5)")
    with col3:
        test_unit = st.text_input("Unit (e.g. g/dL)")

    if st.button("Add test value"):
        name = test_name.strip()
        # BUG FOUND: this used to append unconditionally, so an empty click
        # (blank name) added a ghost {"raw_name": "", ...} row, and adding
        # the SAME test name twice (e.g. once without a unit, then again
        # after typing it) produced two separate entries instead of one
        # corrected one. Fixed: require a name, and match on it
        # case-insensitively to REPLACE an existing entry rather than
        # duplicate it.
        try:
            test_value = float(test_value_raw.strip())
        except ValueError:
            test_value = None

        if not name:
            st.warning("Please enter a test name before adding.")
        elif test_value is None:
            st.warning("Please enter a valid numeric value.")
        else:
            existing = next(
                (i for i, r in enumerate(st.session_state.manual_results)
                 if r["raw_name"].strip().lower() == name.lower()),
                None,
            )
            entry = {"raw_name": name, "value": test_value, "unit": test_unit.strip()}
            if existing is not None:
                st.session_state.manual_results[existing] = entry
                st.success(f"Updated: {name} = {test_value} {test_unit}")
            else:
                st.session_state.manual_results.append(entry)
                st.success(f"Added: {name} = {test_value} {test_unit}")

    if st.session_state.manual_results:
        st.write("**Entered values:**")
        for i, r in enumerate(st.session_state.manual_results):
            rcol1, rcol2 = st.columns([5, 1])
            with rcol1:
                st.write(f"- {r['raw_name']} = {r['value']} {r['unit']}")
            with rcol2:
                if st.button("Remove", key=f"remove_manual_{i}"):
                    st.session_state.manual_results.pop(i)
                    st.rerun()

        # BUG FOUND: there was previously no way to actually submit these
        # entries anywhere -- they only ever sat in session_state as a raw
        # JSON dump. Wired to a new backend endpoint (/explain-manual) that
        # mirrors /upload-pdf's own pipeline.
        if st.button("Explain my results", type="primary", key="explain_manual_btn"):
            profile = st.session_state.get("profile", {})
            st.session_state.upload_status = "running"
            st.session_state.upload_error = None
            worker = threading.Thread(
                target=_run_manual_explain,
                args=(
                    list(st.session_state.manual_results),
                    st.session_state.get("user_id", "demo_user"),
                    profile.get("age", 30), profile.get("sex", "unknown"),
                    profile.get("diet_type", "omnivore"),
                ),
                daemon=True,
            )
            add_script_run_ctx(worker)
            worker.start()
            st.rerun()
