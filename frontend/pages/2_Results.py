import os
import time
import streamlit as st
import requests
from theme import inject as inject_theme

API = os.environ.get("MEDREPORT_API_URL", "http://localhost:8000/api/v1")

inject_theme()
st.title("Your Results")

if st.session_state.get("upload_status") == "running":
    st.info("This may take a moment for larger reports -- please wait. This page will update automatically once it's ready.")
    time.sleep(2)
    st.rerun()

report = st.session_state.get("report")
if not report:
    st.warning("No report loaded. Please upload a PDF first.")
    st.stop()

results = report.get("explained_results", [])
status_counts = {"normal": 0, "low": 0, "high": 0, "unknown": 0}
for r in results:
    status_counts[r.get("status", "unknown")] = status_counts.get(r.get("status", "unknown"), 0) + 1

m1, m2, m3, m4 = st.columns(4)
m1.metric("Tests explained", len(results))
m2.metric("Within normal range", status_counts["normal"])
m3.metric("Outside normal range", status_counts["low"] + status_counts["high"])
m4.metric("Needs GP referral", len(set(report.get("degraded_test_ids", []) or []) | set(report.get("ungrounded_test_ids", []) or [])))

st.divider()
st.subheader("Overall Summary")

# The backend now reports when generation failed instead of quietly serving
# placeholder text that reads like a real explanation.
if report.get("summary_degraded"):
    st.warning(
        "The automated summary could not be generated for this report "
        "(no AI provider was reachable). The text below is a generic "
        "placeholder -- please rely on the individual results."
    )
st.write(report.get("overall_summary", ""))

degraded_ids = set(report.get("degraded_test_ids", []) or [])
if degraded_ids:
    st.warning(
        f"{len(degraded_ids)} result(s) could not be explained automatically "
        "and are marked below."
    )

# Explanations that failed post-generation source verification. These are
# replaced with a GP referral rather than shown, which is the project's
# stated behaviour when NHS/NIH sources don't cover something.
ungrounded_ids = set(report.get("ungrounded_test_ids", []) or [])
if ungrounded_ids:
    st.info(
        f"{len(ungrounded_ids)} result(s) are not covered by the approved "
        "NHS UK / NIH MedlinePlus reference sources. Rather than generate an "
        "unsourced explanation, this tool directs you to your GP for those."
    )

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Topics to discuss with your GP:**")
    for t in report.get("top_gp_topics", []):
        st.markdown(f"- {t}")
with col2:
    st.info(f"**Top lifestyle change:** {report.get('top_lifestyle_change','')}")

st.divider()
st.subheader("Results Explained")

EMOJI = {"normal": ":large_green_circle:", "low": ":red_circle:", "high": ":large_yellow_circle:", "unknown": ":white_circle:"}

_GAUGE_GREEN = "#2E8B57"
_GAUGE_AMBER = "#F0A93A"
_GAUGE_TRACK_BG = "#E4E4E4"


def _range_gauge_html(value, nr_min, nr_max):
    """A horizontal position-in-range gauge, styled after the reference
    sample report that prompted this (a coloured track + a dot marking
    where the patient's value falls), rather than just a status badge.
    Returns None (caller falls back to the existing text caption) unless
    the value is numeric and at least one bound exists -- a categorical
    result ("Negative", "NA") or a fully unknown range has nothing to
    plot.
    """
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if nr_min is None and nr_max is None:
        return None

    lo = float(nr_min) if nr_min is not None else min(val, float(nr_max))
    hi = float(nr_max) if nr_max is not None else max(val, float(nr_min))
    if hi <= lo:
        return None

    span = hi - lo
    pad = span * 0.3
    disp_lo = min(lo - pad, val)
    disp_hi = max(hi + pad, val)
    if disp_hi <= disp_lo:
        return None

    def pct(x):
        return max(0.0, min(100.0, (x - disp_lo) / (disp_hi - disp_lo) * 100))

    green_start, green_end = pct(lo), pct(hi)
    marker = pct(val)

    gradient = (
        f"linear-gradient(to right, "
        f"{_GAUGE_AMBER} 0%, {_GAUGE_AMBER} {green_start:.2f}%, "
        f"{_GAUGE_GREEN} {green_start:.2f}%, {_GAUGE_GREEN} {green_end:.2f}%, "
        f"{_GAUGE_AMBER} {green_end:.2f}%, {_GAUGE_AMBER} 100%)"
    )
    return f"""
<div style="margin: 6px 0 10px 0;">
  <div style="position: relative; height: 10px; border-radius: 5px; background: {gradient};">
    <div style="position: absolute; left: {marker:.2f}%; top: -4px; width: 16px; height: 16px;
      margin-left: -8px; border-radius: 50%; background: #262730; border: 2px solid white;
      box-shadow: 0 1px 3px rgba(0,0,0,0.35);"></div>
  </div>
  <div style="display: flex; justify-content: space-between; font-size: 12px; color: #767676; margin-top: 3px;">
    <span>{lo:g}</span><span>{hi:g}</span>
  </div>
</div>
"""

for r in report.get("explained_results", []):
    status = r.get("status", "unknown")
    emoji = EMOJI.get(status, ":white_circle:")
    with st.expander(
        f"{emoji} **{r['raw_name']}** -- {r['value']} {r['unit']} ({status.upper()})",
        expanded=(status != "normal"),
    ):
        if r.get("test_id") in degraded_ids:
            st.error("No automated explanation was generated for this result.")
        elif r.get("test_id") in ungrounded_ids:
            st.info(
                "Not covered by the approved NHS UK / NIH MedlinePlus sources "
                "- please discuss this result with your GP."
            )
        st.markdown(f"**What it measures:** {r['what_it_measures']}")
        st.markdown(f"**Your result:** {r['what_your_result_means']}")
        if r.get("lifestyle_suggestions"):
            st.markdown("**Lifestyle suggestions:**")
            for s in r["lifestyle_suggestions"]:
                st.markdown(f"- {s}")
        st.markdown(f"**Ask your GP:** _{r['gp_question']}_")
        nr_min = r.get("normal_range_min")
        nr_max = r.get("normal_range_max")
        gauge = _range_gauge_html(r.get("value"), nr_min, nr_max)
        if gauge:
            st.markdown(gauge, unsafe_allow_html=True)
        if nr_min is not None and nr_max is not None:
            st.caption(f"Normal range: {nr_min} -- {nr_max} {r['unit']} - Source: {r.get('source','NHS UK')}")
        elif nr_max is not None:
            st.caption(f"Normal range: up to {nr_max} {r['unit']} - Source: {r.get('source','NHS UK')}")
        elif nr_min is not None:
            st.caption(f"Normal range: {nr_min} {r['unit']} and above - Source: {r.get('source','NHS UK')}")
        # The reference pages this explanation was actually grounded in.
        for url in r.get("source_urls", []) or []:
            st.caption(f"Reference: {url}")
        st.caption(r.get("disclaimer", ""))

st.divider()
st.subheader("Download Your Report")

# BUG FOUND: st.download_button used to live INSIDE the `if
# st.button("Generate...")` block. st.button only reads True for the one
# script-run right after it's clicked -- clicking the download button
# itself triggers a full rerun, st.button(...) evaluates False again on
# that rerun, and the whole block (download button included) stops
# rendering. Not an intentional "acknowledge the download" behaviour --
# the button was just an accidental casualty of Streamlit's rerun model.
# Fixed by generating once, caching the bytes in session_state (tagged
# with the report_id so a stale PDF from a PREVIOUS report can't be
# offered for the current one), and rendering the download button
# unconditionally from that cache so it survives reruns/re-downloads.
current_report_id = report.get("report_id")
cached_pdf = st.session_state.get("pdf_bytes")
cached_pdf_report_id = st.session_state.get("pdf_bytes_report_id")

if st.button("Generate downloadable PDF", type="secondary"):
    with st.spinner("Generating your personalised PDF..."):
        resp = requests.post(f"{API}/download-pdf", json=report, timeout=30)
        if resp.status_code == 200:
            st.session_state.pdf_bytes = resp.content
            st.session_state.pdf_bytes_report_id = current_report_id
            cached_pdf = resp.content
            cached_pdf_report_id = current_report_id
        else:
            st.error("Could not generate PDF. Please try again.")

if cached_pdf is not None and cached_pdf_report_id == current_report_id:
    st.download_button(
        label="Download MedReport AI PDF",
        data=cached_pdf,
        file_name="medreport_ai.pdf",
        mime="application/pdf",
    )

st.divider()
# Ported from feature/developv4.2: a second, always-available entry point
# back to Home for starting a fresh report, for a user who has just
# finished reviewing/downloading this one and wants to explain a
# different one next, without scrolling all the way back up.
if st.button("Explain another report"):
    st.session_state.upload_status = None
    st.session_state.upload_error = None
    st.session_state.manual_results = []
    st.session_state.pdf_bytes = None
    st.session_state.pdf_bytes_report_id = None
    st.switch_page("App.py")
