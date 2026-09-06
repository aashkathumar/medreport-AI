import os
import time
import streamlit as st
import requests
from theme import inject as inject_theme

API = os.environ.get("MEDREPORT_API_URL", "http://localhost:8000/api/v1")

inject_theme()
st.title("Your Results")

if st.session_state.get("upload_status") == "running":
    st.info("This may take a moment for larger reports, please wait. This page will update automatically once it's ready.")
    time.sleep(2)
    st.rerun()

report = st.session_state.get("report")
if not report:
    st.warning("No report loaded. Please upload a PDF first.")
    st.stop()

# ETHICS CONSTRAINT (Chris Clarke): explained_results no longer carry a
# status/normal-high-low classification at all on this branch (see
# backend/app/models/schemas.py) -- so there is nothing to count here beyond
# how many tests were explained vs. left to a GP referral for lack of
# NHS/NIH coverage, which is a source-coverage fact, not a clinical judgement
# about any patient's value.
results = report.get("explained_results", [])

m1, m2 = st.columns(2)
m1.metric("Tests explained", len(results))
m2.metric("Needs GP referral", len(set(report.get("degraded_test_ids", []) or []) | set(report.get("ungrounded_test_ids", []) or [])))

st.divider()
st.subheader("Overall Summary")

# The backend now reports when generation failed instead of quietly serving
# placeholder text that reads like a real explanation.
if report.get("summary_degraded"):
    st.warning(
        "The automated summary could not be generated for this report "
        "(no AI provider was reachable). The text below is a generic "
        "placeholder. Please rely on the individual results."
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

# ETHICS CONSTRAINT (Chris Clarke): no status emoji/badge, no range gauge, no
# "Your result:" section, no normal-range caption anywhere below -- this
# branch never states what a specific value/status means, only what the
# test is and general guidance for that test category. `value`/`unit` are
# still shown (what was extracted from the report), just not judged.
for r in report.get("explained_results", []):
    with st.expander(f"**{r['raw_name']}** · {r['value']} {r['unit']}"):
        if r.get("test_id") in degraded_ids:
            st.error("No automated explanation was generated for this result.")
        elif r.get("test_id") in ungrounded_ids:
            st.info(
                "Not covered by the approved NHS UK / NIH MedlinePlus sources "
                "- please discuss this result with your GP."
            )
        st.markdown(f"**What it measures:** {r['what_it_measures']}")
        if r.get("lifestyle_suggestions"):
            st.markdown("**Lifestyle suggestions:**")
            for s in r["lifestyle_suggestions"]:
                st.markdown(f"- {s}")
        st.markdown(f"**Ask your GP:** _{r['gp_question']}_")
        # The reference pages this explanation was actually grounded in.
        for url in r.get("source_urls", []) or []:
            st.caption(f"Reference: {url}")
        st.caption(f"Source: {r.get('source','NHS UK')}")
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
# A second, always-available entry point back to Home for starting a fresh
# report -- not a replacement for the "View results" / "Enter a new set of
# tests" pairing on the Home page itself (kept as-is), but this is the point
# where a user has just finished reviewing/downloading a report and may want
# to explain a different one next, without scrolling all the way back up.
if st.button("Explain another report"):
    st.session_state.upload_status = None
    st.session_state.upload_error = None
    st.session_state.manual_results = []
    st.session_state.pdf_bytes = None
    st.session_state.pdf_bytes_report_id = None
    st.switch_page("app.py")
