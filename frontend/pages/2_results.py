import os
import streamlit as st
import requests

API = os.environ.get("MEDREPORT_API_URL", "http://localhost:8000/api/v1")

st.title("Your Results")

report = st.session_state.get("report")
if not report:
    st.warning("No report loaded. Please upload a PDF first.")
    st.stop()

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

if st.button("Generate downloadable PDF", type="secondary"):
    with st.spinner("Generating your personalised PDF..."):
        resp = requests.post(f"{API}/download-pdf", json=report, timeout=30)
        if resp.status_code == 200:
            st.download_button(
                label="Download MedReport AI PDF",
                data=resp.content,
                file_name="medreport_ai.pdf",
                mime="application/pdf",
            )
        else:
            st.error("Could not generate PDF. Please try again.")
