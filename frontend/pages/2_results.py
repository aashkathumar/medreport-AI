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
st.write(report.get("overall_summary", ""))

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
