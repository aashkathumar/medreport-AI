import streamlit as st
from theme import inject as inject_theme

inject_theme()
st.title("Your Report History")
st.caption(
    "Reports generated during this browser session only. Nothing is stored "
    "on the server -- close this tab and this list is gone."
)

reports = st.session_state.get("session_reports", [])

if not reports:
    st.info("No reports generated yet this session. Upload a report to get started.")
    st.stop()

st.write(f"You have **{len(reports)}** report(s) from this session.")

for item in reports:
    data = item["report_data"]
    # ETHICS CONSTRAINT (Chris Clarke): no status field on explained_results
    # on this branch, so no "flagged value(s)" count -- explained_results
    # count alone, not a high/low tally.
    count = len(data.get("explained_results", []))

    with st.expander(f"{item['time_label']} -- {count} test(s) explained"):
        st.write(data.get("overall_summary", ""))
        if st.button("Load this report", key=item["report_id"]):
            st.session_state.report = data
            st.switch_page("pages/2_Results.py")
