import streamlit as st

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
    flagged = [r for r in data.get("explained_results", []) if r.get("status") != "normal"]

    with st.expander(f"{item['time_label']} -- {len(flagged)} flagged value(s)"):
        st.write(data.get("overall_summary", ""))
        if st.button("Load this report", key=item["report_id"]):
            st.session_state.report = data
            st.switch_page("pages/2_results.py")
