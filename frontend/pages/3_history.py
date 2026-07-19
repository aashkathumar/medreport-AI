import os
import streamlit as st
import requests
from datetime import datetime

API = os.environ.get("MEDREPORT_API_URL", "http://localhost:8000/api/v1")

st.title("Your Report History")

user_id = st.session_state.get("user_id", "demo_user")

try:
    resp = requests.get(f"{API}/reports/{user_id}", timeout=10)
    if resp.status_code != 200 or not resp.json().get("reports"):
        st.info("No past reports found. Upload your first report to get started.")
        st.stop()
except requests.exceptions.ConnectionError:
    st.error("Cannot connect to backend.")
    st.stop()

reports = resp.json()["reports"]
st.write(f"You have **{len(reports)}** past report(s).")

for item in reports:
    try:
        date_str = datetime.fromisoformat(item["timestamp"]).strftime("%d %B %Y, %H:%M")
    except Exception:
        date_str = item.get("timestamp", "Unknown date")

    data = item.get("report_data", {})
    flagged = [r for r in data.get("explained_results", []) if r.get("status") != "normal"]

    with st.expander(f"{date_str} -- {len(flagged)} flagged value(s)"):
        st.write(data.get("overall_summary", ""))
        if st.button("Load this report", key=item["report_id"]):
            st.session_state.report = data
            st.switch_page("pages/2_results.py")
