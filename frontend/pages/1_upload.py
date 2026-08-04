import os
import streamlit as st
import requests

# Set MEDREPORT_API_URL as an env var / Streamlit secret when deployed to the cloud.
API = os.environ.get("MEDREPORT_API_URL", "http://localhost:8000/api/v1")

st.title("Upload Your Report")
st.write("Upload a PDF blood test or urinalysis report.")

uploaded = st.file_uploader(
    "Choose your report PDF", type=["pdf"],
    help="Your file is processed transiently and never stored raw.",
)

if uploaded:
    profile = st.session_state.get("profile", {})
    st.info(
        f"Profile: Age {profile.get('age', 30)}, "
        f"{profile.get('sex','unknown')}, {profile.get('diet_type','omnivore')} diet"
    )
    if st.button("Explain my results", type="primary"):
        with st.spinner("Extracting and explaining your results -- this takes 10-15 seconds..."):
            try:
                resp = requests.post(
                    f"{API}/upload-pdf",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                    params={
                        "user_id": st.session_state.get("user_id", "demo_user"),
                        "age": profile.get("age", 30),
                        "sex": profile.get("sex", "unknown"),
                        "diet_type": profile.get("diet_type", "omnivore"),
                    },
                    timeout=300,
                )
                if resp.status_code == 200:
                    st.session_state.report = resp.json()
                    st.success("Report explained! Go to Results page.")
                    st.switch_page("pages/2_results.py")
                elif resp.status_code == 422:
                    st.warning("Could not extract results automatically. Please enter values manually below.")
                else:
                    st.error(f"Error: {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend. Make sure FastAPI is running on port 8000.")

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
