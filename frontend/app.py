import streamlit as st

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
if "profile" not in st.session_state:
    st.session_state.profile = {}

with st.sidebar:
    st.header("Your Profile")
    st.session_state.profile["age"] = st.slider("Age", 16, 90, 30)
    st.session_state.profile["sex"] = st.selectbox("Sex", ["prefer_not_to_say", "male", "female"])
    st.session_state.profile["diet_type"] = st.selectbox("Diet", ["omnivore", "vegetarian", "vegan"])
    st.caption("Your profile personalises the lifestyle recommendations.")
    st.divider()
    st.session_state.user_id = st.text_input("User ID", value="demo_user")

st.title("MedReport AI")
st.caption(
    "Upload your blood test or urinalysis report and receive plain-English "
    "explanations with personalised lifestyle guidance."
)
st.info("Use the pages in the sidebar to upload a report, view results, or see your history.")
