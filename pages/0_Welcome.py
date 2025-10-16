# pages/0_Welcome.py
import streamlit as st


def clear_old_results():
    """Callback function to clear stale results from session state."""
    keys_to_delete = [
        "last_run_result",
        "processed_data",
        "full_df",
        "recipe",
        "raw_df",
    ]
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]


# --- Page Configuration ---
st.set_page_config(
    page_title="Interactive ML Explorer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Session State Initialization ---
# Initialize session state keys if they don't exist. This is crucial for
# preventing errors when the app reruns.
if "problem_type" not in st.session_state:
    st.session_state.problem_type = None

# --- Page Content ---

st.title("🤖 Welcome to the Interactive Machine Learning Explorer!")

st.markdown(
    """
This application is a hands-on toolkit designed to help you build intuition for how different machine learning algorithms work. 
Navigate through the pages on the left to load your data and explore various ML tasks.
"""
)

st.info("👇 **Start by selecting a machine learning task below.**")

# --- Task Selection ---
# We'll use columns to group supervised and unsupervised tasks.

st.subheader("Supervised Learning: Learning from Labeled Data")
col1, col2 = st.columns(2)
with col1:
    if st.button(
        "**Classification**",
        use_container_width=True,
        on_click=clear_old_results,
        help="Predict a category or label (e.g., Yes/No, Cat/Dog).",
    ):
        st.session_state.problem_type = "Classification"
        st.rerun()  # Rerun the script to update the UI immediately
with col2:
    if st.button(
        "**Regression**",
        use_container_width=True,
        on_click=clear_old_results,
        help="Predict a continuous number (e.g., price, temperature).",
    ):
        st.session_state.problem_type = "Regression"
        st.rerun()

st.subheader("Unsupervised Learning: Finding Patterns in Unlabeled Data")
col3, col4, col5 = st.columns(3)
with col3:
    if st.button(
        "**Clustering**",
        use_container_width=True,
        on_click=clear_old_results,
        help="Discover natural groups or segments in your data.",
    ):
        st.session_state.problem_type = "Clustering"
        st.rerun()
with col4:
    if st.button(
        "**Dimensionality Reduction**",
        use_container_width=True,
        on_click=clear_old_results,
        help="Simplify your data by reducing the number of features for visualization or modeling.",
    ):
        st.session_state.problem_type = "Dimensionality Reduction"
        st.rerun()
with col5:
    if st.button(
        "**Anomaly Detection**",
        use_container_width=True,
        on_click=clear_old_results,
        help="Identify rare or unusual data points that deviate from the norm.",
    ):
        st.session_state.problem_type = "Anomaly Detection"
        st.rerun()

# --- Display Current Selection ---
if st.session_state.problem_type:
    st.success(f"### You have selected: **{st.session_state.problem_type}**")
    st.markdown(
        "#### Please proceed to the **`Data_Loader`** page on the left sidebar to begin!"
    )
else:
    st.warning("No task selected yet.")

# --- Footer ---
st.markdown("---")
st.markdown("Created by Samir | [GitHub](https://github.com/shassen14)")
