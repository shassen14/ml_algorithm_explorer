# pages/3_Model_Explorer.py
import streamlit as st
from src.config.problem_config import PROBLEM_CONFIG
from src.schemas import ExplainerData
from ui.components import sidebar, results

st.title("🔬 Model Explorer")

# --- 1. State Checks ---
if "problem_type" not in st.session_state or st.session_state.problem_type is None:
    st.warning("Please select a problem type on the Welcome page first.")
elif "processed_data" not in st.session_state:
    st.warning("Please load and process your data on the 'Data Loader' page first.")
else:
    # --- 2. Get State & Config ---
    problem_type = st.session_state["problem_type"]
    config = PROBLEM_CONFIG[problem_type]

    # --- 3. Render Sidebar ---
    model_config = sidebar.render_sidebar(config)

    # --- 4. Run Logic ---
    if st.button(f"Train {model_config.model_name}"):
        data = st.session_state["processed_data"]
        pipeline_function = config["pipeline"]
        with st.spinner("Training in progress..."):
            pipeline_result = pipeline_function(
                data["X_train"],
                data["X_test"],
                data["y_train"],
                data["y_test"],
                model_config,
            )

        # Store the entire result object in session state
        if pipeline_result:
            st.success("Model training complete!")
            st.session_state["last_run_result"] = pipeline_result
        else:
            st.error("Model training failed. Check the terminal for logs.")
            if "last_run_result" in st.session_state:
                del st.session_state["last_run_result"]

    # --- 5. Call the Results Dispatcher ---
    if "last_run_result" in st.session_state:
        # Retrieve the result object from session state
        last_run_result_object = st.session_state["last_run_result"]

        # We gather the necessary raw/intermediate data from session state and
        # package it into our formal ExplainerData schema.
        try:
            explainer_data = ExplainerData(
                full_df=st.session_state.get("full_df"),
                target_column=st.session_state.get("target_column"),
                processed_data=st.session_state.get("processed_data"),
            )
        except Exception as e:
            st.error(f"Failed to create explainer data object: {e}")
            st.stop()

        results.display_results(last_run_result_object, explainer_data)
