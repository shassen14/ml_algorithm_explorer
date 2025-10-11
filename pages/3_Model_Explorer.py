# pages/3_Model_Explorer.py
import streamlit as st
from src.config.problem_config import PROBLEM_CONFIG
from src.schemas import DisplayContext
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

            # Store the result object AND the name of the model that generated it
            st.session_state["last_run_result"] = pipeline_result
            st.session_state["last_run_model_name"] = model_config.model_name
        else:
            st.error("Model training failed. Check the terminal for logs.")
            if "last_run_result" in st.session_state:
                del st.session_state["last_run_result"]
            if "last_run_model_name" in st.session_state:
                del st.session_state["last_run_model_name"]

    # --- 5. Call the Results Dispatcher ---
    if "last_run_result" in st.session_state:

        if st.session_state.get("last_run_model_name") == model_config.model_name:

            # We gather the necessary raw/intermediate data from session state and
            # package it into our formal DisplayContext schema.
            try:
                display_context = DisplayContext(
                    result=st.session_state["last_run_result"],
                    full_df=st.session_state.get("full_df"),
                    target_column=st.session_state.get("target_column"),
                    processed_data=st.session_state.get("processed_data"),
                )
                results.display_results(display_context)

            except Exception as e:
                st.error(f"Failed to create display context: {e}")
                del st.session_state["last_run_result"]
                del st.session_state["last_run_model_name"]
                st.rerun()
