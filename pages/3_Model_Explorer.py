# pages/3_Model_Explorer.py
import streamlit as st
from src.config.problem_config import PROBLEM_CONFIG
from src.schemas import (
    ClassificationPipelineConfig,
    DisplayContext,
    RegressionPipelineConfig,
)
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
    model_run_config = sidebar.render_sidebar(config)

    # --- 4. Run Logic ---

    transform_method = st.session_state.get("target_transform_method", "None")

    if st.button(f"Train {model_run_config.model_name}"):
        data = st.session_state["processed_data"]
        pipeline_function = config["pipeline"]
        pipeline_config = None

        base_params = {
            "X_train": data["X_train"],
            "X_test": data["X_test"],
            "y_train": data["y_train"],
            "y_test": data["y_test"],
            "model_run_config": model_run_config,
        }

        if problem_type == "Classification":
            pipeline_config = ClassificationPipelineConfig(**base_params)

        elif problem_type == "Regression":
            transform_method = st.session_state.get("target_transform_method", "None")
            pipeline_config = RegressionPipelineConfig(
                **base_params,
                target_transform_method=transform_method,  # Pass the specific arg here
            )

        with st.spinner("Training in progress..."):
            pipeline_result = pipeline_function(pipeline_config)

        # Store the entire result object in session state
        if pipeline_result:
            st.success("Model training complete!")

            # Store the result object AND the name of the model that generated it
            st.session_state["last_run_result"] = pipeline_result
            st.session_state["last_run_model_name"] = model_run_config.model_name
        else:
            st.error("Model training failed. Check the terminal for logs.")
            if "last_run_result" in st.session_state:
                del st.session_state["last_run_result"]
            if "last_run_model_name" in st.session_state:
                del st.session_state["last_run_model_name"]

    # --- 5. Call the Results Dispatcher ---
    if "last_run_result" in st.session_state:

        if st.session_state.get("last_run_model_name") == model_run_config.model_name:

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
