# pages/3_Model_Explorer.py
import streamlit as st
from src.config.problem_config import PROBLEM_CONFIG
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
    if st.button(f"Train {model_config['model_name']}"):
        data = st.session_state["processed_data"]
        pipeline_function = config["pipeline"]
        with st.spinner("Training in progress..."):
            run_results, trained_pipeline = pipeline_function(
                data["X_train"],
                data["X_test"],
                data["y_train"],
                data["y_test"],
                model_config,
            )

        if run_results:
            st.success("Model training complete!")
            st.session_state["last_run_results"] = run_results
            st.session_state["last_run_pipeline"] = trained_pipeline
            st.session_state["last_run_model_name"] = model_config["model_name"]
        else:
            st.error("Model training failed. Check the terminal for logs.")

    # --- 5. Call the Results Dispatcher ---
    if "last_run_results" in st.session_state:
        results.display_results(
            run_results=st.session_state["last_run_results"],
            model_name=st.session_state["last_run_model_name"],
            problem_type=st.session_state["problem_type"],
            full_df=st.session_state["full_df"],
            target_column=st.session_state["target_column"],
        )
