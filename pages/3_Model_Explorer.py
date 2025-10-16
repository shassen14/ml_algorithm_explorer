# pages/model_explorer.py
import streamlit as st
from src.config.problem_config import PROBLEM_CONFIG
from src.schemas import DisplayContext
from ui.components import sidebar, results

st.title("🔬 Model Explorer")

# --- 1. Prerequisites Check ---
if "problem_type" not in st.session_state or st.session_state.problem_type is None:
    st.warning("Please select a problem type on the Welcome page first.")
    st.stop()
if "processed_data" not in st.session_state:
    st.warning("Please prepare your data on the 🛠️ Data Preparation page first.")
    st.stop()

# --- Load necessary data from session state ---
problem_type = st.session_state["problem_type"]
config = PROBLEM_CONFIG[problem_type]
data_dict = st.session_state["processed_data"]

# --- 2. Get the UI Configuration from the Sidebar ---
# The sidebar component returns a Pydantic `ModelConfig` object.
# Let's rename the variable for clarity as we discussed.
model_run_config = sidebar.render_sidebar(config)

# --- 3. The Main Action: Training the Model ---
if st.button(f"Train {model_run_config.model_name}"):

    # a. Get the correct pipeline and config CLASSES from our problem_config.
    pipeline_class = config["pipeline_class"]
    config_class = config["config_class"]

    # b. Prepare the dictionary of parameters needed to build the config object.
    base_params = {
        "X_train": data_dict["X_train"],
        "X_test": data_dict["X_test"],
        "y_train": data_dict["y_train"],
        "y_test": data_dict["y_test"],
        "model_run_config": model_run_config,
    }

    # c. Instantiate the specific pipeline CONFIGURATION object.
    #    This is where we would add regression-specific args in the future.
    if problem_type == "Classification":
        pipeline_config = config_class(**base_params)
    elif problem_type == "Regression":
        # Example for the future:
        transform_method = st.session_state.get("target_transform_method", "None")
        pipeline_config = config_class(
            **base_params, target_transform_method=transform_method
        )
    else:
        pipeline_config = None
        st.error(f"Pipeline for '{problem_type}' not yet implemented.")

    # d. If the config was built successfully, instantiate and run the pipeline.
    if pipeline_config:
        with st.spinner("Training model and generating results..."):
            try:
                # i. Create an INSTANCE of the pipeline class (e.g., ClassificationPipeline)
                pipeline_instance = pipeline_class(pipeline_config)

                # ii. Call the .run() method to execute the entire process
                pipeline_result = pipeline_instance.run()

                # iii. Save the final result to session state
                st.success("Model training complete!")
                st.session_state["last_run_result"] = pipeline_result

            except Exception as e:
                st.error(f"An error occurred during the pipeline run: {e}")
                if "last_run_result" in st.session_state:
                    del st.session_state["last_run_result"]

# --- 4. Displaying the Results ---
if "last_run_result" in st.session_state:

    # We add a check to ensure the displayed result matches the selected model
    # to prevent showing stale results, as we discussed.
    last_run_model_name = st.session_state.last_run_result.model_name
    if last_run_model_name == model_run_config.model_name:
        st.markdown("---")

        try:
            display_context = DisplayContext(
                result=st.session_state["last_run_result"],
                full_df=st.session_state.get("full_df"),
                target_column=st.session_state.get("target_column"),
                processed_data=st.session_state.get("processed_data"),
            )

            results.display_results(display_context)

        except Exception as e:
            st.error(f"An error occurred while preparing the display: {e}")
