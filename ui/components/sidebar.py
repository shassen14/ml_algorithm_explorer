# ui/components/sidebar.py
import streamlit as st


def render_sidebar(problem_config):
    """
    Renders the model selection and hyperparameter tuning sidebar.
    This version is robust against models with no defined hyperparameters.
    """
    with st.sidebar:
        st.header("⚙️ Model Configuration")

        available_models = list(problem_config["models"].keys())
        selected_model_name = st.selectbox("Choose a Model", available_models)

        st.subheader("Hyperparameters")

        user_hyperparameters = {}
        # Get the hyperparameter UI definitions for the currently selected model
        model_params_config = problem_config["hyperparameters"].get(
            selected_model_name, {}
        )

        # Check if there are any hyperparameters to display for this model
        if not model_params_config:
            st.info("This model has no tunable hyperparameters.")
        else:
            # Loop through the defined hyperparameters and create the appropriate widget
            for param, settings in model_params_config.items():
                # Defensive check: ensure the 'widget' key exists before proceeding
                if "widget" not in settings:
                    st.error(f"Configuration error for {param}: missing 'widget' key.")
                    continue  # Skip this parameter and move to the next

                widget_type = settings["widget"]
                label = settings["label"]

                if widget_type == "slider":
                    user_hyperparameters[param] = st.slider(
                        label=label,
                        min_value=settings["min_value"],
                        max_value=settings["max_value"],
                        value=settings["value"],
                        step=settings.get("step"),
                        help=settings.get("help"),
                    )

                elif widget_type == "number_input":
                    val = st.number_input(
                        label=label,
                        min_value=settings["min_value"],
                        max_value=settings["max_value"],
                        value=settings["value"],
                        step=settings.get("step"),
                        help=settings.get("help"),
                    )
                    # Convert 0 from UI to None for models where it's a valid setting (e.g., max_depth)
                    user_hyperparameters[param] = (
                        val if param != "max_depth" or val != 0 else None
                    )

                elif widget_type == "selectbox":
                    user_hyperparameters[param] = st.selectbox(
                        label=label,
                        options=settings["options"],
                        index=settings.get("index", 0),
                        help=settings.get("help"),
                    )

        # Assemble the final model config object for the pipeline
        model_config_for_pipeline = {
            "model_name": selected_model_name,
            "model_class": problem_config["models"][selected_model_name],
            "hyperparameters": user_hyperparameters,
        }

        return model_config_for_pipeline
