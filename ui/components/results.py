# ui/components/results.py
import streamlit as st
from .results_display import classification


def display_results(run_results, model_name, problem_type, full_df, target_column):
    """
    Top-level dispatcher component that calls the appropriate results display
    based on the problem type.

    Args:
        run_results (dict): The dictionary from the pipeline run.
        model_name (str): The name of the model that was run.
        problem_type (str): "Classification" or "Regression".
        full_df (pd.DataFrame): The complete, original dataframe for explainers.
        target_column (str): The name of the target column.
    """
    if problem_type == "Classification":
        classification.render(
            run_results=run_results,
            model_name=model_name,
            full_df=full_df,
            target_column=target_column,
        )
    elif problem_type == "Regression":
        # regression.render(...) # TODO: We'll build this later
        st.info("Regression results display is not yet implemented.")
    else:
        st.error(f"No results display component found for problem type: {problem_type}")
