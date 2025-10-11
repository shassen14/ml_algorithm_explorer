# ui/components/results.py
import streamlit as st

from src.schemas import (
    BasePipelineResult,
    ClassificationPipelineResult,
    ExplainerData,
    RegressionPipelineResult,
)
from .results_display import classification


def display_results(run_results: BasePipelineResult, explainer_data: ExplainerData):
    """
    Renders the results dashboard by checking the specific type of the result object.
    """

    # Check child class for correct render
    if isinstance(run_results, ClassificationPipelineResult):
        classification.render(run_results, explainer_data)
    # elif isinstance(run_results, RegressionPipelineResult):
    #     regression.render(run_results, explainer_data)
    else:
        st.error("Unknown result type received.")
