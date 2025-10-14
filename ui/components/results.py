# ui/components/results.py
import streamlit as st

from src.schemas import (
    ClassificationPipelineResult,
    DisplayContext,
    RegressionPipelineResult,
)
from .results_display import classification, regression


def display_results(context: DisplayContext):
    """
    Renders the results dashboard by checking the specific type of the result object.
    """

    # Check child class for correct render
    if isinstance(context.result, ClassificationPipelineResult):
        classification.render(context)
    elif isinstance(context.result, RegressionPipelineResult):
        regression.render(context.result)
    else:
        st.error(f"Unknown result type: {type(context.result)}")
