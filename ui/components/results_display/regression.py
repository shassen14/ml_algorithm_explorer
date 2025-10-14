# ui/components/results_display/regression.py
import streamlit as st
import pandas as pd
from src.schemas import RegressionPipelineResult


def render(result: RegressionPipelineResult):
    """
    Renders the full results dashboard for a regression model run.
    """
    st.header(f"Results for Regression Model: {result.model_name}")

    if result.target_transform_method != "None":
        st.info(
            f"""**Note:** The target variable was trained using a 
            **{result.target_transform_method}**. 
            All metrics and plots below have been inverse-transformed back to the original scale for interpretation."""
        )

    tab1, tab2 = st.tabs(["📈 Performance (What)", "🔍 Diagnosis (Why)"])

    # --- Performance Tab ---
    with tab1:
        st.subheader("Model Performance Summary")

        st.markdown("##### **Summary Metrics**")
        st.dataframe(pd.DataFrame([result.metrics]))
        st.markdown("---")

        st.markdown("##### **Performance Visualizations**")
        col1, col2 = st.columns(2)
        with col1:
            if result.actual_vs_predicted_fig:
                st.pyplot(result.actual_vs_predicted_fig)
        with col2:
            if result.residuals_fig:
                st.pyplot(result.residuals_fig)

        with st.expander("How to Interpret These Results"):
            st.markdown(
                """
            - **R-squared (R²):** The proportion of the variance in the target variable that is predictable from the features. Ranges from 0 to 1 (or can be negative for very poor models). Higher is better.
            - **Mean Absolute Error (MAE):** The average absolute difference between the predicted and actual values. It's in the same units as the target variable. Lower is better.
            - **Root Mean Squared Error (RMSE):** Similar to MAE, but it penalizes larger errors more heavily. Lower is better.
            - **Actual vs. Predicted Plot:** For a good model, the points should fall closely along the diagonal red line.
            - **Residuals Plot:** For a good model, the points should form a random, structureless cloud around the horizontal zero line. Patterns in this plot indicate problems with the model.
            """
            )

    # --- Diagnosis Tab (Placeholder for now) ---
    with tab2:
        st.info(
            "Diagnosis features like Feature Importance for regression models can be added here."
        )
        # if result.feature_importance_fig:
        #     st.pyplot(result.feature_importance_fig)
