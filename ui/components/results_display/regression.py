# ui/components/results_display/regression.py
import streamlit as st
import pandas as pd
from src.schemas import DisplayContext


def render(context: DisplayContext):
    """Renders the full results dashboard for a regression model run."""

    result = context.result  # The RegressionPipelineResult object
    st.header(f"Results for Regression Model: {result.model_name}")

    if result.target_transform_method != "None":
        st.info(
            f"""**Note:** The target variable was trained using a **{result.target_transform_method}**. 
            All metrics and plots below are in the original, untransformed scale."""
        )

    tab1, tab2, tab3 = st.tabs(
        ["📈 Performance (What)", "🔍 Diagnosis (Why)", "⚙️ Algorithm Explainers (How)"]
    )

    # --- TAB 1: PERFORMANCE ---
    with tab1:
        st.subheader("Model Performance Summary")

        st.markdown("##### **Summary Metrics**")
        st.dataframe(pd.DataFrame([result.metrics]))

        with st.expander("How to Interpret These Metrics"):
            st.markdown(
                """
            - **R-squared (R²):** Represents the percentage of the target variable's variance that the model successfully explains. A score of 1.0 is perfect; 0 means the model is no better than just predicting the average value.
            - **Mean Absolute Error (MAE):** The average absolute difference between the model's prediction and the actual value. This is your average prediction error in the original units (e.g., dollars).
            - **Root Mean Squared Error (RMSE):** Similar to MAE, but it penalizes larger errors more heavily. A large difference between MAE and RMSE suggests the model has some significant outlier errors.
            """
            )
        st.markdown("---")

        st.markdown("##### **Performance Visualizations**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Actual vs. Predicted Values")
            st.info(
                """
            **Goal:** For a perfect model, all the blue dots would lie exactly on the red dashed line.
            
            This plot shows the model's predictions against the true values. The spread of the dots around the line gives you a visual sense of the model's accuracy and where it makes the biggest errors.
            """
            )
            if result.actual_vs_predicted_fig:
                st.pyplot(result.actual_vs_predicted_fig)
        with col2:
            st.markdown("#### Residuals vs. Predicted Values")
            st.info(
                """
            **Goal:** For a good model, the blue dots should form a random, shapeless cloud with no clear pattern, centered around the red "Zero Error" line.
            
            This is a key diagnostic plot. A "residual" is simply the prediction error (`Actual - Predicted`). Patterns in this plot reveal systematic problems with the model. For example:
            - **A 'fanning out' shape (Megaphone):** The model's errors get larger as the predicted value increases (called heteroscedasticity).
            - **A curved shape:** The model is failing to capture a non-linear relationship in the data.
            """
            )
            if result.residuals_fig:
                st.pyplot(result.residuals_fig)

    # --- TAB 2: DIAGNOSIS ---
    with tab2:
        st.subheader("Diagnosing the Trained Model")

        if result.coefficient_plot_fig:
            st.markdown("##### **Coefficient Inspector**")
            st.pyplot(result.coefficient_plot_fig)
        elif result.feature_importance_fig:
            st.markdown("##### **Feature Importance**")
            st.pyplot(result.feature_importance_fig)

        # Error analysis could be added here in the future

    # --- TAB 3: ALGORITHM EXPLAINERS ---
    with tab3:
        st.subheader(f"How a {result.model_name} Works")
        st.info("Interactive explainers for regression models are coming soon!")
