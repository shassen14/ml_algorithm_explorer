# ui/components/results_display/classification.py
import streamlit as st
import pandas as pd
from src.schemas import ClassificationPipelineResult, DisplayContext
from ui.components import linear_model_explainer, knn_explainer


def render(context: DisplayContext):
    """Renders the full results dashboard for a classification model run."""

    result = context.result  # The ClassificationPipelineResult object
    st.header(f"Results for Model: {result.model_name}")

    tab1, tab2, tab3 = st.tabs(
        ["📈 Performance (What)", "🔍 Diagnosis (Why)", "⚙️ Algorithm Explainers (How)"]
    )

    # --- TAB 1: PERFORMANCE ---
    with tab1:
        st.subheader("Model Performance Summary")
        st.markdown("##### **Summary Metrics**")
        st.dataframe(pd.DataFrame([result.metrics]))

        st.markdown("##### **Performance Visualizations**")
        col1, col2 = st.columns(2)
        with col1:
            if result.confusion_matrix_fig:
                st.pyplot(result.confusion_matrix_fig)
        with col2:
            if result.roc_curve_fig:
                st.pyplot(result.roc_curve_fig)

        if result.classification_report:
            with st.expander("View Detailed Classification Report"):
                st.code(result.classification_report, language=None)

    # --- TAB 2: DIAGNOSIS ---
    with tab2:
        st.subheader("Diagnosing the Trained Model")

        st.markdown("#### **Geometric & Feature Analysis**")
        if result.decision_boundary_fig:
            st.pyplot(result.decision_boundary_fig)

        # Display the most relevant feature plot
        if result.coefficient_plot_fig:
            st.markdown("##### **Coefficient Inspector**")
            st.pyplot(result.coefficient_plot_fig)
        elif result.feature_importance_fig:
            st.markdown("##### **Feature Importance**")
            st.pyplot(result.feature_importance_fig)

        st.markdown("---")
        st.markdown("#### **Error Analysis**")
        if result.false_positives_df is not None:
            with st.expander("Show Top 5 Worst False Positives"):
                st.dataframe(result.false_positives_df)
        if result.false_negatives_df is not None:
            with st.expander("Show Top 5 Worst False Negatives"):
                st.dataframe(result.false_negatives_df)

    # --- TAB 3: ALGORITHM EXPLAINERS ---
    with tab3:
        st.subheader(f"How a {result.model_name} Works")
        if result.model_name == "Logistic Regression":
            if context.full_df is not None:
                linear_model_explainer.render(context.full_df, context.target_column)
        elif result.model_name == "K-Nearest Neighbors":
            if context.processed_data:
                knn_explainer.render(context)
