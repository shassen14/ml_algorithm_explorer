# ui/components/results_display/classification.py
import streamlit as st
import pandas as pd
from ui.components import (
    linear_model_explainer,
    pros_cons_display,
)


def render(run_results, model_name, full_df, target_column):
    """
    Renders the full results dashboard specifically for CLASSIFICATION problems.
    """
    tab_titles = [
        "📈 Performance (What)",
        "🔍 Diagnosis (Why)",
        "⚙️ Algorithm Explainers (How)",
    ]
    tab1, tab2, tab3 = st.tabs(tab_titles)

    # ==============================================================================
    # TAB 1: PERFORMANCE ("WHAT") - The high-level verdict
    # ==============================================================================
    with tab1:
        st.subheader("Model Performance Summary")

        metrics = run_results.get("metrics", {})
        if not metrics:
            st.warning("No metrics were generated for this run.")
            return  # Exit early if there's nothing to show

        st.write("**Summary Metrics:**")
        st.dataframe(pd.DataFrame([metrics]))

        col1, col2 = st.columns(2)
        with col1:
            if "confusion_matrix_fig" in run_results:
                st.write("**Confusion Matrix:**")
                st.pyplot(run_results["confusion_matrix_fig"])
        with col2:
            if "roc_curve_fig" in run_results:
                st.markdown("---")  # Add a separator
                st.write("**Receiver Operating Characteristic (ROC) Curve:**")
                st.pyplot(run_results["roc_curve_fig"])

        # Detailed, Collapsible Information
        st.markdown("---")
        st.markdown("##### **Detailed Reports & Explanations**")

        with st.expander("How to Interpret These Results"):
            st.markdown(
                """
            **Summary Metrics:**
            - **Accuracy:** The overall percentage of correct predictions. Can be misleading on imbalanced datasets.
            - **Precision:** Of the predictions made for the positive class, how many were correct? (Measures "false alarms").
            - **Recall:** Of all the actual positive instances, how many did the model find? (Measures "missed cases").
            - **F1-Score:** The harmonic mean of Precision and Recall, providing a balanced score.
            - **AUC (Area Under Curve):** A measure of the model's ability to distinguish between classes (1.0 is perfect, 0.5 is a random guess).

            **Confusion Matrix:**
            - A direct breakdown of the model's predictions: True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).

            **ROC Curve:**
            - Visualizes the trade-off between the True Positive Rate (Recall) and the False Positive Rate. A better model has a curve that bows towards the top-left corner.
            """
            )

        if run_results.get("classification_report"):
            st.write("**Detailed Classification Report:**")
            st.code(run_results["classification_report"], language=None)

    # ==============================================================================
    # TAB 2: DIAGNOSIS ("WHY") - Understanding the specific model run
    # This tab is now the home for our diagnostic plots.
    # ==============================================================================
    with tab2:
        st.subheader("Diagnosing the Trained Model")

        st.markdown("#### **Geometric Analysis**")
        if run_results.get("decision_boundary_fig"):
            st.info(
                "This plot is the ultimate summary of the model. It visualizes how the model has partitioned the data space, projected into 2D using PCA."
            )
            st.pyplot(run_results["decision_boundary_fig"])
        else:
            st.info(
                "A decision boundary plot could not be generated for this model or data."
            )
        st.markdown("---")

        st.markdown("#### **Feature Analysis**")
        st.info(
            "These plots explain *which features* the model found important for making its predictions on your data."
        )

        # --- Feature Importance / Coefficient Inspector ---
        # We can use columns for a cleaner layout
        col1, col2 = st.columns(2)
        with col1:
            # For Tree-Based models
            if run_results.get("feature_importance_fig"):
                st.write("**Feature Importance (MDI):**")
                st.pyplot(run_results["feature_importance_fig"])

            # For Linear Models
            if run_results.get("coefficient_plot_fig"):
                st.write("**Coefficient Inspector:**")
                st.pyplot(run_results["coefficient_plot_fig"])
        with col2:
            # Placeholder for future, more advanced plots like Permutation Importance or SHAP
            st.write("")  # Empty column for now, can be filled later
            st.write("")

        st.markdown("---")

        st.markdown("#### **Error Analysis**")
        st.info(
            "Here we can inspect the specific data points where the model was most confident, yet incorrect."
        )

        # --- Error Analysis Tables ---
        fp_df = run_results.get("false_positives_df")
        fn_df = run_results.get("false_negatives_df")

        if fp_df is not None and not fp_df.empty:
            with st.expander("Show Top 5 Worst False Positives"):
                st.write(
                    "Examples the model incorrectly predicted as 'Positive' with high confidence."
                )
                st.dataframe(fp_df)

        if fn_df is not None and not fn_df.empty:
            with st.expander("Show Top 5 Worst False Negatives"):
                st.write(
                    "'Positive' examples the model missed, predicting them as 'Negative' with high confidence."
                )
                st.dataframe(fn_df)

    # ==============================================================================
    # TAB 3: ALGORITHM EXPLAINERS ("HOW") - The educational deep dive
    # This tab is now clean and focused only on teaching the algorithm.
    # ==============================================================================
    with tab3:
        st.subheader(f"How a {model_name} Works")
        if model_name == "Logistic Regression":
            if full_df is not None and target_column is not None:
                linear_model_explainer.render(full_df, target_column)

        elif model_name == "K-Nearest Neighbors":
            st.info(
                "(Placeholder) The interactive K-NN Neighbor Inspector will be displayed here."
            )
        elif model_name in ["Random Forest", "XGBoost"]:
            st.info(
                "(Placeholder) The Decision Tree visualizer and from-scratch code will be displayed here."
            )

        # General pros and cons
        pros_cons_display.render(model_name)
