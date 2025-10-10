# ui/components/eda_summary_report.py
import streamlit as st
from src.analysis.auto_eda import generate_insights


def render(df, target_column, problem_type):
    """
    Renders the Automated EDA 'Key Insights' report section.
    """
    st.markdown("### 🤖 Automated Key Insights Report")

    # Button to trigger the analysis
    if st.button("Generate Report", key="generate_eda_report"):
        with st.spinner("Analyzing your dataset... This might take a moment."):
            # Call the backend analysis function
            insights = generate_insights(df, target_column, problem_type)

            # Store the results in session state to persist them
            st.session_state["eda_insights"] = insights

    # If insights have been generated, display them
    if "eda_insights" in st.session_state:
        insights = st.session_state["eda_insights"]
        if not insights:
            st.info(
                "No specific insights were automatically generated for this dataset."
            )
        else:
            st.write("Here are the key findings from the automated analysis:")
            for insight in insights:
                if insight["type"] == "info":
                    st.info(insight["text"], icon="ℹ️")
                elif insight["type"] == "warning":
                    st.warning(insight["text"], icon="⚠️")
                elif insight["type"] == "error":
                    st.error(insight["text"], icon="❗️")
                elif insight["type"] == "success":
                    st.success(insight["text"], icon="✅")
