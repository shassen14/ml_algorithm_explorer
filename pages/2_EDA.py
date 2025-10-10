# pages/2_EDA.py
import streamlit as st
from ui.components import (
    eda_summary_report,
    eda_univariate,
    eda_bivariate,
    eda_multivariate,
)

st.set_page_config(layout="wide")  # Ensure wide layout for better plot viewing

st.title("📊 Exploratory Data Analysis (EDA)")

st.markdown(
    """
Welcome to the EDA workbench! 
Exploratory Data Analysis is the crucial first step in any machine learning project. 
The goal here is to understand the fundamental characteristics of your dataset. 
Use the tools below to investigate feature distributions, spot relationships, and form hypotheses about what drives your target variable. 
A deep understanding of your data is the key to building an effective model.
    """
)

# Check if data is loaded into session state
if "full_df" not in st.session_state:
    st.warning("Please load your data on the 💾 Data Loader page to begin.")
else:
    df = st.session_state["full_df"]
    target_column = st.session_state.get("target_column", None)
    problem_type = st.session_state.get("problem_type", None)

    st.markdown("---")
    eda_summary_report.render(df, target_column, problem_type)
    st.markdown("---")

    with st.expander("Univariate Analysis: Inspecting Single Features"):
        eda_univariate.render(df)
    with st.expander(
        "Bivariate Analysis: Exploring Relationships Between Two Features"
    ):
        eda_bivariate.render(df, target_column)
    with st.expander("Multivariate Analysis: Getting the Big Picture"):
        eda_multivariate.render(df)
