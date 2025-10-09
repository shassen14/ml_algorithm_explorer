# ui/components/eda_multivariate.py
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def render(df):
    """Renders the multivariate analysis section (correlation heatmap)."""

    st.markdown("### See the Big Picture with a Correlation Heatmap")

    numerical_df = df.select_dtypes(include=np.number)

    if numerical_df.shape[1] < 2:
        st.info(
            "Not enough numerical features (at least 2 required) to generate a correlation heatmap."
        )
        return

    st.markdown(
        "This plot shows the linear correlation between numerical features. Values close to 1 (bright) mean a strong positive correlation, while values near -1 (dark) mean a strong negative correlation."
    )

    # Checkbox to toggle annotations
    show_values = st.checkbox("Show correlation values on the map", value=True)

    try:
        corr_matrix = numerical_df.corr()

        # Disable annotations if the matrix is too large, to prevent clutter
        if numerical_df.shape[1] > 15:
            show_values = False
            st.warning("Hiding correlation values as the number of features is large.")

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=show_values,
            fmt=".2f",
            cmap="coolwarm",
            linewidths=0.5,
            ax=ax,
        )
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"An error occurred while creating the heatmap: {e}")
