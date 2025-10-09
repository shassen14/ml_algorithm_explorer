# ui/components/eda_bivariate.py
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

SAMPLING_THRESHOLD = 2000  # Set a reasonable limit for scatter plots


def render(df):
    """Renders the bivariate analysis section of the EDA page."""

    st.markdown("### Explore Relationships Between Two Features")

    # Get column types for filtering
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    plot_type = st.selectbox(
        "Select Plot Type:",
        ["Scatter Plot", "Box Plot", "Violin Plot", "Count Plot (with Hue)"],
    )

    # Dynamic plotting for selection
    if plot_type == "Scatter Plot":
        st.markdown(
            "Use a scatter plot to see the relationship between two numerical features."
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            x_axis = st.selectbox("X-axis:", numerical_cols, key="bivar_scatter_x")
        with col2:
            y_axis = st.selectbox(
                "Y-axis:",
                numerical_cols,
                index=min(1, len(numerical_cols) - 1),
                key="bivar_scatter_y",
            )
        with col3:
            hue = st.selectbox(
                "Hue (Color by):", [None] + categorical_cols, key="bivar_scatter_hue"
            )

    elif plot_type in ["Box Plot", "Violin Plot"]:
        st.markdown(
            f"Use a {plot_type} to compare the distribution of a numerical feature across different categories."
        )
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox(
                "X-axis (Categorical):", categorical_cols, key="bivar_box_x"
            )
        with col2:
            y_axis = st.selectbox(
                "Y-axis (Numerical):", numerical_cols, key="bivar_box_y"
            )
        hue = None

    elif plot_type == "Count Plot (with Hue)":
        st.markdown(
            "Use a count plot to see the frequency of a category, broken down by another category."
        )
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox(
                "X-axis (Primary Category):", categorical_cols, key="bivar_count_x"
            )
        with col2:
            hue = st.selectbox(
                "Hue (Secondary Category):",
                categorical_cols,
                index=min(1, len(categorical_cols) - 1),
                key="bivar_count_hue",
            )
        y_axis = None

    # Plotting Logic
    df_to_plot = df
    if len(df) > SAMPLING_THRESHOLD and plot_type == "Scatter Plot":
        st.info(
            f"Dataset is large. Displaying a random sample of {SAMPLING_THRESHOLD} rows for the scatter plot."
        )
        df_to_plot = df.sample(n=SAMPLING_THRESHOLD, random_state=42)

    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        if plot_type == "Scatter Plot":
            sns.scatterplot(data=df_to_plot, x=x_axis, y=y_axis, hue=hue, ax=ax)
        elif plot_type == "Box Plot":
            sns.boxplot(data=df_to_plot, x=x_axis, y=y_axis, ax=ax)
        elif plot_type == "Violin Plot":
            sns.violinplot(data=df_to_plot, x=x_axis, y=y_axis, ax=ax)
        elif plot_type == "Count Plot (with Hue)":
            sns.countplot(data=df_to_plot, x=x_axis, hue=hue, ax=ax)
            plt.xticks(rotation=45)

        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred while creating the plot: {e}")
