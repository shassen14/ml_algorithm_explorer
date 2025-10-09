# ui/components/eda_univariate.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def render(df):
    """Renders the univariate analysis section of the EDA page."""

    st.markdown("### Inspect a Single Feature")

    # Selectbox for the user to choose a feature
    selected_column = st.selectbox(
        "Select a feature to analyze:", df.columns, key="univariate_select"
    )

    if selected_column:
        st.markdown("---")

        # Dynamically render
        # Check if the selected column is numerical
        if pd.api.types.is_numeric_dtype(df[selected_column]):
            render_numerical_analysis(df, selected_column)
        else:  # Assume it's categorical
            render_categorical_analysis(df, selected_column)


def render_numerical_analysis(df, column_name):
    """Displays statistics and a histogram for a numerical feature."""

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Summary Statistics")
        st.write(df[column_name].describe())

    with col2:
        st.markdown("#### Distribution Plot")
        try:
            fig, ax = plt.subplots()
            sns.histplot(df[column_name], kde=True, ax=ax)
            ax.set_title(f"Distribution of {column_name}")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not plot {column_name}. Error: {e}")


def render_categorical_analysis(df, column_name):
    """Displays value counts and a bar chart for a categorical feature."""

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Frequency Count")
        counts = df[column_name].value_counts()
        percentages = (
            df[column_name].value_counts(normalize=True).mul(100).round(2).astype(str)
            + "%"
        )
        summary_df = pd.DataFrame({"Count": counts, "Percentage": percentages})
        st.dataframe(summary_df)

    with col2:
        st.markdown("#### Bar Chart")
        try:
            fig, ax = plt.subplots()
            sns.countplot(
                y=df[column_name], ax=ax, order=df[column_name].value_counts().index
            )
            ax.set_title(f"Count of {column_name}")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not plot {column_name}. Error: {e}")
