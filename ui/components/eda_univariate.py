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
        if pd.api.types.is_numeric_dtype(df[selected_column]):
            # For numerical columns, use a tabbed interface
            tab1, tab2 = st.tabs(["📊 Distribution Analysis", "⚠️ Outlier Detection"])
            with tab1:
                render_numerical_distribution(df, selected_column)
            with tab2:
                render_outlier_analysis(df, selected_column)
        else:
            # For categorical columns, display directly
            render_categorical_analysis(df, selected_column)


def render_numerical_distribution(df, column_name):
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


def render_outlier_analysis(df, column_name):
    """
    Performs and displays outlier analysis using the IQR method.
    """
    st.markdown("#### What are Outliers?")
    st.markdown(
        """
    Think of outliers as the 'unusual suspects' in your dataset. 
    They are data points that are significantly different from the majority of the other data points. 
    Identifying them is important because they can sometimes negatively affect a machine learning model's 
    performance or reveal interesting, rare events.
    """
    )

    st.markdown("---")

    st.markdown("#### The Interquartile Range (IQR) Method")
    st.info(
        """
    This is a common statistical method to programmatically identify outliers. Here’s the step-by-step logic:

    1.  **Order the Data:** Imagine all the values for **`{}`** are sorted from smallest to largest.
    2.  **Find the Quartiles:**
        - **Q1 (25th percentile):** The value below which 25% of the data falls.
        - **Q3 (75th percentile):** The value below which 75% of the data falls.
    3.  **Calculate the IQR:** The Interquartile Range (IQR) is the distance between Q3 and Q1. 
    This represents the 'middle 50%' of your data, or the 'typical' range of values.
    4.  **Define the 'Fences':** We create 'fences' by extending 1.5 times the IQR from Q1 (downwards) and Q3 (upwards).
    5.  **Identify Outliers:** Any value that falls outside these fences is considered a potential outlier.
        """.format(
            column_name
        )
    )

    # --- 1. Calculate IQR and bounds ---
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # --- 2. Display the calculated values ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Lower Bound", f"{lower_bound:.2f}")
    col2.metric("Upper Bound", f"{upper_bound:.2f}")
    col3.metric("IQR", f"{IQR:.2f}")

    # --- 3. Identify and count outliers ---
    outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]
    num_outliers = len(outliers)
    percentage_outliers = (num_outliers / len(df[column_name].dropna())) * 100

    if num_outliers > 0:
        st.warning(
            f"Found **{num_outliers}** potential outliers ({percentage_outliers:.2f}% of the data)."
        )

        # --- 4. Display the outlier data ---
        show_outliers = st.checkbox("Show outlier data points")
        if show_outliers:
            st.dataframe(outliers)
    else:
        st.success("No potential outliers were detected using the IQR method.")

    # --- 4. Add Contextual Advice ---
    st.markdown("##### What should I do with outliers?")
    st.markdown(
        """
    There's no single answer! It depends on the context:
    - **Are they data entry errors?** 
    If a value is impossible (e.g., a human age of 200), you might correct or remove it.
    - **Are they legitimate, rare events?** 
    Sometimes outliers are the most interesting data points (e.g., a fraudulent transaction). 
    In this case, you might want to study them specifically.
    - **Will they harm my model?** 
    Some models (like Linear Regression) are sensitive to outliers, while others (like tree-based models) are more robust.
    
    Common strategies include removing them, transforming the data (e.g., with a log transformation), 
    or simply accepting them and choosing a robust model. This tool helps you **identify** 
    them so you can make an informed decision.
    """
    )
