# src/analysis/auto_eda.py
import pandas as pd
from scipy import stats
import numpy as np


def generate_insights(df, target_column=None, problem_type=None):
    """
    Performs an automated exploratory data analysis and generates a list of key insights.
    """
    insights = []

    # --- 1. High-Level Dataset Overview ---
    insights.append(
        {
            "type": "info",
            "text": f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.",
        }
    )

    # --- 2. Check for Zero-Variance (Constant) Features ---
    zero_variance_cols = df.columns[df.nunique() == 1]
    if not zero_variance_cols.empty:
        insights.append(
            {
                "type": "error",
                "text": f"**Zero-Variance Features:** The following columns have only one unique value and should be removed: `{', '.join(zero_variance_cols)}`.",
            }
        )

    # --- 3. Check for High Cardinality Features ---
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    high_cardinality_cols = [
        col
        for col in categorical_cols
        if df[col].nunique() > 50 and col != target_column
    ]
    if high_cardinality_cols:
        insights.append(
            {
                "type": "warning",
                "text": f"**High Cardinality Features:** The following categorical features have a large number of unique values (>50) and may require special handling (e.g., feature hashing, target encoding, or being dropped): `{', '.join(high_cardinality_cols)}`.",
            }
        )

    # --- 4. Find High Correlations between Numerical Features ---
    numerical_cols = df.select_dtypes(include=np.number)
    if numerical_cols.shape[1] > 1:
        corr_matrix = numerical_cols.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        high_corr_pairs = [
            (col1, col2)
            for col1 in upper_tri.columns
            for col2 in upper_tri.index
            if upper_tri.loc[col2, col1] > 0.8
        ]
        if high_corr_pairs:
            pair_strings = [f"`{p[0]}` & `{p[1]}`" for p in high_corr_pairs]
            insights.append(
                {
                    "type": "warning",
                    "text": f"**High Correlation:** Strong linear relationships (>0.8) were found between the following feature pairs, which could indicate multicollinearity: {', '.join(pair_strings)}.",
                }
            )

    # --- 5. Find Most Predictive Features (if target is provided) ---
    if target_column and problem_type:
        predictive_insights = find_predictive_features(df, target_column, problem_type)
        if predictive_insights:
            insights.append(predictive_insights)

    return insights


def find_predictive_features(df, target_column, problem_type):
    """Helper function to find features with strong statistical links to the target."""

    significant_features = []

    for col in df.columns:
        if col == target_column:
            continue

        # Drop rows where target or feature is NaN for statistical tests
        test_df = df[[col, target_column]].dropna()
        if test_df.empty:
            continue

        if problem_type == "Classification":
            # Chi-squared test for categorical feature vs. categorical target
            if pd.api.types.is_categorical_dtype(
                test_df[col]
            ) or pd.api.types.is_object_dtype(test_df[col]):
                contingency_table = pd.crosstab(test_df[col], test_df[target_column])
                chi2, p, _, _ = stats.chi2_contingency(contingency_table)
                if p < 0.05:  # Using a significance level of 5%
                    significant_features.append(f"`{col}` (p={p:.3f})")

            # ANOVA F-test for numerical feature vs. categorical target
            elif pd.api.types.is_numeric_dtype(test_df[col]):
                groups = [
                    test_df[col][test_df[target_column] == c]
                    for c in test_df[target_column].unique()
                ]
                if len(groups) > 1:
                    f_val, p = stats.f_oneway(*groups)
                    if p < 0.05:
                        significant_features.append(f"`{col}` (p={p:.3f})")

        # Regression logic here (e.g., using Pearson correlation)

    if significant_features:
        return {
            "type": "success",
            "text": f"**Strong Predictors Found:** The following features have a statistically significant relationship with the target variable `{target_column}`: {', '.join(significant_features)}.",
        }
    return None
