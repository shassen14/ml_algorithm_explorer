# src/analysis/profiler.py
import pandas as pd


def profile_dataframe(df: pd.DataFrame) -> list:
    """Performs a quick profiling of a dataframe and returns a list of insights."""
    insights = []

    # High-level overview
    insights.append(
        {
            "type": "info",
            "text": f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.",
        }
    )

    # Missing values
    missing_summary = df.isnull().mean() * 100
    high_missing_cols = missing_summary[missing_summary > 50].index.tolist()
    if high_missing_cols:
        insights.append(
            {
                "type": "warning",
                "text": f"High Missing Values: The following columns have >50% missing data: `{', '.join(high_missing_cols)}`.",
            }
        )

    # Unique ID columns
    id_cols = [
        col
        for col in df.select_dtypes(include="object").columns
        if df[col].nunique() >= len(df) * 0.95
    ]
    if id_cols:
        insights.append(
            {
                "type": "info",
                "text": f"Potential Identifiers: The following columns have a high number of unique values and might be IDs: `{', '.join(id_cols)}`.",
            }
        )

    # Constant columns
    constant_cols = df.columns[df.nunique() == 1].tolist()
    if constant_cols:
        insights.append(
            {
                "type": "error",
                "text": f"Constant Columns: The following columns have only one value and should be dropped: `{', '.join(constant_cols)}`.",
            }
        )

    return insights
