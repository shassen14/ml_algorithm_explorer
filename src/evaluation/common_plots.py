# src/evaluation/common_plots.py
from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import logging


logger = logging.getLogger(__name__)


def plot_feature_importance(pipeline, preprocessor, model_step_name: str):
    """Generates and returns the feature importance plot for tree-based models."""
    model = pipeline.named_steps[model_step_name]

    if not hasattr(model, "feature_importances_"):
        return None  # Not a tree-based model

    # Get feature names from the preprocessor
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        # Fallback for older scikit-learn versions
        feature_names = [
            f"feature_{i}" for i in range(model.feature_importances_.shape[0])
        ]

    importances = model.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names).sort_values(
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    forest_importances.head(20).plot.barh(ax=ax)  # Show top 20
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    return fig


def plot_linear_coefficients(
    pipeline, model_step_name: str, class_names: Optional[List[str]] = None
):
    """
    Extracts and plots coefficients from a linear model.
    This function is now generic and handles Regression, Binary Classification,
    and Multiclass Classification by inspecting the model's .coef_ attribute.
    """
    try:
        model = pipeline.named_steps[model_step_name]
        preprocessor = pipeline.named_steps["preprocessor"]

        if not hasattr(model, "coef_"):
            return None

        feature_names = preprocessor.get_feature_names_out()
        coefficients = model.coef_

        # Case 1: Regression (coef_ is 1D array) or Binary Classification (coef_ is 2D with 1 row)
        if len(coefficients.shape) == 1 or coefficients.shape[0] == 1:
            # Flatten to a 1D array to handle both cases uniformly
            coef_series = pd.Series(coefficients.flatten(), index=feature_names)

            # Determine the title based on whether it's regression or classification
            if class_names and len(class_names) == 2:
                title = f"Top 20 Feature Influences on Predicting '{class_names[1]}'"
            else:
                title = "Top 20 Feature Influences on Prediction"

            # Find top 20 most influential features by absolute value
            top_n = 20
            top_features = coef_series.abs().nlargest(top_n).index
            plotting_df = coef_series[top_features].sort_values(ascending=True)

            fig, ax = plt.subplots(figsize=(12, 10))
            colors = ["red" if c < 0 else "blue" for c in plotting_df]
            ax.barh(plotting_df.index, plotting_df.values, color=colors)
            ax.axvline(0, color="grey", linewidth=0.8)

        # Case 2: Multiclass Classification (coef_ is 2D with >1 row)
        else:
            if not class_names or len(class_names) != coefficients.shape[0]:
                raise ValueError(
                    "For multiclass models, class_names must be provided and match the number of classes."
                )

            title = "Top 20 Feature Coefficients per Class"
            coef_df = pd.DataFrame(
                coefficients.T, columns=class_names, index=feature_names
            )

            # Find top 20 features based on the maximum absolute coefficient across all classes
            coef_df["abs_max_coef"] = coef_df.abs().max(axis=1)
            top_n = 20
            top_features_df = coef_df.nlargest(top_n, "abs_max_coef").drop(
                "abs_max_coef", axis=1
            )

            # Sort for plotting
            plotting_df = top_features_df.sort_values(class_names[0], ascending=True)

            fig, ax = plt.subplots(figsize=(12, 10))
            plotting_df.plot(kind="barh", ax=ax)
            ax.axvline(0, color="grey", linewidth=0.8)

        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Coefficient Value (Impact on Prediction)")
        ax.set_ylabel("Feature")
        fig.tight_layout()
        return fig

    except Exception as e:
        # Use a logger if you have one configured
        print(f"Could not generate coefficient plot: {e}")
        return None
