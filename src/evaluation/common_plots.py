# src/evaluation/common_plots.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import logging


logger = logging.getLogger(__name__)


def plot_feature_importance(pipeline, preprocessor):
    """Generates and returns the feature importance plot for tree-based models."""
    model = pipeline.named_steps["classifier"]

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


def plot_decision_boundary(X_train, y_train_encoded, class_names, original_pipeline):
    """
    Generates a 2D decision boundary plot using a dedicated visualization pipeline.
    """
    try:
        # --- Step 1: Create a dedicated pipeline for visualization ---
        # This pipeline includes the original preprocessor, PCA, and the trained classifier.
        preprocessor = original_pipeline.named_steps["preprocessor"]
        classifier = original_pipeline.named_steps["classifier"]

        vis_pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("pca", PCA(n_components=2, random_state=42)),
                ("classifier", classifier),  # Use the already-trained classifier
            ]
        )

        # We don't need to refit the classifier, but we need to fit the PCA.
        # So we fit the preprocessor and PCA on the training data.
        X_train_processed = preprocessor.fit_transform(X_train)
        vis_pipeline.named_steps["pca"].fit(X_train_processed)
        X_pca = vis_pipeline.named_steps["pca"].transform(X_train_processed)

        # --- Step 2: Create a meshgrid in the 2D PCA space ---
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

        # --- Step 3: Predict on the meshgrid ---
        # We need a classifier that can predict from the 2D PCA space.
        # To do this, we'll train a new classifier just on the 2D data.
        # This is a standard approach for visualization.
        vis_classifier = type(classifier)(**classifier.get_params())
        vis_classifier.fit(X_pca, y_train_encoded)
        Z = vis_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # --- Step 4: Plot the boundary and the data points ---
        fig, ax = plt.subplots(figsize=(10, 8))

        unique_labels = np.unique(y_train_encoded)
        colors = plt.cm.get_cmap("viridis", len(unique_labels))

        ax.contourf(xx, yy, Z, alpha=0.4, cmap=colors)

        scatter = ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=y_train_encoded,
            alpha=0.8,
            edgecolor="k",
            cmap=colors,
        )

        legend_elements = [
            mpatches.Patch(color=colors(i), label=class_names[i]) for i in unique_labels
        ]
        ax.legend(handles=legend_elements, title="Classes")

        ax.set_title("Decision Boundary (visualized in 2D PCA space)")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        return fig

    except Exception as e:
        logger.error(f"Could not generate decision boundary plot: {e}", exc_info=True)
        return None  # Return None on failure


def plot_linear_coefficients(pipeline, class_names=None):
    """
    Extracts and plots the coefficients from a trained linear model pipeline.

    Handles both binary and multiclass classification coefficients.
    """
    try:

        # Step 1: Extract the trained classifier and the preprocessor
        classifier = pipeline.named_steps["classifier"]
        preprocessor = pipeline.named_steps["preprocessor"]

        # Check if the model has the 'coef_' attribute
        if not hasattr(classifier, "coef_"):
            return None  # Not a linear model with coefficients

        # Step 2: Get the feature names from the preprocessor
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            # Fallback for older scikit-learn or complex transformers
            return None  # Cannot reliably get feature names

        # Step 3: Create a DataFrame of coefficients and plot
        if classifier.coef_.shape[0] == 1:
            # --- Binary Classification Case ---
            coef_df = pd.DataFrame(
                {"feature": feature_names, "coefficient": classifier.coef_[0]}
            )

            # 1. Calculate absolute coefficients to find the most influential features.
            coef_df["abs_coef"] = coef_df["coefficient"].abs()

            # 2. Get the top 20 most influential features (largest absolute value).
            top_n = 20
            top_features_df = coef_df.nlargest(top_n, "abs_coef")

            # 3. Sort this subset by the actual coefficient value for plotting.
            # Ascending=True ensures the most negative is first (plotted at bottom)
            # and most positive is last (plotted at top).
            plotting_df = top_features_df.sort_values("coefficient", ascending=True)

            title = f"Top {top_n} Feature Influences on Predicting '{class_names[1]}'"

            fig, ax = plt.subplots(figsize=(12, 10))

            colors = ["red" if c < 0 else "blue" for c in plotting_df["coefficient"]]

            # Plot using the correctly sorted DataFrame
            ax.barh(plotting_df["feature"], plotting_df["coefficient"], color=colors)

        else:
            # --- Multiclass Classification Case (remains the same, but let's ensure it's clean) ---
            coef_df = pd.DataFrame(
                classifier.coef_.T, columns=class_names, index=feature_names
            )
            coef_df["abs_max_coef"] = coef_df.abs().max(axis=1)

            top_n = 20
            top_features_df = coef_df.nlargest(top_n, "abs_max_coef").drop(
                "abs_max_coef", axis=1
            )

            # Sort by the first class's coefficient for a consistent visual order
            plotting_df = top_features_df.sort_values(class_names[0], ascending=True)

            title = f"Top {top_n} Feature Coefficients per Class"

            fig, ax = plt.subplots(figsize=(12, 10))
            plotting_df.plot(
                kind="barh", ax=ax
            )  # DataFrame.plot() handles this internally

        ax.axvline(0, color="grey", linewidth=0.8)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Coefficient Value (Magnitude of Influence)")
        ax.set_ylabel("Feature")
        fig.tight_layout()

        return fig

    except Exception as e:
        logger.error(f"Could not generate coefficient plot: {e}", exc_info=True)
        return None
