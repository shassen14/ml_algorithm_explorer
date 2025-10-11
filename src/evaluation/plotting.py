# src/evaluation/plotting.py
from collections import Counter
import logging
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.calibration import label_binarize
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def plot_confusion_matrix(y_test, y_pred, class_names):
    """
    Generates and returns a Matplotlib figure for the confusion matrix.
    """
    numerical_labels = range(len(class_names))
    cm = confusion_matrix(y_test, y_pred, labels=numerical_labels)

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(6, 5))

    # Use Seaborn's heatmap to plot the confusion matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=class_names,
        yticklabels=class_names,
    )

    ax.set_title("Confusion Matrix", fontsize=16)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)

    # Ensure the plot is drawn correctly
    plt.tight_layout()

    return fig


def plot_roc_curve(y_test_encoded, y_pred_proba, class_names):
    """
    Generates and returns the ROC curve figure for both binary and multiclass cases.
    For multiclass, it plots a "One-vs-Rest" ROC curve for each class.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if len(class_names) == 2:
        # --- Binary Case ---
        fpr, tpr, _ = roc_curve(y_test_encoded, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f'ROC curve for class "{class_names[1]}" (AUC = {roc_auc:0.2f})',
        )
    else:
        # --- Multiclass Case (One-vs-Rest) ---
        # Binarize the output labels
        y_test_binarized = label_binarize(
            y_test_encoded, classes=range(len(class_names))
        )

        # Plot ROC curve for each class
        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr,
                tpr,
                lw=2,
                label=f'ROC curve for "{class_name}" (AUC = {roc_auc:0.2f})',
            )

    # --- Common plot settings ---
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    return fig


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


def plot_linear_coefficients(pipeline, class_names):
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


def plot_knn_neighbors(X_train, y_train_encoded, X_test, point_index, k, class_names):
    """
    Generates a 2D PCA plot to visualize the k-NN decision for a single test point.
    """
    try:
        if point_index >= len(X_test):
            logger.warning("Selected point_index is out of bounds for X_test.")
            return None

        # Combine data for consistent PCA transformation
        X_combined = pd.concat([X_train, X_test], ignore_index=True)

        # Create a simple preprocessor for visualization purposes
        # This ensures PCA works even if data has missing values or isn't scaled
        vis_preprocessor = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),  # Impute numerical
                # Note: We are ignoring categorical for this PCA plot for simplicity
                ("scaler", StandardScaler()),
            ]
        )

        # Select only numerical columns for this visualization
        numerical_cols_train = X_train.select_dtypes(include=np.number).columns
        numerical_cols_combined = X_combined.select_dtypes(include=np.number).columns

        X_processed = vis_preprocessor.fit_transform(
            X_combined[numerical_cols_combined]
        )

        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_processed)

        X_train_pca = X_pca[: len(X_train)]
        X_test_pca = X_pca[len(X_train) :]

        test_point_pca = X_test_pca[point_index]

        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X_train_pca)
        _, indices = nn.kneighbors([test_point_pca])

        neighbor_indices = indices[0]
        neighbor_labels = y_train_encoded.iloc[neighbor_indices].values

        fig, ax = plt.subplots(figsize=(12, 8))

        sns.scatterplot(
            x=X_train_pca[:, 0],
            y=X_train_pca[:, 1],
            hue=y_train_encoded,
            palette="viridis",
            alpha=0.3,
            ax=ax,
            legend="full",
        )

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, class_names, title="Classes")

        ax.scatter(
            X_train_pca[neighbor_indices, 0],
            X_train_pca[neighbor_indices, 1],
            s=150,
            facecolors="none",
            edgecolors="red",
            linewidth=2,
            label=f"{k} Nearest Neighbors",
        )

        ax.scatter(
            test_point_pca[0],
            test_point_pca[1],
            marker="*",
            s=300,
            c="black",
            edgecolors="white",
            linewidth=1,
            label=f"Test Point #{point_index}",
        )

        for neighbor_idx in neighbor_indices:
            con = mpatches.ConnectionPatch(
                xyA=test_point_pca,
                xyB=X_train_pca[neighbor_idx],
                coordsA="data",
                coordsB="data",
                axesA=ax,
                axesB=ax,
                color="red",
                linestyle="--",
                alpha=0.6,
            )
            ax.add_artist(con)

        vote_counts = Counter(neighbor_labels)
        prediction_encoded = vote_counts.most_common(1)[0][0]

        vote_text = f"Neighbor Votes (k={k}):\n"
        for label_encoded, count in vote_counts.items():
            vote_text += f"- {class_names[label_encoded]}: {count}\n"
        vote_text += f"\nPrediction: {class_names[prediction_encoded]}"

        ax.text(
            0.05,
            0.95,
            vote_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.7),
        )

        ax.set_title(f"k-NN Neighbor Inspector for Test Point #{point_index}")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")

        return fig
    except Exception as e:
        logger.error(f"Could not generate k-NN neighbor plot: {e}", exc_info=True)
        return None
