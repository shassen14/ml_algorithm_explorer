# src/pipelines/classification_pipeline.py
from typing import Optional
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import logging

from src.analysis import model_analyzer
from src.evaluation import plotting
from src.schemas import ClassificationPipelineResult, ModelConfig

logger = logging.getLogger(__name__)


def run_classification_pipeline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_config: ModelConfig,
) -> Optional[ClassificationPipelineResult]:
    try:
        # The config now directly provides the class and its string name
        model_class = model_config.model_class
        model_name = model_config.model_name
        hyperparameters = model_config.hyperparameters

        logger.info(f"Starting classification pipeline for model: {model_name}")
        logger.info(f"Hyperparameters: {hyperparameters}")

        if y_train.dtype == "object" or y_train.dtype.name == "category":
            logger.info("Target variable is categorical. Applying LabelEncoder.")
            le = LabelEncoder()

            # Fit on the training data and transform both train and test data
            # LabelEncoder returns NumPy arrays
            y_train_encoded_np = le.fit_transform(y_train)
            y_test_encoded_np = le.transform(y_test)

            # Convert NumPy arrays back to Pandas Series
            # We preserve the original index, which is good practice.
            y_train_encoded = pd.Series(
                y_train_encoded_np, index=y_train.index, name=y_train.name
            )
            y_test_encoded = pd.Series(
                y_test_encoded_np, index=y_test.index, name=y_test.name
            )

            # Keep track of the original labels for plotting and metrics
            class_names = le.classes_
            logger.info(
                f"LabelEncoder mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}"
            )
        else:
            # If it's already numerical, just use it as is
            logger.info("Target variable is already numerical.")
            y_train_encoded = y_train
            y_test_encoded = y_test
            class_names = sorted(y_train.unique())

        # Add random_state for reproducibility where applicable
        if model_name in ["Logistic Regression", "Random Forest", "XGBoost"]:
            hyperparameters["random_state"] = 42

        model = model_class(**hyperparameters)

        numerical_features = X_train.select_dtypes(include=["int64", "float64"]).columns
        categorical_features = X_train.select_dtypes(include=["object"]).columns

        logger.info(f"Identified {len(numerical_features)} numerical features.")
        logger.info(f"Identified {len(categorical_features)} categorical features.")

        # Pipeline for numerical features: impute with median, then scale.
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        # Pipeline for categorical features: impute with most frequent, then one-hot encode.
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        main_pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", model)]
        )

        logger.info("Training the pipeline...")
        main_pipeline.fit(X_train, y_train_encoded)
        logger.info("Training complete.")

        y_pred_encoded = main_pipeline.predict(X_test)
        y_pred_proba = main_pipeline.predict_proba(X_test)

        # Initialize results dictionary
        metrics = {}

        # Calculate metrics
        if len(class_names) == 2:
            pos_label = 1
            logger.info(
                f"Binary classification detected. Using '{pos_label}' as the positive label."
            )

            metrics = {
                "Accuracy": accuracy_score(y_test_encoded, y_pred_encoded),
                "Precision": precision_score(
                    y_test_encoded, y_pred_encoded, pos_label=pos_label, zero_division=0
                ),
                "Recall": recall_score(
                    y_test_encoded, y_pred_encoded, pos_label=pos_label, zero_division=0
                ),
                "F1-Score": f1_score(
                    y_test_encoded, y_pred_encoded, pos_label=pos_label, zero_division=0
                ),
            }
        else:
            logger.info(
                "Multi-class classification detected. Calculating weighted-average metrics."
            )
            metrics = {
                "Accuracy": accuracy_score(y_test_encoded, y_pred_encoded),
                "F1-Score (Weighted)": f1_score(
                    y_test_encoded, y_pred_encoded, average="weighted", zero_division=0
                ),
            }

        # Get report
        report_str = classification_report(
            y_test_encoded,
            y_pred_encoded,
            target_names=[str(c) for c in class_names],
            zero_division=0,
        )

        # Plotting
        confusion_matrix_fig = plotting.plot_confusion_matrix(
            y_test_encoded, y_pred_encoded, class_names=class_names
        )
        roc_curve_fig = plotting.plot_roc_curve(
            y_test_encoded, y_pred_proba, class_names
        )
        feature_importance_fig = plotting.plot_feature_importance(
            main_pipeline, main_pipeline.named_steps["preprocessor"]
        )
        decision_boundary_fig = plotting.plot_decision_boundary(
            X_train, y_train_encoded, class_names, main_pipeline
        )

        coefficient_plot_fig = plotting.plot_linear_coefficients(
            main_pipeline, class_names
        )

        # Diagnostics
        fp_df, fn_df = model_analyzer.get_error_analysis(
            X_test, y_test, y_pred_encoded, y_pred_proba, class_names
        )

        logger.info(f"Calculated metrics: {metrics}")

        return ClassificationPipelineResult(
            pipeline=main_pipeline,
            model_name=model_name,
            feature_importance_fig=feature_importance_fig,
            metrics=metrics,
            class_names=class_names,
            y_train_encoded=y_train_encoded,
            confusion_matrix_fig=confusion_matrix_fig,
            roc_curve_fig=roc_curve_fig,
            classification_report=report_str,
            decision_boundary_fig=decision_boundary_fig,
            coefficient_plot_fig=coefficient_plot_fig,
            false_positives_df=fp_df,
            false_negatives_df=fn_df,
        )

    except Exception as e:
        logger.error(
            f"An error occurred in the classification pipeline for model {model_config.model_name}: {e}",
            exc_info=True,
        )
        return None, None
