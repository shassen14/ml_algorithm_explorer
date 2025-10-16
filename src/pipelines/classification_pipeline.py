# src/pipelines/classification_pipeline.py
from typing import Optional
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from .base_pipeline import BasePipeline
from src.schemas import ClassificationPipelineConfig, ClassificationPipelineResult
from src.evaluation import common_plots
from src.evaluation.specific_plots import classification as classification_plots
from src.analysis import model_analyzer


class ClassificationPipeline(BasePipeline):
    """Concrete pipeline for executing classification tasks."""

    def __init__(self, config: ClassificationPipelineConfig):
        # Must call the parent constructor to build the pipeline
        super().__init__(config)

        # Initialize classification-specific state
        self.le = LabelEncoder()
        self.y_train_encoded: pd.Series = None
        self.y_test_encoded: pd.Series = None
        self.class_names: list = []

    @property
    def model_step_name(self) -> str:
        """Implements the abstract property from BasePipeline."""
        return "classifier"

    def _fit(self):
        """
        Overrides the base _fit method to handle label encoding of the target variable
        before fitting the scikit-learn pipeline.
        """
        y_train = self.config.y_train

        # Encode string labels to integers, which is required by some models (like XGBoost)
        # and is good practice for all.
        if y_train.dtype == "object" or y_train.dtype.name == "category":
            self.y_train_encoded = pd.Series(
                self.le.fit_transform(y_train), index=y_train.index
            )
            self.y_test_encoded = pd.Series(
                self.le.transform(self.config.y_test), index=self.config.y_test.index
            )
            self.class_names = list(self.le.classes_)
        else:  # If target is already numeric
            self.y_train_encoded = y_train
            self.y_test_encoded = self.config.y_test
            self.class_names = sorted(y_train.unique().tolist())

        # The actual fitting happens on the encoded target data
        self.pipeline.fit(self.config.X_train, self.y_train_encoded)

    def _generate_results(self) -> Optional[ClassificationPipelineResult]:
        """
        Implements the abstract method to generate a rich set of classification-specific
        results, including metrics, plots, and diagnostics.
        """
        y_pred = self.pipeline.predict(self.config.X_test)
        y_pred_proba = self.pipeline.predict_proba(self.config.X_test)

        # --- Calculate Metrics ---
        metrics = {}
        if len(self.class_names) == 2:  # Binary classification
            pos_label = 1  # The encoded positive class
            metrics = {
                "Accuracy": accuracy_score(self.y_test_encoded, y_pred),
                "Precision": precision_score(
                    self.y_test_encoded, y_pred, pos_label=pos_label, zero_division=0
                ),
                "Recall": recall_score(
                    self.y_test_encoded, y_pred, pos_label=pos_label, zero_division=0
                ),
                "F1-Score": f1_score(
                    self.y_test_encoded, y_pred, pos_label=pos_label, zero_division=0
                ),
            }
        else:  # Multiclass classification
            metrics = {
                "Accuracy": accuracy_score(self.y_test_encoded, y_pred),
                "F1-Score (Weighted)": f1_score(
                    self.y_test_encoded, y_pred, average="weighted", zero_division=0
                ),
                "Precision (Weighted)": precision_score(
                    self.y_test_encoded, y_pred, average="weighted", zero_division=0
                ),
                "Recall (Weighted)": recall_score(
                    self.y_test_encoded, y_pred, average="weighted", zero_division=0
                ),
            }
        report_str = classification_report(
            self.y_test_encoded,
            y_pred,
            target_names=[str(c) for c in self.class_names],
            zero_division=0,
        )

        # --- Generate Plots ---
        cm_fig = classification_plots.plot_confusion_matrix(
            self.y_test_encoded, y_pred, class_names=self.class_names
        )
        roc_fig = classification_plots.plot_roc_curve(
            self.y_test_encoded, y_pred_proba, class_names=self.class_names
        )
        db_fig = classification_plots.plot_decision_boundary(
            self.config.X_train, self.y_train_encoded, self.class_names, self.pipeline
        )

        # Common plots (can be None if not applicable to the model)
        fi_fig = common_plots.plot_feature_importance(
            self.pipeline,
            self.pipeline.named_steps["preprocessor"],
            self.model_step_name,
        )
        coef_fig = common_plots.plot_linear_coefficients(
            self.pipeline, self.model_step_name, self.class_names
        )

        # --- Generate Diagnostics ---
        fp_df, fn_df = model_analyzer.get_error_analysis(
            self.config.X_test,
            self.config.y_test,
            y_pred,
            y_pred_proba,
            self.class_names,
        )

        # --- Package into the Pydantic Result Object ---
        return ClassificationPipelineResult(
            pipeline=self.pipeline,
            model_name=self.config.model_run_config.model_name,
            metrics=metrics,
            class_names=self.class_names,
            y_train_encoded=self.y_train_encoded,
            classification_report=report_str,
            confusion_matrix_fig=cm_fig,
            roc_curve_fig=roc_fig,
            decision_boundary_fig=db_fig,
            feature_importance_fig=fi_fig,
            coefficient_plot_fig=coef_fig,
            false_positives_df=fp_df,
            false_negatives_df=fn_df,
        )
