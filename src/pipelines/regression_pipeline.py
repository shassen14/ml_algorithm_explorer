# src/pipelines/regression_pipeline.py
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from .base_pipeline import BasePipeline
from src.schemas import RegressionPipelineConfig, RegressionPipelineResult
from src.evaluation import common_plots
from src.evaluation.specific_plots import regression as regression_plots
from src.processing.target_transforms import TRANSFORMATION_REGISTRY


class RegressionPipeline(BasePipeline):
    """Concrete pipeline for executing regression tasks."""

    def __init__(self, config: RegressionPipelineConfig):
        # Must call the parent constructor
        super().__init__(config)

        # Regression-specific state
        self.apply_transform, self.inverse_transform = TRANSFORMATION_REGISTRY[
            self.config.target_transform_method
        ]
        self.transform_state = None  # To store lambda for Box-Cox

    @property
    def model_step_name(self) -> str:
        """Implements the abstract property from BasePipeline."""
        return "regressor"

    def _fit(self):
        """
        Overrides the base _fit method to handle optional target variable transformation
        before fitting the scikit-learn pipeline.
        """
        y_train_to_fit = self.config.y_train

        if self.apply_transform:
            y_train_to_fit, self.transform_state = self.apply_transform(
                self.config.y_train
            )

        self.pipeline.fit(self.config.X_train, y_train_to_fit)

    def _generate_results(self) -> Optional[RegressionPipelineResult]:
        """Implements the generation of a rich set of regression-specific results."""
        y_pred_transformed = self.pipeline.predict(self.config.X_test)
        y_pred_transformed = pd.Series(
            y_pred_transformed, index=self.config.X_test.index
        )

        # --- Inverse transform results to original scale for interpretation ---
        y_test_orig = self.config.y_test
        y_pred_orig = y_pred_transformed

        if self.inverse_transform:
            y_pred_orig = self.inverse_transform(
                y_pred_transformed, self.transform_state
            )

        # --- Calculate Metrics (on the original scale) ---
        metrics = {
            "R-squared": r2_score(y_test_orig, y_pred_orig),
            "Mean Absolute Error (MAE)": mean_absolute_error(y_test_orig, y_pred_orig),
            "Mean Squared Error (MSE)": mean_squared_error(y_test_orig, y_pred_orig),
            "Root Mean Squared Error (RMSE)": np.sqrt(
                mean_squared_error(y_test_orig, y_pred_orig)
            ),
        }

        # --- Generate Plots (on the original scale) ---
        actual_vs_predicted_fig = regression_plots.plot_actual_vs_predicted(
            y_test_orig, y_pred_orig
        )
        residuals_fig = regression_plots.plot_residuals(y_test_orig, y_pred_orig)

        # Plot common plots
        fi_fig = common_plots.plot_feature_importance(
            self.pipeline,
            self.pipeline.named_steps["preprocessor"],
            self.model_step_name,
        )
        coef_fig = common_plots.plot_linear_coefficients(
            self.pipeline, self.model_step_name
        )

        # --- Package into the Pydantic Result Object ---
        return RegressionPipelineResult(
            pipeline=self.pipeline,
            model_name=self.config.model_run_config.model_name,
            metrics=metrics,
            target_transform_method=self.config.target_transform_method,
            actual_vs_predicted_fig=actual_vs_predicted_fig,
            residuals_fig=residuals_fig,
            feature_importance_fig=fi_fig,
            coefficient_plot_fig=coef_fig,
        )
