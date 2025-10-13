# src/pipelines/regression_pipeline.py
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import logging

from src.processing.target_transforms import TRANSFORMATION_REGISTRY
from src.schemas import RegressionPipelineConfig, RegressionPipelineResult

from src.evaluation.specific_plots import regression as regression_plots

logger = logging.getLogger(__name__)


def run_regression_pipeline(
    config: RegressionPipelineConfig,
) -> RegressionPipelineResult:
    try:
        # Unpack config variables
        X_train = config.X_train
        X_test = config.X_test
        y_train = config.y_train
        y_test = config.y_test
        model_config = config.model_run_config
        target_transform_method = config.target_transform_method

        # Unpack model variables
        model_class = model_config.model_class
        model_name = model_config.model_name
        hyperparameters = model_config.hyperparameters

        logger.info(f"Starting regression pipeline for model: {model_name}")
        logger.info(f"Hyperparameters: {hyperparameters}")

        # Add random_state for reproducibility where applicable
        if model_name in ["Random Forest Regressor", "XGBoost Regressor"]:
            hyperparameters["random_state"] = 42

        model = model_class(**hyperparameters)

        # Preprocessing steps are identical to classification
        numerical_features = X_train.select_dtypes(include=np.number).columns
        categorical_features = X_train.select_dtypes(
            include=["object", "category"]
        ).columns

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
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

        apply_transform, inverse_transform = TRANSFORMATION_REGISTRY[
            target_transform_method
        ]

        y_train_transformed = y_train
        transform_state = None  # For storing lambda in Box-Cox

        if apply_transform:
            logger.info(f"Applying '{target_transform_method}' to the target variable.")
            y_train_transformed, transform_state = apply_transform(y_train)

        main_pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("regressor", model)]
        )

        logger.info("Training the regression pipeline...")
        main_pipeline.fit(X_train, y_train_transformed)
        logger.info("Training complete.")

        y_pred_transformed = main_pipeline.predict(X_test)
        y_pred_transformed = pd.Series(y_pred_transformed, index=X_test.index)

        # Inverse so we can have original units
        y_test_orig = y_test
        y_pred_orig = y_pred_transformed

        if inverse_transform:
            logger.info(
                "Inverse transforming the target and predictions for evaluation."
            )
            # Note: We inverse transform the original y_test for a fair comparison
            y_test_transformed, _ = apply_transform(y_test)
            y_pred_orig = inverse_transform(y_pred_transformed, transform_state)
            # We must also inverse transform the original y_test to get it back to the dollar scale if it was transformed.
            # A simpler way is to just use the original y_test that was never transformed.
            # Let's stick with the original y_test.

            # The predictions need to be inversed from the transformed space
            y_pred_transformed_series = pd.Series(
                y_pred_transformed, index=y_test.index
            )
            y_pred_orig = inverse_transform(y_pred_transformed_series, transform_state)

        # --- Calculate Regression Metrics ---
        metrics = {
            "R-squared": r2_score(y_test_orig, y_pred_orig),
            "Mean Absolute Error (MAE)": mean_absolute_error(y_test_orig, y_pred_orig),
            "Mean Squared Error (MSE)": mean_squared_error(y_test_orig, y_pred_orig),
            "Root Mean Squared Error (RMSE)": np.sqrt(
                mean_squared_error(y_test_orig, y_pred_orig)
            ),
        }
        logger.info(f"Calculated regression metrics: {metrics}")

        # --- Generate Regression Plots ---
        actual_vs_predicted_fig = regression_plots.plot_actual_vs_predicted(
            y_test_orig, y_pred_orig
        )
        residuals_fig = regression_plots.plot_residuals(y_test_orig, y_pred_orig)

        # --- Package results into the Pydantic schema ---
        return RegressionPipelineResult(
            pipeline=main_pipeline,
            model_name=model_name,
            metrics=metrics,
            actual_vs_predicted_fig=actual_vs_predicted_fig,
            residuals_fig=residuals_fig,
            target_transform_method=target_transform_method,
            # feature_importance_fig
        )

    except Exception as e:
        logger.error(
            f"An error occurred in the regression pipeline for model {model_name}: {e}",
            exc_info=True,
        )
        return None
