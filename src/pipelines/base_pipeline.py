# src/pipelines/base_pipeline.py
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod  # Abstract Base Classes
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.schemas import BasePipelineConfig, BasePipelineResult


class BasePipeline(ABC):
    """
    An abstract base class for all machine learning pipelines.
    It encapsulates the common logic for preprocessing, training, and prediction.
    """

    def __init__(self, config: BasePipelineConfig):
        self.config = config
        self.pipeline: Pipeline = self._build_pipeline()

    @property
    @abstractmethod
    def model_step_name(self) -> str:
        """
        Child classes MUST override this property to define the name of their
        final modeling step in the scikit-learn pipeline.
        e.g., 'classifier', 'regressor', 'clusterer'
        """
        pass

    def _build_preprocessor(self) -> ColumnTransformer:
        """Builds the preprocessing steps for features (X). This is common to all pipelines."""
        numerical_features = self.config.X_train.select_dtypes(
            include=np.number
        ).columns
        categorical_features = self.config.X_train.select_dtypes(
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

        return ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

    def _build_pipeline(self) -> Pipeline:
        """Builds the full scikit-learn pipeline, chaining the preprocessor and the model."""
        preprocessor = self._build_preprocessor()
        model_run_config = self.config.model_run_config

        # Add random_state for reproducibility if the model supports it
        if model_run_config.model_name in [
            "Random Forest",
            "XGBoost",
            "Random Forest Regressor",
            "XGBoost Regressor",
            "Logistic Regression",
        ]:
            model_run_config.hyperparameters["random_state"] = 42

        model = model_run_config.model_class(**model_run_config.hyperparameters)

        return Pipeline(
            steps=[("preprocessor", preprocessor), (self.model_step_name, model)]
        )

    def run(self) -> BasePipelineResult:
        """
        The main public method to execute the pipeline: fit, predict, and generate results.
        This is the template method pattern.
        """
        # 1. Fit the pipeline
        self._fit()

        # 2. Generate results (delegated to the abstract method)
        results = self._generate_results()

        return results

    def _fit(self):
        """Fits the pipeline on the training data."""
        # This is where you would handle target transformation for regression
        y_train_to_fit = self.config.y_train  # Default
        if (
            hasattr(self.config, "target_transform_method")
            and self.config.target_transform_method != "None"
        ):
            # (Your target transform logic would go here)
            pass
        self.pipeline.fit(self.config.X_train, y_train_to_fit)

    @abstractmethod
    def _generate_results(self) -> BasePipelineResult:
        """
        Abstract method. Each child pipeline MUST implement this.
        This is where all the specific metrics and plots are generated.
        """
        pass
