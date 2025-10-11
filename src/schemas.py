# src/schemas.py
import numpy as np
from pydantic import BaseModel
from typing import Any, Dict, Optional, List
import pandas as pd
from matplotlib.figure import Figure


# Pydantic needs this to know how to handle non-standard types like DataFrames and Figures
class ArbitraryTypesConfig:
    arbitrary_types_allowed = True


class ProcessedData(BaseModel):
    """Data contract for the output of the data loading and splitting step."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train_encoded: pd.Series
    y_test_encoded: pd.Series
    class_names: List[str]

    class Config(ArbitraryTypesConfig):
        pass


class ModelConfig(BaseModel):
    """Data contract for the configuration of a single model run, built by the sidebar."""

    model_name: str
    model_class: Any
    hyperparameters: Dict[str, Any]


class ExplainerData(BaseModel):
    """
    Data contract for the raw and intermediate data needed by the
    interactive algorithm explainer components. This data is from BEFORE
    the main pipeline is run.
    """

    full_df: Optional[pd.DataFrame] = None
    target_column: Optional[str] = None
    processed_data: Optional[Dict[str, Any]] = None  # Contains X_train, y_train etc.

    class Config(ArbitraryTypesConfig):
        pass


# Base Class containing fields that are common to every pipeline result.
class BasePipelineResult(BaseModel):
    pipeline: Any  # The trained scikit-learn pipeline object
    model_name: str
    feature_importance_fig: Optional[Figure] = None

    class Config(ArbitraryTypesConfig):
        pass


# Each inherits from the base and adds its own unique fields.
class ClassificationPipelineResult(BasePipelineResult):
    metrics: Dict[str, float]
    class_names: List[str]
    y_train_encoded: pd.Series

    confusion_matrix_fig: Optional[Figure] = None
    roc_curve_fig: Optional[Figure] = None
    classification_report: Optional[str] = None
    decision_boundary_fig: Optional[Figure] = None
    coefficient_plot_fig: Optional[Figure] = None
    false_positives_df: Optional[pd.DataFrame] = None
    false_negatives_df: Optional[pd.DataFrame] = None


class RegressionPipelineResult(BasePipelineResult):
    metrics: Dict[str, float]

    actual_vs_predicted_fig: Optional[Figure] = None
    residuals_fig: Optional[Figure] = None
    coefficient_plot_fig: Optional[Figure] = None


class ClusteringPipelineResult(BasePipelineResult):
    metrics: Dict[str, float]  # e.g., Silhouette Score
    cluster_labels: np.ndarray

    cluster_plot_fig: Optional[Figure] = None
