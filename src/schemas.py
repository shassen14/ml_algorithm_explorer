# src/schemas.py
import numpy as np
from pydantic import BaseModel
from typing import Any, Dict, Optional, List, Union
import pandas as pd
from matplotlib.figure import Figure


# --- DATA TRANSFORMATION RECIPE SCHEMAS ---
# Define the parameters for each specific transformation "verb"
class DropColumnsParams(BaseModel):
    columns: List[str]


class RenameColumnParams(BaseModel):
    source_column: str
    new_name: str


class ConvertTypeParams(BaseModel):
    column: str
    target_type: str  # e.g., 'numeric'


class MapValuesParams(BaseModel):
    column: str
    new_column_name: str
    mapping_dict: Dict[str, Any]
    default_value: Optional[Any] = None


class ExtractTextParams(BaseModel):
    source_column: str
    new_column_name: str
    regex_pattern: str


class MathOperationParams(BaseModel):
    new_column_name: str
    formula: str  # e.g., "2024 - {year}"


class BooleanFlagParams(BaseModel):
    source_column: str
    keyword: str
    new_column_name: str
    case_sensitive: bool = False  # Add an option for case sensitivity


class MapKeywordsParams(BaseModel):
    source_column: str
    new_column_name: str
    # A dictionary mapping the keyword to search for to the new value
    keyword_mapping: Dict[str, str]
    default_value: str


# This Union type lists all possible transformation parameter schemas
TransformationParams = Union[
    DropColumnsParams,
    RenameColumnParams,
    ConvertTypeParams,
    MapValuesParams,
    ExtractTextParams,
    MathOperationParams,
    BooleanFlagParams,
    MapKeywordsParams,
]


# A generic model for a single step in our recipe
class TransformationStep(BaseModel):
    step_type: str
    params: TransformationParams
    is_active: bool = True


# The full recipe is a list of these steps
class TransformationRecipe(BaseModel):
    steps: List[TransformationStep] = []


# --- MODEL EXPLORER SCHEMAS ---
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
    target_transform_method: str  # e.g., "Log Transform (log1p)"


class ClusteringPipelineResult(BasePipelineResult):
    metrics: Dict[str, float]  # e.g., Silhouette Score
    cluster_labels: np.ndarray

    cluster_plot_fig: Optional[Figure] = None


class DisplayContext(BaseModel):
    """
    A single, unified object that contains ALL data needed to render any
    results page or component. This is the 'Single Source of Truth'.
    """

    # The core result from the pipeline run
    result: BasePipelineResult

    # The raw/intermediate data needed for explainers
    full_df: Optional[pd.DataFrame] = None
    processed_data: Optional[Dict[str, Any]] = None
    target_column: Optional[str] = None

    class Config(ArbitraryTypesConfig):
        pass
