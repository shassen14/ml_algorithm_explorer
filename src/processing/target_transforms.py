# src/processing/target_transforms.py
import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Any


def apply_log_transform(series: pd.Series) -> Tuple[pd.Series, Any]:
    """Applies a safe log(1+x) transform."""
    transformed_series = np.log1p(series)
    # The 'state' for log transform is simple: it has no parameters.
    return transformed_series, None


def inverse_log_transform(series: pd.Series, state: Any) -> pd.Series:
    """Applies the inverse of log1p, which is exp(x) - 1."""
    return np.expm1(series)


def apply_sqrt_transform(series: pd.Series) -> Tuple[pd.Series, Any]:
    """Applies a square root transform."""
    transformed_series = np.sqrt(series)
    return transformed_series, None


def inverse_sqrt_transform(series: pd.Series, state: Any) -> pd.Series:
    """Applies the inverse of square root, which is squaring."""
    return series**2


def apply_boxcox_transform(series: pd.Series) -> Tuple[pd.Series, Any]:
    """Applies a Box-Cox transform and returns the transformed series and the lambda value."""
    # Box-Cox requires positive data, so we add a small constant if there are zeros or negatives
    if series.min() <= 0:
        series = series + abs(series.min()) + 0.01

    transformed_series, optimal_lambda = stats.boxcox(series)
    # The 'state' for Box-Cox is the lambda value, which is needed for the inverse transform.
    return pd.Series(transformed_series, index=series.index), optimal_lambda


def inverse_boxcox_transform(series: pd.Series, state: Any) -> pd.Series:
    """Applies the inverse Box-Cox transform using the saved lambda."""
    from scipy.special import inv_boxcox

    optimal_lambda = state
    if optimal_lambda == 0:
        return np.expm1(series)  # Log transform is a special case
    else:
        return pd.Series(inv_boxcox(series, optimal_lambda), index=series.index)


# A registry to hold our transformation functions
TRANSFORMATION_REGISTRY = {
    "None": (None, None),
    "Log Transform (log1p)": (apply_log_transform, inverse_log_transform),
    "Square Root Transform": (apply_sqrt_transform, inverse_sqrt_transform),
    "Box-Cox Transform": (apply_boxcox_transform, inverse_boxcox_transform),
}
