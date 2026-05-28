import logging
from typing import Any, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


logger = logging.getLogger(__name__)

####

# TODO: is there a type hint for all sklearn models? BaseEstimator is odd to put here: https://stackoverflow.com/questions/54868698/what-type-is-a-sklearn-model
Classifier = Union[Pipeline, BaseEstimator]


def is_clf_a_sklearn_pipeline(clf: Classifier) -> bool:
    # clf may be an individual estimator, or it may be a pipeline, in which case the estimator is the final pipeline step
    return isinstance(clf, Pipeline)


def _get_final_estimator_if_pipeline(clf: Classifier) -> BaseEstimator:
    """If this is a pipeline, return final step (after any transformations). Otherwise pass through."""
    if is_clf_a_sklearn_pipeline(clf):
        return clf.steps[-1][1]
    else:
        return clf


def validate_boolean_mask(
    mask: np.ndarray, expected_length: int, value_name: str = "mask"
) -> np.ndarray:
    """Validate a positional boolean mask and return it as a NumPy array."""
    mask_array = np.asarray(mask)
    if mask_array.ndim != 1:
        raise ValueError(f"{value_name} must be a one-dimensional boolean mask")
    if mask_array.dtype != bool:
        raise TypeError(f"{value_name} must be a boolean mask")
    if mask_array.shape[0] != expected_length:
        raise ValueError(
            f"Must supply boolean mask, but got {value_name}.shape[0] ({mask_array.shape[0]}) != expected length ({expected_length})"
        )
    return mask_array


def index_rows_by_mask(value: Any, mask: np.ndarray):
    """Index rows positionally for pandas and array-like values."""
    if value is None:
        return None
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return value.iloc[mask].copy()
    return np.asarray(value)[mask]
