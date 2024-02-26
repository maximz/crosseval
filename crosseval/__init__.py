"""crosseval."""

import logging
from logging import NullHandler

# Import these first to avoid circular imports:
from .utils import (
    Classifier,
    is_clf_a_sklearn_pipeline,
    _get_final_estimator_if_pipeline,
)

from .defaults import DEFAULT_LABEL_SCORERS, DEFAULT_PROBABILITY_SCORERS
from .featurized_data import FeaturizedData
from .train import train_classifier
from .metric import Metric

# Import single fold performance class first to avoid circular import issue:
from .model_single_fold_performance import ModelSingleFoldPerformance
from .model_global_performance import ModelGlobalPerformance, Y_TRUE_VALUES

from .experiment_set_global_performance import ExperimentSetGlobalPerformance
from .experiment_set import ExperimentSet

__all__ = [
    "DEFAULT_LABEL_SCORERS",
    "DEFAULT_PROBABILITY_SCORERS",
    "FeaturizedData",
    "train_classifier",
    "Classifier",
    "is_clf_a_sklearn_pipeline",
    "_get_final_estimator_if_pipeline",
    "Metric",
    "ModelGlobalPerformance",
    "Y_TRUE_VALUES",
    "ModelSingleFoldPerformance",
    "ExperimentSetGlobalPerformance",
    "ExperimentSet",
]


__author__ = """Maxim Zaslavsky"""
__email__ = "maxim@maximz.com"
__version__ = "0.0.2"

# Set default logging handler to avoid "No handler found" warnings.
logging.getLogger(__name__).addHandler(NullHandler())
