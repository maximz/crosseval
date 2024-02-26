import logging
import time
import inspect
from pathlib import Path
from typing import Tuple, Union, Optional

import joblib
import numpy as np

from crosseval import Classifier
from crosseval.utils import is_clf_a_sklearn_pipeline

logger = logging.getLogger(__name__)


def train_classifier(
    clf: Classifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = "(unknown name)",
    train_sample_weights: Optional[np.ndarray] = None,
    train_groups: Optional[np.ndarray] = None,
    export_clf_fname: Optional[Union[str, Path]] = None,
) -> Tuple[Classifier, float]:
    itime = time.time()

    # clf may be an individual estimator, or it may be a pipeline, in which case the estimator is the final pipeline step
    is_pipeline = is_clf_a_sklearn_pipeline(clf)

    # check if the estimator (or final pipeline step, if pipeline) accepts sample weights
    fit_parameters = inspect.signature(
        clf[-1].fit if is_pipeline else clf.fit
    ).parameters
    estimator_supports_sample_weight = "sample_weight" in fit_parameters.keys()
    estimator_supports_groups = "groups" in fit_parameters.keys()
    estimator_supports_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in fit_parameters.values()
    )
    extra_kwargs_warning_message = " (Classifier does support kwargs, but we don't pass any because it may cause issues down the call stack.)"

    def make_kwarg_name(name):
        if is_pipeline:
            # Fitting a pipeline with sample weights requires this odd syntax.
            # https://stackoverflow.com/a/36224909/130164
            # https://github.com/scikit-learn/scikit-learn/issues/18159
            last_step_name = clf.steps[-1][0]
            return last_step_name + "__" + name
        else:
            # Just a plain-old estimator, not a pipeline.
            # No parameter renaming necessary.
            return name

    fit_kwargs = {}
    if train_sample_weights is not None:
        # User wants to use sample weights
        if not estimator_supports_sample_weight:
            # Classifier does not support sample weights
            msg = f"Classifier {model_name} does not support sample weights -- fitting without them."
            if estimator_supports_kwargs:
                # But classifier does support arbitrary kwargs, which may be used to pass sample weights, but we're not sure
                msg += extra_kwargs_warning_message
            logger.warning(msg)
        else:
            # Fit with sample weights.
            fit_kwargs[make_kwarg_name("sample_weight")] = train_sample_weights

    if train_groups is not None:
        # User wants to use groups
        if not estimator_supports_groups:
            # Classifier does not support sample weights
            msg = f"Classifier {model_name} does not support groups parameter -- fitting without it."
            if estimator_supports_kwargs:
                # But classifier does support arbitrary kwargs, which may be used to pass groups, but we're not sure
                msg += extra_kwargs_warning_message
            logger.warning(msg)
        else:
            # Fit with groups parameter.
            fit_kwargs[make_kwarg_name("groups")] = train_groups

    # Fit.
    clf = clf.fit(X_train, y_train, **fit_kwargs)

    elapsed_time = time.time() - itime

    if export_clf_fname is not None:
        # Save clf (or pipeline) to disk
        try:
            joblib.dump(clf, export_clf_fname)
        except Exception as err:
            logger.error(f"Error in saving classifier {model_name} to disk: {err}")

    return clf, elapsed_time
