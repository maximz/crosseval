import logging
import dataclasses
from dataclasses import InitVar, dataclass, fields, field
from functools import cached_property
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union, Optional
from typing_extensions import Self

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


from crosseval import Classifier, Metric
from crosseval.scores import compute_classification_scores
from crosseval.utils import _get_final_estimator_if_pipeline, is_clf_a_sklearn_pipeline

logger = logging.getLogger(__name__)


def _get_feature_names(clf: Classifier) -> Union[List[str], List[int], None]:
    """Get feature names from classifier"""
    # If this is a pipeline, get feature names from the last step (after any transformations).
    final_estimator: BaseEstimator = _get_final_estimator_if_pipeline(clf)

    # Get number of features inputted to the final pipeline step,
    # which may be different than the number of features inputted to the whole pipeline originally (X_test.shape[1]).
    # Edge case: n_features_in_ can be None (sklearn <1.2) or unset (sklearn >= 1.2) for DummyClassifier.
    if not hasattr(final_estimator, "n_features_in_"):
        # In sklearn >= 1.2, DummyClassifier does not have this attribute (https://github.com/scikit-learn/scikit-learn/pull/24386)
        return None
    n_features: Optional[int] = final_estimator.n_features_in_
    if n_features is None:
        # In sklearn < 1.2, DummyClassifier has this attribute set to None (https://github.com/scikit-learn/scikit-learn/pull/24386)
        return None

    feature_names = None
    if is_clf_a_sklearn_pipeline(clf):
        # for pipelines, this is an alternate way of getting feature names after all transformations
        # unfortunately hasattr(clf[:-1], "get_feature_names_out") does not guarantee no error,
        # can still hit estimator does not provide get_feature_names_out?
        try:
            feature_names = clf[:-1].get_feature_names_out()
        except AttributeError:
            pass

    if feature_names is not None:
        return feature_names

    # Above approach failed - still None
    if hasattr(final_estimator, "feature_names_in_"):
        # Get feature names from classifier
        # feature_names_in_ can be undefined, would throw AttributeError
        return final_estimator.feature_names_in_
    else:
        # Feature names are not available.
        return list(range(n_features))


def _extract_feature_importances(clf: Classifier) -> Optional[np.ndarray]:
    """
    get feature importances or coefficients from a classifier.
    does not support multiclass OvR/OvO.
    """
    # If this is a pipeline, get feature importances from the last step (after any transformations).
    final_estimator: BaseEstimator = _get_final_estimator_if_pipeline(clf)

    if hasattr(final_estimator, "feature_importances_"):
        # random forest
        # one feature importance vector per fold
        return np.ravel(final_estimator.feature_importances_)
    elif hasattr(final_estimator, "coef_") and final_estimator.coef_.shape[0] == 1:
        # linear model - access coef_
        # coef_ is ndarray of shape (1, n_features) if binary or (n_classes, n_features) if multiclass

        # Here we handle the case of a linear model with a single feature importance vector
        # Multiclass OvR/OvO will be handled separately

        # we will flatten each (1, n_features)-shaped vector to a (n_features,)-shaped vector
        return np.ravel(final_estimator.coef_)
    else:
        # Model has no attribute 'feature_importances_' or 'coef_',
        # or is a multiclass linear model
        return None


def _extract_multiclass_feature_importances(clf: Classifier) -> Optional[np.ndarray]:
    """get feature importances or coefficients from a multiclass OvR/OvO classifier."""
    # If this is a pipeline, use the final estimator
    final_estimator: BaseEstimator = _get_final_estimator_if_pipeline(clf)

    if hasattr(final_estimator, "coef_") and final_estimator.coef_.shape[0] > 1:
        # coef_ is ndarray of shape (1, n_features) if binary,
        # or (n_classes, n_features) if multiclass OvR,
        # or (n_classes * (n_classes - 1) / 2, n_features) if multiclass OvO.
        if final_estimator.coef_.shape[0] == len(final_estimator.classes_):
            return final_estimator.coef_

        else:
            # TODO: Handle multiclass OvO?
            # Note: sometimes n_classes * (n_classes - 1) / 2 == n_classes, e.g. for n_classes = 3.
            # So we might still fall into the case above, thinking it's OvR when really it's OvO.
            # Let's add a warning for the user.
            return None

    # Not a multiclass linear model
    return None


@dataclass(eq=False)
class ModelSingleFoldPerformance:
    """Evaluate trained classifier. Gets performance of one model on one fold."""

    model_name: str
    fold_id: int
    y_true: Union[np.ndarray, pd.Series]
    fold_label_train: str
    fold_label_test: Optional[
        str
    ]  # Optional if we're evaluating on a dataset with no "fold label". But there should still be a fold label for the training set that was used.

    # InitVar means these variables are used in __post_init__ but not stored in the dataclass afterwards.
    # Mark these optional. If they aren't provided, the user should directly provide the fields normally filled in by post_init (see below).
    clf: InitVar[Classifier] = None
    X_test: InitVar[np.ndarray] = None

    # These fields are normally filled in by post_init.
    # However we are not going to do the conventional ` = field(init=False)`
    # because for backwards-compatability, we want ability to initialize with these directly, rather than providing clf and X_test.
    # So we provide them default values of None so that we can call init() without passing,
    # but we don't set their types to Optional because, by the time __post_init__ is done, we want these to be set.
    # (They are not truly optional!)
    y_pred: Union[np.ndarray, pd.Series] = None
    class_names: Union[List[str], List[int], np.ndarray] = None
    # These can be optional, if we avoid the standard path of passing clf and X_test:
    X_test_shape: Optional[Tuple[int, int]] = None
    y_decision_function: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None
    y_preds_proba: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None
    feature_names: Optional[Union[List[str], List[int]]] = None
    feature_importances: Optional[pd.Series] = None
    multiclass_feature_importances: Optional[
        pd.DataFrame
    ] = None  # n_classes x n_features, for one-vs-rest models.

    # Truly optional parameters (Express this way to avoid mutable default value):
    test_metadata: pd.DataFrame = field(default_factory=pd.DataFrame)
    # ground truth "y" labels for abstained examples
    test_abstentions: Union[np.ndarray, pd.Series] = field(
        default_factory=lambda: np.empty(0, dtype="object")
    )
    # metadata for abstained examples
    test_abstention_metadata: pd.DataFrame = field(default_factory=pd.DataFrame)
    train_time: Optional[float] = None  # elapsed time for training
    test_sample_weights: Optional[np.ndarray] = None
    # sample weights for abstained examples. must be provided if we have abstentions and if test_sample_weights are provided.
    test_abstention_sample_weights: Optional[np.ndarray] = None
    # this can record where the classifier was saved, so it can be reloaded later
    export_clf_fname: Optional[Union[str, Path]] = None

    def __post_init__(self, clf: Classifier = None, X_test: np.ndarray = None):
        if clf is None or X_test is None:
            # If we are not provided with clf and X_test, the user must provide the following fields themselves.
            # Confirm that they have.
            if any(x is None for x in [self.y_pred, self.class_names]):
                raise ValueError(
                    "Must provide clf and X_test to initialize ModelSingleFoldPerformance, or initialize with pre-computed fields directly."
                )
        else:
            self.y_pred = clf.predict(X_test)
            self.class_names = clf.classes_
            self.X_test_shape = X_test.shape

            # Set optional properties
            if hasattr(clf, "decision_function"):
                self.y_decision_function = clf.decision_function(X_test)
            if hasattr(clf, "predict_proba"):
                # Get predicted class probabilities
                self.y_preds_proba = clf.predict_proba(X_test)

            self.feature_names = _get_feature_names(clf)
            if self.feature_names is not None:
                # Get feature importances - this may return None:
                feature_importances: Optional[
                    np.ndarray
                ] = _extract_feature_importances(clf)
                if feature_importances is not None:
                    # Sanity check the shape
                    n_features = len(self.feature_names)
                    if feature_importances.shape[0] != n_features:
                        raise ValueError(
                            f"Feature importances shape {feature_importances.shape} does not match expected n_features = {n_features}"
                        )

                    # Transform into Series
                    self.feature_importances = pd.Series(
                        feature_importances, index=self.feature_names
                    )
                else:
                    self.feature_importances = None

                # Special case: multiclass OvR feature importances
                # This too might return None
                multiclass_feature_importances: Optional[
                    np.ndarray
                ] = _extract_multiclass_feature_importances(clf)
                if multiclass_feature_importances is not None:
                    # Sanity check the shape
                    n_features = len(self.feature_names)
                    if multiclass_feature_importances.shape[1] != n_features:
                        raise ValueError(
                            f"Multiclass feature importances shape {multiclass_feature_importances.shape} does not match expected n_features = {n_features}"
                        )

                    n_classes = len(self.class_names)
                    if multiclass_feature_importances.shape[0] != n_classes:
                        raise ValueError(
                            f"Multiclass feature importances shape {multiclass_feature_importances.shape} does not match expected n_classes = {n_classes}"
                        )

                    # Transform into DataFrame
                    self.multiclass_feature_importances = pd.DataFrame(
                        multiclass_feature_importances,
                        index=self.class_names,
                        columns=self.feature_names,
                    )
                else:
                    self.multiclass_feature_importances = None

        # Convert explicitly provided None values to default_factory generated values, for consistency.
        # For example, if constructor given test_metadata=None, replace with empty DataFrame as we would if test_metadata kwarg was missing.
        # https://stackoverflow.com/a/55839223/130164
        for f in fields(self):
            if (
                getattr(self, f.name) is None
            ) and f.default_factory is not dataclasses.MISSING:
                setattr(self, f.name, f.default_factory())

        # Validation
        if self.y_true.shape[0] != self.y_pred.shape[0]:
            raise ValueError(
                f"y_true shape {self.y_true.shape} does not match y_pred shape {self.y_pred.shape}"
            )
        if (
            not self.test_metadata.empty
            and self.y_true.shape[0] != self.test_metadata.shape[0]
        ):
            raise ValueError(
                "Metadata was supplied but does not match y_true or y_pred length."
            )
        if self.n_abstentions > 0:
            if not self.test_abstention_metadata.empty:
                if self.test_abstention_metadata.shape[0] != self.n_abstentions:
                    raise ValueError(
                        "If test_abstentions_metadata is provided, it must match the number of abstention ground truth labels (test_abstentions) provided."
                    )
            else:  # self.test_abstention_metadata is empty
                if not self.test_metadata.empty:
                    # test_abstention_metadata must be provided if test_abstentions and test_metadata provided
                    raise ValueError(
                        "If there are abstentions and metadata was provided for non-abstained examples, then test_abstentions_metadata (metadata dataframe) must be provided alongside test_abstentions (ground truth labels)"
                    )

            if self.test_abstention_sample_weights is not None:
                if self.test_abstention_sample_weights.shape[0] != self.n_abstentions:
                    raise ValueError(
                        "If test_abstention_sample_weights is provided, it must match the number of abstention ground truth labels (test_abstentions) provided."
                    )
            else:  # self.test_abstention_sample_weights is None
                if self.test_sample_weights is not None:
                    raise ValueError(
                        "If there are abstentions and sample weights was provided for non-abstained examples, then test_abstention_sample_weights must be provided alongside test_abstentions (ground truth labels)"
                    )

    def export(self, metadata_fname: Union[str, Path]):
        # Export this object
        joblib.dump(self, metadata_fname)

    def copy(self) -> Self:
        """Return a copy of this ModelSingleFoldPerformance. This is NOT a deep copy. The fields are not replaced."""
        # Running dataclasses.replace() without any changes gives:
        # ValueError: InitVar 'clf' must be specified with replace()
        return dataclasses.replace(self, clf=None, X_test=None)

    @cached_property
    def classifier(self) -> Classifier:
        """load original classifier from disk, if export_clf_fname was provided"""
        if self.export_clf_fname is None:
            raise ValueError(
                "Could not load classifier from disk because export_clf_fname is not set"
            )
        return joblib.load(self.export_clf_fname)

    @property
    def n_abstentions(self) -> int:
        if self.test_abstentions is None:
            return 0
        return len(self.test_abstentions)

    def scores(
        self,
        label_scorers: Optional[Dict[str, Tuple[Callable, str, dict]]] = None,
        probability_scorers: Optional[Dict[str, Tuple[Callable, str, dict]]] = None,
        with_abstention=True,
        exclude_metrics_that_dont_factor_in_abstentions=False,
        abstain_label="Unknown",
    ) -> Dict[str, Metric]:
        # Compute probability scores without abstention.
        # (Probability-based scores do not support abstention.)
        # (We have to separate out computing of the probability scores from the label scores, because the probability scores require y_true without abstention, i.e. not exxpanded)
        computed_scores = compute_classification_scores(
            y_true=self.y_true,
            y_preds=self.y_pred,
            y_preds_proba=self.y_preds_proba,
            y_preds_proba_classes=self.class_names,
            sample_weights=self.test_sample_weights,
            label_scorers={},  # Disable label scorers
            probability_scorers=probability_scorers,
        )

        # Compute label scores with abstention if supplied and requested.
        to_add = {}
        y_true = self.y_true
        y_pred = self.y_pred
        sample_weights = self.test_sample_weights
        if with_abstention:
            if exclude_metrics_that_dont_factor_in_abstentions:
                # We are asked to discard probability-based scores that did not factor in abstentions
                computed_scores = {}

            if self.n_abstentions > 0:
                # TODO: warn if abstain_label conflicts with name of existing class
                # TODO: if y_true classes are all numeric, then use a numeric abstain_label too
                y_true = np.hstack([y_true, self.test_abstentions])
                y_pred = np.hstack([y_pred, [abstain_label] * self.n_abstentions])
                # We are guaranteed that self.test_sample_weights and self.test_abstention_sample_weights are both given or both null
                sample_weights = (
                    np.hstack([sample_weights, self.test_abstention_sample_weights])
                    if sample_weights is not None
                    else None
                )

            # Record abstention rate even if it is 0, so we properly aggregate the abstention rate over all folds.
            to_add["abstention_rate"] = Metric(
                value=self.n_abstentions / len(y_true),
                friendly_name="Unknown/abstention proportion",
            )

        # Merge dictionaries
        return (
            computed_scores
            | compute_classification_scores(
                y_true=y_true,
                y_preds=y_pred,
                sample_weights=sample_weights,
                label_scorers=label_scorers,
                # Explicitly disable probability_scorers and do not pass y_preds_proba
                probability_scorers={},
            )
            | to_add
        )

    def apply_abstention_mask(self, mask: np.ndarray) -> Self:
        """Pass a boolean mask. Returns a copy of self with the mask samples turned into abstentions."""
        # TODO: consider exposing this functionality on an ExperimentSetSummary, and on a ModelGlobalPerformance -> basically regenerates by modifying inner ModelSingleFoldPerformances?
        if mask.shape[0] != self.y_true.shape[0]:
            raise ValueError(
                f"Must supply boolean mask, but got mask.shape[0] ({mask.shape[0]}) != self.y_true.shape[0] ({self.y_true.shape[0]})"
            )
        return dataclasses.replace(
            self,
            # Pass in null InitVars to make a copy (see comments in `.copy()` method above)
            clf=None,
            X_test=None,
            # Make changes
            y_true=self.y_true[~mask],
            y_pred=self.y_pred[~mask],
            X_test_shape=(
                self.X_test_shape[0] - np.sum(mask),
                self.X_test_shape[1],
            )
            if self.X_test_shape is not None
            else None,
            y_decision_function=self.y_decision_function[~mask]
            if self.y_decision_function is not None
            else None,
            y_preds_proba=self.y_preds_proba[~mask]
            if self.y_preds_proba is not None
            else None,
            # special case if self.test_metadata is empty and thus cannot be masked further
            test_metadata=self.test_metadata[~mask]
            if not self.test_metadata.empty
            else self.test_metadata,
            test_sample_weights=self.test_sample_weights[~mask]
            if self.test_sample_weights is not None
            else None,
            test_abstentions=np.hstack(
                [
                    self.test_abstentions,
                    self.y_true[mask],
                ]
            ),
            test_abstention_metadata=pd.concat(
                [
                    self.test_abstention_metadata,
                    # special case if self.test_metadata is empty and thus cannot be masked further
                    self.test_metadata[mask]
                    if not self.test_metadata.empty
                    else self.test_metadata,
                ],
                axis=0,
            ),
            test_abstention_sample_weights=np.hstack(
                [
                    self.test_abstention_sample_weights
                    if self.test_abstention_sample_weights is not None
                    else [],
                    self.test_sample_weights[mask],
                ]
            )
            if self.test_sample_weights is not None
            else self.test_abstention_sample_weights,
            # All other fields stay the same
        )
