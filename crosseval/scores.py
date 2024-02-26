import logging
from typing import Callable, Dict, Tuple, Optional

import numpy as np

from crosseval import Metric, DEFAULT_LABEL_SCORERS, DEFAULT_PROBABILITY_SCORERS

logger = logging.getLogger(__name__)


def compute_classification_scores(
    y_true: np.ndarray,
    y_preds: np.ndarray,
    y_preds_proba: Optional[np.ndarray] = None,
    y_preds_proba_classes: Optional[np.ndarray] = None,
    sample_weights: Optional[np.ndarray] = None,
    label_scorers: Optional[Dict[str, Tuple[Callable, str, dict]]] = None,
    probability_scorers: Optional[Dict[str, Tuple[Callable, str, dict]]] = None,
) -> Dict[str, Metric]:
    """Get classification scores.
    Pass in metrics: map output dictionary key name to (scoring function, friendly name) or (scoring function, friendly name, kwargs dictionary) tuple.
    """
    if len(y_true) == 0:
        raise ValueError("Cannot compute scores when y_true is empty.")

    # Default metrics
    if label_scorers is None:
        label_scorers = DEFAULT_LABEL_SCORERS
    if probability_scorers is None:
        probability_scorers = DEFAULT_PROBABILITY_SCORERS

    output = {}
    for label_scorer_name, (
        label_scorer_func,
        label_scorer_friendly_name,
        label_scorer_kwargs,
    ) in label_scorers.items():
        try:
            output[label_scorer_name] = Metric(
                value=label_scorer_func(
                    y_true, y_preds, sample_weight=sample_weights, **label_scorer_kwargs
                ),
                friendly_name=label_scorer_friendly_name,
            )
        except Exception as err:
            logger.error(
                f"Error in evaluating label-based metric {label_scorer_name}: {err}"
            )
    if y_preds_proba is not None:
        if y_preds_proba_classes is None:
            raise ValueError(
                "y_preds_proba_classes must be provided if y_preds_proba is provided"
            )

        # defensive cast to numpy array
        y_score = np.array(y_preds_proba)

        # handle binary classification case for roc-auc score
        # (multiclass_probabilistic_score_with_missing_labels handles this for us, but still doing in case we use other evaluation approaches.)
        y_score = y_score[:, 1] if len(y_preds_proba_classes) == 2 else y_score
        for probability_scorer_name, (
            probability_scorer_func,
            probability_scorer_friendly_name,
            probability_scorer_kwargs,
        ) in probability_scorers.items():
            try:
                # TODO(later): Detect if the returned metric is a list (indexed by `labels` - make sure lengths match)
                # or a dict (keep its index),
                # and explode into a set of metrics with the appropriate names.
                output[probability_scorer_name] = Metric(
                    value=probability_scorer_func(
                        y_true=y_true,
                        y_score=y_score,
                        labels=y_preds_proba_classes,
                        sample_weight=sample_weights,
                        **probability_scorer_kwargs,
                    ),
                    friendly_name=probability_scorer_friendly_name,
                )
            except Exception as err:
                logger.error(
                    f"Error in evaluating predict-proba-based metric {probability_scorer_name}: {err}"
                )
    return output
