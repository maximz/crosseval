import numpy as np
import pytest

from crosseval.scores import compute_classification_scores


def test_compute_classification_scores_raises_label_scorer_errors():
    def broken_label_scorer(y_true, y_pred, sample_weight=None):
        raise ValueError("boom")

    with pytest.raises(RuntimeError, match="label-based metric broken"):
        compute_classification_scores(
            y_true=np.array(["a", "b"]),
            y_preds=np.array(["a", "b"]),
            label_scorers={"broken": (broken_label_scorer, "Broken", {})},
            probability_scorers={},
        )


def test_compute_classification_scores_raises_probability_scorer_errors():
    def broken_probability_scorer(y_true, y_score, labels, sample_weight=None):
        raise ValueError("boom")

    with pytest.raises(RuntimeError, match="predict-proba-based metric broken"):
        compute_classification_scores(
            y_true=np.array(["a", "b"]),
            y_preds=np.array(["a", "b"]),
            y_preds_proba=np.array([[0.9, 0.1], [0.1, 0.9]]),
            y_preds_proba_classes=np.array(["a", "b"]),
            label_scorers={},
            probability_scorers={"broken": (broken_probability_scorer, "Broken", {})},
        )


def test_compute_classification_scores_handles_mixed_label_types():
    scores = compute_classification_scores(
        y_true=np.array([0, 1, 1], dtype=object),
        y_preds=np.array([0, 1, "Unknown"], dtype=object),
        probability_scorers={},
    )

    assert scores["accuracy"].value == 2 / 3
