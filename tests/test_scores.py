import numpy as np
import pytest

from crosseval import scores as scores_module
from crosseval.scores import (
    coerce_incomparable_label_arrays,
    compute_classification_scores,
)


def test_compute_classification_scores_skips_label_scorer_errors(caplog):
    def broken_label_scorer(y_true, y_pred, sample_weight=None):
        raise ValueError("boom")

    scores = compute_classification_scores(
        y_true=np.array(["a", "b"]),
        y_preds=np.array(["a", "b"]),
        label_scorers={"broken": (broken_label_scorer, "Broken", {})},
        probability_scorers={},
    )

    assert scores == {}
    assert "Error in evaluating label-based metric broken" in caplog.text


def test_compute_classification_scores_skips_probability_scorer_errors(caplog):
    def broken_probability_scorer(y_true, y_score, labels, sample_weight=None):
        raise ValueError("boom")

    scores = compute_classification_scores(
        y_true=np.array(["a", "b"]),
        y_preds=np.array(["a", "b"]),
        y_preds_proba=np.array([[0.9, 0.1], [0.1, 0.9]]),
        y_preds_proba_classes=np.array(["a", "b"]),
        label_scorers={},
        probability_scorers={"broken": (broken_probability_scorer, "Broken", {})},
    )

    assert scores == {}
    assert "Error in evaluating predict-proba-based metric broken" in caplog.text


def test_compute_classification_scores_handles_mixed_label_types():
    scores = compute_classification_scores(
        y_true=np.array([0, 1, 1], dtype=object),
        y_preds=np.array([0, 1, "Unknown"], dtype=object),
        probability_scorers={},
    )

    assert scores["accuracy"].value == 2 / 3


@pytest.mark.parametrize(
    ("y_true", "y_pred", "labels"),
    [
        (
            np.array([0, 1], dtype=np.int64),
            np.array([1.0, 0.0], dtype=np.float64),
            None,
        ),
        (
            np.array(["a", "b"]),
            np.array(["b", "a"]),
            np.array(["a", "b"]),
        ),
    ],
)
def test_coerce_incomparable_label_arrays_skips_probe_for_sortable_dtypes(
    monkeypatch, y_true, y_pred, labels
):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("np.unique probe should not run for sortable dtypes")

    monkeypatch.setattr(scores_module.np, "unique", fail_if_called)

    coerced_y_true, coerced_y_pred, coerced_labels = coerce_incomparable_label_arrays(
        y_true, y_pred, labels
    )

    assert coerced_y_true is y_true
    assert coerced_y_pred is y_pred
    assert coerced_labels is labels


def test_coerce_incomparable_label_arrays_coerces_mixed_kinds_to_strings():
    y_true = np.array([0, 1])
    y_pred = np.array([1, 0])
    labels = np.array(["0", "1"])

    coerced_y_true, coerced_y_pred, coerced_labels = coerce_incomparable_label_arrays(
        y_true, y_pred, labels
    )

    assert coerced_y_true.dtype.kind in {"S", "U"}
    assert coerced_y_pred.dtype.kind in {"S", "U"}
    assert coerced_labels.dtype.kind in {"S", "U"}
    assert coerced_y_true.tolist() == ["0", "1"]
    assert coerced_y_pred.tolist() == ["1", "0"]
    assert coerced_labels.tolist() == ["0", "1"]
