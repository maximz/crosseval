import crosseval


import copy
import pickle
from dataclasses import FrozenInstanceError

import numpy as np
import pytest


def test_sentinel_value():
    # we want to make sure our Y_TRUE_VALUES behaves

    assert (
        str(crosseval.Y_TRUE_VALUES)
        == repr(crosseval.Y_TRUE_VALUES)
        == "<default y_true column>"
    )
    assert crosseval.Y_TRUE_VALUES is crosseval.Y_TRUE_VALUES
    assert crosseval.Y_TRUE_VALUES is not object()
    assert crosseval.Y_TRUE_VALUES is pickle.loads(
        pickle.dumps(crosseval.Y_TRUE_VALUES)
    )
    assert copy.deepcopy(crosseval.Y_TRUE_VALUES) is crosseval.Y_TRUE_VALUES


def test_aggregated_per_fold_scores_all_nan_metric_does_not_raise():
    def nan_scorer(y_true, y_pred, sample_weight=None):
        return np.nan

    perf = crosseval.ModelGlobalPerformance(
        model_name="model",
        per_fold_outputs={
            fold_id: crosseval.ModelSingleFoldPerformance(
                model_name="model",
                fold_id=fold_id,
                y_true=np.array(["a", "b"]),
                y_pred=np.array(["a", "b"]),
                class_names=np.array(["a", "b"]),
                fold_label_train="train",
                fold_label_test="test",
            )
            for fold_id in [0, 1]
        },
        abstain_label="Unknown",
    )

    formatted_scores = perf.aggregated_per_fold_scores(
        label_scorers={"nan_metric": (nan_scorer, "NaN metric", {})},
        probability_scorers={},
    )
    raw_scores = perf.aggregated_per_fold_scores(
        label_scorers={"nan_metric": (nan_scorer, "NaN metric", {})},
        probability_scorers={},
        formatted=False,
    )

    assert formatted_scores["NaN metric"] == "nan +/- 0.000 (in 0 folds)"
    assert np.isnan(raw_scores["NaN metric"])


def test_missing_classes_detects_equal_size_different_label_sets():
    perf = crosseval.ModelGlobalPerformance(
        model_name="model",
        per_fold_outputs={
            0: crosseval.ModelSingleFoldPerformance(
                model_name="model",
                fold_id=0,
                y_true=np.array(["a", "b"]),
                y_pred=np.array(["a", "c"]),
                class_names=np.array(["a", "b", "c"]),
                fold_label_train="train",
                fold_label_test="test",
            )
        },
        abstain_label="Unknown",
    )

    assert perf._get_stats(
        label_scorers={
            "accuracy": (lambda y_true, y_pred, sample_weight=None: 0, "Accuracy", {})
        },
        probability_scorers={},
    )["missing_classes"]


def test_model_comparison_stats_formatted_false_returns_numeric_scores():
    perf = crosseval.ModelGlobalPerformance(
        model_name="model",
        per_fold_outputs={
            0: crosseval.ModelSingleFoldPerformance(
                model_name="model",
                fold_id=0,
                y_true=np.array(["a", "b"]),
                y_pred=np.array(["a", "b"]),
                class_names=np.array(["a", "b"]),
                fold_label_train="train",
                fold_label_test="test",
            )
        },
        abstain_label="Unknown",
    )
    experiment = crosseval.ExperimentSetGlobalPerformance({"model": perf})

    stats = experiment.get_model_comparison_stats(
        label_scorers={
            "accuracy": (lambda y_true, y_pred, sample_weight=None: 1, "Accuracy", {})
        },
        probability_scorers={},
        formatted=False,
    )

    assert isinstance(stats.loc["model", "Accuracy per fold"], float)
    assert isinstance(stats.loc["model", "Accuracy global"], float)
    assert stats.loc["model", "Accuracy per fold"] == 1.0
    assert stats.loc["model", "Accuracy global"] == 1.0


def test_aggregated_per_fold_scores_supports_distinct_scorer_arguments():
    perf = crosseval.ModelGlobalPerformance(
        model_name="model",
        per_fold_outputs={
            0: crosseval.ModelSingleFoldPerformance(
                model_name="model",
                fold_id=0,
                y_true=np.array(["a", "b"]),
                y_pred=np.array(["a", "b"]),
                class_names=np.array(["a", "b"]),
                fold_label_train="train",
                fold_label_test="test",
            )
        },
        abstain_label="Unknown",
    )

    first_scores = perf.aggregated_per_fold_scores(
        label_scorers={
            "accuracy": (
                lambda y_true, y_pred, sample_weight=None: 0.25,
                "Accuracy",
                {},
            )
        },
        probability_scorers={},
        formatted=False,
    )
    second_scores = perf.aggregated_per_fold_scores(
        label_scorers={
            "accuracy": (
                lambda y_true, y_pred, sample_weight=None: 1.0,
                "Accuracy",
                {},
            )
        },
        probability_scorers={},
        formatted=False,
    )

    assert first_scores["Accuracy"] == 0.25
    assert second_scores["Accuracy"] == 1.0


def test_default_per_fold_scores_cached_across_formatting_modes():
    calls = []
    fold = crosseval.ModelSingleFoldPerformance(
        model_name="model",
        fold_id=0,
        y_true=np.array(["a", "b"]),
        y_pred=np.array(["a", "b"]),
        class_names=np.array(["a", "b"]),
        fold_label_train="train",
        fold_label_test="test",
    )

    def scores(**kwargs):
        calls.append(kwargs.copy())
        return {"accuracy": crosseval.Metric(0.5, "Accuracy")}

    fold.scores = scores
    perf = crosseval.ModelGlobalPerformance(
        model_name="model",
        per_fold_outputs={0: fold},
        abstain_label="Unknown",
    )

    assert perf.aggregated_per_fold_scores()["Accuracy"] == (
        "0.500 +/- 0.000 (in 1 folds)"
    )
    assert perf.aggregated_per_fold_scores(formatted=False)["Accuracy"] == 0.5
    assert len(calls) == 1


def test_default_per_fold_score_cache_detects_default_scorer_mutations(monkeypatch):
    calls = []
    fold = crosseval.ModelSingleFoldPerformance(
        model_name="model",
        fold_id=0,
        y_true=np.array(["a", "b"]),
        y_pred=np.array(["a", "b"]),
        class_names=np.array(["a", "b"]),
        fold_label_train="train",
        fold_label_test="test",
    )

    def scores(**kwargs):
        calls.append(kwargs.copy())
        return {"accuracy": crosseval.Metric(float(len(calls)), "Accuracy")}

    fold.scores = scores
    perf = crosseval.ModelGlobalPerformance(
        model_name="model",
        per_fold_outputs={0: fold},
        abstain_label="Unknown",
    )

    assert perf.aggregated_per_fold_scores(formatted=False)["Accuracy"] == 1.0

    monkeypatch.setitem(
        crosseval.DEFAULT_LABEL_SCORERS,
        "cache_probe",
        (lambda y_true, y_pred, sample_weight=None: 1.0, "Cache probe", {}),
    )

    assert perf.aggregated_per_fold_scores(formatted=False)["Accuracy"] == 2.0
    assert len(calls) == 2


def test_custom_per_fold_scorers_bypass_default_score_cache():
    calls = []
    fold = crosseval.ModelSingleFoldPerformance(
        model_name="model",
        fold_id=0,
        y_true=np.array(["a", "b"]),
        y_pred=np.array(["a", "b"]),
        class_names=np.array(["a", "b"]),
        fold_label_train="train",
        fold_label_test="test",
    )

    def scores(**kwargs):
        calls.append(kwargs.copy())
        return {"accuracy": crosseval.Metric(float(len(calls)), "Accuracy")}

    fold.scores = scores
    perf = crosseval.ModelGlobalPerformance(
        model_name="model",
        per_fold_outputs={0: fold},
        abstain_label="Unknown",
    )

    first_scores = perf.aggregated_per_fold_scores(
        label_scorers={},
        probability_scorers={},
        formatted=False,
    )
    second_scores = perf.aggregated_per_fold_scores(
        label_scorers={},
        probability_scorers={},
        formatted=False,
    )

    assert first_scores["Accuracy"] == 1.0
    assert second_scores["Accuracy"] == 2.0
    assert len(calls) == 2


def test_default_per_fold_score_cache_keeps_abstention_modes_separate():
    calls = []
    fold = crosseval.ModelSingleFoldPerformance(
        model_name="model",
        fold_id=0,
        y_true=np.array(["a", "b"]),
        y_pred=np.array(["a", "b"]),
        class_names=np.array(["a", "b"]),
        fold_label_train="train",
        fold_label_test="test",
    )

    def scores(**kwargs):
        calls.append(kwargs.copy())
        return {"accuracy": crosseval.Metric(0.5, "Accuracy")}

    fold.scores = scores
    perf = crosseval.ModelGlobalPerformance(
        model_name="model",
        per_fold_outputs={0: fold},
        abstain_label="Unknown",
    )

    perf.aggregated_per_fold_scores(with_abstention=False)
    perf.aggregated_per_fold_scores(with_abstention=True)
    perf.aggregated_per_fold_scores(with_abstention=True)
    perf.aggregated_per_fold_scores(
        with_abstention=True,
        exclude_metrics_that_dont_factor_in_abstentions=True,
    )

    assert [
        (
            call["with_abstention"],
            call["exclude_metrics_that_dont_factor_in_abstentions"],
        )
        for call in calls
    ] == [(False, False), (True, False), (True, True)]


def test_model_global_performance_is_shallow_frozen_snapshot():
    fold = crosseval.ModelSingleFoldPerformance(
        model_name="model",
        fold_id=0,
        y_true=np.array(["a", "b"]),
        y_pred=np.array(["a", "a"]),
        class_names=np.array(["a", "b"]),
        fold_label_train="train",
        fold_label_test="test",
    )
    perf = crosseval.ModelGlobalPerformance(
        model_name="model",
        per_fold_outputs={0: fold},
        abstain_label="Unknown",
    )

    with pytest.raises(FrozenInstanceError):
        perf.per_fold_outputs = {}

    cached_y_pred = perf.cv_y_pred_without_abstention.copy()
    fold.y_pred = np.array(["a", "b"])

    assert np.array_equal(perf.per_fold_outputs[0].y_pred, np.array(["a", "b"]))
    assert np.array_equal(perf.cv_y_pred_without_abstention, cached_y_pred)
    assert perf.global_scores(formatted=False)["Accuracy"] == 0.5


def test_global_scores_ignores_probability_scorers_for_backwards_compatibility():
    perf = crosseval.ModelGlobalPerformance(
        model_name="model",
        per_fold_outputs={
            0: crosseval.ModelSingleFoldPerformance(
                model_name="model",
                fold_id=0,
                y_true=np.array(["a", "b"]),
                y_pred=np.array(["a", "b"]),
                y_preds_proba=np.array([[0.9, 0.1], [0.1, 0.9]]),
                class_names=np.array(["a", "b"]),
                fold_label_train="train",
                fold_label_test="test",
            )
        },
        abstain_label="Unknown",
    )

    scores = perf.global_scores(
        probability_scorers={
            "custom": (
                lambda y_true, y_score, labels, sample_weight=None: 1,
                "Custom probability",
                {},
            )
        },
        formatted=False,
    )

    assert scores["Accuracy"] == 1.0
    assert "Custom probability" not in scores


def test_full_report_scores_keep_probability_scorers_per_fold_only():
    def custom_probability_scorer(y_true, y_score, labels, sample_weight=None):
        return 0.5

    perf = crosseval.ModelGlobalPerformance(
        model_name="model",
        per_fold_outputs={
            0: crosseval.ModelSingleFoldPerformance(
                model_name="model",
                fold_id=0,
                y_true=np.array(["a", "b"]),
                y_pred=np.array(["a", "b"]),
                y_preds_proba=np.array([[0.9, 0.1], [0.1, 0.9]]),
                class_names=np.array(["a", "b"]),
                fold_label_train="train",
                fold_label_test="test",
            )
        },
        abstain_label="Unknown",
    )

    scores = perf._full_report_scores(
        probability_scorers={
            "custom": (
                custom_probability_scorer,
                "Custom probability",
                {},
            )
        },
        formatted=False,
    )

    assert scores["per_fold"]["Custom probability"] == 0.5
    assert "Custom probability" not in scores["global"]


def test_one_class_fold_skips_unavailable_probability_metrics():
    perf = crosseval.ModelGlobalPerformance(
        model_name="model",
        per_fold_outputs={
            0: crosseval.ModelSingleFoldPerformance(
                model_name="model",
                fold_id=0,
                y_true=np.array(["a", "a"]),
                y_pred=np.array(["a", "a"]),
                y_preds_proba=np.array([[0.9, 0.1], [0.8, 0.2]]),
                class_names=np.array(["a", "b"]),
                fold_label_train="train",
                fold_label_test="test",
            ),
            1: crosseval.ModelSingleFoldPerformance(
                model_name="model",
                fold_id=1,
                y_true=np.array(["a", "b"]),
                y_pred=np.array(["a", "b"]),
                y_preds_proba=np.array([[0.9, 0.1], [0.1, 0.9]]),
                class_names=np.array(["a", "b"]),
                fold_label_train="train",
                fold_label_test="test",
            ),
        },
        abstain_label="Unknown",
    )

    scores = perf.aggregated_per_fold_scores(
        label_scorers={
            "accuracy": (
                lambda y_true, y_pred, sample_weight=None: np.mean(y_true == y_pred),
                "Accuracy",
                {},
            )
        }
    )

    assert scores["Accuracy"] == "1.000 +/- 0.000 (in 2 folds)"
    assert scores["ROC-AUC (weighted OvO)"] == "1.000 +/- 0.000 (in 1 folds)"


def test_numeric_labels_with_string_abstention_label_score_correctly():
    perf = crosseval.ModelGlobalPerformance(
        model_name="model",
        per_fold_outputs={
            0: crosseval.ModelSingleFoldPerformance(
                model_name="model",
                fold_id=0,
                y_true=np.array([0, 1]),
                y_pred=np.array([0, 1]),
                class_names=np.array([0, 1]),
                y_preds_proba=np.array([[0.9, 0.1], [0.1, 0.9]]),
                test_abstentions=np.array([1]),
                fold_label_train="train",
                fold_label_test="test",
            )
        },
        abstain_label="Unknown",
    )

    scores = perf.global_scores(with_abstention=True, formatted=False)

    assert scores["Accuracy"] == 2 / 3
    assert "Unknown" in perf.classification_report
    assert "Unknown" in perf.confusion_matrix_label_ordering
    assert list(perf.cv_y_preds_proba.columns) == ["0", "1", "Unknown"]
    assert perf.cv_y_preds_proba["Unknown"].sum() == 0
    assert perf.get_all_entries().shape[0] == 3


def test_get_stats_handles_mixed_label_types_without_abstention():
    perf = crosseval.ModelGlobalPerformance(
        model_name="model",
        per_fold_outputs={
            0: crosseval.ModelSingleFoldPerformance(
                model_name="model",
                fold_id=0,
                y_true=np.array([0, "one"], dtype=object),
                y_pred=np.array([0, "one"], dtype=object),
                class_names=np.array([0, "one"], dtype=object),
                fold_label_train="train",
                fold_label_test="test",
            )
        },
        abstain_label="Unknown",
    )

    stats = perf._get_stats(formatted=False)

    assert stats["Accuracy global"] == 1.0
    assert stats["missing_classes"] is False
