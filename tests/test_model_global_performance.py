import crosseval


import copy
import pickle

import numpy as np


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
