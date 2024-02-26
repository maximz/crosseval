import numpy as np
import pandas as pd
import pytest

import crosseval
from .conftest import all_model_names


def test_ModelSingleFoldPerformance_constructor_patterns(sample_data, models_factory):
    (X_train, y_train, X_test, y_test) = sample_data
    clf = models_factory()["logistic_multinomial"].fit(X_train, y_train)

    # typical: pass clf and X_test
    crosseval.ModelSingleFoldPerformance(
        model_name="logistic_multinomial",
        fold_id=0,
        clf=clf,
        X_test=X_test,
        y_true=y_test,
        fold_label_train="train",
        fold_label_test="test",
    )

    # atypical: pass computed fields directly
    crosseval.ModelSingleFoldPerformance(
        model_name="logistic_multinomial",
        fold_id=0,
        y_true=y_test,
        y_pred=clf.predict(X_test),
        class_names=clf.classes_,
        X_test_shape=X_test.shape,
        y_decision_function=clf.decision_function(X_test),
        y_preds_proba=clf.predict_proba(X_test),
        fold_label_train="train",
        fold_label_test="test",
    )


@pytest.mark.xfail
def test_ModelSingleFoldPerformance_constructor_requires_one_of_two_patterns(
    sample_data,
):
    (X_train, y_train, X_test, y_test) = sample_data

    # passing none of the above does not work
    crosseval.ModelSingleFoldPerformance(
        model_name="logistic_multinomial",
        fold_id=0,
        y_true=y_test,
        fold_label_train="train",
        fold_label_test="test",
    )


def test_ModelSingleFoldPerformance_copy(sample_data, models_factory):
    (X_train, y_train, X_test, y_test) = sample_data
    model_name, clf = next(iter(models_factory().items()))
    clf = clf.fit(X_train, y_train)
    single_perf = crosseval.ModelSingleFoldPerformance(
        model_name=model_name,
        fold_id=0,
        clf=clf,
        X_test=X_test,
        y_true=y_test,
        fold_label_train="train",
        fold_label_test="test",
    )
    single_perf_copy = single_perf.copy()
    assert np.array_equal(single_perf.y_pred, single_perf_copy.y_pred)
    # This was not a deep copy. The fields are not replaced. Same memory addresses.
    assert single_perf.y_pred is single_perf_copy.y_pred
    assert id(single_perf.y_pred) == id(single_perf_copy.y_pred)


def test_ModelSingleFoldPerformance_apply_abstention_mask(
    sample_data, sample_data_two, models_factory
):
    model_outputs = []
    for fold_id, (X_train, y_train, X_test, y_test) in zip(
        [0, 1], [sample_data, sample_data_two]
    ):
        for model_name, clf in models_factory().items():
            clf = clf.fit(X_train, y_train)
            single_perf = crosseval.ModelSingleFoldPerformance(
                model_name=model_name,
                fold_id=fold_id,
                clf=clf,
                X_test=np.vstack([X_test, X_test[:2]]),
                y_true=np.hstack([y_test, ["Healthy1", "HIV1"]]),
                fold_label_train="train",
                fold_label_test="test",
                test_metadata=pd.DataFrame(
                    {
                        "patient_id": np.hstack([np.arange(X_test.shape[0]), [-1, -2]]),
                    }
                ),
                test_sample_weights=np.hstack([np.ones(X_test.shape[0]), [1.1, 1.1]]),
            )

            # Confirm initial state
            assert (
                single_perf.y_true.shape[0]
                == single_perf.y_pred.shape[0]
                == single_perf.X_test_shape[0]
                == single_perf.test_metadata.shape[0]
                == single_perf.test_sample_weights.shape[0]
                == y_test.shape[0] + 2
            )
            if single_perf.y_decision_function is not None:
                # it's None for "dummy" model
                assert single_perf.y_decision_function.shape[0] == y_test.shape[0] + 2
            if single_perf.y_preds_proba is not None:
                # it's None for "linearsvm" model
                assert single_perf.y_preds_proba.shape[0] == y_test.shape[0] + 2
            assert single_perf.test_abstentions.shape[0] == 0
            assert single_perf.test_abstention_metadata.empty
            assert single_perf.test_abstention_sample_weights is None

            # Apply mask
            single_perf = single_perf.apply_abstention_mask(
                single_perf.test_metadata["patient_id"] < 0
            )

            # Confirm new sizes
            assert (
                single_perf.y_true.shape[0]
                == single_perf.y_pred.shape[0]
                == single_perf.X_test_shape[0]
                == single_perf.test_metadata.shape[0]
                == single_perf.test_sample_weights.shape[0]
                == y_test.shape[0]
            )
            if single_perf.y_decision_function is not None:
                # it's None for "dummy" model
                assert single_perf.y_decision_function.shape[0] == y_test.shape[0]
            if single_perf.y_preds_proba is not None:
                # it's None for "linearsvm" model
                assert single_perf.y_preds_proba.shape[0] == y_test.shape[0]
            assert (
                single_perf.test_abstentions.shape[0]
                == single_perf.test_abstention_metadata.shape[0]
                == single_perf.test_abstention_sample_weights.shape[0]
                == 2
            )
            assert all(single_perf.test_metadata["patient_id"] >= 0)
            assert all(single_perf.test_abstention_metadata["patient_id"] < 0)
            assert np.array_equal(single_perf.test_abstentions, ["Healthy1", "HIV1"])
            assert all(single_perf.test_abstention_sample_weights == 1.1)
            assert all(single_perf.test_sample_weights == 1.0)

            model_outputs.append(single_perf)
    experiment_set = crosseval.ExperimentSet(model_outputs=model_outputs)
    experiment_set_global_performance = experiment_set.summarize()
    for model in all_model_names:
        print(model)
        print(
            experiment_set_global_performance.model_global_performances[
                model
            ].full_report()
        )
        experiment_set_global_performance.model_global_performances[
            model
        ].confusion_matrix_fig()
        print()

    all_entries = experiment_set_global_performance.model_global_performances[
        "logistic_multinomial"
    ].get_all_entries()
    print(all_entries)
    assert not all_entries["y_true"].isna().any()
    assert all_entries["difference_between_top_two_predicted_probas"].isna().sum() == 4
    assert all_entries.shape[0] == 14

    combined_stats = experiment_set_global_performance.get_model_comparison_stats()
    assert set(combined_stats.index) == set(all_model_names)
    expected_series = pd.Series(
        {
            "ROC-AUC (weighted OvO) per fold": "1.000 +/- 0.000 (in 2 folds)",
            "ROC-AUC (macro OvO) per fold": "1.000 +/- 0.000 (in 2 folds)",
            "au-PRC (weighted OvO) per fold": "1.000 +/- 0.000 (in 2 folds)",
            "au-PRC (macro OvO) per fold": "1.000 +/- 0.000 (in 2 folds)",
            "Accuracy per fold": "1.000 +/- 0.000 (in 2 folds)",
            "MCC per fold": "1.000 +/- 0.000 (in 2 folds)",
            "Accuracy global": "1.000",
            "MCC global": "1.000",
            "Accuracy per fold with abstention": "0.694 +/- 0.000 (in 2 folds)",
            "MCC per fold with abstention": "0.704 +/- 0.000 (in 2 folds)",
            "Unknown/abstention proportion per fold with abstention": "0.286 +/- 0.000 (in 2 folds)",
            "Accuracy global with abstention": "0.694",
            "MCC global with abstention": "0.704",
            "Unknown/abstention proportion global with abstention": "0.286",
            "Abstention label global with abstention": "Unknown",
            "sample_size": 10,
            "n_abstentions": 4,
            "sample_size including abstentions": 14,
            "abstention_rate": 4 / 14,
            "missing_classes": False,
        },
    )
    print(
        combined_stats.loc["logistic_multinomial"].to_dict()
    )  # if the below assertion fails, scroll up to see the print out to update expected_series
    pd.testing.assert_series_equal(
        combined_stats.loc["logistic_multinomial"], expected_series, check_names=False
    )


def test_ModelSingleFoldPerformance_apply_abstention_mask_empty_mask(
    sample_data, models_factory
):
    # Edge case: mask nothing
    (X_train, y_train, X_test, y_test) = sample_data
    model_name = "logistic_multinomial"
    clf = models_factory()[model_name]
    clf = clf.fit(X_train, y_train)
    single_perf = crosseval.ModelSingleFoldPerformance(
        model_name=model_name,
        fold_id=0,
        clf=clf,
        X_test=X_test,
        y_true=y_test,
        fold_label_train="train",
        fold_label_test="test",
    )

    original_scores = single_perf.scores()

    # Apply mask of all False
    single_perf = single_perf.apply_abstention_mask(np.full(y_test.shape[0], False))

    # Confirm new sizes
    assert (
        single_perf.y_true.shape[0]
        == single_perf.y_pred.shape[0]
        == single_perf.y_decision_function.shape[0]
        == single_perf.y_preds_proba.shape[0]
        == single_perf.X_test_shape[0]
        == y_test.shape[0]
    )
    assert single_perf.test_abstentions.shape[0] == 0
    assert single_perf.test_abstention_metadata.empty
    assert single_perf.test_abstention_sample_weights is None

    assert original_scores == single_perf.scores()


def test_ModelSingleFoldPerformance_apply_abstention_mask_entire_mask(
    sample_data, sample_data_two, models_factory
):
    # Edge case: mask everything
    model_outputs = []
    for fold_id, (X_train, y_train, X_test, y_test) in zip(
        [0, 1], [sample_data, sample_data_two]
    ):
        for model_name, clf in models_factory().items():
            clf = clf.fit(X_train, y_train)
            single_perf = crosseval.ModelSingleFoldPerformance(
                model_name=model_name,
                fold_id=fold_id,
                clf=clf,
                X_test=X_test,
                y_true=y_test,
                fold_label_train="train",
                fold_label_test="test",
            )

            # Apply mask of all True
            single_perf = single_perf.apply_abstention_mask(
                np.full(y_test.shape[0], True)
            )

            # Confirm new sizes
            assert (
                single_perf.y_true.shape[0]
                == single_perf.y_pred.shape[0]
                == single_perf.X_test_shape[0]
                == 0
            )
            if single_perf.y_decision_function is not None:
                # it's None for "dummy" model
                assert single_perf.y_decision_function.shape[0] == 0
            if single_perf.y_preds_proba is not None:
                # it's None for "linearsvm" model
                assert single_perf.y_preds_proba.shape[0] == 0
            assert single_perf.test_abstentions.shape[0] == y_test.shape[0]

            model_outputs.append(single_perf)
    experiment_set = crosseval.ExperimentSet(model_outputs=model_outputs)
    experiment_set_global_performance = experiment_set.summarize()

    # Carrying around len(y_true) == 0 objects should be fine, until it's time to compute scores
    with pytest.raises(ValueError, match="Cannot compute scores when y_true is empty"):
        experiment_set_global_performance.get_model_comparison_stats()


def test_ModelSingleFoldPerformance_post_init_replace_explicit_None_values_with_default_factory(
    sample_data, models_factory
):
    (X_train, y_train, X_test, y_test) = sample_data
    model_name = "logistic_multinomial"
    clf = models_factory()[model_name]
    clf = clf.fit(X_train, y_train)
    single_perf = crosseval.ModelSingleFoldPerformance(
        model_name=model_name,
        fold_id=0,
        clf=clf,
        X_test=X_test,
        y_true=y_test,
        fold_label_train="train",
        fold_label_test="test",
        test_abstentions=None,
    )
    assert single_perf.test_abstentions is not None
