#!/usr/bin/env python

import numpy as np
import pandas as pd
import pytest

import crosseval

from .conftest import (
    all_model_names,
    tree_models,
    ovr_models,
    ovo_models,
    dummy_models,
    no_coefs_models,
)


def test_crosseval(sample_data, sample_data_two, models_factory, tmp_path):
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
            print(single_perf.scores())
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

    combined_stats = experiment_set_global_performance.get_model_comparison_stats()
    assert set(combined_stats.index) == set(all_model_names)
    assert combined_stats.loc["logistic_multinomial"].equals(
        pd.Series(
            {
                "ROC-AUC (weighted OvO) per fold": "1.000 +/- 0.000 (in 2 folds)",
                "ROC-AUC (macro OvO) per fold": "1.000 +/- 0.000 (in 2 folds)",
                "au-PRC (weighted OvO) per fold": "1.000 +/- 0.000 (in 2 folds)",
                "au-PRC (macro OvO) per fold": "1.000 +/- 0.000 (in 2 folds)",
                "Accuracy per fold": "1.000 +/- 0.000 (in 2 folds)",
                "MCC per fold": "1.000 +/- 0.000 (in 2 folds)",
                "Accuracy global": "1.000",
                "MCC global": "1.000",
                "sample_size": 10,
                "n_abstentions": 0,
                "sample_size including abstentions": 10,
                "abstention_rate": 0.0,
                "missing_classes": False,
            }
        )
    ), f"Observed: {combined_stats.loc['logistic_multinomial'].to_dict()}"
    assert (
        experiment_set_global_performance.model_global_performances[
            "logistic_multinomial"
        ].full_report()
        == """Per-fold scores:
ROC-AUC (weighted OvO): 1.000 +/- 0.000 (in 2 folds)
ROC-AUC (macro OvO): 1.000 +/- 0.000 (in 2 folds)
au-PRC (weighted OvO): 1.000 +/- 0.000 (in 2 folds)
au-PRC (macro OvO): 1.000 +/- 0.000 (in 2 folds)
Accuracy: 1.000 +/- 0.000 (in 2 folds)
MCC: 1.000 +/- 0.000 (in 2 folds)

Global scores:
Accuracy: 1.000
MCC: 1.000

Global classification report:
              precision    recall  f1-score   support

       Covid       1.00      1.00      1.00         4
       Ebola       1.00      1.00      1.00         2
         HIV       1.00      1.00      1.00         2
     Healthy       1.00      1.00      1.00         2

    accuracy                           1.00        10
   macro avg       1.00      1.00      1.00        10
weighted avg       1.00      1.00      1.00        10
"""
    )

    experiment_set_global_performance.export_all_models(
        func_generate_classification_report_fname=lambda model_name: tmp_path
        / f"{model_name}.classification_report.txt",
        func_generate_confusion_matrix_fname=lambda model_name: tmp_path
        / f"{model_name}.confusion_matrix.png",
        confusion_matrix_figsize=(4, 4),
        dpi=72,
    )


def test_crosseval_with_abstention(
    sample_data, sample_data_two, models_factory, tmp_path
):
    # Some folds will have abstentions, while others don't.
    model_outputs = []
    for fold_id, (X_train, y_train, X_test, y_test), add_abstentions in zip(
        [0, 1], [sample_data, sample_data_two], [False, True]
    ):
        for model_name, clf in models_factory().items():
            clf = clf.fit(X_train, y_train)
            test_metadata = pd.DataFrame(
                {
                    "patient_id": range(X_test.shape[0]),
                    "alternate_ground_truth_column": y_test,
                }
            )
            single_perf = crosseval.ModelSingleFoldPerformance(
                model_name=model_name,
                fold_id=fold_id,
                clf=clf,
                X_test=X_test,
                y_true=y_test,
                fold_label_train="train",
                fold_label_test="test",
                test_metadata=test_metadata,
                test_abstentions=["Healthy", "HIV"] if add_abstentions else None,
                test_abstention_metadata=pd.DataFrame(
                    {
                        "patient_id": [-1, -2],
                        "alternate_ground_truth_column": ["Healthy", "HIV"],
                    }
                )
                if add_abstentions
                else None,
                test_sample_weights=np.ones(X_test.shape[0]),
                test_abstention_sample_weights=np.ones(2) if add_abstentions else None,
            )
            print(single_perf.scores())
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
    assert all_entries["difference_between_top_two_predicted_probas"].isna().sum() == 2
    assert all_entries.shape[0] == 12

    combined_stats = experiment_set_global_performance.get_model_comparison_stats()
    assert set(combined_stats.index) == set(all_model_names)
    expected_series = pd.Series(
        {
            "ROC-AUC (weighted OvO) per fold": "1.000 +/- 0.000 (in 2 folds)",
            "ROC-AUC (macro OvO) per fold": "1.000 +/- 0.000 (in 2 folds)",
            "au-PRC (weighted OvO) per fold": "1.000 +/- 0.000 (in 2 folds)",
            "au-PRC (macro OvO) per fold": "1.000 +/- 0.000 (in 2 folds)",
            # Excluding abstentions: each fold has 5/5 correct predictions.
            "Accuracy per fold": "1.000 +/- 0.000 (in 2 folds)",
            "MCC per fold": "1.000 +/- 0.000 (in 2 folds)",
            "Accuracy global": "1.000",
            "MCC global": "1.000",
            # We have one fold with 5/5 correct predictions, and another fold with 5/7 correct predictions where the two mistakes are abstentions:
            # Global accuracy should be 10/12 = 0.833.
            # Per-fold accuracy should be 5/5 = 1.0 for the first fold, and 5/7 = 0.714 for the second fold, which averages to 0.857.
            # Global abstention rate is 2/12 = 0.167.
            # Per-fold abstention rate is 0/5 = 0.0 for the first fold, and 2/7 = 0.286 for the second fold, which averages to 0.143.
            "Accuracy per fold with abstention": "0.857 +/- 0.202 (in 2 folds)",
            "MCC per fold with abstention": "0.851 +/- 0.210 (in 2 folds)",
            # See comment above for why the per-fold abstention rate is expected to be 0.143.
            "Unknown/abstention proportion per fold with abstention": "0.143 +/- 0.202 (in 2 folds)",
            # See comment above for why the global accuracy is expected to be 0.833.
            "Accuracy global with abstention": "0.833",
            "MCC global with abstention": "0.808",
            # See comment above for why the global abstention rate is expected to be 0.167.
            "Unknown/abstention proportion global with abstention": "0.167",
            "Abstention label global with abstention": "Unknown",
            "sample_size": 10,
            "n_abstentions": 2,
            "sample_size including abstentions": 12,
            "abstention_rate": 2 / 12,
            "missing_classes": False,
        },
    )
    print(
        combined_stats.loc["logistic_multinomial"].to_dict()
    )  # if the below assertion fails, scroll up to see the print out to update expected_series
    pd.testing.assert_series_equal(
        combined_stats.loc["logistic_multinomial"], expected_series, check_names=False
    )
    assert (
        experiment_set_global_performance.model_global_performances[
            "logistic_multinomial"
        ].full_report()
        == """Per-fold scores without abstention:
ROC-AUC (weighted OvO): 1.000 +/- 0.000 (in 2 folds)
ROC-AUC (macro OvO): 1.000 +/- 0.000 (in 2 folds)
au-PRC (weighted OvO): 1.000 +/- 0.000 (in 2 folds)
au-PRC (macro OvO): 1.000 +/- 0.000 (in 2 folds)
Accuracy: 1.000 +/- 0.000 (in 2 folds)
MCC: 1.000 +/- 0.000 (in 2 folds)

Global scores without abstention:
Accuracy: 1.000
MCC: 1.000

Per-fold scores with abstention (note that abstentions not included in probability-based scores):
Accuracy: 0.857 +/- 0.202 (in 2 folds)
MCC: 0.851 +/- 0.210 (in 2 folds)
Unknown/abstention proportion: 0.143 +/- 0.202 (in 2 folds)

Global scores with abstention:
Accuracy: 0.833
MCC: 0.808
Unknown/abstention proportion: 0.167
Abstention label: Unknown

Global classification report with abstention:
              precision    recall  f1-score   support

       Covid       1.00      1.00      1.00         4
       Ebola       1.00      1.00      1.00         2
         HIV       1.00      0.67      0.80         3
     Healthy       1.00      0.67      0.80         3
     Unknown       0.00      0.00      0.00         0

    accuracy                           0.83        12
   macro avg       0.80      0.67      0.72        12
weighted avg       1.00      0.83      0.90        12
"""
    )

    experiment_set_global_performance.export_all_models(
        func_generate_classification_report_fname=lambda model_name: tmp_path
        / f"{model_name}.classification_report.txt",
        func_generate_confusion_matrix_fname=lambda model_name: tmp_path
        / f"{model_name}.confusion_matrix.png",
        dpi=72,
    )


def test_crosseval_with_alternate_column_name(
    sample_data, sample_data_two, models_factory, tmp_path
):
    """test scenario where we provide metadata object and an alternate column name, including metadata for abstentions."""
    model_outputs = []
    for fold_id, (X_train, y_train, X_test, y_test) in zip(
        [0, 1], [sample_data, sample_data_two]
    ):
        for model_name, clf in models_factory().items():
            clf = clf.fit(X_train, y_train)
            test_metadata = pd.DataFrame(
                {
                    "patient_id": range(X_test.shape[0]),
                    "alternate_ground_truth_column": y_test,
                }
            )
            single_perf = crosseval.ModelSingleFoldPerformance(
                model_name=model_name,
                fold_id=fold_id,
                clf=clf,
                X_test=X_test,
                y_true=y_test,
                fold_label_train="train",
                fold_label_test="test",
                test_metadata=test_metadata,
                test_abstentions=["Healthy", "HIV"],
                test_abstention_metadata=pd.DataFrame(
                    {
                        "patient_id": [-1, -2],
                        "alternate_ground_truth_column": ["Healthy", "HIV"],
                    }
                ),
            )
            print(single_perf.scores())
            model_outputs.append(single_perf)

    experiment_set = crosseval.ExperimentSet(model_outputs=model_outputs)
    experiment_set_global_performance = experiment_set.summarize(
        global_evaluation_column_name="alternate_ground_truth_column"
    )

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

    combined_stats = experiment_set_global_performance.get_model_comparison_stats()
    assert set(combined_stats.index) == set(all_model_names)
    assert combined_stats.loc["logistic_multinomial"].equals(
        pd.Series(
            {
                "ROC-AUC (weighted OvO) per fold": "1.000 +/- 0.000 (in 2 folds)",
                "ROC-AUC (macro OvO) per fold": "1.000 +/- 0.000 (in 2 folds)",
                "au-PRC (weighted OvO) per fold": "1.000 +/- 0.000 (in 2 folds)",
                "au-PRC (macro OvO) per fold": "1.000 +/- 0.000 (in 2 folds)",
                "Accuracy per fold": "1.000 +/- 0.000 (in 2 folds)",
                "MCC per fold": "1.000 +/- 0.000 (in 2 folds)",
                "Accuracy global": "1.000",
                "MCC global": "1.000",
                "Global evaluation column name global": "alternate_ground_truth_column",
                "Accuracy per fold with abstention": "0.714 +/- 0.000 (in 2 folds)",
                "MCC per fold with abstention": "0.703 +/- 0.000 (in 2 folds)",
                "Unknown/abstention proportion per fold with abstention": "0.286 +/- 0.000 (in 2 folds)",
                "Accuracy global with abstention": "0.714",
                "MCC global with abstention": "0.703",
                "Unknown/abstention proportion global with abstention": "0.286",
                "Abstention label global with abstention": "Unknown",
                "Global evaluation column name global with abstention": "alternate_ground_truth_column",
                "sample_size": 10,
                "n_abstentions": 4,
                "sample_size including abstentions": 14,
                "abstention_rate": 4 / 14,
                "missing_classes": False,
            },
        )
    ), f"Observed: {combined_stats.loc['logistic_multinomial'].to_dict()}"
    assert (
        experiment_set_global_performance.model_global_performances[
            "logistic_multinomial"
        ].full_report()
        == """Per-fold scores without abstention:
ROC-AUC (weighted OvO): 1.000 +/- 0.000 (in 2 folds)
ROC-AUC (macro OvO): 1.000 +/- 0.000 (in 2 folds)
au-PRC (weighted OvO): 1.000 +/- 0.000 (in 2 folds)
au-PRC (macro OvO): 1.000 +/- 0.000 (in 2 folds)
Accuracy: 1.000 +/- 0.000 (in 2 folds)
MCC: 1.000 +/- 0.000 (in 2 folds)

Global scores without abstention using column name alternate_ground_truth_column:
Accuracy: 1.000
MCC: 1.000
Global evaluation column name: alternate_ground_truth_column

Per-fold scores with abstention (note that abstentions not included in probability-based scores):
Accuracy: 0.714 +/- 0.000 (in 2 folds)
MCC: 0.703 +/- 0.000 (in 2 folds)
Unknown/abstention proportion: 0.286 +/- 0.000 (in 2 folds)

Global scores with abstention using column name alternate_ground_truth_column:
Accuracy: 0.714
MCC: 0.703
Unknown/abstention proportion: 0.286
Abstention label: Unknown
Global evaluation column name: alternate_ground_truth_column

Global classification report with abstention using column name alternate_ground_truth_column:
              precision    recall  f1-score   support

       Covid       1.00      1.00      1.00         4
       Ebola       1.00      1.00      1.00         2
         HIV       1.00      0.50      0.67         4
     Healthy       1.00      0.50      0.67         4
     Unknown       0.00      0.00      0.00         0

    accuracy                           0.71        14
   macro avg       0.80      0.60      0.67        14
weighted avg       1.00      0.71      0.81        14
"""
    )

    experiment_set_global_performance.export_all_models(
        func_generate_classification_report_fname=lambda model_name: tmp_path
        / f"{model_name}.classification_report.txt",
        func_generate_confusion_matrix_fname=lambda model_name: tmp_path
        / f"{model_name}.confusion_matrix.png",
        dpi=72,
    )


@pytest.mark.xfail
def test_metadata_object_required_for_alternate_column(
    sample_data, sample_data_two, models_factory
):
    model_outputs = []
    for fold_id, (X_train, y_train, X_test, y_test) in zip(
        [0, 1], [sample_data, sample_data_two]
    ):
        for model_name, clf in models_factory().items():
            # TODO: try the train method from crosseval, along with export
            clf = clf.fit(X_train, y_train)
            single_perf = crosseval.ModelSingleFoldPerformance(
                model_name=model_name,
                fold_id=fold_id,
                clf=clf,
                X_test=X_test,
                y_true=y_test,
                fold_label_train="train",
                fold_label_test="test",
                test_abstentions=["Healthy", "HIV"],
            )
            print(single_perf.scores())
            model_outputs.append(single_perf)
    experiment_set = crosseval.ExperimentSet(model_outputs=model_outputs)
    experiment_set.summarize(
        global_evaluation_column_name=[crosseval.Y_TRUE_VALUES, "column_dne"]
    )


def test_crosseval_with_multiple_alternate_column_names(
    sample_data, sample_data_two, models_factory, tmp_path
):
    """test scenario where we provide metadata object and an alternate column name, including metadata for abstentions."""
    model_outputs = []
    for fold_id, (X_train, y_train, X_test, y_test) in zip(
        [0, 1], [sample_data, sample_data_two]
    ):
        for model_name, clf in models_factory().items():
            clf = clf.fit(X_train, y_train)
            test_metadata = pd.DataFrame(
                {
                    "patient_id": range(X_test.shape[0]),
                    "alternate_ground_truth_column": [f"alternate_{s}" for s in y_test],
                }
            )
            single_perf = crosseval.ModelSingleFoldPerformance(
                model_name=model_name,
                fold_id=fold_id,
                clf=clf,
                X_test=X_test,
                y_true=y_test,
                fold_label_train="train",
                fold_label_test="test",
                test_metadata=test_metadata,
                test_abstentions=["Healthy", "HIV"],
                test_abstention_metadata=pd.DataFrame(
                    {
                        "patient_id": [-1, -2],
                        "alternate_ground_truth_column": [
                            "alternate_Healthy",
                            "alternate_HIV",
                        ],
                    }
                ),
            )
            print(single_perf.scores())
            model_outputs.append(single_perf)

    experiment_set = crosseval.ExperimentSet(model_outputs=model_outputs)
    experiment_set_global_performance = experiment_set.summarize(
        global_evaluation_column_name=[
            crosseval.Y_TRUE_VALUES,
            "alternate_ground_truth_column",
        ]
    )

    for model in all_model_names:
        print(model)
        cm = experiment_set_global_performance.model_global_performances[
            model
        ].confusion_matrix()
        print(cm)
        print()

        assert np.array_equal(
            cm.index,
            [
                "Covid, alternate_Covid",
                "Ebola, alternate_Ebola",
                "HIV, alternate_HIV",
                "Healthy, alternate_Healthy",
            ],
        )

        # Confirm repr of Y_TRUE_VALUES is printed correctly
        assert (
            "Global classification report with abstention using column name [<default y_true column>, 'alternate_ground_truth_column']"
            in experiment_set_global_performance.model_global_performances[
                model
            ].full_report()
        )


def test_feature_importances_and_names_with_default_feature_names(
    sample_data, sample_data_two, models_factory
):
    """Numpy data -> default feature names"""
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

            ## Test feature names and feature importances for single fold
            if model_name in dummy_models:
                # dummy has no coefs, and also swallows feature names
                assert single_perf.feature_names is None
                assert single_perf.feature_importances is None
                assert single_perf.multiclass_feature_importances is None
            elif model_name in ovr_models:
                # multiclass OvR linear model is special cased
                assert np.array_equal(single_perf.feature_names, [0, 1, 2, 3, 4])
                assert single_perf.feature_importances is None
                assert type(single_perf.multiclass_feature_importances) == pd.DataFrame
                assert np.array_equal(
                    single_perf.multiclass_feature_importances.index, clf.classes_
                )
                assert np.array_equal(
                    single_perf.multiclass_feature_importances.columns,
                    single_perf.feature_names,
                )
            elif model_name in ovo_models or model_name in no_coefs_models:
                # multiclass OvO linear model does not support feature importances,
                # but does store feature names.
                assert np.array_equal(single_perf.feature_names, [0, 1, 2, 3, 4])
                assert single_perf.feature_importances is None
                assert single_perf.multiclass_feature_importances is None
            elif model_name in tree_models:
                # These work just like binary linear models
                assert np.array_equal(single_perf.feature_names, [0, 1, 2, 3, 4])
                assert type(single_perf.feature_importances) == pd.Series
                assert np.array_equal(
                    single_perf.feature_importances.index, single_perf.feature_names
                )
                assert single_perf.multiclass_feature_importances is None
            else:
                raise ValueError("Did not expect other model types")
            model_outputs.append(single_perf)

    ## Test with multiple folds at aggregated level
    experiment_set = crosseval.ExperimentSet(model_outputs=model_outputs)
    experiment_set_global_performance = experiment_set.summarize()

    for model in dummy_models:
        # no feature importances
        assert (
            experiment_set_global_performance.model_global_performances[
                model
            ].feature_importances
            is None
        )
        assert (
            experiment_set_global_performance.model_global_performances[
                model
            ].multiclass_feature_importances
            is None
        )

    for model in ovr_models:
        # multiclass OvR
        assert (
            experiment_set_global_performance.model_global_performances[
                model
            ].feature_importances
            is None
        )
        dict_of_dfs = experiment_set_global_performance.model_global_performances[
            model
        ].multiclass_feature_importances
        assert np.array_equal(list(dict_of_dfs.keys()), [0, 1])
        for df in dict_of_dfs.values():
            assert np.array_equal(df.index, ["Covid", "Ebola", "HIV", "Healthy"])
            assert np.array_equal(df.columns, [0, 1, 2, 3, 4])

    for model in ovo_models:
        # multiclass OvO is not supported
        assert (
            experiment_set_global_performance.model_global_performances[
                model
            ].feature_importances
            is None
        )
        assert (
            experiment_set_global_performance.model_global_performances[
                model
            ].multiclass_feature_importances
            is None
        )

    for model in no_coefs_models:
        assert (
            experiment_set_global_performance.model_global_performances[
                model
            ].feature_importances
            is None
        )
        assert (
            experiment_set_global_performance.model_global_performances[
                model
            ].multiclass_feature_importances
            is None
        )

    for model in tree_models:
        # These work just like binary linear models
        df = experiment_set_global_performance.model_global_performances[
            model
        ].feature_importances
        assert df.index.name == "fold_id"
        assert np.array_equal(df.index, [0, 1])
        assert np.array_equal(df.columns, [0, 1, 2, 3, 4])
        assert (
            experiment_set_global_performance.model_global_performances[
                model
            ].multiclass_feature_importances
            is None
        )


def test_feature_importances_and_names_with_custom_feature_names(
    sample_data, sample_data_two, models_factory
):
    """Pandas data -> custom feature names"""
    model_outputs = []
    for fold_id, (X_train, y_train, X_test, y_test) in zip(
        [0, 1], [sample_data, sample_data_two]
    ):
        for model_name, clf in models_factory().items():
            X_train_df = pd.DataFrame(X_train).rename(columns=lambda s: f"feature_{s}")
            X_test_df = pd.DataFrame(X_test).rename(columns=lambda s: f"feature_{s}")

            clf = clf.fit(X_train_df, y_train)
            single_perf = crosseval.ModelSingleFoldPerformance(
                model_name=model_name,
                fold_id=fold_id,
                clf=clf,
                X_test=X_test_df,
                y_true=y_test,
                fold_label_train="train",
                fold_label_test="test",
            )

            ## Test feature names and feature importances for single fold

            if model_name in dummy_models:
                # dummy has no coefs, and also swallows feature names
                assert single_perf.feature_names is None
                assert single_perf.feature_importances is None
                assert single_perf.multiclass_feature_importances is None
            elif model_name in ovr_models:
                # multiclass OvR linear model is special cased
                assert np.array_equal(
                    single_perf.feature_names,
                    ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"],
                )
                assert single_perf.feature_importances is None
                assert type(single_perf.multiclass_feature_importances) == pd.DataFrame
                assert np.array_equal(
                    single_perf.multiclass_feature_importances.index, clf.classes_
                )
                assert np.array_equal(
                    single_perf.multiclass_feature_importances.columns,
                    single_perf.feature_names,
                )
            elif model_name in ovo_models or model_name in no_coefs_models:
                # multiclass OvO linear model does not support feature importances,
                # but does store feature names.
                assert np.array_equal(
                    single_perf.feature_names,
                    ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"],
                )
                assert single_perf.feature_importances is None
                assert single_perf.multiclass_feature_importances is None
            elif model_name in tree_models:
                # These work just like binary linear models
                assert np.array_equal(
                    single_perf.feature_names,
                    ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"],
                )
                assert type(single_perf.feature_importances) == pd.Series
                assert np.array_equal(
                    single_perf.feature_importances.index, single_perf.feature_names
                )
                assert single_perf.multiclass_feature_importances is None
            else:
                raise ValueError("Did not expect other model types")
            model_outputs.append(single_perf)

    ## Test with multiple folds at aggregated level
    experiment_set = crosseval.ExperimentSet(model_outputs=model_outputs)
    experiment_set_global_performance = experiment_set.summarize()

    for model in dummy_models:
        # no feature importances
        assert (
            experiment_set_global_performance.model_global_performances[
                model
            ].feature_importances
            is None
        )
        assert (
            experiment_set_global_performance.model_global_performances[
                model
            ].multiclass_feature_importances
            is None
        )

    for model in ovr_models:
        # multiclass OvR
        assert (
            experiment_set_global_performance.model_global_performances[
                model
            ].feature_importances
            is None
        )
        dict_of_dfs = experiment_set_global_performance.model_global_performances[
            model
        ].multiclass_feature_importances
        assert np.array_equal(list(dict_of_dfs.keys()), [0, 1])
        for df in dict_of_dfs.values():
            assert np.array_equal(df.index, ["Covid", "Ebola", "HIV", "Healthy"])
            assert np.array_equal(
                df.columns,
                ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"],
            )

    for model in ovo_models:
        # multiclass OvO is not supported
        assert (
            experiment_set_global_performance.model_global_performances[
                model
            ].feature_importances
            is None
        )
        assert (
            experiment_set_global_performance.model_global_performances[
                model
            ].multiclass_feature_importances
            is None
        )

    for model in no_coefs_models:
        assert (
            experiment_set_global_performance.model_global_performances[
                model
            ].feature_importances
            is None
        )
        assert (
            experiment_set_global_performance.model_global_performances[
                model
            ].multiclass_feature_importances
            is None
        )

    for model in tree_models:
        # These work just like binary linear models
        df = experiment_set_global_performance.model_global_performances[
            model
        ].feature_importances
        assert df.index.name == "fold_id"
        assert np.array_equal(df.index, [0, 1])
        assert np.array_equal(
            df.columns,
            ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"],
        )
        assert (
            experiment_set_global_performance.model_global_performances[
                model
            ].multiclass_feature_importances
            is None
        )
