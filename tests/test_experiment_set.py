import crosseval
import numpy as np


def _single_fold_performance(model_name, fold_id):
    return crosseval.ModelSingleFoldPerformance(
        model_name=model_name,
        fold_id=fold_id,
        y_true=np.array(["a"]),
        y_pred=np.array(["a"]),
        class_names=np.array(["a"]),
        fold_label_train="train",
        fold_label_test="test",
    )


def test_experiment_set_copy(sample_data, sample_data_two, models_factory, tmp_path):
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
    experiment_set_copy = experiment_set.copy()
    # confirm it was a deep copy
    assert experiment_set_copy is not experiment_set
    assert id(experiment_set_copy) != id(experiment_set)
    first_key = next(iter(experiment_set.model_outputs.keys()))
    assert id(experiment_set.model_outputs[first_key]) != id(
        experiment_set_copy.model_outputs[first_key]
    )
    assert id(experiment_set.model_outputs[first_key].y_true) != id(
        experiment_set_copy.model_outputs[first_key].y_true
    )


def test_experiment_set_incomplete_detection_uses_fold_ids_not_counts():
    experiment_set = crosseval.ExperimentSet(
        model_outputs=[
            _single_fold_performance("model_a", 0),
            _single_fold_performance("model_b", 1),
        ]
    )

    assert set(experiment_set.incomplete_models) == {"model_a", "model_b"}
    assert set(experiment_set.incomplete_folds) == {0, 1}
    assert experiment_set.summarize().model_global_performances == {}


def test_experiment_set_incomplete_detection_keeps_shared_folds():
    experiment_set = crosseval.ExperimentSet(
        model_outputs=[
            _single_fold_performance("model_a", 0),
            _single_fold_performance("model_a", 1),
            _single_fold_performance("model_b", 0),
        ]
    )

    assert experiment_set.incomplete_models == ["model_b"]
    assert experiment_set.incomplete_folds == [1]

    drop_models = experiment_set.remove_incomplete(inplace=False)
    assert set(drop_models.model_outputs.keys(dimensions=0)) == {"model_a"}

    drop_folds = experiment_set.remove_incomplete(
        inplace=False,
        remove_incomplete_strategy=crosseval.RemoveIncompleteStrategy.DROP_INCOMPLETE_FOLDS,
    )
    assert set(drop_folds.model_outputs.keys(dimensions=0)) == {"model_a", "model_b"}
    assert set(drop_folds.model_outputs.keys(dimensions=1)) == {0}
