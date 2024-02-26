import crosseval


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
