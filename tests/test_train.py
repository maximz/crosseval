import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

import crosseval


def test_train_classifier_raises_when_requested_export_fails(tmp_path):
    clf = DummyClassifier(strategy="most_frequent")

    with pytest.raises(RuntimeError, match="saving classifier"):
        crosseval.train_classifier(
            clf=clf,
            X_train=np.array([[0], [1], [2]]),
            y_train=np.array(["a", "a", "b"]),
            model_name="dummy",
            export_clf_fname=tmp_path / "missing" / "clf.joblib",
        )
