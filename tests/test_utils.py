from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from crosseval.utils import is_clf_a_sklearn_pipeline


def test_is_clf_a_sklearn_pipeline_accepts_pipeline_subclasses():
    class CustomPipeline(Pipeline):
        pass

    clf = CustomPipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression()),
        ]
    )

    assert is_clf_a_sklearn_pipeline(clf)
