# crosseval

[![](https://img.shields.io/pypi/v/crosseval.svg)](https://pypi.python.org/pypi/crosseval)
[![CI](https://github.com/maximz/crosseval/actions/workflows/ci.yaml/badge.svg?branch=master)](https://github.com/maximz/crosseval/actions/workflows/ci.yaml)
[![](https://img.shields.io/badge/docs-here-blue.svg)](https://crosseval.maximz.com)
[![](https://img.shields.io/github/stars/maximz/crosseval?style=social)](https://github.com/maximz/crosseval)

crosseval summarizes machine-learning classifier performance across cross-validation folds. It stores per-fold predictions, probabilities, metadata, abstentions, feature importances, and sample weights, then aggregates them into model-level reports and model-comparison tables.

## Installation

```bash
pip install crosseval
```

## Core Types

`Metric` stores one score value plus its display name.

`ModelSingleFoldPerformance` stores the predictions and scores for one trained model on one fold.

`ModelGlobalPerformance` combines all folds for one model and produces per-fold aggregates, global scores, confusion matrices, and full text reports.

`ExperimentSet` stores many `(model_name, fold_id)` results and summarizes them into an `ExperimentSetGlobalPerformance` comparison.

## Example

```python
import crosseval
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

X, y = load_iris(return_X_y=True, as_frame=True)

models = {
    "logistic": LogisticRegression(max_iter=1000),
    "forest": RandomForestClassifier(n_estimators=100, random_state=0),
}

folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
per_fold = []

for fold_id, (train_idx, test_idx) in enumerate(folds.split(X, y)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    for model_name, estimator in models.items():
        clf = clone(estimator).fit(X_train, y_train)
        per_fold.append(
            crosseval.ModelSingleFoldPerformance(
                model_name=model_name,
                fold_id=fold_id,
                clf=clf,
                X_test=X_test,
                y_true=y_test,
                fold_label_train=f"fold-{fold_id}-train",
                fold_label_test=f"fold-{fold_id}-test",
            )
        )

experiment = crosseval.ExperimentSet(per_fold)
summary = experiment.summarize(abstain_label="Unknown")

print(summary.get_model_comparison_stats().to_string())
print(summary.get_model_comparison_stats(formatted=False).to_string())
```

By default, `get_model_comparison_stats()` returns display strings such as `"0.973 +/- 0.025 (in 3 folds)"` and `"0.980"`. Use `formatted=False` when downstream code needs floats for sorting, thresholding, or further aggregation.

## sklearn

crosseval works with fitted sklearn-style classifiers that expose `predict()` and `classes_`. If the classifier exposes `predict_proba()`, crosseval computes probability-based metrics such as ROC-AUC and au-PRC per fold. If the estimator exposes `feature_importances_` or linear-model `coef_`, crosseval stores feature-importance tables; sklearn `Pipeline` objects are handled by inspecting the final estimator.

## Development

```bash
uv sync
uv run pre-commit install

uv run pytest
uv run pre-commit run --all-files --show-diff-on-failure
```

When developing crosseval and genetools side by side, install the local checkout
after syncing:

```bash
uv pip install -e ../genetools
```
