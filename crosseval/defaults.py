import logging

from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
)


import multiclass_metrics

logger = logging.getLogger(__name__)


DEFAULT_LABEL_SCORERS = {
    "accuracy": (accuracy_score, "Accuracy", {}),
    "mcc": (matthews_corrcoef, "MCC", {}),
}

DEFAULT_PROBABILITY_SCORERS = {
    "rocauc": (
        multiclass_metrics.roc_auc_score,
        "ROC-AUC (weighted OvO)",
        {"average": "weighted", "multi_class": "ovo"},
    ),
    "rocauc_macro_average": (
        multiclass_metrics.roc_auc_score,
        "ROC-AUC (macro OvO)",
        {"average": "macro", "multi_class": "ovo"},
    ),
    "auprc": (
        multiclass_metrics.auprc,
        "au-PRC (weighted OvO)",
        {"average": "weighted", "multi_class": "ovo"},
    ),
    "auprc_macro_average": (
        multiclass_metrics.auprc,
        "au-PRC (macro OvO)",
        {"average": "macro", "multi_class": "ovo"},
    ),
}
