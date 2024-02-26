import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple, Union, Optional

import pandas as pd

from crosseval import ModelGlobalPerformance

logger = logging.getLogger(__name__)


@dataclass
class ExperimentSetGlobalPerformance:
    """Summarizes performance of many models across all CV folds."""

    model_global_performances: Dict[
        str, ModelGlobalPerformance
    ]  # map model name -> ModelGlobalPerformance

    def get_model_comparison_stats(
        self,
        label_scorers: Optional[Dict[str, Tuple[Callable, str, dict]]] = None,
        probability_scorers: Optional[Dict[str, Tuple[Callable, str, dict]]] = None,
        sort=True,
    ):
        if len(self.model_global_performances) == 0:
            # Edge case: empty
            return pd.DataFrame()

        # Put all scores in one table, and optionally sort by first score column.
        combined_stats = pd.DataFrame.from_dict(
            {
                model_name: model_global_performance._get_stats(
                    label_scorers=label_scorers, probability_scorers=probability_scorers
                )
                for model_name, model_global_performance in self.model_global_performances.items()
            },
            orient="index",
        )

        if sort:
            combined_stats.sort_values(
                by=combined_stats.columns[0], ascending=False, inplace=True
            )

        return combined_stats

    def export_all_models(
        self,
        func_generate_classification_report_fname: Callable[[str], Union[str, Path]],
        func_generate_confusion_matrix_fname: Callable[[str], Union[str, Path]],
        confusion_matrix_figsize: Optional[Tuple[float, float]] = None,
        confusion_matrix_true_label="Patient of origin",
        confusion_matrix_pred_label="Predicted label",
        dpi=72,
        label_scorers: Optional[Dict[str, Tuple[Callable, str, dict]]] = None,
        probability_scorers: Optional[Dict[str, Tuple[Callable, str, dict]]] = None,
    ):
        """Export global results for each model.
        func_generate_classification_report_fname and func_generate_confusion_matrix_fname should be functions that accept model_name str and return a file name.
        """
        for (
            model_name,
            model_global_performance,
        ) in self.model_global_performances.items():
            model_global_performance.export(
                classification_report_fname=func_generate_classification_report_fname(
                    model_name
                ),
                confusion_matrix_fname=func_generate_confusion_matrix_fname(model_name),
                confusion_matrix_figsize=confusion_matrix_figsize,
                confusion_matrix_true_label=confusion_matrix_true_label,
                confusion_matrix_pred_label=confusion_matrix_pred_label,
                dpi=dpi,
                label_scorers=label_scorers,
                probability_scorers=probability_scorers,
            )
