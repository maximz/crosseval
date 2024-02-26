import collections.abc
import logging
from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union, Optional

import genetools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sentinels
from sklearn.metrics import (
    classification_report,
)


from genetools.plots import plot_confusion_matrix
from genetools.stats import make_confusion_matrix


import multiclass_metrics

from crosseval import ModelSingleFoldPerformance, Metric, Classifier
from crosseval.utils import support_list_and_dict_arguments_in_cache_decorator
from crosseval.scores import compute_classification_scores


logger = logging.getLogger(__name__)


# Sentinel value
Y_TRUE_VALUES: sentinels.Sentinel = sentinels.Sentinel("default y_true column")


def _wrap_as_list(list_or_single_value: Union[Any, Iterable]) -> Iterable[Any]:
    """Wrap as list, even if given single value"""
    if isinstance(list_or_single_value, str) or not isinstance(
        list_or_single_value, collections.abc.Iterable
    ):
        return [list_or_single_value]

    # already iterable, and is not a string
    return list_or_single_value


def _stack_numpy_arrays_horizontally_into_string_array(
    arrs: Iterable[np.ndarray],
) -> np.ndarray:
    """create combined tuples of chosen columns (essentially zip)"""
    return np.array([", ".join(item) for item in np.column_stack(arrs).astype(str)])


@dataclass(eq=False)
class ModelGlobalPerformance:
    """Summarizes performance of one model across all CV folds."""

    model_name: str
    per_fold_outputs: Dict[
        int, ModelSingleFoldPerformance
    ]  # map fold ID -> ModelSingleFoldPerformance
    abstain_label: str
    # Optional override of the default column name for global scores.
    # Can pass multiple columns.
    # If provided, each column must either match a metadata column name, or must be crosseval.Y_TRUE_VALUES (a sentinel indicating use default y_true).
    # Example use case: override y_true to show confusion matrix to delineate disease groups further into "past exposure" or "active exposure" groups, even though classifier trained on full disease label.
    global_evaluation_column_name: Optional[
        Union[str, sentinels.Sentinel, List[Union[str, sentinels.Sentinel]]]
    ] = None

    def __post_init__(self):
        """Data validation on initialization.
        See https://stackoverflow.com/a/60179826/130164"""

        # Computing all these properties is expensive.
        # TODO: Should we disable most validation for higher performance through lazy evaluation?

        for fold_output in self.per_fold_outputs.values():
            if self.model_name != fold_output.model_name:
                raise ValueError(
                    "All folds must be from the same model. Model_name was different."
                )

        # sanity checks / validation
        # TODO: can we remove some of these since these are now being validated at ModelSingleFoldPerformance level?
        if self.has_abstentions:
            # TODO: warn if abstain_label conflicts with name of existing class
            if self.cv_abstentions_metadata is not None:
                if self.cv_abstentions_metadata.shape[0] != self.n_abstentions:
                    raise ValueError(
                        "If test_abstentions_metadata is provided, it must match the number of abstention ground truth labels (test_abstentions) provided."
                    )
            else:  # self.cv_abstentions_metadata is None
                if self.cv_metadata is not None:
                    raise ValueError(
                        "If there are abstentions and metadata was provided for non-abstained examples, then test_abstentions_metadata (metadata dataframe) must be provided alongside test_abstentions (ground truth labels)"
                    )

            if self._cv_abstentions_sample_weights is not None:
                if self._cv_abstentions_sample_weights.shape[0] != self.n_abstentions:
                    raise ValueError(
                        "If test_abstention_sample_weights is provided, it must match the number of abstention ground truth labels (test_abstentions) provided."
                    )
            else:  # self._cv_abstentions_sample_weights is None
                if self.cv_sample_weights_without_abstention is not None:
                    raise ValueError(
                        "If there are abstentions and sample weights was provided for non-abstained examples, then test_abstention_sample_weights must be provided alongside test_abstentions (ground truth labels)"
                    )

        if self.global_evaluation_column_name is not None:
            # loop over provided column names
            # wrap as list even if given a single value
            colnames = _wrap_as_list(self.global_evaluation_column_name)
            if len(colnames) == 0:
                raise ValueError(
                    "global_evaluation_column_name cannot be an empty list."
                )
            for colname in colnames:
                if colname == Y_TRUE_VALUES:
                    # skip any that match sentinel like crosseval.Y_TRUE_COLUMN
                    continue
                # validate that cv_metadata exists and column name is in there
                if self.cv_metadata is None:
                    raise ValueError(
                        "If a global_evaluation_column_name is provided (that is not Y_TRUE_VALUES), then cv_metadata must be provided."
                    )
                elif colname not in self.cv_metadata.columns:
                    raise ValueError(
                        f"global_evaluation_column_name {colname} is not a column in the metadata dataframe"
                    )

                if self.has_abstentions:
                    # and if we have abstentions, validate that cv_abstentions_metadata exists and column name in there
                    if self.cv_abstentions_metadata is None:
                        raise ValueError(
                            "If global_evaluation_column_name is provided (that is not Y_TRUE_VALUES) and there are abstentions, then cv_abstentions_metadata must be provided."
                        )
                    elif colname not in self.cv_abstentions_metadata.columns:
                        # validate that all provided column names are valid abstention metadata column names,
                        # except for a sentinel like crosseval.Y_TRUE_COLUMN
                        raise ValueError(
                            f"global_evaluation_column_name {colname} is not a column in the abstentions metadata dataframe"
                        )

        if (
            self.cv_y_true_without_abstention.shape[0]
            != self.cv_y_pred_without_abstention.shape[0]
        ):
            raise ValueError("cv_y_true and cv_y_pred must have same shape")

        if (
            self.cv_y_true_with_abstention.shape[0]
            != self.cv_y_pred_with_abstention.shape[0]
        ):
            raise ValueError("cv_y_true and cv_y_pred must have same shape")

        # If any metadata was supplied, make sure all entries (all folds) had metadata supplied.
        if (
            self.cv_metadata is not None
            and self.cv_y_true_without_abstention.shape[0] != self.cv_metadata.shape[0]
        ):
            raise ValueError(
                "Not all folds supplied metadata (cv_y_true_without_abstention and cv_metadata have different lengths)."
            )

        if self.cv_sample_weights_without_abstention is not None:
            if (
                self.cv_sample_weights_without_abstention.shape[0]
                != self.cv_y_pred_without_abstention.shape[0]
            ):
                raise ValueError(
                    "cv_sample_weights_without_abstention and cv_y_pred must have same shape"
                )
            if (
                self.has_abstentions
                and self.cv_sample_weights_without_abstention.shape[0]
                + self._cv_abstentions_sample_weights.shape[0]
                != self.cv_y_pred_with_abstention.shape[0]
            ):
                # sample weights are passed for abstained examples separately,
                # and we already validated above that if non-abstained examples had sample weights, then abstained examples do too.
                # here we check that total counts match.
                raise ValueError(
                    "Sample weights must have same shape as cv_y_pred + # of abstentions"
                )

    @cached_property
    def fold_order(self) -> List[int]:
        return sorted(self.per_fold_outputs.keys())

    @support_list_and_dict_arguments_in_cache_decorator
    @cache
    def aggregated_per_fold_scores(
        self,
        with_abstention=True,
        exclude_metrics_that_dont_factor_in_abstentions=False,
        label_scorers: Optional[Dict[str, Tuple[Callable, str, dict]]] = None,
        probability_scorers: Optional[Dict[str, Tuple[Callable, str, dict]]] = None,
    ) -> Dict[str, str]:
        """return dict mapping friendly-metric-name to "mean +/- std" formatted string (computed across folds)"""
        raw_metrics_per_fold: Dict[int, Dict[str, Metric]] = {
            fold_id: fold_output.scores(
                with_abstention=with_abstention,
                exclude_metrics_that_dont_factor_in_abstentions=exclude_metrics_that_dont_factor_in_abstentions,
                abstain_label=self.abstain_label,
                label_scorers=label_scorers,
                probability_scorers=probability_scorers,
            )
            for fold_id, fold_output in self.per_fold_outputs.items()
        }

        # Put everything in a pandas df,
        # aggregate by (metric_keyname, metric_friendlyname),
        # and run mean, std, and size (i.e. report n_folds present in case < total n_folds)

        # This dataframe has index = fold_id, columns = metric keyname, values = Metric object
        scores_per_fold = pd.DataFrame.from_dict(raw_metrics_per_fold, orient="index")
        # extract metric friendlynames (TODO(later): move away from separation of metric-keyname and metric-friendly-name. use friendly name only?)
        map_metric_keyname_to_friendly_name = {}
        for colname in scores_per_fold.columns:
            friendly_names = (
                scores_per_fold[colname]
                .dropna()
                .apply(lambda metric: metric.friendly_name)
                .values
            )
            # These should be identical between folds (though a metric might not appear in all folds)
            if len(set(friendly_names)) > 1:
                raise ValueError(
                    f"Metric friendly names must be unique for metric keyname {colname}"
                )
            map_metric_keyname_to_friendly_name[colname] = friendly_names[0]
        # now change df to have values = metric value rather than full Metric object (again, note that a metric might not appear in all folds)
        scores_per_fold = scores_per_fold.applymap(
            lambda metric: metric.value if isinstance(metric, Metric) else np.nan
        )

        # aggregate mean, standard deviation, and non-NaN count (columns) for each metric keyname (index)
        scores_per_fold_agg = scores_per_fold.describe().loc[["mean", "std", "count"]].T
        scores_per_fold_agg["std"].fillna(
            0, inplace=True
        )  # if a metric appeared in only one fold, it will have std NaN
        # Add metric friendlyname. (unlike replace()'s pass-through behavior, map() means that if not in the dict, will store NaN)
        scores_per_fold_agg[
            "metric_friendly_name"
        ] = scores_per_fold_agg.index.to_series().map(
            map_metric_keyname_to_friendly_name
        )
        if scores_per_fold_agg.isna().any().any():
            raise ValueError("Scores_per_fold_agg had NaNs")
        if scores_per_fold_agg["metric_friendly_name"].duplicated().any():
            raise ValueError("Some metrics had duplicate friendly names")

        # summarize a range of scores into strings: mean plus-minus one standard deviation (68% interval if normally distributed).
        return {
            row[
                "metric_friendly_name"
            ]: f"""{row['mean']:0.3f} +/- {row['std']:0.3f} (in {row['count']:n} folds)"""
            for _, row in scores_per_fold_agg.iterrows()
        }

    @staticmethod
    def _concatenate_in_fold_order(value_name, model_output, fold_order) -> np.ndarray:
        """Combine numpy array from all folds, in fold order, not original anndata.obs order.
        If data not present in folds, returns empty numpy array, instead of None."""
        vals = (getattr(model_output[fold_id], value_name) for fold_id in fold_order)
        vals = [v for v in vals if v is not None]
        if len(vals) == 0:
            return np.array([])
        return np.concatenate(vals).ravel()

    @staticmethod
    def _concatenate_dataframes_in_fold_order(
        value_name, model_output, fold_order
    ) -> pd.DataFrame:
        """Combine pandas dataframe from all folds, in fold order, not original anndata.obs order.
        Index gets reset (original index stored as a column).
        If data not present in folds, returns empty dataframe, instead of None."""
        vals = (getattr(model_output[fold_id], value_name) for fold_id in fold_order)
        return pd.concat(
            vals,
            axis=0,
        ).reset_index()  # original index gets stored as a column. keep index as a column in case there was valuable info in there.

    @staticmethod
    def _get_column_combination_or_pass_through(
        requested_column_names,
        func_get_metadata_df,
        func_get_default_ground_truth,
        default_ground_truth_sentinel_value=Y_TRUE_VALUES,
    ):
        """create synthetic column that combines all requested column names
        func_get_metadata_df and func_get_default_ground_truth should be functions for lazy evaluation, because they can be expensive to compute
        """
        if requested_column_names is None:
            return func_get_default_ground_truth()

        # wrap as list in case we were given a single value
        requested_column_names = _wrap_as_list(requested_column_names)

        metadata_df = None
        if any(
            colname != default_ground_truth_sentinel_value
            for colname in requested_column_names
        ):
            # lazy evaluation
            metadata_df = func_get_metadata_df()

        values_to_combine = []
        for colname in requested_column_names:
            if colname == default_ground_truth_sentinel_value:
                values_to_combine.append(func_get_default_ground_truth())
            else:
                values_to_combine.append(metadata_df[colname].values)

        if len(values_to_combine) == 1:
            # pass through if only one column
            return values_to_combine[0]

        # combine
        return _stack_numpy_arrays_horizontally_into_string_array(values_to_combine)

    @cached_property
    def cv_y_true_without_abstention(self) -> np.ndarray:
        # sub in self.global_evaluation_column_name if defined
        return self._get_column_combination_or_pass_through(
            requested_column_names=self.global_evaluation_column_name,
            func_get_metadata_df=lambda: self.cv_metadata,
            # standard output if not overriding with a global_evaluation_column_name
            func_get_default_ground_truth=lambda: self._concatenate_in_fold_order(
                "y_true", self.per_fold_outputs, self.fold_order
            ),
        )

    @cached_property
    def cv_y_pred_without_abstention(self) -> np.ndarray:
        return self._concatenate_in_fold_order(
            "y_pred", self.per_fold_outputs, self.fold_order
        )

    @cached_property
    def _model_output_with_abstention(self) -> pd.DataFrame:
        output = pd.DataFrame(
            {
                "y_true": self.cv_y_true_without_abstention,
                "y_pred": self.cv_y_pred_without_abstention,
            }
        )

        if self.has_abstentions:
            # prepare abstention values
            # sub in self.global_evaluation_column_name if defined
            abstention_values = self._get_column_combination_or_pass_through(
                requested_column_names=self.global_evaluation_column_name,
                func_get_metadata_df=lambda: self.cv_abstentions_metadata,
                # standard output if not overriding with a global_evaluation_column_name
                func_get_default_ground_truth=lambda: self.cv_abstentions,
            )

            # concatenate with abstentions
            output = pd.concat(
                [
                    output,
                    pd.DataFrame({"y_true": abstention_values}).assign(
                        y_pred=self.abstain_label
                    ),
                ],
                axis=0,
            ).reset_index(drop=True)

        return output

    def get_all_entries(self) -> pd.DataFrame:
        """Get all predicted (y_pred) and ground-truth (y_true) labels for each example from all folds, along with any metadata if provided.
        Abstentions are included at the end, if available.
        max_predicted_proba, second_highest_predicted_proba, and difference_between_top_two_predicted_probas included for all but abstentions.
        """

        true_vs_pred_labels = (
            self._model_output_with_abstention
        )  # abstentions included. default (0 to n-1) index

        if self.cv_y_preds_proba is not None:
            df_probas_top_two = pd.DataFrame(
                self.cv_y_preds_proba.apply(
                    lambda row: pd.Series(row.nlargest(2).values), axis=1
                )
            )
            df_probas_top_two.columns = [
                "max_predicted_proba",
                "second_highest_predicted_proba",
            ]

            df_probas_top_two["difference_between_top_two_predicted_probas"] = (
                df_probas_top_two["max_predicted_proba"]
                - df_probas_top_two["second_highest_predicted_proba"]
            )

            # combine horizontally, but note that this will have NaNs for abstentions
            true_vs_pred_labels = pd.concat(
                [true_vs_pred_labels, df_probas_top_two], axis=1
            )

        if self.cv_metadata is not None:
            metadata_compiled = pd.concat(
                [self.cv_metadata, self.cv_abstentions_metadata], axis=0
            ).reset_index(drop=True)
            # if these columns already exist, rename them
            metadata_compiled = metadata_compiled.rename(
                columns={"y_true": "metadata_y_true", "y_pred": "metadata_y_pred"}
            )
            # combine horizontally
            true_vs_pred_labels = pd.concat(
                [true_vs_pred_labels, metadata_compiled], axis=1
            )

        if true_vs_pred_labels.shape[0] != self._model_output_with_abstention.shape[0]:
            raise ValueError("Shape changed unexpectedly")

        return true_vs_pred_labels

    @property
    def cv_y_true_with_abstention(self) -> np.ndarray:
        """includes abstained ground truth labels"""
        return self._model_output_with_abstention["y_true"].values

    @property
    def cv_y_pred_with_abstention(self) -> np.ndarray:
        """includes "Unknown" or similar when abstained on an example"""
        return self._model_output_with_abstention["y_pred"].values

    @cached_property
    def cv_sample_weights_without_abstention(self) -> Union[np.ndarray, None]:
        """Combine test-set sample weights in fold order, if they were supplied"""
        if any(
            fold_output.test_sample_weights is None
            for fold_output in self.per_fold_outputs.values()
        ):
            # Require test_sample_weights to be defined in every fold, otherwise stop
            return None
        # TODO: Switch to returning empty array, for consistency?
        return self._concatenate_in_fold_order(
            "test_sample_weights", self.per_fold_outputs, self.fold_order
        )

    @cached_property
    def _cv_abstentions_sample_weights(self) -> Union[np.ndarray, None]:
        """Concatenate sample weights (if supplied) of abstained test examples from each fold.
        (Abstentions don't necessarily need to occur in each fold, though.)
        """
        if all(
            fold_output.test_abstention_sample_weights is None
            for fold_output in self.per_fold_outputs.values()
        ):
            # Require test_abstention_sample_weights to be defined in at least one fold, otherwise stop
            # (Abstentions might occur only in some folds)
            return None
        # TODO: Switch to returning empty array, for consistency?
        return self._concatenate_in_fold_order(
            "test_abstention_sample_weights", self.per_fold_outputs, self.fold_order
        )

    @cached_property
    def cv_sample_weights_with_abstention(self) -> Union[np.ndarray, None]:
        """Combine test-set sample weights in fold order, then add abstention sample weights if we had abstentions. Returns None if no sample weights supplied."""
        if not self.has_abstentions:
            return self.cv_sample_weights_without_abstention  # might be None
        if self.cv_sample_weights_without_abstention is not None:
            return np.hstack(
                [
                    self.cv_sample_weights_without_abstention,
                    self._cv_abstentions_sample_weights,
                ]
            )
        # TODO: Switch to returning empty array, for consistency?
        return None

    @cached_property
    def cv_metadata(self) -> Union[pd.DataFrame, None]:
        """If supplied, concatenate dataframes of metadata for each test example, in fold order, not in original adata.obs order.
        If supplied in any fold, must be supplied for all folds.
        If not supplied in any fold, returns None.
        """
        test_metadata_concat = self._concatenate_dataframes_in_fold_order(
            "test_metadata", self.per_fold_outputs, self.fold_order
        )

        # Return None if no metadata was supplied for any fold.
        # TODO: Switch to returning empty dataframe, for consistency?
        if test_metadata_concat.shape[0] > 0:
            return test_metadata_concat
        return None

    @cached_property
    def cv_abstentions(self) -> np.ndarray:
        """Concatenate ground truth labels of abstained test examples from each fold.
        These don't necessarily need to be provided for each fold though.
        """
        return self._concatenate_in_fold_order(
            "test_abstentions", self.per_fold_outputs, self.fold_order
        )

    @cached_property
    def cv_abstentions_metadata(self) -> Union[pd.DataFrame, None]:
        """Concatenate metadata of abstained test examples from each fold.
        These don't necessarily need to be provided for each fold though.
        Returns None if not supplied for any fold.
        """
        test_metadata_concat = self._concatenate_dataframes_in_fold_order(
            "test_abstention_metadata", self.per_fold_outputs, self.fold_order
        )

        # Return None if no metadata was supplied for any fold.
        # TODO: Switch to returning empty dataframe, for consistency?
        if test_metadata_concat.shape[0] > 0:
            return test_metadata_concat
        return None

    @cached_property
    def sample_size_without_abstentions(self):
        return self.cv_y_true_without_abstention.shape[0]

    @cached_property
    def sample_size_with_abstentions(self):
        return self.cv_y_true_with_abstention.shape[0]

    @cached_property
    def n_abstentions(self):
        return self.cv_abstentions.shape[0]

    @cached_property
    def has_abstentions(self) -> bool:
        return self.cv_abstentions.shape[0] > 0

    @cached_property
    def abstention_proportion(self):
        """abstention proportion: what percentage of predictions were unknown"""
        return self.n_abstentions / self.sample_size_with_abstentions

    @cached_property
    def cv_y_preds_proba(self) -> Union[pd.DataFrame, None]:
        """Concatenate y_preds_proba (if supplied), in fold order, not in original adata.obs order.
        Abstentions never included here."""
        if any(
            fold_output.y_preds_proba is None
            for fold_output in self.per_fold_outputs.values()
        ):
            # Require y_preds_proba to be defined for every fold, otherwise stop.
            return None
        # Confirm class names are in same order across all folds.
        for fold_id in self.fold_order:
            if not np.array_equal(
                self.per_fold_outputs[fold_id].class_names,
                self.per_fold_outputs[self.fold_order[0]].class_names,
            ):
                logger.warning(
                    f"Class names are not the same across folds: {fold_id} vs {self.fold_order[0]} for model {self.model_name}"
                )

        # Convert to dataframes with column names, and concatenate
        y_preds_proba_concat = pd.concat(
            [
                pd.DataFrame(
                    self.per_fold_outputs[fold_id].y_preds_proba,
                    columns=self.per_fold_outputs[fold_id].class_names,
                )
                for fold_id in self.fold_order
            ],
            axis=0,
        )

        # Confirm n_examples by n_classes shape.
        if self.cv_y_pred_without_abstention.shape[0] != y_preds_proba_concat.shape[0]:
            raise ValueError(
                "y_preds_proba has different number of rows than cv_y_pred_without_abstention"
            )

        if y_preds_proba_concat.isna().any().any():
            logger.warning(
                f"Model {self.model_name} has missing probabilities for some samples; may be because class names were not the same across folds. Filling with 0s."
            )
            y_preds_proba_concat.fillna(0.0, inplace=True)

        # So far we have included all class names ever predicted by any fold's model.
        # But it's possible there are other class names seen in the data.
        # Add any missing classes to the probability matrix.
        y_preds_proba_concat, labels = multiclass_metrics._inject_missing_labels(
            y_true=self.confusion_matrix_label_ordering,
            y_score=y_preds_proba_concat.values,
            labels=y_preds_proba_concat.columns,
        )
        # convert back to dataframe (default 0 to n-1 index)
        y_preds_proba_concat = pd.DataFrame(y_preds_proba_concat, columns=labels)

        # Arrange columns in same order as cm_label_order
        if set(y_preds_proba_concat.columns) != set(
            self.confusion_matrix_label_ordering
        ):
            raise ValueError(
                "y_preds_proba has different columns than confusion_matrix_label_ordering (without considering order)"
            )
        y_preds_proba_concat = y_preds_proba_concat[
            self.confusion_matrix_label_ordering
        ]

        return y_preds_proba_concat

    @cached_property
    def classification_report(self) -> str:
        """Classification report"""
        # zero_division=0 is same as "warn" but suppresses this warning for labels with no predicted samples:
        # `UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.``
        return classification_report(
            self.cv_y_true_with_abstention,
            self.cv_y_pred_with_abstention,
            zero_division=0,
        )

    def _full_report_scores(
        self,
        label_scorers: Optional[Dict[str, Tuple[Callable, str, dict]]] = None,
        probability_scorers: Optional[Dict[str, Tuple[Callable, str, dict]]] = None,
    ):
        scores = {
            "per_fold": self.aggregated_per_fold_scores(
                with_abstention=False,
                label_scorers=label_scorers,
                probability_scorers=probability_scorers,
            ),
            "global": self.global_scores(
                with_abstention=False,
                label_scorers=label_scorers,
                probability_scorers=probability_scorers,
            ),
        }
        if self.has_abstentions:
            scores["per_fold_with_abstention"] = self.aggregated_per_fold_scores(
                with_abstention=True,
                exclude_metrics_that_dont_factor_in_abstentions=True,
                label_scorers=label_scorers,
                probability_scorers=probability_scorers,
            )
            scores["global_with_abstention"] = self.global_scores(
                with_abstention=True,
                label_scorers=label_scorers,
                probability_scorers=probability_scorers,
            )
        return scores

    def full_report(
        self,
        label_scorers: Optional[Dict[str, Tuple[Callable, str, dict]]] = None,
        probability_scorers: Optional[Dict[str, Tuple[Callable, str, dict]]] = None,
    ) -> str:
        def formatted_scores(header, score_dict):
            return "\n".join(
                [header]
                + [
                    f"{metric_friendly_name}: {metric_value}"
                    for metric_friendly_name, metric_value in score_dict.items()
                ]
            )

        # Return per-fold scores, global scores without abstention, and global scores with abstention.
        scores = self._full_report_scores(
            label_scorers=label_scorers, probability_scorers=probability_scorers
        )
        global_column_usage = (
            f" using column name {self.global_evaluation_column_name}"
            if self.global_evaluation_column_name is not None
            else ""
        )
        pieces = [
            formatted_scores(
                f"Per-fold scores{' without abstention' if self.has_abstentions else ''}:",
                scores["per_fold"],
            ),
            formatted_scores(
                f"Global scores{' without abstention' if self.has_abstentions else ''}{global_column_usage}:",
                scores["global"],
            ),
        ]
        if self.has_abstentions:
            pieces.append(
                formatted_scores(
                    "Per-fold scores with abstention (note that abstentions not included in probability-based scores):",
                    scores["per_fold_with_abstention"],
                ),
            )
            pieces.append(
                formatted_scores(
                    f"Global scores with abstention{global_column_usage}:",
                    scores["global_with_abstention"],
                ),
            )
        pieces.append(
            "\n".join(
                [
                    f"Global classification report{' with abstention' if self.has_abstentions else ''}{global_column_usage}:",
                    self.classification_report,
                ]
            )
        )
        return "\n\n".join(pieces)

    @support_list_and_dict_arguments_in_cache_decorator
    @cache
    def global_scores(
        self,
        with_abstention=True,
        label_scorers: Optional[Dict[str, Tuple[Callable, str, dict]]] = None,
        probability_scorers: Optional[Dict[str, Tuple[Callable, str, dict]]] = None,
    ) -> dict:
        """Calculate global scores with or without abstention. Global scores should not include probabilistic scores."""
        scores = compute_classification_scores(
            y_true=self.cv_y_true_with_abstention
            if with_abstention
            else self.cv_y_true_without_abstention,
            y_preds=self.cv_y_pred_with_abstention
            if with_abstention
            else self.cv_y_pred_without_abstention,
            y_preds_proba=None,
            y_preds_proba_classes=None,
            sample_weights=self.cv_sample_weights_with_abstention
            if with_abstention
            else self.cv_sample_weights_without_abstention,
            label_scorers=label_scorers,
            probability_scorers=probability_scorers,
        )
        # Format
        scores_formatted = [
            (metric.friendly_name, f"{metric.value:0.3f}")
            for metric_keyname, metric in scores.items()
        ]
        # confirm all metric friendly names are unique
        all_metric_friendly_names = [v[0] for v in scores_formatted]
        if len(set(all_metric_friendly_names)) != len(all_metric_friendly_names):
            raise ValueError("Metric friendly names are not unique")

        # then convert to dict
        scores_formatted = dict(scores_formatted)

        if with_abstention:
            scores_formatted[
                "Unknown/abstention proportion"
            ] = f"{self.abstention_proportion:0.3f}"
            scores_formatted["Abstention label"] = self.abstain_label

        if self.global_evaluation_column_name is not None:
            scores_formatted[
                "Global evaluation column name"
            ] = self.global_evaluation_column_name

        return scores_formatted

    def _get_stats(
        self,
        label_scorers: Optional[Dict[str, Tuple[Callable, str, dict]]] = None,
        probability_scorers: Optional[Dict[str, Tuple[Callable, str, dict]]] = None,
    ):
        """Get overall stats for table"""
        scores = self._full_report_scores(
            label_scorers=label_scorers, probability_scorers=probability_scorers
        )

        # Combine all scores into single dictionary.
        scores_dict = {}
        for suffix, scores_dict_part in [
            ("per fold", "per_fold"),
            ("global", "global"),
            ("per fold with abstention", "per_fold_with_abstention"),
            ("global with abstention", "global_with_abstention"),
        ]:
            if scores_dict_part not in scores:
                continue
            for k, v in scores[scores_dict_part].items():
                scores_dict[f"{k} {suffix}"] = v

        # Other summary stats.
        nunique_predicted_labels = np.unique(self.cv_y_pred_without_abstention).shape[0]
        nunique_true_labels = np.unique(self.cv_y_true_without_abstention).shape[0]

        scores_dict.update(
            {
                "sample_size": self.sample_size_without_abstentions,
                "n_abstentions": self.n_abstentions,
                "sample_size including abstentions": self.sample_size_with_abstentions,
                "abstention_rate": self.abstention_proportion,
                # Flag if number of unique predicted labels is less than number of unique ground truth labels
                "missing_classes": nunique_predicted_labels < nunique_true_labels,
            }
        )
        return scores_dict

    @cached_property
    def confusion_matrix_label_ordering(self) -> List[str]:
        """Order of labels in confusion matrix"""
        return sorted(
            np.unique(
                np.hstack(
                    [
                        np.unique(
                            self.cv_y_true_with_abstention
                        ),  # includes self.cv_abstentions abstained ground truth labels, if any
                        np.unique(
                            self.cv_y_pred_with_abstention
                        ),  # may include self.abstain_label
                    ]
                )
            )
        )

    def confusion_matrix(
        self,
        confusion_matrix_true_label="Patient of origin",
        confusion_matrix_pred_label="Predicted label",
    ) -> pd.DataFrame:
        """Confusion matrix"""
        return make_confusion_matrix(
            y_true=self.cv_y_true_with_abstention,
            y_pred=self.cv_y_pred_with_abstention,
            true_label=confusion_matrix_true_label,
            pred_label=confusion_matrix_pred_label,
            label_order=self.confusion_matrix_label_ordering,
        )

    def confusion_matrix_fig(
        self,
        confusion_matrix_figsize: Optional[Tuple[float, float]] = None,
        confusion_matrix_true_label="Patient of origin",
        confusion_matrix_pred_label="Predicted label",
    ) -> plt.Figure:
        """Confusion matrix figure"""
        fig, ax = plot_confusion_matrix(
            self.confusion_matrix(
                confusion_matrix_true_label=confusion_matrix_true_label,
                confusion_matrix_pred_label=confusion_matrix_pred_label,
            ),
            figsize=confusion_matrix_figsize,
        )
        plt.close(fig)
        return fig

    def export(
        self,
        classification_report_fname: Union[str, Path],
        confusion_matrix_fname: Union[str, Path],
        confusion_matrix_figsize: Optional[Tuple[float, float]] = None,
        confusion_matrix_true_label="Patient of origin",
        confusion_matrix_pred_label="Predicted label",
        dpi=72,
        label_scorers: Optional[Dict[str, Tuple[Callable, str, dict]]] = None,
        probability_scorers: Optional[Dict[str, Tuple[Callable, str, dict]]] = None,
    ):
        # Save classification report
        with open(classification_report_fname, "w") as w:
            w.write(
                self.full_report(
                    label_scorers=label_scorers, probability_scorers=probability_scorers
                )
            )

        # Save confusion matrix contents in text format. Rows are ground truth labels; columns are predicted labels.
        # TODO(later): Create a command line tool to render this saved data as a confusion matrix figure, with user-configurable DPI
        self.confusion_matrix(
            confusion_matrix_true_label=confusion_matrix_true_label,
            confusion_matrix_pred_label=confusion_matrix_pred_label,
        ).to_csv(
            # Replace final extension (eg .png) with .confusion_matrix_data.tsv
            Path(confusion_matrix_fname).with_suffix(".confusion_matrix_data.tsv"),
            sep="\t",
        )

        # Save confusion matrix figure
        genetools.plots.savefig(
            self.confusion_matrix_fig(
                confusion_matrix_figsize=confusion_matrix_figsize,
                confusion_matrix_true_label=confusion_matrix_true_label,
                confusion_matrix_pred_label=confusion_matrix_pred_label,
            ),
            confusion_matrix_fname,
            dpi=dpi,
        )

    @property
    def per_fold_classifiers(self) -> Dict[int, Classifier]:
        """reload classifier objects from disk"""
        return {
            fold_id: model_single_fold_performance.classifier
            for fold_id, model_single_fold_performance in self.per_fold_outputs.items()
        }

    @cached_property
    def feature_importances(self) -> Union[pd.DataFrame, None]:
        """
        Get feature importances for each fold.
        Extracts feature importances or coefficients from sklearn pipelines' inner models.
        Returns fold_ids x feature_names DataFrame, or None if no feature importances are available.
        """
        feature_importances = {
            fold_id: model_single_fold_performance.feature_importances
            for fold_id, model_single_fold_performance in self.per_fold_outputs.items()
        }

        if all(fi is None for fi in feature_importances.values()):
            # Model had no feature importances in any fold.
            return None

        # Combine all
        return pd.DataFrame.from_dict(feature_importances, orient="index").rename_axis(
            index="fold_id"
        )

    @cached_property
    def multiclass_feature_importances(self) -> Union[Dict[int, pd.DataFrame], None]:
        """
        Get One-vs-Rest multiclass feature importances for each fold.
        Extracts feature importances or coefficients from sklearn pipelines' inner models.
        Returns dict mapping fold_id to a classes x features DataFrame,
        or returns None if no feature importances are available.
        """
        feature_importances = {
            fold_id: model_single_fold_performance.multiclass_feature_importances
            for fold_id, model_single_fold_performance in self.per_fold_outputs.items()
        }

        if all(fi is None for fi in feature_importances.values()):
            # Model had no feature importances in any fold.
            return None

        return feature_importances
