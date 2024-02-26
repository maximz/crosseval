import collections.abc
import glob
import logging
from dataclasses import fields
from typing import Dict, List, Union, Optional
from typing_extensions import Self
import copy

import joblib
import sentinels
from kdict import kdict
from enum import Enum, auto

from crosseval import (
    ModelSingleFoldPerformance,
    ModelGlobalPerformance,
    ExperimentSetGlobalPerformance,
)


from enum_mixins import ValidatableEnumMixin


logger = logging.getLogger(__name__)


class RemoveIncompleteStrategy(ValidatableEnumMixin, Enum):
    """
    How to handle incomplete ExperimentSets: remove incomplete models or remove incomplete folds.

    These represent two strategies for removing incomplete results to compare models apples-to-apples on the same data (i.e. same collection of cross validation folds):

    1. Keep only those models that have results for all folds (DROP_INCOMPLETE_MODELS; sensible default).
    (How this works: Find the maximum number of folds any model has results for; keep only models analyzed for that number of folds.)
    This is relevant when you have some finnicky models that may give up on a fold out of the blue.

    2. Keep only folds that have results for all models (DROP_INCOMPLETE_FOLDS).
    This is relevant when you have a finnicky fold that fails for some but not all models.
    For example, if one cross validation fold's training set somehow only has samples of a single class (perhaps the splits were stratified for one target variable, and now you are evaluating how well you can model another classification target without changing the cross validation structure),
    many models will fail and cite that there was only a single class in the data — but some models might still succeed.
    The right answer may be to drop this broken fold altogether, rather than restricting the analysis to only those models that happen to handle this edge case.
    """

    DROP_INCOMPLETE_MODELS = auto()
    DROP_INCOMPLETE_FOLDS = auto()


class ExperimentSet:
    """Store ModelSingleFoldPerformance objects for many (model_name, fold_id) combinations."""

    model_outputs: kdict  # map (model_name, fold_id) to ModelSingleFoldPerformance

    @classmethod
    def _unwrap_nested_dict(
        cls,
        model_outputs: Union[
            collections.abc.Sequence,
            collections.abc.Mapping,
            ModelSingleFoldPerformance,
        ],
    ):
        if isinstance(model_outputs, ModelSingleFoldPerformance):
            yield model_outputs
        elif isinstance(model_outputs, collections.abc.Mapping):
            if "model_name" in model_outputs.keys():
                # stop
                yield model_outputs
            else:
                for key, value in model_outputs.items():
                    # this is a nested dict, not a dict that should become a ModelSingleFoldPerformance dataclass
                    yield from cls._unwrap_nested_dict(value)
        elif isinstance(model_outputs, collections.abc.Sequence):
            for model_output in model_outputs:
                yield from cls._unwrap_nested_dict(model_output)

    def __init__(
        self,
        model_outputs: Union[
            collections.abc.Sequence, collections.abc.Mapping, None
        ] = None,
    ):
        """stores ModelSingleFoldPerformance objects for each fold and model name.
        accepts existing single-fold model outputs as a list, a kdict, or a nested dict.
        """
        if model_outputs is None:
            model_outputs = []
        self.model_outputs = kdict()
        for model_output in self._unwrap_nested_dict(model_outputs):
            self.add(model_output)

    def add(self, model_output: Union[ModelSingleFoldPerformance, Dict]) -> None:
        def cast_to_dataclass(model_output):
            if isinstance(model_output, ModelSingleFoldPerformance):
                return model_output
            if isinstance(model_output, collections.abc.Mapping):
                # If dict, wrap dict as dataclass,
                # but only the subset of dict keys that match dataclass field names (and are not non-init fields - which are forbidden to pass)
                all_valid_field_names = [
                    field.name
                    for field in fields(ModelSingleFoldPerformance)
                    if field.init
                ]
                return ModelSingleFoldPerformance(
                    **{
                        k: v
                        for k, v in model_output.items()
                        if k in all_valid_field_names
                    }
                )
            raise ValueError(f"Unrecognized model output type: {type(model_output)}")

        data: ModelSingleFoldPerformance = cast_to_dataclass(model_output)
        self.model_outputs[data.model_name, data.fold_id] = data

    @property
    def incomplete_models(self) -> List[str]:
        if len(self.model_outputs) == 0:
            # edge case: empty
            return []

        n_folds_per_model = {
            model_name: len(self.model_outputs[model_name, :])
            for model_name in self.model_outputs.keys(dimensions=0)
        }
        max_n_folds_per_model = max(n_folds_per_model.values())
        return [
            model_name
            for model_name, n_folds in n_folds_per_model.items()
            if n_folds != max_n_folds_per_model
        ]

    @property
    def incomplete_folds(self) -> List[int]:
        if len(self.model_outputs) == 0:
            # edge case: empty
            return []
        n_models_per_fold = {
            fold_id: len(self.model_outputs[:, fold_id])
            for fold_id in self.model_outputs.keys(dimensions=1)
        }
        max_n_models_per_fold = max(n_models_per_fold.values())
        return [
            fold_id
            for fold_id, n_models in n_models_per_fold.items()
            if n_models != max_n_models_per_fold
        ]

    def remove_incomplete(
        self,
        inplace: bool = True,
        remove_incomplete_strategy: RemoveIncompleteStrategy = RemoveIncompleteStrategy.DROP_INCOMPLETE_MODELS,
    ) -> Self:
        """
        Removes incomplete results, in-place by default (which can be disabled with inplace=False).
        remove_incomplete_strategy determines whether we remove incomplete models (default) or remove incomplete folds. See RemoveIncompleteStrategy for details.
        """
        RemoveIncompleteStrategy.validate(remove_incomplete_strategy)

        if not inplace:
            clone = self.copy()
            return clone.remove_incomplete(
                inplace=True, remove_incomplete_strategy=remove_incomplete_strategy
            )

        if (
            remove_incomplete_strategy
            == RemoveIncompleteStrategy.DROP_INCOMPLETE_MODELS
        ):
            for model_name in self.incomplete_models:
                # TODO: make kdict support: del self.model_outputs[model_name, :] (and vice versa for other branch below)
                for key in self.model_outputs[model_name, :].keys():
                    logger.info(
                        f"Removing {key} because model {model_name} is incomplete."
                    )
                    del self.model_outputs[key]
        elif (
            remove_incomplete_strategy == RemoveIncompleteStrategy.DROP_INCOMPLETE_FOLDS
        ):
            for fold_id in self.incomplete_folds:
                for key in self.model_outputs[:, fold_id].keys():
                    logger.info(f"Removing {key} because fold {fold_id} is incomplete.")
                    del self.model_outputs[key]
        else:
            raise NotImplementedError(
                f"remove_incomplete_strategy={remove_incomplete_strategy} not implemented."
            )

        return self

    def copy(self) -> Self:
        """
        Returns deep copy of self.
        The underlying ModelSingleFoldPerformance objects and their attributes are copied too.
        This allows mutation of the copied object without affecting any objects nested inside the original.
        """
        return self.__class__(copy.deepcopy(self.model_outputs))

    @classmethod
    def load_from_disk(cls, output_prefix):
        """alternate constructor: reload all fit models (including partial results) from disk"""
        return cls(
            model_outputs=[
                joblib.load(metadata_fname)
                for metadata_fname in glob.glob(f"{output_prefix}*.metadata_joblib")
            ]
        )

    def summarize(
        self,
        abstain_label: str = "Unknown",
        global_evaluation_column_name: Optional[
            Union[str, sentinels.Sentinel, List[Union[str, sentinels.Sentinel]]]
        ] = None,
        remove_incomplete_strategy: RemoveIncompleteStrategy = RemoveIncompleteStrategy.DROP_INCOMPLETE_MODELS,
    ) -> ExperimentSetGlobalPerformance:
        """
        Summarize classification performance with all models across all folds (ignoring any incomplete models trained on some but not all folds).
        To override default confusion matrix ground truth values, pass global_evaluation_column_name to evaluate on a specific metadata column or combination of columns.
        (You can incorporate the default ground truth values in combination with metadata columns by using the special value `crosseval.Y_TRUE_VALUES`)

        The remove_incomplete_strategy parameters controls whether to ignore incomplete models (default) or incomplete folds (False). See RemoveIncompleteStrategy for more details.
        """
        # don't summarize incomplete models, because indices will be distorted by missing fold(s)
        # so clone self and remove incomplete
        self_without_incomplete_models = self.copy().remove_incomplete(
            remove_incomplete_strategy=remove_incomplete_strategy
        )

        return ExperimentSetGlobalPerformance(
            model_global_performances={
                model_name: self_without_incomplete_models._summarize_single_model_across_folds(
                    model_name=model_name,
                    abstain_label=abstain_label,
                    global_evaluation_column_name=global_evaluation_column_name,
                )
                for model_name in self_without_incomplete_models.model_outputs.keys(
                    dimensions=0
                )
            }
        )

    def _summarize_single_model_across_folds(
        self,
        model_name: str,
        abstain_label: str,
        global_evaluation_column_name: Optional[
            Union[str, sentinels.Sentinel, List[Union[str, sentinels.Sentinel]]]
        ] = None,
    ) -> ModelGlobalPerformance:
        return ModelGlobalPerformance(
            model_name=model_name,
            per_fold_outputs={
                fold_id: val
                for (model_name, fold_id), val in self.model_outputs[
                    model_name, :
                ].items()
            },
            abstain_label=abstain_label,
            global_evaluation_column_name=global_evaluation_column_name,
        )
