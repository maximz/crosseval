import logging
import dataclasses
from dataclasses import dataclass, fields, field
from typing import Any, Dict, List, Type, Union, Optional
from typing_extensions import Self

import numpy as np
import pandas as pd

from crosseval.utils import index_rows_by_mask, validate_boolean_mask


logger = logging.getLogger(__name__)


def _row_count(value: Any) -> int:
    return value.shape[0] if hasattr(value, "shape") else len(value)


@dataclass(eq=False)
class FeaturizedData:
    """Container for featurized data for models."""

    X: Union[np.ndarray, pd.DataFrame]
    y: Union[np.ndarray, pd.Series]  # Ground truth

    # Optional if this information is unavailable:
    # TODO: should these also have default_factories, at least for metadata?
    sample_names: Optional[Union[np.ndarray, pd.Series, pd.Index, List[str]]]
    metadata: Optional[pd.DataFrame]

    sample_weights: Optional[Union[np.ndarray, pd.Series]] = None

    # Optional fields (expressed this way to avoid mutable default value):
    abstained_sample_names: Union[np.ndarray, pd.Series, pd.Index, List[str]] = field(
        default_factory=lambda: np.empty(0, dtype="object")
    )
    abstained_sample_y: Union[np.ndarray, pd.Series] = field(
        default_factory=lambda: np.empty(0)
    )
    abstained_sample_metadata: pd.DataFrame = field(default_factory=pd.DataFrame)
    # TODO(Python 3.10): Mark extras as kw_only field to allow subclassing FeaturizedData and declaring extras to be a specific dataclass type custom to a particular model. https://stackoverflow.com/a/69822584/130164
    extras: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Convert explicitly provided None values to default_factory generated values, for consistency.
        # For example, if constructor given test_metadata=None, replace with empty DataFrame as we would if test_metadata kwarg was missing.
        # https://stackoverflow.com/a/55839223/130164
        for f in fields(self):
            if (
                getattr(self, f.name) is None
            ) and f.default_factory is not dataclasses.MISSING:
                setattr(self, f.name, f.default_factory())

        if not isinstance(self.X, pd.DataFrame):
            self.X = np.asarray(self.X)
        n_samples = _row_count(self.X)

        if not isinstance(self.y, pd.Series):
            self.y = np.asarray(self.y)
        if _row_count(self.y) != n_samples:
            raise ValueError("X and y must have the same number of rows.")

        if self.sample_names is None:
            self.sample_names = np.arange(n_samples)
        elif not isinstance(self.sample_names, (pd.Series, pd.Index)):
            self.sample_names = np.asarray(self.sample_names)
        if _row_count(self.sample_names) != n_samples:
            raise ValueError("sample_names must match X and y row count.")

        if self.metadata is None:
            self.metadata = pd.DataFrame(index=np.arange(n_samples))
        elif not isinstance(self.metadata, pd.DataFrame):
            raise TypeError("metadata must be a pandas DataFrame or None.")
        if self.metadata.shape[0] != n_samples:
            raise ValueError("metadata must match X and y row count.")

        if self.sample_weights is not None:
            if not isinstance(self.sample_weights, pd.Series):
                self.sample_weights = np.asarray(self.sample_weights)
            if _row_count(self.sample_weights) != n_samples:
                raise ValueError("sample_weights must match X and y row count.")

        if not isinstance(self.abstained_sample_names, (pd.Series, pd.Index)):
            self.abstained_sample_names = np.asarray(self.abstained_sample_names)
        if not isinstance(self.abstained_sample_y, pd.Series):
            self.abstained_sample_y = np.asarray(self.abstained_sample_y)
        if not isinstance(self.abstained_sample_metadata, pd.DataFrame):
            raise TypeError("abstained_sample_metadata must be a pandas DataFrame.")

        n_abstained = _row_count(self.abstained_sample_y)
        if _row_count(self.abstained_sample_names) != n_abstained:
            raise ValueError(
                "abstained_sample_names and abstained_sample_y must have the same length."
            )
        if self.abstained_sample_metadata.shape[0] != n_abstained:
            raise ValueError(
                "abstained_sample_metadata must match abstained_sample_y row count."
            )

    # TODO: add abstained_sample_weights?

    def copy(self) -> Self:
        """Clone the object. This is NOT a deep copy. The fields are not replaced."""
        return dataclasses.replace(self)

    def apply_abstention_mask(self, mask: np.ndarray) -> Self:
        """Pass a boolean mask. Returns a copy of self with the mask samples turned into abstentions."""
        mask = validate_boolean_mask(
            mask, expected_length=_row_count(self.X), value_name="mask"
        )
        return dataclasses.replace(
            self,
            X=index_rows_by_mask(self.X, ~mask),
            y=index_rows_by_mask(self.y, ~mask),
            sample_names=index_rows_by_mask(self.sample_names, ~mask),
            metadata=index_rows_by_mask(self.metadata, ~mask),
            sample_weights=index_rows_by_mask(self.sample_weights, ~mask),
            abstained_sample_names=np.hstack(
                [
                    self.abstained_sample_names,
                    index_rows_by_mask(self.sample_names, mask),
                ]
            ),
            abstained_sample_y=np.hstack(
                [self.abstained_sample_y, index_rows_by_mask(self.y, mask)]
            ),
            abstained_sample_metadata=pd.concat(
                [
                    self.abstained_sample_metadata,
                    index_rows_by_mask(self.metadata, mask),
                ],
                axis=0,
            ),
            # All other fields (i.e. "extras") stay the same
        )

    @classmethod
    def concat(cls: Type[Self], lst: List[Self]) -> Self:
        """Concatenate multiple FeaturizedData objects into one. Extras are dropped."""
        # sample_weights must be None or available for all
        if len(lst) == 0:
            raise ValueError("Cannot concatenate empty list of FeaturizedData objects")
        is_sample_weights_available = [
            featurized_data.sample_weights is not None for featurized_data in lst
        ]
        if any(is_sample_weights_available) and not all(is_sample_weights_available):
            raise ValueError(
                "sample_weights must be None or available for all FeaturizedData objects"
            )
        return cls(
            X=pd.concat(
                [pd.DataFrame(featurized_data.X) for featurized_data in lst], axis=0
            )
            if isinstance(lst[0].X, pd.DataFrame)
            else np.vstack([featurized_data.X for featurized_data in lst]),
            y=np.hstack([featurized_data.y for featurized_data in lst]),
            sample_names=np.hstack(
                [featurized_data.sample_names for featurized_data in lst]
            ),
            metadata=pd.concat(
                [featurized_data.metadata for featurized_data in lst], axis=0
            ),
            sample_weights=np.hstack(
                [featurized_data.sample_weights for featurized_data in lst]
            )
            if any(is_sample_weights_available)
            else None,
            abstained_sample_names=np.hstack(
                [featurized_data.abstained_sample_names for featurized_data in lst]
            ),
            abstained_sample_y=np.hstack(
                [featurized_data.abstained_sample_y for featurized_data in lst]
            ),
            abstained_sample_metadata=pd.concat(
                [featurized_data.abstained_sample_metadata for featurized_data in lst],
                axis=0,
            ),
        )
