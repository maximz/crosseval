import logging
import functools
from typing import Any, Union

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


logger = logging.getLogger(__name__)

####

# TODO: is there a type hint for all sklearn models? BaseEstimator is odd to put here: https://stackoverflow.com/questions/54868698/what-type-is-a-sklearn-model
Classifier = Union[Pipeline, BaseEstimator]


def is_clf_a_sklearn_pipeline(clf: Classifier) -> bool:
    # clf may be an individual estimator, or it may be a pipeline, in which case the estimator is the final pipeline step
    return type(clf) == Pipeline


def _get_final_estimator_if_pipeline(clf: Classifier) -> BaseEstimator:
    """If this is a pipeline, return final step (after any transformations). Otherwise pass through."""
    if is_clf_a_sklearn_pipeline(clf):
        return clf.steps[-1][1]
    else:
        return clf


####


def support_list_and_dict_arguments_in_cache_decorator(func):
    """Cache decorator normally fails with list or dict arguments, throwing e.g. "TypeError: unhashable type: 'dict'"
    This additional decorator makes dict/list arguments immutable, recursing into making their values immutable too.
    From https://stackoverflow.com/a/53394430/130164 and https://stackoverflow.com/a/66729248/130164
    Alternative: https://stackoverflow.com/a/44776960/130164
    """

    def _make_immutable(obj: Any):
        from frozendict import frozendict
        from collections.abc import Collection, Mapping, Hashable

        if isinstance(obj, str):
            # short circuit for strings, which are iterable but we don't want to recurse into
            return obj
        if isinstance(obj, Mapping):
            # dict -> frozendict, recursing inwards on values
            return frozendict({k: _make_immutable(v) for k, v in obj.items()})
        elif isinstance(obj, Collection):
            # list -> tuple, recursing inwards on values
            return tuple(_make_immutable(i) for i in obj)
        elif not isinstance(obj, Hashable):
            # other unhashable type - we don't know what to do
            raise TypeError(f"Unhashable type: {type(obj)}")
        # already hashable
        return obj

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        # make the args (tuple) and kwargs (dict) have immutable contents
        return func(*_make_immutable(args), **_make_immutable(kwargs))

    # Preserve lru_cache functionality (https://stackoverflow.com/questions/6358481/using-functools-lru-cache-with-dictionary-arguments#comment88158142_44776960)
    wrapped.cache_info = func.cache_info
    wrapped.cache_clear = func.cache_clear

    return wrapped
