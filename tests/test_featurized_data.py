import numpy as np
import pandas as pd
import pytest

import crosseval


def test_featurized_data_apply_abstention_mask():
    fd = crosseval.FeaturizedData(
        X=pd.DataFrame(np.ones((3, 5))),
        y=np.array([0, 1, 2]),
        # note the mixed types
        sample_names=np.array([1, "2", 3], dtype="object"),
        metadata=pd.DataFrame({"sample_name": [1, 2, 3]}),
        sample_weights=np.array([0.1, 0.2, 0.3]),
        extras={"key": "value"},
    )
    fd.X.iloc[2, :] = 0
    fd_new = fd.apply_abstention_mask(mask=(np.array(fd.X) == 0).all(axis=1))
    assert fd_new.X.shape == (2, 5)
    assert (np.array(fd_new.X) == 1).all()
    assert np.array_equal(fd_new.y, [0, 1])
    assert np.array_equal(fd_new.sample_names, np.array([1, "2"], dtype="object"))
    assert np.array_equal(fd_new.metadata["sample_name"].values, [1, 2])
    assert np.array_equal(fd_new.abstained_sample_names, [3])
    assert np.array_equal(fd_new.abstained_sample_y, [2])
    assert np.array_equal(fd_new.abstained_sample_metadata["sample_name"].values, [3])
    assert np.array_equal(fd_new.sample_weights, [0.1, 0.2])
    assert fd_new.extras == {"key": "value"}


def test_FeaturizedData_post_init_replace_explicit_None_values_with_default_factory():
    fd = crosseval.FeaturizedData(
        X=pd.DataFrame(np.ones((3, 5))),
        y=np.array([0, 1, 2]),
        sample_names=None,
        metadata=None,
        abstained_sample_names=None,
    )
    assert fd.abstained_sample_names is not None
    assert np.array_equal(fd.sample_names, np.arange(3))
    assert fd.metadata.shape == (3, 0)


def test_featurized_data_validates_row_counts():
    with pytest.raises(ValueError, match="X and y"):
        crosseval.FeaturizedData(
            X=np.ones((3, 2)),
            y=np.array([0, 1]),
            sample_names=np.array(["a", "b", "c"]),
            metadata=pd.DataFrame(index=np.arange(3)),
        )

    with pytest.raises(ValueError, match="sample_names"):
        crosseval.FeaturizedData(
            X=np.ones((3, 2)),
            y=np.array([0, 1, 2]),
            sample_names=np.array(["a", "b"]),
            metadata=pd.DataFrame(index=np.arange(3)),
        )

    with pytest.raises(ValueError, match="metadata"):
        crosseval.FeaturizedData(
            X=np.ones((3, 2)),
            y=np.array([0, 1, 2]),
            sample_names=np.array(["a", "b", "c"]),
            metadata=pd.DataFrame(index=np.arange(2)),
        )

    with pytest.raises(ValueError, match="sample_weights"):
        crosseval.FeaturizedData(
            X=np.ones((3, 2)),
            y=np.array([0, 1, 2]),
            sample_names=np.array(["a", "b", "c"]),
            metadata=pd.DataFrame(index=np.arange(3)),
            sample_weights=np.array([1, 1]),
        )


def test_featurized_data_apply_abstention_mask_with_default_optional_fields():
    fd = crosseval.FeaturizedData(
        X=np.ones((3, 2)),
        y=[0, 1, 2],
        sample_names=None,
        metadata=None,
    )

    fd_new = fd.apply_abstention_mask(np.array([False, True, False]))

    assert fd_new.X.shape == (2, 2)
    assert np.array_equal(fd_new.y, [0, 2])
    assert np.array_equal(fd_new.sample_names, [0, 2])
    assert fd_new.metadata.shape == (2, 0)
    assert np.array_equal(fd_new.abstained_sample_names, [1])
    assert np.array_equal(fd_new.abstained_sample_y, [1])
    assert fd_new.abstained_sample_metadata.shape == (1, 0)


def test_featurized_data_apply_abstention_mask_rejects_non_boolean_mask():
    fd = crosseval.FeaturizedData(
        X=np.ones((3, 2)),
        y=np.array([0, 1, 2]),
        sample_names=np.array(["a", "b", "c"]),
        metadata=pd.DataFrame(index=np.arange(3)),
    )

    with pytest.raises(TypeError, match="boolean mask"):
        fd.apply_abstention_mask(np.array([0, 1, 0]))
