import numpy as np
import pandas as pd

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
    fd.X.values[2, :] = 0
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
