import crosseval


import copy
import pickle


def test_sentinel_value():
    # we want to make sure our Y_TRUE_VALUES behaves

    assert (
        str(crosseval.Y_TRUE_VALUES)
        == repr(crosseval.Y_TRUE_VALUES)
        == "<default y_true column>"
    )
    assert crosseval.Y_TRUE_VALUES is crosseval.Y_TRUE_VALUES
    assert crosseval.Y_TRUE_VALUES is not object()
    assert crosseval.Y_TRUE_VALUES is pickle.loads(
        pickle.dumps(crosseval.Y_TRUE_VALUES)
    )
    assert copy.deepcopy(crosseval.Y_TRUE_VALUES) is crosseval.Y_TRUE_VALUES
