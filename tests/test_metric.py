import crosseval


def test_metric_comparison():
    # comparison by value, not by name
    assert crosseval.Metric(value=1.0, friendly_name="metric1") > crosseval.Metric(
        value=0.5, friendly_name="metric1"
    )
    assert crosseval.Metric(value=1.0, friendly_name="metric1") > crosseval.Metric(
        value=0.5, friendly_name="metric2"
    )
    assert crosseval.Metric(value=1.0, friendly_name="metric1") == crosseval.Metric(
        value=1.0, friendly_name="metric2"
    )
