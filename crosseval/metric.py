import logging
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass(frozen=True, order=True)
class Metric:
    """Store metric value along with friendly name. Comparison operators of Metric objects only compare the metric values, not the names."""

    value: float
    # don't use friendly_name in equality and inequality (e.g. greater than) comparisons
    friendly_name: str = field(compare=False)
