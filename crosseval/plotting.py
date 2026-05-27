import os
import textwrap
from typing import List, Optional, Tuple, Union

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def savefig(fig: matplotlib.figure.Figure, *args, **kwargs):
    """Save a Matplotlib figure with deterministic vector-output defaults."""
    kwargs = {"bbox_inches": "tight", **kwargs}

    original_source_date_epoch = os.environ.pop("SOURCE_DATE_EPOCH", None)
    os.environ["SOURCE_DATE_EPOCH"] = "946684800"
    try:
        with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
            fig.savefig(*args, **kwargs)
    finally:
        if original_source_date_epoch is None:
            os.environ.pop("SOURCE_DATE_EPOCH", None)
        else:
            os.environ["SOURCE_DATE_EPOCH"] = original_source_date_epoch


def make_confusion_matrix(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    true_label: str,
    pred_label: str,
    label_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Make a confusion matrix with rows as truth labels and columns as predictions."""
    cm = pd.crosstab(
        np.array(y_true),
        np.array(y_pred),
        rownames=[true_label],
        colnames=[pred_label],
    )

    if label_order is None:
        label_order = cm.index.union(cm.columns)

    resulting_row_order, resulting_col_order = [
        pd.Index(label_order).intersection(source_list).tolist()
        for source_list in [cm.index, cm.columns]
    ]

    return cm.loc[resulting_row_order][resulting_col_order]


def plot_confusion_matrix(
    df: pd.DataFrame,
    figsize: Optional[Tuple[float, float]] = None,
    wrap_labels_amount: Optional[int] = 15,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Render a confusion-matrix DataFrame as a heatmap."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df, annot=True, fmt="g", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel(df.columns.name or "")
    ax.set_ylabel(df.index.name or "")

    if wrap_labels_amount is not None and wrap_labels_amount > 0:
        wrap_tick_labels(ax, wrap_amount=wrap_labels_amount)

    fig.tight_layout()
    return fig, ax


def wrap_tick_labels(
    ax: matplotlib.axes.Axes,
    wrap_amount: int,
    wrap_x_axis: bool = True,
    wrap_y_axis: bool = True,
):
    """Wrap tick label text on either axis."""
    if wrap_x_axis:
        ax.set_xticklabels(
            [
                textwrap.fill(label.get_text(), wrap_amount)
                for label in ax.get_xticklabels()
            ],
            rotation=0,
        )
    if wrap_y_axis:
        ax.set_yticklabels(
            [
                textwrap.fill(label.get_text(), wrap_amount)
                for label in ax.get_yticklabels()
            ],
            rotation=0,
        )
