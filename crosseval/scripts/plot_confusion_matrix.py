"""plotconfusion CLI command."""

import pandas as pd
import genetools
from genetools.plots import plot_confusion_matrix
from typing import Optional
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import typer

app = typer.Typer()


@app.command()
def plot(
    input: str,
    output: Optional[str] = None,
    separator="\t",
    pred_label: str = "Predicted label",
    # true_label defaults to "True label" if not provided in the input csv/tsv file
    true_label: Optional[str] = None,
    width: float = 4,
    height: float = 4,
    wrapx: int = 15,
    wrapy: int = 15,
    dpi: int = 300,
    # datascale controls the size of the font for the data in the cells
    datascale: float = 1.0,
    # labelscale controls the size of the font for the labels
    labelscale: float = 1.0,
):
    """Render a .confusion_matrix.tsv file as a confusion matrix image."""
    input_fname: Path = Path(input)
    df = pd.read_csv(
        input_fname,
        sep=separator,
        index_col=0,
    )
    df.columns.name = pred_label
    if true_label is not None:
        df.index.name = true_label
    elif df.index.name is None or df.index.name == "":
        # Set default only if not already set (may have been provided in the input file)
        df.index.name = "True label"
    sns.set_context(
        "paper",
        rc={
            "axes.labelsize": 15 * labelscale,
            "ytick.labelsize": 10 * labelscale,
            "xtick.labelsize": 10 * labelscale,
            "font.size": 10 * datascale,
        },
    )
    fig, ax = plot_confusion_matrix(
        df,
        figsize=(width, height),
        wrap_labels_amount=None,  # disable
    )
    # Wrap labels if needed
    if wrapx > 0:
        genetools.plots.wrap_tick_labels(
            ax,
            wrap_amount=wrapx,
            wrap_x_axis=True,
            wrap_y_axis=False,
        )
    if wrapy > 0:
        genetools.plots.wrap_tick_labels(
            ax,
            wrap_amount=wrapy,
            wrap_x_axis=False,
            wrap_y_axis=True,
        )
    if output is None:
        # Make output fname if not specified
        if str(input_fname).lower().endswith(".confusion_matrix_data.tsv"):
            # Special case for crosseval's ".confusion_matrix_data.tsv" files:
            # Remove the ".confusion_matrix_data.tsv" suffix and replace with ".png"
            output: Path = input_fname.with_suffix("").with_suffix(".png")
        else:
            # Replace suffix with ".png"
            output: Path = input_fname.with_suffix(".png")
    genetools.plots.savefig(
        fig,
        output,
        dpi=dpi,
    )
    plt.close(fig)


if __name__ == "__main__":
    app()
