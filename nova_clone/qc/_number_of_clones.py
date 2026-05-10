"""
This module contains QC functionality for the output of the clone calling algorithm.
"""

from collections import defaultdict
from typing import Dict, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from nova_clone.algorithm._types import CellBCType, CloneCallType, CloneType

PLT_SAVEFIG_KWARGS = {
    "bbox_inches": "tight",
    "dpi": 300,
}


def qc_number_of_clones(
    clone_call: CloneCallType,
    show_plot: bool = True,
    fig_path: Optional[str] = None,
) -> Tuple[Dict[CloneType, int], Dict[CloneType, float]]:
    """
    Basic QC on the number of clones recovered.

    We plot one circle for each clone. The attributes are:
    - x-axis: Clone size.
    - y-axis: Average number of intBCs per cell in that clone.
    """
    print(f"************************************************")
    print(f"****** Clone sizes and tracing capacities ******")
    print(f"************************************************")

    clone_to_size = {
        clone: len(cellbcs)
        for (clone, cellbcs) in clone_call.clone_to_cells_mapping.items()
    }
    clones = list(clone_call.clone_to_cells_mapping.keys())
    clone_to_average_intbc_per_cellbc = {
        clone: np.mean(
            [
                len(clone_call.cell_intbc_sets[cellbc])
                for cellbc in clone_call.clone_to_cells_mapping[clone]
            ]
        )
        for clone in clones
    }
    clone_sizes = [clone_to_size[clone] for clone in clones]
    clone_average_cell_intbcs = [
        clone_to_average_intbc_per_cellbc[clone] for clone in clones
    ]

    # Create figure with gridspec to accommodate the histograms on the margins
    fig = plt.figure(figsize=(10, 8))  # Increase figure size further for more space
    gs = fig.add_gridspec(
        5, 5, height_ratios=[1, 6, 1, 1, 0.5], width_ratios=[1, 6, 1, 0.5, 1]
    )

    # Main scatter plot
    ax_main = fig.add_subplot(gs[1:3, 1:3])
    sc = ax_main.scatter(
        clone_sizes,
        clone_average_cell_intbcs,
        s=clone_sizes,
        c=clone_average_cell_intbcs,
        cmap="gray_r",
        edgecolors="k",
        linewidth=0.5,
        alpha=0.8,
    )
    ax_main.set_xscale("log")
    ax_main.set_title(
        f"{len(clones)} clones recovered\nClone size (median {np.median(clone_sizes):.0f}) and lineage tracing capacity (median {np.median(clone_average_cell_intbcs):.1f})"
    )
    ax_main.grid(True)

    # Place xlabel and ylabel directly next to the scatter plot
    ax_main.set_xlabel("Clone size (circle size)")
    ax_main.set_ylabel("Average intBCs per cell (circle color)")

    # Color bar for the scatter plot
    # cbar = plt.colorbar(sc, ax=ax_main, label="Average intBCs per cell")

    # Histograms on the x and y margins
    ax_histx = fig.add_subplot(gs[3, 1:3], sharex=ax_main)  # Histogram under the plot
    ax_histy = fig.add_subplot(
        gs[1:3, 0], sharey=ax_main
    )  # Histogram to the left of the plot

    # Create log-spaced bins for the x histogram
    log_bins = np.logspace(
        np.log10(min(clone_sizes)), np.log10(max(clone_sizes)), num=30
    )
    ax_histx.hist(clone_sizes, bins=log_bins, color="gray", alpha=0.6)
    ax_histx.set_xscale("log")
    ax_histx.set_xlabel("Clone size (circle size)")

    # Plot histogram for the y-axis (colors)
    ax_histy.hist(
        clone_average_cell_intbcs,
        bins=30,
        color="gray",
        alpha=0.6,
        orientation="horizontal",
    )
    ax_histy.set_ylabel("Average intBCs per cell (circle color)")

    # Copy x and y ticks from the scatter plot (ax_main) to histograms
    ax_histx.tick_params(
        axis="x", labelbottom=True
    )  # Ensure x ticks are on the bottom histogram
    ax_histy.tick_params(
        axis="y", labelleft=True
    )  # Ensure y ticks are on the left histogram

    # Remove ticks on the histograms
    ax_histx.tick_params(
        axis="x", labelbottom=True
    )  # Keep x ticks for the bottom histogram
    ax_histy.tick_params(
        axis="y", labelleft=True
    )  # Keep y ticks for the left histogram

    # Automatically adjust the layout to avoid clipping
    plt.tight_layout(pad=2.0)  # Increase padding between subplots

    if fig_path is not None:
        plt.savefig(
            fig_path,
            **PLT_SAVEFIG_KWARGS,
        )
    if show_plot:
        plt.show()
    plt.close()
    return clone_to_size, clone_to_average_intbc_per_cellbc


def test_qc_number_of_clones():
    clone_call = CloneCallType(
        clone_to_cells_mapping={"clone_0": {"c1", "c2"}},
        cell_to_clones_mapping={"c1": {"clone_0"}, "c2": {"clone_0"}},
        clone_intbc_sets={"clone_0": {"A", "B"}},
        cell_intbc_sets={"c1": {"A"}, "c2": {"A", "B"}},
    )
    clone_to_size, clone_to_avg = qc_number_of_clones(
        clone_call=clone_call,
        show_plot=False,
        fig_path=None,
    )
    assert clone_to_size == {"clone_0": 2}
    assert clone_to_avg == {"clone_0": 1.5}
