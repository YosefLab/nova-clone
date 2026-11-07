"""
This module contains QC functionality for the output of the clone calling algorithm.
"""

from collections import defaultdict
from typing import Dict, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from nova_clone.algorithm._types import CellBCType, CloneType


def qc_multiplet_rate(
    cell_to_clones_mapping: Dict[CellBCType, Set[CloneType]],
    verbose: bool = True,
) -> Dict[int, float]:
    """
    Get the multiplet rate of the clone call.

    res[0] contains the rate of unmapped cellbcs (which is between 0 and 1).
    res[1] contains the rate of cellbcs mapped uniquely to a clone (which
    is what we want to maximize).
    res[2] contains the rate of doublets.
    res[3] contains the rate of triplets.
    etc.

    Args:
        cell_to_clones_mapping: The cellbc to clone mapping.
        verbose: If True, the multiplet rates will be printed.
    """
    multiplets = defaultdict(int)
    multiplets[2] = 0  # To make sure this appears.
    num_cellbc = len(cell_to_clones_mapping)
    for cellbc, clones in cell_to_clones_mapping.items():
        multiplets[len(clones)] += 1
    if verbose:
        print(f"******************************")
        print(f"****** Doublet Analysis ******")
        print(f"******************************")
        for k, v in sorted(multiplets.items()):
            if k == 0:
                print(
                    f"Unmapped cellBCs: {100.0 * v / num_cellbc:.2f}% = {v} / {num_cellbc} [the lower the better]"
                )
            elif k == 1:
                print(
                    f"Uniquely (i.e. successfully) mapped cellBCs: {100.0 * v / num_cellbc:.2f}% = {v} / {num_cellbc} [the higher the better]"
                )
            elif k == 2:
                print(
                    f"Doublet cellBCs: {100.0 * v / num_cellbc:.2f}% = {v} / {num_cellbc} [should match expectations]"
                )
            else:
                assert k >= 3
                print(
                    f"Multiplet cellBCs (degree {k}): {100.0 * v / num_cellbc:.2f}% = {v} / {num_cellbc} [should match expectations]"
                )
    return dict(multiplets)


def qc_multiplet_rate_per_clone(
    cell_to_clones_mapping: Dict[CellBCType, Set[CloneType]],
    show_plot: bool = True,
    plot_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Get the multiplet rate of each clone, which is a number between 0 and 1.

    Args:
        cell_to_clones_mapping: The cellbc to clone mapping.
        show_plot: If True, the plot will be shown.
        plot_path: If not None, the plot will additionally be saved to this
            file.
    """
    numerator = defaultdict(int)
    denominator = defaultdict(int)
    all_clones = set([clone for clone_set in cell_to_clones_mapping.values() for clone in clone_set])
    multiplets_per_clone = {clone: defaultdict(int) for clone in all_clones}
    for cellbc, clones in cell_to_clones_mapping.items():
        clones = list(clones)
        if len(clones) == 1:
            clone = clones[0]
            denominator[clone] += 1
            numerator[clone] += 0
        else:
            for clone in clones:
                denominator[clone] += 1
                numerator[clone] += 1
                for other_clone in clones:
                    if clone < other_clone:
                        multiplets_per_clone[clone][other_clone] += 1
                        multiplets_per_clone[other_clone][clone] += 1
    multiplet_rate_per_clone = {
        clone: numerator[clone] / denominator[clone] for clone in denominator
    }

    # Make plot
    sns.histplot(list(multiplet_rate_per_clone.values()))
    plt.xlim(0, 1)
    num_clones = len(multiplet_rate_per_clone)
    max_multiplet_rate = max(list(multiplet_rate_per_clone.values()))
    mean_multiplet_rate = np.mean(list(multiplet_rate_per_clone.values()))
    median_multiplet_rate = np.median(list(multiplet_rate_per_clone.values()))
    plt.xlabel("Multiplet rate")
    plt.ylabel("Number of clones")
    plt.title(
        f"Multiplet rate per clone ({num_clones} clones)\nmax multiplet rate = {100*max_multiplet_rate:.2f}% [Lower is better]\nmean multiplet rate = {100*mean_multiplet_rate:.2f}% [Lower is better]\nmedian multiplet rate = {100*median_multiplet_rate:.2f}% [Lower is better]"
    )
    plt.tight_layout()
    if plot_path is not None:
        plt.savefig(plot_path, dpi=300)
    if show_plot:
        print(f"******************************")
        print(f"* Per-Clone Doublet Analysis *")
        print(f"******************************")
        # Show the worse offender
        worst_clone = max([(multiplet_rate_per_clone[clone], clone) for clone in multiplet_rate_per_clone])[1]
        stats_to_show = "{"
        for other_clone, n_multiplets in multiplets_per_clone[worst_clone].items():
            stats_to_show += f"{other_clone} (size {denominator[other_clone]}): {n_multiplets}, "
        stats_to_show += "}"
        print(
            f"Clone with the highest multiplet rate: {worst_clone} . Clone size: {denominator[worst_clone]}. "
            f"Number of multiplets shared with other clones:\n{stats_to_show}. "
            "You may want to discard this clone by using nova_clone.algorithm.drop_clones ."
        )
        plt.show()
    plt.close()

    return multiplet_rate_per_clone


def test_qc_multiplet_rate():
    cell_to_clones_mapping = {
        "unmapped": set(),
        "unique": {"clone_0"},
        "doublet": {"clone_0", "clone_1"},
    }
    res = qc_multiplet_rate(cell_to_clones_mapping=cell_to_clones_mapping, verbose=False)
    assert res[0] == 1 and res[1] == 1 and res[2] == 1


def test_qc_multiplet_rate_per_clone():
    cell_to_clones_mapping = {
        "a": {"clone_0"},
        "b": {"clone_0"},
        "c": {"clone_0", "clone_1"},
    }
    rates = qc_multiplet_rate_per_clone(
        cell_to_clones_mapping=cell_to_clones_mapping,
        show_plot=False,
        plot_path=None,
    )
    assert set(rates.keys()) == {"clone_0", "clone_1"}
    assert rates["clone_0"] == 1 / 3
    assert rates["clone_1"] == 1.0
