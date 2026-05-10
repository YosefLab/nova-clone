"""
Module for performing QC on the raw data that is given to the clone calling
algorithm.
"""

from collections import defaultdict
from typing import Dict, Optional

import editdistance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

from nova_clone.algorithm._types import (
    IntBCTableType,
    IntBCType,
    SampleNameType,
    validate_intbc_table_per_sample,
)


def pre_qc_intbcs_per_cellbc(
    intbc_table_per_sample: Dict[SampleNameType, IntBCTableType],
    min_intbc_umi_count: int = 10,
    show_plot: bool = True,
    fig_path: Optional[str] = None,
) -> Dict[int, int]:
    """
    Plot the number of intBCs per cellBC.

    `res[i]` contains the number of cellBC with `i` distinct intBCs.

    intBCs with less than the given `min_intbc_umi_count` are filtered out.
    """
    validate_intbc_table_per_sample(intbc_table_per_sample)
    res = defaultdict(int)
    for intbc_table in intbc_table_per_sample.values():
        for cellbc, sub_df in intbc_table.groupby("cellBC"):
            res[len(set(sub_df[sub_df.UMI_count >= min_intbc_umi_count].intBC))] += 1

    # Extract keys and values
    x_values = list(res.keys())
    heights = list(res.values())

    # Plot histogram
    plt.bar(
        x_values, heights, width=0.4, align="center", color="blue", edgecolor="blue"
    )

    plt.title(
        f"Number of distinct intBCs per cellBC\nUMI threshold: {min_intbc_umi_count}\n[The larger the number of intBCs, the better]\n[A left peak is typical (and expected) from e.g. empty droplets]"
    )
    plt.xlabel("Number of intBCs")
    plt.ylabel("Number of cellBCs")
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    if show_plot:
        plt.show()
    plt.close()


def test_pre_qc_intbcs_per_cellbc():
    intbc_table = pd.DataFrame(
        [
            ["cell_a", "I1", 10],
            ["cell_a", "I2", 10],
            ["cell_b", "I1", 10],
        ],
        columns=["cellBC", "intBC", "UMI_count"],
    )
    pre_qc_intbcs_per_cellbc(
        intbc_table_per_sample={"s0": intbc_table},
        min_intbc_umi_count=5,
        show_plot=False,
        fig_path=None,
    )


def pre_qc_intbc_frequencies(
    intbc_table_per_sample: Dict[SampleNameType, IntBCTableType],
    min_intbc_umi_count: int = 10,
    num_intbcs_to_plot: int = 10,
    show_plot: bool = True,
    fig_path: Optional[str] = None,
) -> Dict[IntBCType, int]:
    """
    Plot the frequency of each intBC.

    `res[x]` contains the fraction of cellBC containing the intBC `x`.

    intBCs with less than the given `min_intbc_umi_count` are filtered out.

    Only the top `num_intbcs_to_plot` most frequent intBCs will be plotted.
    """
    validate_intbc_table_per_sample(intbc_table_per_sample)
    res = defaultdict(int)
    cellbcs = set()
    for intbc_table in intbc_table_per_sample.values():
        intbc_table = intbc_table.copy()
        intbc_table = intbc_table[
            ["cellBC", "intBC", "UMI_count"]
        ]  # make sure order is correct as we will convert to numpy
        intbc_table_np = intbc_table.to_numpy()
        for i in range(intbc_table_np.shape[0]):
            cellbc = intbc_table_np[i, 0]
            intbc = intbc_table_np[i, 1]
            umi_count = intbc_table_np[i, 2]
            cellbcs.add(cellbc)
            if umi_count >= min_intbc_umi_count:
                res[intbc] += 1
    num_cellbcs = len(cellbcs)
    for intbc in res:
        res[intbc] = res[intbc] / num_cellbcs

    # Sort dictionary by values in descending order and take top `num_intbcs_to_plot`
    top_10 = dict(
        sorted(res.items(), key=lambda item: item[1], reverse=True)[:num_intbcs_to_plot]
    )

    # Extract keys and values
    x_values = list(top_10.keys())
    heights = list(top_10.values())

    # Plot bar chart
    plt.bar(x_values, heights, color="blue", edgecolor="black")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    # Labels and title
    plt.xlabel("IntBC")
    plt.ylabel("Fraction of cellBCs containing this intBC")
    plt.title(
        f"Top {num_intbcs_to_plot} intBCs by Frequency\nUMI threshold: {min_intbc_umi_count}"
    )
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    if show_plot:
        plt.show()
    plt.close()

    return res


def test_pre_qc_intbc_frequencies():
    intbc_table = pd.DataFrame(
        [
            ["cell_a", "I1", 10],
            ["cell_a", "I2", 10],
            ["cell_b", "I1", 10],
        ],
        columns=["cellBC", "intBC", "UMI_count"],
    )
    res = pre_qc_intbc_frequencies(
        intbc_table_per_sample={"s0": intbc_table},
        min_intbc_umi_count=5,
        num_intbcs_to_plot=2,
        show_plot=False,
        fig_path=None,
    )
    assert "I1" in res and "I2" in res
    assert res["I1"] == 2 / 2
    assert res["I2"] == 1 / 2


def pre_qc_max_umi_count_per_cellbc_quantiles(
    intbc_table_per_sample: Dict[SampleNameType, IntBCTableType],
    show_plot: bool = True,
    fig_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Plot the max UMI count for each cellBC.

    E.g. if a cellbc has intBCs {"AA": 10, "BB": 20}, the max UMI count is 20.

    We want this distribution to be shifted to the right as much as possible,
    as it means that a lot of cellBCs show strong expression of at least one
    intBC. Importantly, any cellBCs below level `x` will lose all their data
    when the `min_intbc_umi_count` is set above `x`.

    Returns:
        A DataFrame with the quantiles of the max UMI count for each cellBC.
    """
    res = []
    for intbc_table in intbc_table_per_sample.values():
        res += list(
            intbc_table.groupby("cellBC").apply(
                lambda sub_df: float(sub_df.UMI_count.max())
            )
        )

    quantiles = np.quantile(res, np.arange(0.05, 1.0, 0.05))
    quantiles_df = pd.DataFrame(
        {"Quantile": np.arange(0.05, 1.0, 0.05), "Max UMI_count": quantiles}
    )
    quantiles_df = quantiles_df.set_index("Quantile")
    res = quantiles_df.T
    res.columns = [f"{int(q * 100)}%" for q in res.columns]
    display(res)
    if fig_path is not None:
        fig, ax = plt.subplots(figsize=(len(res.columns) * 1.2, len(res) * 0.5))
        ax.axis('off')  # Turn off axis

        # Create table and render
        table = pd.plotting.table(ax, res, loc='center', cellLoc='center')

        # Optional: Resize font
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        # Save to file
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    if show_plot:
        plt.show()
    plt.close()
    return res


def test_pre_qc_max_umi_count_per_cellbc_quantiles():
    intbc_table = pd.DataFrame(
        [
            ["cell_a", "I1", 10],
            ["cell_a", "I2", 30],
            ["cell_b", "I1", 20],
        ],
        columns=["cellBC", "intBC", "UMI_count"],
    )
    out = pre_qc_max_umi_count_per_cellbc_quantiles(
        intbc_table_per_sample={"s0": intbc_table},
        show_plot=False,
        fig_path=None,
    )
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] >= 1


def pre_qc_umi_per_intbc_distribution(
    intbc_table_per_sample: Dict[SampleNameType, IntBCTableType],
    show_plot: bool = True,
    fig_path: Optional[str] = None,
):
    """
    Plot the distribution of UMI counts per intBC.

    This is useful to see how many intBCs have a high enough UMI count.
    Those with a low UMI count are likely to be e.g. sequencing errors
    and will be filtered out by the `min_intbc_umi_count` parameter.
    """
    validate_intbc_table_per_sample(intbc_table_per_sample)

    # Concatenate all intBC tables into one DataFrame
    all_intbc_data = pd.concat(intbc_table_per_sample.values(), ignore_index=True)
    umi_count_list = sorted(all_intbc_data.UMI_count.tolist())

    res = defaultdict(int)
    for umi_count in umi_count_list:
        res[umi_count] += 1
    # Extract keys and values
    x_values = list(res.keys())
    heights = list(res.values())
    # Create a boxplot of UMI_count per intBC
    plt.bar(
        x_values, heights, width=0.4, align="center", color="blue", edgecolor="blue"
    )
    plt.title(
        f"Number of UMIs per intBC\n[The larger the number of UMIs, the better]\n[A left peak is typical of sequencing errors and background RNA]"
    )
    plt.xlabel("Number of UMIs")
    plt.ylabel("Number of intBCs")

    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    if show_plot:
        plt.show()
    plt.close()

    return umi_count_list


def test_pre_qc_umi_per_intbc_distribution():
    intbc_table_0 = pd.DataFrame(
        data=[
            ["cellBC_1", "AAA", 40],
            ["cellBC_1", "AAB", 5],
            ["cellBC_1", "BBB", 41],
            ["cellBC_1", "CCC", 42],
            ["cellBC_2", "AAA", 10],
            ["cellBC_2", "BBB", 20],
            ["cellBC_2", "BBZ", 2],
            ["cellBC_2", "ZZZ", 4],
            ["cellBC_3", "DDD", 40],
            ["cellBC_3", "EEE", 40],
        ],
        columns=["cellBC", "intBC", "UMI_count"],
    )
    intbc_table_1 = pd.DataFrame(
        data=[
            ["cellBC_3", "FFF", 40],
            ["cellBC_4", "DDD", 40],
            ["cellBC_4", "EEE", 40],
            ["cellBC_4", "FFF", 40],
            ["cellBC_5", "DDD", 40],
            ["cellBC_5", "EEE", 40],
            ["cellBC_5", "FFF", 40],
            ["cellBC_5", "GGG", 40],
            ["cellBC_6", "AAA", 30],  # an unmapped cell
            ["cellBC_7", "AAA", 40],  # a doublet
            ["cellBC_7", "DDD", 40],  # a doublet
            ["cellBC_7", "DDE", 1],  # a doublet
            ["cellBC_8", "ZZZ", 33],  # noise
        ],
        columns=["cellBC", "intBC", "UMI_count"],
    )
    intbc_table_per_sample = {"sample0": intbc_table_0, "sample1": intbc_table_1}
    res = pre_qc_umi_per_intbc_distribution(intbc_table_per_sample=intbc_table_per_sample, show_plot=False, fig_path=None)
    assert(res == [1, 2, 4, 5, 10, 20, 30, 33, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 41, 42])


def pre_qc_edit_distance_distribution(
    intbc_table_per_sample: Dict[SampleNameType, IntBCTableType],
    show_plot: bool = True,
    fig_path: Optional[str] = None,
    max_pairs_per_cell: Optional[int] = None,
    min_intbc_umi_count: int = 0,
):
    """
    Plot the distribution of edit distances between intBCs in the same cell.

    This is useful to see what the edit distance of sequencing errors
    looks like. Indeed, the plot will show two modalities:
    1. A left peak, which is caused by sequencing errors.
    2. A right distribution, which is the edit distance between a
        random pair of intBC sequences.

    To speed this up, we can only sample a limited number of pairs
    of intBCs per cellBC, controlled by the `max_pairs_per_cell` parameter.
    By default, it is `None` no subsetting is performed.

    By passing in `min_intbc_umi_count`, one can see the edit distance
    distribution after filtering out intBCs with a low UMI count.
    This should significantly wipe out the left peak, as
    sequence errors are typically low in UMI count.

    Thus, this plot both confirms the effectiveness of the
    `min_intbc_umi_count` parameter at wiping out sequencing errors,
    and suggests what value of the edit distance cutoff to use to
    prune the few sequencing errors that remain after the
    `min_intbc_umi_count` filter. Generally, an edit distance cutoff
    of 2 should be good, or ceiling(intBC length / 4) more generally.
    """
    validate_intbc_table_per_sample(intbc_table_per_sample)

    # Filter intBCs with less than the given `min_intbc_umi_count`
    intbc_table_per_sample = {
        sample_name: intbc_table[intbc_table.UMI_count >= min_intbc_umi_count]
        for sample_name, intbc_table in intbc_table_per_sample.items()
    }

    # Instantiate random number generator
    rng = np.random.default_rng(seed=42)

    edit_distances = []
    for intbc_table in intbc_table_per_sample.values():
        for cellbc, sub_df in intbc_table.groupby("cellBC"):
            intbcs = sub_df.intBC.tolist()
            all_pairs = [(i, j) for i in range(len(intbcs)) for j in range(i + 1, len(intbcs))]
            if max_pairs_per_cell is not None:
                random_pairs = rng.choice(all_pairs, size=min(max_pairs_per_cell, len(all_pairs)), replace=False)
            else:
                random_pairs = all_pairs
            for i, j in random_pairs:
                intbc1 = intbcs[i]
                intbc2 = intbcs[j]
                assert(intbc1 != intbc2), "intBCs should not be the same"
                edit_distance = editdistance.eval(intbc1, intbc2)
                edit_distances.append(edit_distance)

    # Plot barplot with logarithmic y-scale.
    plt.figure(figsize=(6, 4))
    plt.title(
        "Distribution of edit distances between intBCs in the same cell\n"
        f"min_intbc_umi_count = {min_intbc_umi_count}\n"
        "[A left mode is typical of sequencing errors.]"
    )
    plt.bar(
        *np.unique(edit_distances, return_counts=True),
        color="blue",
        edgecolor="black",
    )
    plt.xlabel("Edit distance", fontsize=12)
    plt.ylabel("Number of intBC pairs", fontsize=12)
    plt.yscale("log")  # Use logarithmic scale for better visibility of the

    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    if show_plot:
        plt.show()
    plt.close()

    return sorted(edit_distances)


def test_pre_qc_edit_distance_distribution():
    intbc_table_0 = pd.DataFrame(
        data=[
            ["cellBC_1", "AAA", 40],
            ["cellBC_1", "AAB", 5],
            ["cellBC_1", "BBB", 41],
            ["cellBC_2", "AAA", 10],
            ["cellBC_2", "BBB", 20],
            ["cellBC_3", "DDD", 40],
            ["cellBC_3", "EEE", 40],
            ["cellBC_4", "DDD", 40],
            ["cellBC_4", "DDE", 40],
            ["cellBC_5", "AAA", 40],
            ["cellBC_5", "BBB", 40],
            ["cellBC_5", "CCC", 40],
            ["cellBC_5", "DDD", 40],
            ["cellBC_5", "EEE", 40],
            ["cellBC_5", "FFF", 40],
            ["cellBC_5", "GGG", 40],
        ],
        columns=["cellBC", "intBC", "UMI_count"],
    )
    intbc_table_1 = pd.DataFrame(
        data=[
            ["cellBC_1", "AAA", 40],
            ["cellBC_1", "AAB", 15],
            ["cellBC_1", "BBB", 41],
            ["cellBC_2", "AAA", 10],
            ["cellBC_2", "BBB", 20],
            ["cellBC_3", "DDD", 40],
            ["cellBC_3", "EEE", 40],
        ],
        columns=["cellBC", "intBC", "UMI_count"],
    )
    intbc_table_per_sample = {"sample0": intbc_table_0, "sample1": intbc_table_1}
    edit_distances = pre_qc_edit_distance_distribution(intbc_table_per_sample=intbc_table_per_sample, show_plot=False, fig_path=None, max_pairs_per_cell=10)
    assert(edit_distances == [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])


def test_pre_qc_edit_distance_distribution_min_umi_count():
    intbc_table_0 = pd.DataFrame(
        data=[
            ["cellBC_1", "AAA", 40],
            ["cellBC_1", "AAB", 55],
            ["cellBC_1", "ABB", 5],
            ["cellBC_1", "BBB", 41],
            ["cellBC_2", "AAA", 10],
            ["cellBC_2", "BBB", 20],
            ["cellBC_4", "DDD", 40],
            ["cellBC_4", "DDE", 40],
            ["cellBC_5", "AAA", 40],
            ["cellBC_5", "BBB", 40],
            ["cellBC_5", "CCC", 40],
            ["cellBC_5", "DDD", 40],
            ["cellBC_5", "EEE", 40],
            ["cellBC_5", "FFF", 40],
            ["cellBC_5", "GGG", 40],
        ],
        columns=["cellBC", "intBC", "UMI_count"],
    )
    intbc_table_1 = pd.DataFrame(
        data=[
            ["cellBC_1", "AAA", 40],
            ["cellBC_1", "AAB", 15],
            ["cellBC_1", "BBB", 41],
            ["cellBC_2", "AAA", 10],
            ["cellBC_2", "BBB", 20],
            ["cellBC_3", "DDD", 40],
            ["cellBC_3", "EEE", 40],
        ],
        columns=["cellBC", "intBC", "UMI_count"],
    )
    intbc_table_per_sample = {"sample0": intbc_table_0, "sample1": intbc_table_1}
    edit_distances = pre_qc_edit_distance_distribution(
        intbc_table_per_sample=intbc_table_per_sample,
        show_plot=False,
        fig_path=None,
        min_intbc_umi_count=10,
        max_pairs_per_cell=10
    )
    assert(edit_distances == [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
