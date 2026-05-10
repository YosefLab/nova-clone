import os
import editdistance
from nova_clone.algorithm._types import *
from ._compare_clone_calls import get_clones_and_sizes_from_cell_to_clones_mapping
from typing import Dict, List, Optional, Set, Union
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib

def qc_clone_intbc_sets(
    intbc_table_per_sample: Dict[SampleNameType, IntBCTableType],
    cell_to_clones_mapping: Dict[CellBCType, Set[CloneType]],
    clone_intbc_sets: Dict[CloneType, Set[IntBCType]],
    min_intbc_umi_count: int = 1,
    clones_to_plot: Optional[Union[int, List[CloneType]]] = 10,
    max_intbcs_per_clone: int = 10,
    show_plot: bool = True,
    fig_path: Optional[str] = None,
):
    """
    For each clone, plots its intBC sets sorted by frequency.

    If `clones_to_plot` is None, all clones are plotted.
    If `clones_to_plot` is an integer, the top N clones are plotted.
    If `clones_to_plot` is a list, only the clones in the list are plotted.
    """
    validate_intbc_table_per_sample(intbc_table_per_sample)
    res = {}
    clones_and_sizes_dict = get_clones_and_sizes_from_cell_to_clones_mapping(
        cell_to_clones_mapping
    )
    top_clones = sorted(clones_and_sizes_dict.items(), key=lambda x: (-x[1], x[0]))
    top_clones = [x[0] for x in top_clones]
    if clones_to_plot is None:
        clones_to_plot = top_clones  # all clones
    elif isinstance(clones_to_plot, int):
        clones_to_plot = top_clones[:clones_to_plot]
    else:
        clones_to_plot = [x for x in clones_to_plot if x in top_clones]

    for clone in clones_to_plot:
        intbc_and_count_and_color_and_edit_distance_and_clone_size_and_umi = []
        intbc_counts = defaultdict(int)
        intbc_umis = defaultdict(list)
        cells_in_clone = {
            cellbc.split("_", 1)[1]  # Remove sample prefix
            for cellbc, clones in cell_to_clones_mapping.items()
            if clones == {clone}
        }
        for sample, intbc_table in intbc_table_per_sample.items():
            # Subset the intbc_table to the cells in the clone
            intbc_table = intbc_table[intbc_table["cellBC"].isin(cells_in_clone)]
            # Count intBCs
            for _, row in intbc_table.iterrows():
                intbc = row["intBC"]
                if row["UMI_count"] >= min_intbc_umi_count:
                    intbc_counts[intbc] += 1
                    intbc_umis[intbc].append(row["UMI_count"])
        intbcs_sorted = sorted(
            intbc_counts.items(), key=lambda x: (-x[1], x[0])
        )
        intbcs_sorted = [x[0] for x in intbcs_sorted]
        for i, intbc in enumerate(intbcs_sorted):
            # Find smallest edit distance to previous intBCs
            if i > 0:
                previous_intbcs = intbcs_sorted[:i]
                min_edit_distance = min(
                    [editdistance.eval(intbc, prev) for prev in previous_intbcs]
                )
            else:
                min_edit_distance = len(intbc)  # No previous intBCs, use length of current intBC
            # We use blue color for intBCs which are in the clone's intBC set, and red for those that are not.
            color = "#0C234B" if intbc in clone_intbc_sets[clone] else "#AB0520"
            intbc_and_count_and_color_and_edit_distance_and_clone_size_and_umi.append(
                (intbc, intbc_counts[intbc], color, min_edit_distance, len(cells_in_clone), float(np.median(intbc_umis[intbc])))
            )
        res[clone] = intbc_and_count_and_color_and_edit_distance_and_clone_size_and_umi

    # Plotting
    # We plot a barplot as follows:
    # - x-axis: intBCs, in the following format: "{intbc}-{median_umis}-{min_edit_distance}"
    # - y-axis: intbc_count
    # - color: blue if intbc is in the clone's intBC set, red otherwise.
    # The number of intBCs per clone is limited to `max_intbcs_per_clone`.
    # - We add a horizontal line equal to the number of cells in the clone.
    plots_dfs = dict()
    matplotlib.pyplot.rcdefaults()
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['font.size'] = 13
    if show_plot:
        for clone, intbc_data in res.items():
            if len(intbc_data) > max_intbcs_per_clone:
                intbc_data = intbc_data[:max_intbcs_per_clone]
            x_values = [
                f"{intbc} {int(median_umis)}u {min_edit_distance}d" for (intbc, _, _, min_edit_distance, _, median_umis) in intbc_data
            ]
            heights = [count for (_, count, _, _, _, _) in intbc_data]
            colors = [color for (_, _, color, _, _, _) in intbc_data]

            clone_size = intbc_data[0][4]
            clone_df = pd.DataFrame.from_dict({'x_values': x_values, 'heights': heights, 'colors': colors, 'total_clone_size':clone_size}, orient='columns').T
            plots_dfs[clone] = clone_df
            plt.figure(figsize=(len(x_values) * 0.4, 4))
            plt.bar(x_values, heights, color=colors, edgecolor="black")
            plt.xticks(rotation=45, ha="right")
            plt.title(f"Top {max_intbcs_per_clone} IntBCs for clone {clone} (Size {int(clone_size)})\n(Blue = in clone's intBC set, Red = not in clone's intBC set)")
            plt.xlabel("IntBC and Median UMI Count and Min Edit Distance")
            plt.ylabel("IntBC Abundance in Clone")
            plt.axhline(
                y=clone_size,
                color="green",
                linestyle="--",
                label=f"Clone size: {clone_size}",
            )
            sns.despine()
            if fig_path:
                # Create directory if it does not exist
                os.makedirs(fig_path, exist_ok=True)
                plt.savefig(f"{fig_path}/{clone}.pdf", bbox_inches='tight')
            if show_plot:
                plt.show()
            plt.close()
    return res, plots_dfs


def test_qc_clone_intbc_sets():
    intbc_table = pd.DataFrame(
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
    clone_call = CloneCallType(
        cell_to_clones_mapping={
            "sample0_cellBC_1": {"clone_1"},
            "sample0_cellBC_2": {"clone_1"},
            "sample0_cellBC_3": {"clone_2"},
            "sample0_cellBC_4": {"clone_2"},
            "sample0_cellBC_5": {"clone_2"},
            "sample0_cellBC_6": set(),
            "sample0_cellBC_7": {"clone_1", "clone_2"},
            "sample0_cellBC_8": {"clone_1"},
        },
        clone_intbc_sets={
            "clone_1": {"AAA", "BBB", "CCC"},
            "clone_2": {"DDD", "EEE", "FFF"},
        },
        cell_intbc_sets={
            "sample0_cellBC_1": {"AAA", "BBB", "CCC"},
            "sample0_cellBC_2": {"AAA", "BBB", "CCC"},
            "sample0_cellBC_3": {"DDD", "EEE", "FFF"},
            "sample0_cellBC_4": {"DDD", "EEE", "FFF"},
            "sample0_cellBC_5": {"DDD", "EEE", "FFF"},
            "sample0_cellBC_6": set(),
            "sample0_cellBC_7": {"AAA", "DDD"},
            "sample0_cellBC_8": set(),
        },
        clone_to_cells_mapping=None,
    )
    res = qc_clone_intbc_sets(
        intbc_table_per_sample={"sample0": intbc_table},
        cell_to_clones_mapping=clone_call.cell_to_clones_mapping,
        clone_intbc_sets=clone_call.clone_intbc_sets,
        show_plot=False,
        min_intbc_umi_count=1,
    )
    print(f"res = {res}")
    assert(
        res[0] == {
            'clone_1': [
                ('AAA', 2, '#0C234B', 3, 3, 25.0),
                ('BBB', 2, '#0C234B', 3, 3, 30.5),
                ('ZZZ', 2, '#AB0520', 3, 3, 18.5),
                ('AAB', 1, '#AB0520', 1, 3, 5.0),
                ('BBZ', 1, '#AB0520', 1, 3, 2.0),
                ('CCC', 1, '#0C234B', 3, 3, 42.0),
            ],
            'clone_2': [
                ('DDD', 3, '#0C234B', 3, 3, 40.0),
                ('EEE', 3, '#0C234B', 3, 3, 40.0),
                ('FFF', 3, '#0C234B', 3, 3, 40.0),
                ('GGG', 1, '#AB0520', 3, 3, 40.0),
            ]
        }
    )


    res = qc_clone_intbc_sets(
        intbc_table_per_sample={"sample0": intbc_table},
        cell_to_clones_mapping=clone_call.cell_to_clones_mapping,
        clone_intbc_sets=clone_call.clone_intbc_sets,
        show_plot=False,
        min_intbc_umi_count=10,
    )
    assert(
        res[0] == {
            'clone_1': [
                ('AAA', 2, '#0C234B', 3, 3, 25.0),
                ('BBB', 2, '#0C234B', 3, 3, 30.5),
                ('CCC', 1, '#0C234B', 3, 3, 42.0),
                ('ZZZ', 1, '#AB0520', 3, 3, 33.0),
            ],
            'clone_2': [
                ('DDD', 3, '#0C234B', 3, 3, 40.0),
                ('EEE', 3, '#0C234B', 3, 3, 40.0),
                ('FFF', 3, '#0C234B', 3, 3, 40.0),
                ('GGG', 1, '#AB0520', 3, 3, 40.0),
            ]
        }
    )
