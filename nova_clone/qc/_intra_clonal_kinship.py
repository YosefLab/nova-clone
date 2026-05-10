from collections import defaultdict
from typing import Dict, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm

from nova_clone.algorithm._types import CellBCType, CloneType, IntBCType

PLT_SAVEFIG_KWARGS = {
    "bbox_inches": "tight",
    "dpi": 300,
}


def _qc_intra_clonal_kinship_distribution(
    cell_intbc_sets: Dict[str, Set[str]],
    cell_to_clones_mapping: Dict[str, Set[str]],
    clone_to_cells_mapping: Dict[str, Set[str]],
    max_overlap: int = 10,
    subsample_ncells_per_clone: Optional[int] = None,
):
    all_overlaps = {clone: [] for clone in clone_to_cells_mapping}
    all_clones = [
        x[0] for x in sorted(clone_to_cells_mapping.items(), key=lambda x: -len(x[1]))
    ]
    res = pd.DataFrame(
        np.zeros(shape=(len(all_clones), max_overlap + 1)),
        index=all_clones,
        columns=list(range(max_overlap + 1)),
    )
    for clone in tqdm.tqdm(all_clones):
        row_dict = {i: 0 for i in range(max_overlap + 1)}
        clone_cells = clone_to_cells_mapping[clone]
        if subsample_ncells_per_clone is not None:
            rng = np.random.default_rng(subsample_ncells_per_clone)
            clone_cells = rng.choice(
                list(clone_cells),
                size=min(subsample_ncells_per_clone, len(clone_cells)),
                replace=False,
            )
        for cell_1 in clone_cells:
            cell_intbc_set_1 = cell_intbc_sets[cell_1]
            for cell_2 in clone_cells:
                if cell_1 <= cell_2:
                    cell_intbc_set_2 = cell_intbc_sets[cell_2]
                    overlap = len(cell_intbc_set_1.intersection(cell_intbc_set_2))
                    overlap = min(overlap, max_overlap)
                    row_dict[overlap] += 1
                    all_overlaps[clone].append(overlap)
        for i in range(max_overlap + 1):
            res.loc[clone, i] = row_dict[i] / (
                len(clone_cells) * (len(clone_cells) + 1) / 2
            )
    overlap_quantiles = {
        clone: {q: np.quantile(all_overlaps[clone], q) for q in [0.1, 0.5, 0.9]}
        for clone in all_overlaps
    }
    means = {clone: np.mean(all_overlaps[clone]) for clone in all_overlaps}
    return res, overlap_quantiles, means


def qc_intra_clonal_kinship(
    clone_to_cells_mapping: Dict[CloneType, Set[CellBCType]],
    cell_to_clones_mapping: Dict[CellBCType, Set[CloneType]],
    cell_intbc_sets: Dict[CellBCType, Set[IntBCType]],
    max_overlap: int = 10,
    subsample_ncells_per_clone: int = None,
    fig_path: Optional[str] = None,
    figsize: Optional[Tuple] = None,
    plot: bool = True,
) -> pd.DataFrame:
    """
    For each clone (row) the percent of cells with a given intBC overlap (column)
    """
    res, overlap_quantiles, means = _qc_intra_clonal_kinship_distribution(
        cell_intbc_sets=cell_intbc_sets,
        cell_to_clones_mapping=cell_to_clones_mapping,
        clone_to_cells_mapping=clone_to_cells_mapping,
        max_overlap=max_overlap,
        subsample_ncells_per_clone=subsample_ncells_per_clone,
    )
    if figsize:
        plt.figure(figsize=figsize)
    res.index = res.index.map(
        lambda clone: clone + "-" + str(len(clone_to_cells_mapping[clone]))
    )
    sns.heatmap(res, cmap="Blues")
    plt.title("Intra-clonal kinship distribution")
    plt.ylabel("clone")
    plt.xlabel("kinship (i.e. #overlapping intBCs)")
    plt.tight_layout()
    if fig_path is not None:
        plt.savefig(
            fig_path,
            **PLT_SAVEFIG_KWARGS,
        )
    if plot:
        plt.show()
    plt.close()
    # for clone in overlap_quantiles:
    #     if overlap_quantiles[clone][0.5] < 3:
    #         print(
    #             f"Clone {clone} has a median kinship of "
    #             f"{overlap_quantiles[clone][0.5]} which is less than 3. "
    #             f"This suggests low lineage tracing capacity or a clone with "
    #             f"an incoherent set of cells. "
    #             f"You might want to exclude this clone from further analysis."
    #         )
    for clone in means:
        if means[clone] < 2:
            print(
                f"Clone {clone} has a mean kinship of "
                f"{means[clone]:.2f} which is less than 2.\n"
                f"This suggests low lineage tracing capacity or a clone with "
                f"an incoherent set of cells.\n"
                f"You might want to exclude this clone from further analysis."
            )
    return res


def _compute_clone_to_cells_mapping(
    cell_to_clones_mapping: Dict[CellBCType, Set[CloneType]],
):
    res = defaultdict(set)
    for cell, clones in cell_to_clones_mapping.items():
        if len(clones) == 1:
            res[list(clones)[0]].add(cell)
    return res


def test__qc_intra_clonal_kinship_distribution(
    fig_path: str = None,
):
    cell_to_clones_mapping = {
        "putative_cell_1": {"estimated_clone_1"},
        "putative_cell_2": {"estimated_clone_1"},
        "putative_cell_3": {"estimated_clone_1"},
        "putative_cell_4": {"estimated_clone_2"},
        "putative_cell_5": {"estimated_clone_2"},
        "putative_cell_6": {"estimated_clone_3"},
    }
    cell_intbc_sets = {
        "putative_cell_1": {"AA", "BB", "CC"},
        "putative_cell_2": {"AA", "BB"},
        "putative_cell_3": {"CC"},
        "putative_cell_4": {"DD", "EE", "FF"},
        "putative_cell_5": {"GG", "HH", "II"},
        "putative_cell_6": {str(i) for i in range(100)},
    }
    max_overlap = 10
    res, _, _ = _qc_intra_clonal_kinship_distribution(
        cell_intbc_sets=cell_intbc_sets,
        cell_to_clones_mapping=cell_to_clones_mapping,
        clone_to_cells_mapping=_compute_clone_to_cells_mapping(cell_to_clones_mapping),
        max_overlap=max_overlap,
    )
    expected_res = pd.DataFrame(
        np.zeros(shape=(3, max_overlap + 1)),
        index=["estimated_clone_1", "estimated_clone_2", "estimated_clone_3"],
        columns=list(range(max_overlap + 1)),
    )
    expected_res.loc["estimated_clone_1", 0] = 1 / 6
    expected_res.loc["estimated_clone_1", 1] = 2 / 6
    expected_res.loc["estimated_clone_1", 2] = 2 / 6
    expected_res.loc["estimated_clone_1", 3] = 1 / 6
    expected_res.loc["estimated_clone_2", 0] = 1 / 3
    expected_res.loc["estimated_clone_2", 3] = 2 / 3
    expected_res.loc["estimated_clone_3", max_overlap] = 1 / 1
    pd.testing.assert_frame_equal(res, expected_res)
