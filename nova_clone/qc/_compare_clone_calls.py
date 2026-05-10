from collections import defaultdict
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd
import tqdm

from nova_clone.algorithm import clone_calling_algorithm
from nova_clone.algorithm._types import CellBCType, CloneType
from nova_clone.algorithm.cassiopeia import call_lineage_groups


def get_clones_and_sizes_from_cell_to_clones_mapping(
    cell_to_clones_mapping: Dict[CellBCType, Set[CloneType]],
) -> Dict[CloneType, int]:
    res = defaultdict(int)
    for cell, clones in cell_to_clones_mapping.items():
        if len(clones) == 1:
            res[list(clones)[0]] += 1
    return res


def qc_compare_clone_calls__overlap_matrix(
    cell_to_clones_mapping_1: Dict[CellBCType, Set[CloneType]],
    cell_to_clones_mapping_2: Dict[CellBCType, Set[CloneType]],
    row_conditional: bool = False,
    col_conditional: bool = False,
) -> pd.DataFrame:
    """
    Compare two clone mappings.

    Returns the overlap matrix between the two clones.
    """
    assert not (row_conditional and col_conditional)
    clones_and_sizes_1 = get_clones_and_sizes_from_cell_to_clones_mapping(
        cell_to_clones_mapping_1
    )  # type: Dict[CloneType, int]
    clones_and_sizes_2 = get_clones_and_sizes_from_cell_to_clones_mapping(
        cell_to_clones_mapping_2
    )  # type: Dict[CloneType, int]
    sorted_clones_1 = [
        x[0] for x in sorted(clones_and_sizes_1.items(), key=lambda x: (-x[1], x[0]))
    ]
    sorted_clones_2 = [
        x[0] for x in sorted(clones_and_sizes_2.items(), key=lambda x: (-x[1], x[0]))
    ]
    res = pd.DataFrame(
        np.zeros(
            shape=(
                1 + len(sorted_clones_1),
                1 + len(sorted_clones_2),
            ),
            dtype=float,
        ),
        index=["junk"] + sorted_clones_1,
        columns=["junk"] + sorted_clones_2,
    )
    cells_1_set = set(cell_to_clones_mapping_1.keys())
    cells_2_set = set(cell_to_clones_mapping_2.keys())
    for cell in tqdm.tqdm(cells_1_set.union(cells_2_set)):
        clones_1 = cell_to_clones_mapping_1.get(cell, [])
        clones_2 = cell_to_clones_mapping_2.get(cell, [])
        if len(clones_1) == 1 and len(clones_2) == 1:
            res.loc[list(clones_1)[0], list(clones_2)[0]] += 1
        elif len(clones_1) == 1:
            res.loc[list(clones_1)[0], "junk"] += 1
        elif len(clones_2) == 1:
            res.loc["junk", list(clones_2)[0]] += 1
        else:
            res.loc["junk", "junk"] += 1
    if row_conditional:
        for i in range(res.shape[0]):
            res.iloc[i, :] /= float(res.iloc[i, :].sum())
    if col_conditional:
        for i in range(res.shape[1]):
            res.iloc[:, i] /= float(res.iloc[:, i].sum())
    return res


def qc_compare_clone_calls__align_clones(
    cell_to_clones_mapping_1: Dict[CellBCType, Set[CloneType]],
    cell_to_clones_mapping_2: Dict[CellBCType, Set[CloneType]],
    threshold: float = 0.5,
    min_clone_size: int = 10,
) -> Tuple[Dict[CloneType, Set[CloneType]], Dict[CloneType, Set[CloneType]]]:
    """
    Given two clone calls, determine how clones map ('align') accross them.

    Here `threshold` is the conditional probability threshold for determining
    whether one clone aligns to another one.

    Here `min_clone_size` is the minimum clone size to call an alignment.
    """
    overlap_matrix_row_conditional = qc_compare_clone_calls__overlap_matrix(
        cell_to_clones_mapping_1,
        cell_to_clones_mapping_2,
        row_conditional=True,
    )
    clones_1 = list(overlap_matrix_row_conditional.index)
    clones_2 = list(overlap_matrix_row_conditional.columns)

    overlap_matrix_col_conditional = qc_compare_clone_calls__overlap_matrix(
        cell_to_clones_mapping_1,
        cell_to_clones_mapping_2,
        col_conditional=True,
    )

    align_1_to_2 = {clone_1: set() for clone_1 in clones_1}
    align_2_to_1 = {clone_2: set() for clone_2 in clones_2}

    clone_sizes_1 = get_clones_and_sizes_from_cell_to_clones_mapping(
        cell_to_clones_mapping_1
    )
    clone_sizes_2 = get_clones_and_sizes_from_cell_to_clones_mapping(
        cell_to_clones_mapping_2
    )

    for overlap_matrix in [
        overlap_matrix_row_conditional,
        overlap_matrix_col_conditional,
    ]:
        xs, ys = np.where(overlap_matrix >= threshold)
        for x, y in zip(xs, ys):
            clone_1 = clones_1[x]
            clone_2 = clones_2[y]
            if (clone_sizes_1[clone_1] >= min_clone_size) and (
                clone_sizes_2[clone_2] >= min_clone_size
            ):
                align_1_to_2[clone_1].add(clone_2)
                align_2_to_1[clone_2].add(clone_1)
    return align_1_to_2, align_2_to_1


def qc_compare_clone_calls__detect_undermerging(
    cell_to_clones_mapping_1: Dict[CellBCType, Set[CloneType]],
    cell_to_clones_mapping_2: Dict[CellBCType, Set[CloneType]],
) -> None:
    """
    Simple prints the undermerging of clones between two clone calls.
    """
    align_1_to_2, align_2_to_1 = qc_compare_clone_calls__align_clones(
        cell_to_clones_mapping_1,
        cell_to_clones_mapping_2,
        threshold=0.9,
        min_clone_size=10,
    )

    clone_sizes_1 = get_clones_and_sizes_from_cell_to_clones_mapping(
        cell_to_clones_mapping_1
    )
    clone_sizes_2 = get_clones_and_sizes_from_cell_to_clones_mapping(
        cell_to_clones_mapping_2
    )

    def f(align_x_to_y, clone_sizes_x, clone_sizes_y):
        for clone_x in align_x_to_y:
            if clone_x == "junk":
                continue
            clones_y = align_x_to_y[clone_x]
            if len(clones_y) >= 2:
                print(
                    clone_x,
                    f"({clone_sizes_x[clone_x]})",
                    "is composed of",
                    [(clone_y, clone_sizes_y[clone_y]) for clone_y in clones_y],
                )

    print(f"align_1_to_2: ")
    f(align_1_to_2, clone_sizes_1, clone_sizes_2)
    print(f"align_2_to_1: ")
    f(align_2_to_1, clone_sizes_2, clone_sizes_1)


def test_qc_compare_clone_calls():
    """
    Just tests that the QC functions for comparing two clone calls do not crash.
    """
    intbc_table = pd.DataFrame(
        data=[
            ["cellBC_1", "AAA", 40],
            ["cellBC_1", "BBB", 40],
            ["cellBC_1", "CCC", 40],
            ["cellBC_2", "AAA", 40],
            ["cellBC_2", "BBB", 40],
            ["cellBC_2", "CCC", 40],
            ["cellBC_3", "DDD", 40],
            ["cellBC_3", "EEE", 40],
            ["cellBC_3", "FFF", 40],
            ["cellBC_4", "DDD", 40],
            ["cellBC_4", "EEE", 40],
            ["cellBC_4", "FFF", 40],
            ["cellBC_5", "DDD", 40],
            ["cellBC_5", "EEE", 40],
            ["cellBC_5", "FFF", 40],
        ],
        columns=["cellBC", "intBC", "UMI_count"],
    )

    clone_call_1 = clone_calling_algorithm(
        intbc_table_per_sample={"sample1": intbc_table},
        min_intbc_umi_count=10,
        custom_intbc_filtering_function=lambda intbc: False,
        min_intbc_cellbc_count=1,
        edit_distance_cutoff=0,
        _sequencing_error_pruning_step__co_occurrence_cutoff=1.1,
        _homoplastic_pruning_step__p_max_cutoff=-0.1,
        _reliable_edges_step__ratio_cutoff=-0.1,
        _preliminary_coreset_building_step__min_coreset_size=2,
        _preliminary_coreset_building_step__min_overlap_for_merging_coresets=2,
        _coreset_dealiasing_step__min_overlap_for_assigning_cellbc=2,
        _coreset_dealiasing_step__co_occurrence_cutoff=1.1,
        min_cells_assigned=1,
        _coreset_disjointing_step__make_disjoint=True,
        _final_cell_assignment_step__min_overlap_for_assigning_cellbc=2,
        _call_clones_from_coresets_step__min_intbc_counts_for_adding_intbc_to_clone_intbc_set=1,
        min_intbc_fraction=-0.1,
    ).clone_call
    clone_call_2 = call_lineage_groups(
        intbc_table_per_sample={"sample1": intbc_table},
    )
    qc_compare_clone_calls__overlap_matrix(
        clone_call_1.cell_to_clones_mapping, clone_call_2.cell_to_clones_mapping
    )
    qc_compare_clone_calls__align_clones(
        clone_call_1.cell_to_clones_mapping, clone_call_2.cell_to_clones_mapping
    )
    qc_compare_clone_calls__detect_undermerging(
        clone_call_1.cell_to_clones_mapping, clone_call_2.cell_to_clones_mapping
    )
