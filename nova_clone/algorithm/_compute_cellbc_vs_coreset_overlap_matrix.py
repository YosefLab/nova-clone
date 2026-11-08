"""
Module for computing the overlap (intersection) between intBC sets (of cellBCs)
and coresets.

The main function exposed by this module is `compute_cellbc_vs_coreset_overlap_matrix`.
"""

import logging
import sys
import time
from typing import *

from multiset import Multiset
import numpy as np
import tqdm

from ._types import *
from ._utils import _cell_intbc_sets_from_intbc_tables


def _init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)


_init_logger()


class ComputeCellBCvsCoresetOverlapMatrixReturnType(NamedTuple):
    """
    Object returned by the `compute_cellbc_vs_coreset_overlap_matrix` function.

    Attributes:
        cellbc_vs_coreset_overlap_matrix: Of size #cellBC x #coresets, the
            size of the intersection between the intBC set of each cellBC and
            each coreset.
        total_time: Total time taken.
    """

    cellbc_vs_coreset_overlap_matrix: pd.DataFrame
    total_time: float


def compute_cellbc_vs_coreset_overlap_matrix(
    cellbc_intbc_sets: Dict[IntBCType, Set[IntBCType]],
    coresets: List[Coreset[IntBCType]],
) -> ComputeCellBCvsCoresetOverlapMatrixReturnType:
    """
    Pandas dataframe of cell vs coreset.

    Entry (i, j) contains the overlap of cell i's intBC set with coreset j.
    """
    st = time.time()
    logger = logging.getLogger(__name__)
    logger.info("Going to compute_cellbc_vs_coreset_overlap_matrix ...")
    cells = sorted(cellbc_intbc_sets.keys())
    res_array = np.zeros(shape=(len(cells), len(coresets)))

    for cell_idx, cell in tqdm.tqdm(list(enumerate(cells))):
        cell_intbc_set = cellbc_intbc_sets[cell]
        for coreset_idx, coreset in enumerate(coresets):
            overlap = len([intbc for intbc in coreset if intbc in cell_intbc_set])  # Handles multiset coresets (for single-intBC support)
            res_array[cell_idx, coreset_idx] = overlap

    cellbc_vs_coreset_overlap_matrix = pd.DataFrame(
        res_array,
        index=cells,
        columns=list(range(len(coresets))),
    )
    total_time = time.time() - st
    res = ComputeCellBCvsCoresetOverlapMatrixReturnType(
        cellbc_vs_coreset_overlap_matrix=cellbc_vs_coreset_overlap_matrix,
        total_time=total_time,
    )
    logger.info(
        f"Done with compute_cellbc_vs_coreset_overlap_matrix! Total time: {total_time}"
    )
    return res


def test_compute_cell_vs_coreset_overlap_matrix():
    cellbc_intbc_sets = {
        "cell_1": {"1", "2", "3"},
        "cell_2": {"4", "5", "6"},
        "cell_3": {"1", "2", "3", "4", "5", "6"},
        "cell_4": {"1", "2", "4"},
    }
    coresets = [
        {"1", "2", "3"},
        {"4", "5", "6"},
    ]
    mat = compute_cellbc_vs_coreset_overlap_matrix(
        cellbc_intbc_sets=cellbc_intbc_sets,
        coresets=coresets,
    ).cellbc_vs_coreset_overlap_matrix
    np.testing.assert_equal(
        mat,
        np.array(
            [
                [3, 0],
                [0, 3],
                [3, 3],
                [2, 1],
            ]
        ),
    )


def test_compute_cell_vs_coreset_overlap_matrix_with_multisets():
    cellbc_intbc_sets = {
        "cell_1": {"1", "2"},
        "cell_2": {"1", "3"},
        "cell_3": {"1", "2", "3"},
        "cell_4": {"1", "2", "4"},
        "cell_5": {"4"},
    }
    coresets = [
        {"1", "2"},
        {"1", "3"},
        Multiset(["4"]) + Multiset(["4"]),
    ]
    mat = compute_cellbc_vs_coreset_overlap_matrix(
        cellbc_intbc_sets=cellbc_intbc_sets,
        coresets=coresets,
    ).cellbc_vs_coreset_overlap_matrix
    np.testing.assert_equal(
        mat,
        np.array(
            [
                [2, 1, 0],
                [1, 2, 0],
                [2, 2, 0],
                [2, 1, 2],
                [0, 0, 2],
            ]
        ),
    )
