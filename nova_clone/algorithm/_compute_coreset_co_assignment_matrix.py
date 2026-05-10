"""
Module for computing the number of cellBCs assigned to two coresets.

The main function exposed by this module is `compute_coreset_co_assignment_matrix`.
"""

import logging
import sys
import time
from typing import *

import numpy as np
import tqdm

from ._types import *


def _init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)


_init_logger()


class ComputeCoresetCoAssignmentMatrixReturnType(NamedTuple):
    """
    Object returned by the `compute_coreset_co_assignment_matrix` function.

    Attributes:
        coreset_co_assignment_matrix: #coresets x #coresets, a numpy array
            containing the number of cellBCs mapping to those two coresets.
            (In particular, the diagonal contains the number of cellBC mapping
            to that coerset).
        total_time: Total time taken.
    """

    coreset_co_assignment_matrix: pd.DataFrame
    total_time: float


def compute_coreset_co_assignment_matrix(
    cellbc_intbc_sets: Dict[CellBCType, Set[IntBCType]],
    coresets: List[Set[IntBCType]],
    min_overlap_for_assigning_cellbc: int,
) -> ComputeCoresetCoAssignmentMatrixReturnType:
    """
    Map cells to coresets to build the coreset co-assignment matrix.

    The coreset co-assignment matrix M contains the statistics:
    M_{i,j} = #(cell assigned to coresets i and j)
    """
    st = time.time()
    logger = logging.getLogger(__name__)
    logger.info("Going to compute_coreset_co_assignment_matrix ...")
    num_coresets = len(coresets)
    res = np.zeros(shape=(num_coresets, num_coresets))
    for cell, cell_intbc_set in tqdm.tqdm(cellbc_intbc_sets.items()):
        coresets_assigned = []
        for coreset_idx, coreset in enumerate(coresets):
            overlap = len(cell_intbc_set.intersection(coreset))
            if overlap >= min_overlap_for_assigning_cellbc:
                coresets_assigned.append(coreset_idx)
        for coreset_idx_1 in coresets_assigned:
            for coreset_idx_2 in coresets_assigned:
                if coreset_idx_1 < coreset_idx_2:
                    res[coreset_idx_1, coreset_idx_2] += 1
                    res[coreset_idx_2, coreset_idx_1] += 1
                elif coreset_idx_1 == coreset_idx_2:
                    res[coreset_idx_1, coreset_idx_2] += 1
    total_time = time.time() - st
    res = ComputeCoresetCoAssignmentMatrixReturnType(
        coreset_co_assignment_matrix=res,
        total_time=total_time,
    )
    logger.info(
        f"Done with compute_coreset_co_assignment_matrix! Total time: {total_time}"
    )
    return res


def test_compute_coreset_co_assignment_matrix():
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
    mat = compute_coreset_co_assignment_matrix(
        cellbc_intbc_sets=cellbc_intbc_sets,
        coresets=coresets,
        min_overlap_for_assigning_cellbc=3,
    ).coreset_co_assignment_matrix
    np.testing.assert_equal(
        mat,
        np.array(
            [
                [2, 1],
                [1, 2],
            ]
        ),
    )
    mat = compute_coreset_co_assignment_matrix(
        cellbc_intbc_sets=cellbc_intbc_sets,
        coresets=coresets,
        min_overlap_for_assigning_cellbc=2,
    ).coreset_co_assignment_matrix
    np.testing.assert_equal(
        mat,
        np.array(
            [
                [3, 1],
                [1, 2],
            ]
        ),
    )
    mat = compute_coreset_co_assignment_matrix(
        cellbc_intbc_sets=cellbc_intbc_sets,
        coresets=coresets,
        min_overlap_for_assigning_cellbc=1,
    ).coreset_co_assignment_matrix
    np.testing.assert_equal(
        mat,
        np.array(
            [
                [3, 2],
                [2, 3],
            ]
        ),
    )
