"""
Module for pruning aliased coresets.

The main function exposed by this module is `prune_coresets`.
"""

import logging
import sys
import time

import numpy as np

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


class PruneCoresetsReturnType(NamedTuple):
    """
    Object returned by the `prune_coresets` function.

    Attributes:
        coresets: List of coresets after pruning aliased coresets.
        total_time: Total time taken.
    """

    coresets: pd.DataFrame
    total_time: float


def prune_coresets(
    coresets: List[Coreset[IntBCType]],
    coreset_co_assignment_matrix: np.array,
    co_occurrence_cutoff: float,
    min_cells_assigned: int,
) -> List[Coreset[IntBCType]]:
    """
    Prune coresets.

    A coreset is pruned if:
    - It co-occurs (in the sense of p(y|x)) with another strictly larger coreset
        (in the sense of number of cells assigned), or of same size but smaller
        index. The coreset is pruned if p(larger coreset | coreset) >
        co_occurrence_cutoff and said size condition (with index tiebreaking) is
        met.
    - It is too small, in the sense that the number of (putative) cells assigned
        to it is < min_cells_assigned.

    Returns the new list of coresets.
    """
    st = time.time()
    logger = logging.getLogger(__name__)
    logger.info("Going to prune_coresets ...")
    num_coresets = coreset_co_assignment_matrix.shape[0]
    res = []
    pruned_because_too_small = 0
    pruned_because_aliased = 0
    for coreset_idx_1 in range(num_coresets):
        # Decide whether we keep coreset_idx_1
        keep = True
        c_1 = coreset_co_assignment_matrix[coreset_idx_1, coreset_idx_1]
        if c_1 < min_cells_assigned:
            keep = False  # We don't keep.
            pruned_because_too_small += 1
        else:
            for coreset_idx_2 in range(num_coresets):
                if coreset_idx_1 == coreset_idx_2:
                    continue
                c_2 = coreset_co_assignment_matrix[coreset_idx_2, coreset_idx_2]
                c_12 = coreset_co_assignment_matrix[coreset_idx_1, coreset_idx_2]
                co_occurrence = c_12 / c_1
                if co_occurrence > co_occurrence_cutoff and (
                    (c_1 < c_2) or ((c_1 == c_2) and (coreset_idx_1 > coreset_idx_2))
                ):
                    keep = False
                    pruned_because_aliased += 1
        if keep:
            res.append(coresets[coreset_idx_1])
    total_time = time.time() - st
    logger.info(
        f"Pruned {num_coresets - len(res)} coresets ({pruned_because_too_small} too small, {pruned_because_aliased} aliased). There remain {len(res)} coresets. Total time: {total_time}"
    )
    res = PruneCoresetsReturnType(
        coresets=res,
        total_time=total_time,
    )
    return res


def test_prune_coresets():
    coresets = [
        {"1", "2", "3"},
        {"4", "5", "6"},
        {"7", "8"},
        {"9"},
    ]
    coreset_co_assignment_matrix = np.array(
        [
            [10, 2, 4, 1],
            [2, 10, 1, 2],
            [4, 1, 5, 0],
            [1, 2, 1, 3],
        ]
    )
    pruned_coresets = prune_coresets(
        coresets=coresets,
        coreset_co_assignment_matrix=coreset_co_assignment_matrix,
        co_occurrence_cutoff=0.9,
        min_cells_assigned=0,
    ).coresets
    assert pruned_coresets == coresets
    pruned_coresets = prune_coresets(
        coresets=coresets,
        coreset_co_assignment_matrix=coreset_co_assignment_matrix,
        co_occurrence_cutoff=0.7,
        min_cells_assigned=0,
    ).coresets
    assert pruned_coresets == [
        {"1", "2", "3"},
        {"4", "5", "6"},
        {"9"},
    ]
    pruned_coresets = prune_coresets(
        coresets=coresets,
        coreset_co_assignment_matrix=coreset_co_assignment_matrix,
        co_occurrence_cutoff=0.6,
        min_cells_assigned=0,
    ).coresets
    assert pruned_coresets == [
        {"1", "2", "3"},
        {"4", "5", "6"},
    ]
    pruned_coresets = prune_coresets(
        coresets=coresets,
        coreset_co_assignment_matrix=coreset_co_assignment_matrix,
        co_occurrence_cutoff=0.7,
        min_cells_assigned=4,
    ).coresets
    assert pruned_coresets == [
        {"1", "2", "3"},
        {"4", "5", "6"},
    ]
    pruned_coresets = prune_coresets(
        coresets=coresets,
        coreset_co_assignment_matrix=coreset_co_assignment_matrix,
        co_occurrence_cutoff=1.0,
        min_cells_assigned=6,
    ).coresets
    assert pruned_coresets == [
        {"1", "2", "3"},
        {"4", "5", "6"},
    ]
