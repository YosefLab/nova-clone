"""
Module for removing homoplastic intBCs from the coresets, thus making then disjoint.

The main function exposed by this module is `make_coresets_disjoint`.
"""

import logging
import sys
import time

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


class MakeCoresetsDisjointReturnType(NamedTuple):
    """
    Object returned by the `make_coresets_disjoint` function.

    Attributes:
        coresets: List of coresets after removing homopalstic intBCs.
        total_time: Total time taken.
    """

    coresets: List[Set[IntBCType]]
    total_time: float


def make_coresets_disjoint(
    coresets: List[Set[IntBCType]],
) -> MakeCoresetsDisjointReturnType:
    """
    Elements that appear in more than one coreset are removed from all the
    coresets.
    """
    st = time.time()
    logger = logging.getLogger(__name__)
    logger.info("Going to make_coresets_disjoint ...")
    intbc_counts = defaultdict(int)
    for coreset in coresets:
        for intbc in coreset:
            intbc_counts[intbc] += 1
    disjoint_coresets = []
    homoplastic_intbcs = [intbc for intbc, cnt in intbc_counts.items() if cnt > 1]
    for coreset in coresets:
        disjoint_coreset = set()
        for intbc in coreset:
            cnt = intbc_counts[intbc]
            assert cnt >= 1
            if cnt == 1:
                disjoint_coreset.add(intbc)
        disjoint_coresets.append(disjoint_coreset)
    total_time = time.time() - st
    res = MakeCoresetsDisjointReturnType(
        coresets=disjoint_coresets,
        total_time=total_time,
    )
    logger.info(
        f"Found {len(homoplastic_intbcs)} homoplastic intBCs while making the coresets disjoint. Total time: {total_time}"
    )
    return res


def test_make_coresets_disjoint():
    coresets = [
        {"1", "2", "3"},
        {"3", "4", "5", "6"},
        {"6"},
        {"7"},
    ]
    coresets_disjoint = make_coresets_disjoint(
        coresets=coresets,
    ).coresets
    assert coresets_disjoint == [
        {"1", "2"},
        {"4", "5"},
        set(),
        {"7"},
    ]
