"""
Module for building an IntBC graph from a list of intBC tables.

The main function exposed by this module is `build_intbc_graph`.
"""

import logging
import sys
import time
from collections import defaultdict
from typing import Set

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


class BuildIntBCGraphReturnType(NamedTuple):
    """
    Object returned by the build_intbc_graph function.

    Attributes:
        intbc_graph: The intBC graph, of type GraphType.
        total_time: Total time taken.
    """

    intbc_graph: GraphType
    total_time: float


def build_intbc_graph(
    intbc_table_per_sample: Dict[SampleNameType, IntBCTableType],
) -> BuildIntBCGraphReturnType:
    """
    Build the intBC graph from a list of intBC tables.

    Args:
        intbc_table_per_sample: intBC table per sample.

    Returns:
        intbc_graph: The intBC graph, of type GraphType.
    """
    st = time.time()
    logger = logging.getLogger(__name__)
    logger.info("Going to build_intbc_graph ...")
    edge_counts = defaultdict(lambda: defaultdict(int))
    node_counts = defaultdict(int)
    cell_intbc_sets = _cell_intbc_sets_from_intbc_tables(
        intbc_table_per_sample=intbc_table_per_sample,
    )
    for cell_intbc_set in cell_intbc_sets.values():
        for intbc_1 in cell_intbc_set:
            node_counts[intbc_1] += 1
            for intbc_2 in cell_intbc_set:
                if intbc_1 == intbc_2:
                    continue
                edge_counts[intbc_1][intbc_2] += 1
    # Make sure that intBCs with no neighbors are present in the edge dict!
    for intbc in node_counts:
        edge_counts[intbc]
    # Convert to a normal Dict
    edge_counts = {x: {y: z for (y, z) in d.items()} for (x, d) in edge_counts.items()}
    node_counts = {x: y for (x, y) in node_counts.items()}
    intbc_graph = (edge_counts, node_counts)
    total_time = time.time() - st
    res = BuildIntBCGraphReturnType(
        intbc_graph=intbc_graph,
        total_time=total_time,
    )
    num_nodes = len(node_counts)
    num_edges = sum([len(x) for x in edge_counts.values()]) // 2
    logger.info(
        f"IntBC graph has {num_nodes} nodes and {num_edges} edges. Density level: {100.0 * num_edges / (num_nodes * (num_nodes - 1.0) / 2.0 + 1e-9):.2f}%"
    )
    logger.info(f"Done with build_intbc_graph! Total time: {total_time}")
    return res


def test_build_intbc_graph():
    """
    Basic test.
    """
    intbc_table_1 = pd.DataFrame(
        data=[
            ["cellBC_1", "AAA", 40],
            ["cellBC_1", "BBB", 40],
            ["cellBC_2", "AAA", 40],
            ["cellBC_2", "BBB", 40],
            ["cellBC_2", "CCC", 40],
            ["cellBC_3", "DDD", 40],
            ["cellBC_3", "EEE", 40],
            ["cellBC_3", "FFF", 40],
        ],
        columns=["cellBC", "intBC", "UMI_count"],
    )
    intbc_table_2 = pd.DataFrame(
        data=[
            ["cellBC_1", "EEE", 10],
            ["cellBC_1", "FFF", 40],
            ["cellBC_2", "GGG", 40],  # Just observed once
        ],
        columns=["cellBC", "intBC", "UMI_count"],
    )
    graph = build_intbc_graph(
        intbc_table_per_sample={
            "sample1": intbc_table_1,
            "sample2": intbc_table_2,
        },
    ).intbc_graph
    expected_graph = (
        {
            "AAA": {"BBB": 2, "CCC": 1},
            "BBB": {"AAA": 2, "CCC": 1},
            "CCC": {"AAA": 1, "BBB": 1},
            "DDD": {"EEE": 1, "FFF": 1},
            "EEE": {"DDD": 1, "FFF": 2},
            "FFF": {"DDD": 1, "EEE": 2},
            "GGG": {},
        },
        {"AAA": 2, "BBB": 2, "CCC": 1, "DDD": 1, "EEE": 2, "FFF": 2, "GGG": 1},
    )
    assert graph == expected_graph
