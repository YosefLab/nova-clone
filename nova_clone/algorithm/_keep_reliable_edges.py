"""
This module implements filtering of edges in the intBC graph which are
unreliable, i.e. likely due to noise rather than true co-occurrence
in some clone.

The main function exposed by this module is `keep_reliable_edges`.
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


class KeepReliableEdgesReturnType(NamedTuple):
    """
    Object returned by the keep_reliable_edges function.

    Attributes:
        intbc_graph: the new graph after pruning away the unreliable edges.
        total_time: Total time taken.
    """

    intbc_graph: GraphType
    total_time: float


def keep_reliable_edges(
    intbc_graph: GraphType,
    ratio_cutoff: float,
) -> KeepReliableEdgesReturnType:
    """
    Keeps only edges in the intBC graph which are reliable.

    We say that an edge in the intBC graph is "reliable" if we are very certain
    that there exists one and only one clone in which its two intBCs are
    integrations. Provided that the graph has been pruned from sequencing
    errors, reliable pairs will have a large ratio. Here the ratio is defined
    as:
    ratio(x, y) = c(x, y) / min(c_x, c_y) = max { p(x|y), p(y|x) }

    Args:
        ratio_cutoff: Only edges with ratio greater than or equal to this will
            be kept.
    """
    st = time.time()
    logger = logging.getLogger(__name__)
    logger.info("Going to keep_reliable_edges ...")
    edge_counts, intbc_counts = intbc_graph
    pruned_edge_counts = {intbc: defaultdict(int) for intbc in intbc_counts}
    total_edges = 0
    number_of_pruned_edges = 0
    for intbc, counts in intbc_counts.items():
        for neighbor, co_counts in edge_counts[intbc].items():
            total_edges += 1
            ratio = co_counts / min(counts, intbc_counts[neighbor])
            if ratio >= ratio_cutoff:
                pruned_edge_counts[intbc][neighbor] = co_counts
            else:
                number_of_pruned_edges += 1

    pruned_edge_counts = {
        x: {y: z for (y, z) in d.items()} for (x, d) in pruned_edge_counts.items()
    }
    pruned_intbc_counts = {x: y for (x, y) in intbc_counts.items()}  # Same as original.
    pruned_graph = (pruned_edge_counts, pruned_intbc_counts)
    total_time = time.time() - st
    res = KeepReliableEdgesReturnType(
        intbc_graph=pruned_graph,
        total_time=total_time,
    )
    edge_counts, node_counts = pruned_graph
    num_nodes = len(node_counts)
    num_edges = sum([len(x) for x in edge_counts.values()]) // 2
    logger.info(
        f"Pruned {(100.0 * number_of_pruned_edges / (total_edges + 1e-9)):.2f}% edges from the intBC graph. Pruned IntBC graph has {num_nodes} nodes and {num_edges} edges.  Density level: {100.0 * num_edges / (num_nodes * (num_nodes - 1.0) / 2.0 + 1e-9):.2f}%"
    )
    logger.info(f"Done with keep_reliable_edges! Total time: {total_time}")
    return res


def test_keep_reliable_edges():
    graph = (
        {
            "1": {"2": 500, "3": 100, "4": 100},
            "2": {"1": 500, "3": 100, "4": 100},
            "3": {"1": 100, "2": 100, "4": 600},
            "4": {"1": 100, "2": 100, "3": 600},
        },
        {
            "1": 1000,
            "2": 1000,
            "3": 1000,
            "4": 1000,
        },
    )
    pruned_graph = keep_reliable_edges(
        intbc_graph=graph,
        ratio_cutoff=0.0,
    ).intbc_graph
    assert graph == pruned_graph
    pruned_graph = keep_reliable_edges(
        intbc_graph=graph,
        ratio_cutoff=0.2,
    ).intbc_graph
    assert pruned_graph == (
        {
            "1": {"2": 500},
            "2": {"1": 500},
            "3": {"4": 600},
            "4": {"3": 600},
        },
        {
            "1": 1000,
            "2": 1000,
            "3": 1000,
            "4": 1000,
        },
    )
    pruned_graph = keep_reliable_edges(
        intbc_graph=graph,
        ratio_cutoff=0.55,
    ).intbc_graph
    assert pruned_graph == (
        {
            "1": {},
            "2": {},
            "3": {"4": 600},
            "4": {"3": 600},
        },
        {
            "1": 1000,
            "2": 1000,
            "3": 1000,
            "4": 1000,
        },
    )
    pruned_graph = keep_reliable_edges(
        intbc_graph=graph,
        ratio_cutoff=0.65,
    ).intbc_graph
    assert pruned_graph == (
        {
            "1": {},
            "2": {},
            "3": {},
            "4": {},
        },
        {
            "1": 1000,
            "2": 1000,
            "3": 1000,
            "4": 1000,
        },
    )
    graph = (
        {
            "1": {"2": 500, "3": 100, "4": 100},
            "2": {"1": 500, "3": 100, "4": 100},
            "3": {"1": 100, "2": 100, "4": 600},
            "4": {"1": 100, "2": 100, "3": 600},
        },
        {
            "1": 1000,
            "2": 1000,
            "3": 1000,
            "4": 800,
        },
    )
    pruned_graph = keep_reliable_edges(
        intbc_graph=graph,
        ratio_cutoff=0.65,
    ).intbc_graph
    assert pruned_graph == (
        {
            "1": {},
            "2": {},
            "3": {"4": 600},
            "4": {"3": 600},
        },
        {
            "1": 1000,
            "2": 1000,
            "3": 1000,
            "4": 800,
        },
    )
