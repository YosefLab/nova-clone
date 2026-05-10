"""
This module implements filtering of intBCs which are likely homoplastic (i.e.
appearing in more than one clone).

The main function exposed by this module is `prune_likely_homoplastic_intbcs`.
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


class PruneLikelyHomoplasticIntBCsReturnType(NamedTuple):
    """
    Object returned by the prune_likely_homoplastic_intbcs function.

    Attributes:
        intbc_graph: the new graph after pruning away the homoplastic intBCs
            found.
        homoplastic_intbcs_filtered: The list of intBCs discovered to be likely
            homoplastic.
        percent_homoplastic_intbcs_found: Percent of intBCs filtered this way.
        total_time: Total time taken.
    """

    intbc_graph: GraphType
    homoplastic_intbcs_filtered: List[IntBCType]
    percent_homoplastic_intbcs_found: float
    total_time: float


def prune_likely_homoplastic_intbcs(
    intbc_graph: GraphType,
    p_max_cutoff: float,
) -> PruneLikelyHomoplasticIntBCsReturnType:
    """
    Prune homoplastic intBCs.

    IntBCs with small p_max are likely homoplastic. Here p_max is:
    p_max(x) = max_y p(y | x) = max_y c(x, y) / c(x).
    The comparisson is with STRICT inqequality, therefore using
    `p_max_cutoff = 0.0` bypasses this filter.

    Args:
        intbc_graph: The graph to prune homoplastic intBCs from.
        p_max_cutoff: IntBCs with p_max lower than this will be pruned.
    """
    st = time.time()
    logger = logging.getLogger(__name__)
    logger.info("Going to prune_likely_homoplastic_intbcs ...")
    intbcs_to_prune = []
    edge_counts, intbc_counts = intbc_graph
    for intbc, counts in intbc_counts.items():
        # Decide if prune this intbc.
        p_max = max(
            [co_counts / counts for neighbor, co_counts in edge_counts[intbc].items()]
            + [0.0]  # In case it is an isolated intBC.
        )
        if p_max < p_max_cutoff:
            intbcs_to_prune.append(intbc)
    pruned_graph = prune_nodes_from_graph(
        graph=intbc_graph,
        nodes_to_prune=intbcs_to_prune,
    )
    percent_homoplastic_intbcs_found = len(intbcs_to_prune) / (
        len(intbc_graph[1]) + 1e-8
    )
    total_time = time.time() - st
    res = PruneLikelyHomoplasticIntBCsReturnType(
        intbc_graph=pruned_graph,
        homoplastic_intbcs_filtered=intbcs_to_prune,
        percent_homoplastic_intbcs_found=percent_homoplastic_intbcs_found,
        total_time=total_time,
    )
    edge_counts, node_counts = pruned_graph
    num_nodes = len(node_counts)
    num_edges = sum([len(x) for x in edge_counts.values()]) // 2
    logger.info(
        f"Pruned {len(intbcs_to_prune)} possibly homoplastic intBCs. Pruned IntBC graph has {num_nodes} nodes and {num_edges} edges.  Density level: {100.0 * num_edges / (num_nodes * (num_nodes - 1.0) / 2.0 + 1e-9):.2f}%"
    )
    logger.info(f"Done with prune_likely_homoplastic_intbcs! Total time: {total_time}")
    return res


def test_prune_likely_homoplastic_intbcs():
    graph = (
        {
            "1": {"2": 10, "3": 10, "4": 10},
            "2": {"1": 10},
            "3": {"1": 10},
            "4": {"1": 10},
        },
        {
            "1": 40,
            "2": 10,
            "3": 10,
            "4": 10,
        },
    )
    pruned_graph = prune_likely_homoplastic_intbcs(
        intbc_graph=graph,
        p_max_cutoff=0.0,
    ).intbc_graph
    assert pruned_graph == graph
    pruned_graph = prune_likely_homoplastic_intbcs(
        intbc_graph=graph,
        p_max_cutoff=0.24,
    ).intbc_graph
    assert pruned_graph == graph
    pruned_graph = prune_likely_homoplastic_intbcs(
        intbc_graph=graph,
        p_max_cutoff=0.26,
    ).intbc_graph
    assert pruned_graph == (
        {
            "2": {},
            "3": {},
            "4": {},
        },
        {
            "2": 10,
            "3": 10,
            "4": 10,
        },
    )
    pruned_graph = prune_likely_homoplastic_intbcs(
        intbc_graph=graph,
        p_max_cutoff=0.99,
    ).intbc_graph
    assert pruned_graph == (
        {
            "2": {},
            "3": {},
            "4": {},
        },
        {
            "2": 10,
            "3": 10,
            "4": 10,
        },
    )
    pruned_graph = prune_likely_homoplastic_intbcs(
        intbc_graph=graph,
        p_max_cutoff=1.1,
    ).intbc_graph
    assert pruned_graph == ({}, {})
