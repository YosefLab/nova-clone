"""
This module implements filtering of intBCs which are sequencing errors.

The main function exposed by this module is `prune_likely_sequencing_errors`.
"""

import logging
import sys
import time

import editdistance

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


class PruneLikelySequencingErrorsReturnType(NamedTuple):
    """
    Object returned by the prune_likely_sequencing_errors function.

    Attributes:
        intbc_graph: the new graph after pruning away the sequencing errors
            found.
        sequencing_errors_filtered: The list of intBCs discovered to be likely
            sequencing errors.
        percent_sequencing_errors_found: What percent of the intBCs were found
            to be sequencing errors.
        total_time: Total time taken.
        sequencing_errors_and_originals: List of
            ((sequencing error found, counts), (original sequenct, counts),
            co-counts)
    """

    intbc_graph: GraphType
    sequencing_errors_filtered: List[IntBCType]
    percent_sequencing_errors_found: float
    total_time: float
    sequencing_errors_and_originals: List[
        Tuple[Tuple[IntBCType, int], Tuple[IntBCType, int], int]
    ]


def prune_likely_sequencing_errors(
    intbc_graph: GraphType,
    edit_distance_cutoff: int,
    co_occurrence_cutoff: float,
) -> PruneLikelySequencingErrorsReturnType:
    """
    Prunes likely sequencing errors.

    Sequencing errors x typically satisfy the following property: There exists
    a neighboring intBC y such that p(y|x) is close to 1; x in close to y in
    edit distance; and x is less frequent than y.

    Args:
        edit_distance_cutoff: How close y needs to be to x to prune x (the
            cutoff is inclusive). The larger that `edit_distance_cutoff` is,
            the more intBCs will be pruned.
        co_occurrence_cutoff: How large p(y|x) needs to be to prune x. The
            smaller that `co_occurrence_cutoff`, the more intBCs will be pruned.
            The cutoff is inclusive.

    Returns:
        pruned_intbc_graph: The pruned intBC graph.
    """
    st = time.time()
    logger = logging.getLogger(__name__)
    logger.info("Going to prune_likely_sequencing_errors ...")
    intbcs_to_prune = []
    sequencing_errors_and_originals = []
    edge_counts, intbc_counts = intbc_graph
    for intbc in intbc_counts:
        intbc_count = intbc_counts[intbc]
        # Decide if intbc is a likely sequencing error.
        for neighbor, co_counts in edge_counts[intbc].items():
            if intbc_counts[neighbor] > intbc_count:
                p_yx = co_counts / intbc_count
                if p_yx >= co_occurrence_cutoff:
                    d = editdistance.eval(intbc, neighbor)
                    if d <= edit_distance_cutoff:
                        intbcs_to_prune.append(intbc)
                        sequencing_errors_and_originals.append(
                            (
                                (intbc, intbc_counts[intbc]),
                                (neighbor, intbc_counts[neighbor]),
                                co_counts,
                                d,
                            )
                        )
                        break
    percent_sequencing_errors_found = len(intbcs_to_prune) / (
        len(intbc_graph[1]) + 1e-8
    )
    pruned_graph = prune_nodes_from_graph(
        graph=intbc_graph,
        nodes_to_prune=intbcs_to_prune,
    )
    total_time = time.time() - st
    res = PruneLikelySequencingErrorsReturnType(
        intbc_graph=pruned_graph,
        sequencing_errors_filtered=sorted(intbcs_to_prune),
        percent_sequencing_errors_found=percent_sequencing_errors_found,
        total_time=total_time,
        sequencing_errors_and_originals=sequencing_errors_and_originals,
    )
    edge_counts, node_counts = pruned_graph
    num_nodes = len(node_counts)
    num_edges = sum([len(x) for x in edge_counts.values()]) // 2
    logger.info(
        f"Pruned {len(intbcs_to_prune)} likely sequencing errors. Pruned IntBC graph has {num_nodes} nodes and {num_edges} edges.  Density level: {100.0 * num_edges / (num_nodes * (num_nodes - 1.0) / 2.0 + 1e-9):.2f}%"
    )
    logger.info(f"Done with prune_likely_sequencing_errors! Total time: {total_time}")
    return res


def test_prune_likely_sequencing_errors():
    graph = (
        {
            "AAAA": {"BBBB": 100, "AAA": 90, "AA": 1},
            "BBBB": {"AAAA": 100},
            "AAA": {"AAAA": 90},
            "AA": {"AAAA": 1},
        },
        {
            "AAAA": 1001,
            "BBBB": 1000,
            "AAA": 100,
            "AA": 10,
        },
    )
    pruned_graph = prune_likely_sequencing_errors(
        intbc_graph=graph,
        edit_distance_cutoff=4,
        co_occurrence_cutoff=0.99,
    ).intbc_graph
    assert pruned_graph == graph
    pruned_graph = prune_likely_sequencing_errors(
        intbc_graph=graph,
        edit_distance_cutoff=1,
        co_occurrence_cutoff=0.85,
    ).intbc_graph
    print(f"pruned_graph = {pruned_graph}")
    assert pruned_graph == (
        {
            "AAAA": {"BBBB": 100, "AA": 1},
            "BBBB": {"AAAA": 100},
            "AA": {"AAAA": 1},
        },
        {
            "AAAA": 1001,
            "BBBB": 1000,
            "AA": 10,
        },
    )
    pruned_graph = prune_likely_sequencing_errors(
        intbc_graph=graph,
        edit_distance_cutoff=2,
        co_occurrence_cutoff=0.05,
    ).intbc_graph
    assert pruned_graph == (
        {
            "AAAA": {"BBBB": 100},
            "BBBB": {"AAAA": 100},
        },
        {
            "AAAA": 1001,
            "BBBB": 1000,
        },
    )
    prune_likely_sequencing_errors_return = prune_likely_sequencing_errors(
        intbc_graph=graph,
        edit_distance_cutoff=4,
        co_occurrence_cutoff=0.05,
    )
    pruned_graph = prune_likely_sequencing_errors_return.intbc_graph
    assert pruned_graph == (
        {
            "AAAA": {},
        },
        {
            "AAAA": 1001,
        },
    )
    assert prune_likely_sequencing_errors_return.sequencing_errors_filtered == [
        "AA",
        "AAA",
        "BBBB",
    ]
