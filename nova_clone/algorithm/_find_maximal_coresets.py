"""
This module implements finding of coresets int the intBC graph.

A "coreset" is a set of intBCs which exist in one and only one clone. Thus,
coresets are signatures left by the clones in the intBC graph. To find
these signatures, we use the following characterization of coresets:

Suppose intbc_1, ..., intbc_k are intBCs all of which co-occurr in _some_
clone. Then, if k is large enough (typically k>=2), then they must co-occurr
in the *same* clone, i.e. {intbc_1, ..., intbc_k} is a coreset. This is because
homoplasy only cause clones to share at most one intBC (with high probability).
Moreover, one can show that if A and B are two coresets that intersect in at
least a certain number of intBCs (typically k >= 2), then they represent the
same clone and their union is a coreset for that clone.

The main function exposed by this module is `keep_reliable_edges`, and it finds
maximal coresets in the intBC graph using the above two properties.
"""

import logging
import sys
import time
from typing import Set

from multiset import Multiset
import networkx as nx
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


class FindMaximalCoresetsReturnType(NamedTuple):
    """
    Object returned by the `find_maximal_coresets` function.

    Attributes:
        maximal_coresets: the maximal coresets found.
        total_time: Total time taken.
    """

    maximal_coresets: List[Set[str]]
    total_time: float


def maximum_cliques_containing_node(
    graph: GraphType,
    node: str,
) -> List[List[str]]:
    """
    Find all maximum cliques containing the given node.

    The `graph` must be a digraph.
    """
    # We need to create a nx.Graph to run the clique finding method.
    edge_counts, node_counts = graph
    neighboring_nodes = sorted(list(edge_counts[node].keys()))
    if len(neighboring_nodes) == 0:
        # Vertex is isolated
        return [tuple([node])]
    G = nx.Graph()
    G.add_nodes_from(neighboring_nodes)
    edges = []
    # Check DiGraph condition for `node` node.
    for x in neighboring_nodes:
        if x != node:
            assert x in edge_counts[node]  # Check DiGraph condition.
            assert node in edge_counts[x]  # Check DiGraph condition.
    for x in neighboring_nodes:
        for y in neighboring_nodes:
            if y in edge_counts[x]:
                assert x in edge_counts[y]  # Check DiGraph condition.
                if x < y:
                    edges.append((x, y))
    G.add_edges_from(edges)
    maximal_cliques = list(nx.find_cliques(G))
    max_clique_size = max([len(x) for x in maximal_cliques])
    maximum_cliques = [x for x in maximal_cliques if len(x) == max_clique_size]
    # We need to add back in the original node.
    # We sort for deterministic output.
    maximum_cliques = sorted([tuple(sorted([node] + x)) for x in maximum_cliques])
    return maximum_cliques


def test_maximum_cliques_containing_node():
    graph = (
        {
            "1": {"2": 10, "6": 10},
            "2": {"1": 10, "3": 10, "4": 10, "5": 10, "6": 10},
            "3": {"2": 10, "4": 10, "5": 10},
            "4": {"2": 10, "3": 10, "5": 10},
            "5": {"2": 10, "3": 10, "4": 10},
            "6": {"1": 10, "2": 10},
        },
        {
            "1": 100,
            "2": 100,
            "3": 100,
            "4": 100,
            "5": 100,
            "6": 100,
        },
    )
    cliques = maximum_cliques_containing_node(
        graph=graph,
        node="2",
    )
    assert cliques == [("2", "3", "4", "5")]
    cliques = maximum_cliques_containing_node(
        graph=graph,
        node="1",
    )
    assert cliques == [("1", "2", "6")]


def find_maximum_cliques_over_all_nodes(
    graph: GraphType,
) -> List[Set[str]]:
    """
    List of all maximum cliques per node. Repetitions excluded.

    If a node has more than one maximum clique, the lexicographically smallest
    one is chosen.
    """
    maximum_clique_for_node = {}
    nodes = list(graph[1].keys())
    cliques_containing_node = {node: set() for node in nodes}
    nodes_with_more_than_one_maximum_clique = {}
    maximum_clique_counts = defaultdict(int)

    for node in tqdm.tqdm(nodes):
        cliques = maximum_cliques_containing_node(graph, node)
        if len(cliques) > 1:
            nodes_with_more_than_one_maximum_clique[node] = cliques
        clique = cliques[0]
        maximum_clique_for_node[node] = clique
        for node in clique:
            cliques_containing_node[node].add(clique)
        maximum_clique_counts[clique] += 1

    # Check that any two maximum cliques are non-included. We do this by
    # iterating over each node, and looking at cliques covering this node.
    for node, maximum_cliques in cliques_containing_node.items():
        maximum_cliques = list(maximum_cliques)
        for i, c1 in enumerate(maximum_cliques):
            for j in range(len(maximum_cliques)):
                if j == i:
                    continue
                c2 = maximum_cliques[j]
                if set(c1).issubset(c2):
                    raise Exception(f"Maximum clique {c1} is a subset of {c2}!")

    cliques = list(maximum_clique_counts.keys())
    cliques = [set(clique) for clique in cliques]
    return cliques


def test_find_maximum_cliques_over_all_nodes():
    graph = (
        {
            "1": {"2": 10, "6": 10},
            "2": {"1": 10, "3": 10, "4": 10, "5": 10, "6": 10},
            "3": {"2": 10, "4": 10, "5": 10},
            "4": {"2": 10, "3": 10, "5": 10},
            "5": {"2": 10, "3": 10, "4": 10},
            "6": {"1": 10, "2": 10},
        },
        {
            "1": 100,
            "2": 100,
            "3": 100,
            "4": 100,
            "5": 100,
            "6": 100,
        },
    )
    cliques = find_maximum_cliques_over_all_nodes(
        graph=graph,
    )
    assert {"1", "2", "6"} in cliques
    assert {"2", "3", "4", "5"} in cliques
    assert len(cliques) == 2


def merge_coresets(
    elements: List[str],
    coresets: List[Set[str]],
    min_overlap_for_merging_coresets: int,
) -> List[Set[str]]:
    """
    Given a list of known coresets, return the maximal coresets.

    Maximal coresets are obtained by repeatedly merging any pair of cliques
    that intersect in at least `min_overlap_for_merging_coresets` elements.
    """
    coresets_original = coresets
    coresets = sorted([tuple(sorted(list(coreset))) for coreset in coresets])
    # print(f"Computing elements ...")  # Too slow, will just require it as an
    # argument.
    coresets_containing_elements = {element: set() for element in elements}
    for coreset in coresets:
        for element in coreset:
            coresets_containing_elements[element].add(coreset)

    # print("Computing coreset_overlap_graph__edge_set ...")
    coreset_overlap_graph__edge_set = set()
    # We iterate over elements to find coresets with high intersection to merge
    for element, coresets_for_element in coresets_containing_elements.items():
        coresets_for_element_list = list(coresets_for_element)
        for i, c1 in enumerate(coresets_for_element_list):
            for j in range(i + 1, len(coresets_for_element_list)):
                c2 = coresets_for_element_list[j]
                if (
                    len(set(c1).intersection(set(c2)))
                    >= min_overlap_for_merging_coresets
                ):
                    edge = (min(c1, c2), max(c1, c2))
                    coreset_overlap_graph__edge_set.add(edge)

    # print("Creating coreset_merge_graph ...")
    # print(f"coreset_merge_graph will have
    # {len(coreset_overlap_graph__edge_set)} edges")
    coreset_merge_graph = {coreset: [] for coreset in coresets}
    for c1, c2 in coreset_overlap_graph__edge_set:
        coreset_merge_graph[c1].append(c2)
        coreset_merge_graph[c2].append(c1)

    # print("Merging cliques ...")
    visited = {coreset: False for coreset in coresets}

    coresets_to_merge = []

    def dfs(curr_coreset):
        visited[curr_coreset] = True
        coresets_to_merge.append(curr_coreset)
        for neighboring_coreset in coreset_merge_graph[curr_coreset]:
            if not visited[neighboring_coreset]:
                dfs(neighboring_coreset)

    final_coresets = []

    for coreset in coresets:
        if not visited[coreset]:
            coresets_to_merge = []
            dfs(coreset)
            final_coresets.append(tuple(sorted(list(set(sum(coresets_to_merge, ()))))))
    final_coresets = sorted(final_coresets)
    final_coresets = [set(coreset) for coreset in final_coresets]

    # Check if there is more reducing to do (if so, we do so recursively)
    if coresets_original != final_coresets:
        # print("Need to recurse!")
        return merge_coresets(
            elements=elements,
            coresets=final_coresets,
            min_overlap_for_merging_coresets=min_overlap_for_merging_coresets,
        )
    else:
        return final_coresets


def test_merge_coresets():
    coresets_test = [
        {"a", "b", "c"},
        {"b", "c", "d"},
        {"a", "d", "e"},
        {
            "a",
        },
    ]
    res = merge_coresets(
        elements=["a", "b", "c", "d", "e"],
        coresets=coresets_test,
        min_overlap_for_merging_coresets=2,
    )
    assert res == [{"a"}, {"a", "b", "c", "d", "e"}]
    coresets_test = [
        {"a", "b", "c"},
        {"b", "c", "d"},
        {"d", "e", "f"},
    ]
    res = merge_coresets(
        elements=["a", "b", "c", "d", "e", "f"],
        coresets=coresets_test,
        min_overlap_for_merging_coresets=3,
    )
    assert res == coresets_test
    res = merge_coresets(
        elements=["a", "b", "c", "d", "e", "f"],
        coresets=coresets_test,
        min_overlap_for_merging_coresets=2,
    )
    assert res == [{"a", "b", "c", "d"}, {"d", "e", "f"}]
    res = merge_coresets(
        elements=["a", "b", "c", "d", "e", "f"],
        coresets=coresets_test,
        min_overlap_for_merging_coresets=1,
    )
    assert res == [{"a", "b", "c", "d", "e", "f"}]


def duplicate_single_intbcs(maximal_coresets: List[set]) -> List[Multiset]:
    """
    Elevates single-intBC coresets into multisets which repeat the intBC twice,
    thereby giving it higher statistical support in all subsequent steps of the algorithm.
    """
    res = []
    for coreset in maximal_coresets:
        if len(coreset) == 1:
            res.append(Multiset(coreset) + Multiset(coreset))
        else:
            res.append(coreset)
    return res


def test_duplicate_single_intbcs():
    res = duplicate_single_intbcs([set(["a"]), set(["b", "c"])])
    assert(
        res == [Multiset(["a", "a"]), {"b", "c"}]
    )


def find_maximal_coresets(
    graph: GraphType,
    min_coreset_size: int,
    min_overlap_for_merging_coresets: int,
    recover_single_intbc_clones: bool = False,
) -> FindMaximalCoresetsReturnType:
    """
    Find maximal coresets.

    Coresets are defined as follows:
    - A k-clique for any k >= `min_coreset_size` is a coreset.
    - If two coresets intersect in at least `min_overlap_for_merging_coresets`
        nodes, then their union is a coreset.

    A maximal coreset is a coreset that is not strictly contained in any larger
    coreset, using the above closure rules.

    If `recover_single_intbc_clones`, then maximal coresets with just 1 intBC are elevated to
    a multiset with 2 intBCs (the same intBC twice).
    """
    if recover_single_intbc_clones and min_coreset_size != 1:
        raise ValueError(
            f"You are trying to recover_single_intbc_clones; this requires `min_coreset_size=1`."
        )
    st = time.time()
    logger = logging.getLogger(__name__)
    logger.info("Going to find_maximal_coresets ...")

    coresets = find_maximum_cliques_over_all_nodes(graph)
    # Throw away singletons, since we need a k-clique with
    # k >= min_coreset_size for a coreset.
    coresets = [coreset for coreset in coresets if len(coreset) >= min_coreset_size]

    maximal_coresets = merge_coresets(
        elements=graph[0].keys(),
        coresets=coresets,
        min_overlap_for_merging_coresets=min_overlap_for_merging_coresets,
    )
    if recover_single_intbc_clones:
        maximal_coresets = duplicate_single_intbcs(maximal_coresets)

    total_time = time.time() - st

    res = FindMaximalCoresetsReturnType(
        maximal_coresets=maximal_coresets,
        total_time=total_time,
    )
    logger.info(
        f"Found {len(maximal_coresets)} coresets, with average size {np.mean([len(x) for x in maximal_coresets]):.2f}"
    )
    logger.info(f"Done with find_maximal_coresets! Total time: {total_time}")

    return res


def test_find_maximal_coresets():
    graph = (
        {
            "1": {"2": 10, "6": 10},
            "2": {"1": 10, "3": 10, "4": 10, "5": 10, "6": 10},
            "3": {"2": 10, "4": 10, "5": 10},
            "4": {"2": 10, "3": 10, "5": 10},
            "5": {"2": 10, "3": 10, "4": 10},
            "6": {"1": 10, "2": 10},
        },
        {
            "1": 100,
            "2": 100,
            "3": 100,
            "4": 100,
            "5": 100,
            "6": 100,
        },
    )
    coresets = find_maximal_coresets(
        graph=graph,
        min_coreset_size=3,
        min_overlap_for_merging_coresets=2,
    ).maximal_coresets
    assert coresets == [{"1", "2", "6"}, {"2", "3", "4", "5"}]
    coresets = find_maximal_coresets(
        graph=graph,
        min_coreset_size=4,
        min_overlap_for_merging_coresets=2,
    ).maximal_coresets
    assert coresets == [{"2", "3", "4", "5"}]
    coresets = find_maximal_coresets(
        graph=graph,
        min_coreset_size=3,
        min_overlap_for_merging_coresets=1,
    ).maximal_coresets
    assert coresets == [{"1", "2", "3", "4", "5", "6"}]
