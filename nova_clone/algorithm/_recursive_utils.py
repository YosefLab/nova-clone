from fsspec.core import infer_compression
from ._types import *


def _is_connected(intbc_graph: GraphType) -> bool:
    """
    Returns True if the graph is connected, False otherwise.
    """
    if len(intbc_graph[1]) == 0:
        return True
    edge_counts, node_counts = intbc_graph
    visited = {node: False for node in node_counts}

    def _dfs(node):
        visited[node] = True
        for neighbor in edge_counts[node]:
            if not visited[neighbor]:
                _dfs(neighbor)

    _dfs(next(iter(node_counts)))
    return all(visited.values())


def test__is_connected():
    intbc_graph = (
        {
            "CCC": {"AAA": 5, "BBB": 5},
            "AAA": {"CCC": 5, "BBB": 10, "DDD": 5},
            "BBB": {"CCC": 5, "AAA": 10, "DDD": 5},
            "DDD": {"AAA": 5, "BBB": 5},
        },
        {"CCC": 5, "AAA": 10, "BBB": 10, "DDD": 5},
    )
    assert _is_connected(intbc_graph)


def test__empty_is_connected():
    intbc_graph = ({}, {})
    assert _is_connected(intbc_graph)


def test__is_connected_2():
    intbc_graph = ({"CCC": {}, "DDD": {}}, {"CCC": 5, "DDD": 5})
    assert not _is_connected(intbc_graph)


def _pop_highest_frequency_vertex(
    intbc_graph: GraphType,
) -> Tuple[IntBCType, GraphType]:
    """
    Pop the highest frequency vertex from the graph.
    """
    edge_counts, node_counts = intbc_graph
    if not node_counts:
        raise ValueError("The graph is empty.")
    highest_frequency_vertex = max(node_counts, key=node_counts.get)
    new_node_counts = {
        node: count
        for node, count in node_counts.items()
        if node != highest_frequency_vertex
    }
    new_edge_counts = {
        node: {
            neighbor: count
            for neighbor, count in neighbors.items()
            if neighbor != highest_frequency_vertex
        }
        for node, neighbors in edge_counts.items()
        if node != highest_frequency_vertex
    }
    return highest_frequency_vertex, (new_edge_counts, new_node_counts)


def test__pop_highest_frequency_vertex():
    intbc_graph = (
        {
            "CCC": {"AAA": 5, "BBB": 5},
            "AAA": {"CCC": 5, "BBB": 10, "DDD": 5},
            "BBB": {"CCC": 5, "AAA": 10, "DDD": 5},
            "DDD": {"AAA": 5, "BBB": 5},
        },
        {"CCC": 5, "AAA": 11, "BBB": 10, "DDD": 6},
    )
    intbc, new_intbc_graph = _pop_highest_frequency_vertex(intbc_graph)
    assert new_intbc_graph == (
        {"CCC": {"BBB": 5}, "BBB": {"CCC": 5, "DDD": 5}, "DDD": {"BBB": 5}},
        {"CCC": 5, "BBB": 10, "DDD": 6},
    )
    assert intbc == "AAA"
    intbc, new_intbc_graph = _pop_highest_frequency_vertex(new_intbc_graph)
    assert new_intbc_graph == ({"CCC": {}, "DDD": {}}, {"CCC": 5, "DDD": 6})
    assert intbc == "BBB"
    intbc, new_intbc_graph = _pop_highest_frequency_vertex(new_intbc_graph)
    assert new_intbc_graph == ({"CCC": {}}, {"CCC": 5})
    assert intbc == "DDD"
    intbc, new_intbc_graph = _pop_highest_frequency_vertex(new_intbc_graph)
    assert new_intbc_graph == ({}, {})
    assert intbc == "CCC"


def test__pop_highest_frequency_vertex_raises():
    intbc_graph = ({}, {})
    try:
        _pop_highest_frequency_vertex(intbc_graph)
        assert False
    except ValueError:
        assert True


def subset_intbc_graph(
    intbc_graph: GraphType,
    nodes_to_keep: list[IntBCType],
) -> GraphType:
    edge_counts, node_counts = intbc_graph
    for node in nodes_to_keep:
        if node not in node_counts:
            raise ValueError(
                f"You tried to subset node {node} which is not in the intBC graph."
            )
    subsetted_edge_counts = {
        node: {
            other_node: edge_counts[node][other_node]
            for other_node in set(nodes_to_keep).intersection(set(edge_counts[node].keys()))
        } for node in nodes_to_keep
    }
    subsetted_node_counts = {
        node: node_counts[node]
        for node in nodes_to_keep
    }
    res = (subsetted_edge_counts, subsetted_node_counts)
    return res


def _get_test_intbc_graph() -> GraphType:
    intbc_graph = (
        {
            "1": {
                "2": 3,
                "3": 4,
            },
            "2": {
                "1": 3,
                "3": 5,
                "4": 6,
            },
            "3": {
                "1": 4,
                "2": 5,
                "4": 7,
            },
            "4": {
                "2": 6,
                "3": 7,
            }
        },
        {
            "1": 11,
            "2": 12,
            "3": 13,
            "4": 10,
        }
    )
    return intbc_graph


def test_subset_intbc_graph():
    intbc_graph = _get_test_intbc_graph()
    subsetted_intbc_graph = subset_intbc_graph(
        intbc_graph=intbc_graph,
        nodes_to_keep=["1", "2", "4"],
    )
    print(subsetted_intbc_graph)
    assert subsetted_intbc_graph == (
        {
            "1": {
                "2": 3,
            },
            "2": {
                "1": 3,
                "4": 6,
            },
            "4": {
                "2": 6,
            }
        },
        {
            "1": 11,
            "2": 12,
            "4": 10,
        }
    )


def infer_progenitor_intbc_set_by_popping(
    subsetted_intbc_graph: GraphType,
) -> Set:
    inferred_progenitor_intbc_set = set()
    while _is_connected(subsetted_intbc_graph):
        if len(subsetted_intbc_graph[1]) == 0:
            # We have popped all the intBCs. I.e. there was no subclonal structure left.
            break
        intbc, subsetted_intbc_graph = _pop_highest_frequency_vertex(
            subsetted_intbc_graph
        )
        inferred_progenitor_intbc_set.add(intbc)
    return inferred_progenitor_intbc_set


def test_infer_progenitor_intbc_set_by_popping():
    intbc_graph = _get_test_intbc_graph()
    inferred_progenitor_intbc_set = infer_progenitor_intbc_set_by_popping(
        subsetted_intbc_graph=intbc_graph,
    )
    assert(inferred_progenitor_intbc_set == set(["2", "3"]))


def test_infer_progenitor_intbc_set_by_popping_empty():
    intbc_graph = ({}, {})
    inferred_progenitor_intbc_set = infer_progenitor_intbc_set_by_popping(
        subsetted_intbc_graph=intbc_graph,
    )
    assert(inferred_progenitor_intbc_set == set())
