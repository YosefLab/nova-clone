"""
Module containing types involved in the clone calling algorithm.

Functions performing data type validation also reside here.
"""

from collections import defaultdict
from typing import Dict, List, NamedTuple, Set, Tuple, Union

from multiset import Multiset
import pandas as pd
from pandas.api.types import is_signed_integer_dtype, is_string_dtype

SampleNameType = str  # e.g. "LM0", "1A", "2B", etc., this is just the name of the sequencing run.  # noqa
IntBCTableType = (
    pd.DataFrame
)  # With 3 columns: "cellBC" (str), "intBC" (str), UMI_count (int)  # noqa

CellBCType = str
IntBCType = str
UMICountType = int
CloneType = str

Coreset = Union[Set, Multiset]

NodeCountsType = Dict[IntBCType, int]  # Nodes of the graph and their weights.
EdgeCountsType = Dict[
    IntBCType, Dict[IntBCType, int]
]  # Edges of the graph and their weights.
GraphType = Tuple[EdgeCountsType, NodeCountsType]


class CloneCallType(NamedTuple):
    clone_to_cells_mapping: Dict[CloneType, Set[CellBCType]]
    cell_to_clones_mapping: Dict[CellBCType, Set[CloneType]]
    clone_intbc_sets: Dict[CloneType, Set[IntBCType]]
    cell_intbc_sets: Dict[CellBCType, Set[IntBCType]]


def validate_intbc_table(intbc_table: pd.DataFrame) -> None:
    """
    Raise error if intbc_table format is wrong.

    The intBC table should have three columns:
    - "cellBC" of type string.
    - "intBC" of type string.
    - "UMI_count" of type int.
    """
    expected_columns = ["cellBC", "intBC", "UMI_count"]
    if list(intbc_table.columns) != expected_columns:
        raise ValueError(
            f"intbc_table should have columns '{expected_columns}', but has "
            f"columns '{intbc_table.columns}' instead."
        )
    if not is_string_dtype(intbc_table["cellBC"]):
        raise ValueError(
            f'"cellBC" column should be of the type string, but is of type: '
            f'{intbc_table["cellBC"].dtype}.'
        )
    if not is_string_dtype(intbc_table["intBC"]):
        raise ValueError(
            f'"intBC" column should be of the type string, but is of type: '
            f'{intbc_table["intBC"].dtype}.'
        )
    if not is_signed_integer_dtype(intbc_table["UMI_count"]):
        raise ValueError(
            f'"UMI_count" column should be of the type string, but is of type:'
            f' {intbc_table["UMI_count"].dtype}.'
        )


def validate_intbc_table_per_sample(
    intbc_table_per_sample: Dict[SampleNameType, IntBCTableType]
) -> None:
    """
    Validate that `intbc_table_per_sample` has the right type.
    """
    if type(intbc_table_per_sample) is not dict:
        raise ValueError(
            f"intbc_table_per_sample should be a dictionary mapping each "
            f"sample to its intBC table. You provided a {type(intbc_table_per_sample)}"
        )
    for intbc_table in intbc_table_per_sample.values():
        validate_intbc_table(intbc_table=intbc_table)


def prune_nodes_from_graph(
    graph: GraphType,
    nodes_to_prune: List[str],
) -> GraphType:
    """
    Prune nodes from the graph.

    Args:
        graph: Graph.
        nodes_to_prune: List of nodes to prune.
    """
    edge_counts, node_counts = graph
    nodes_set = set(node_counts.keys())
    nodes_to_prune_set = set(nodes_to_prune)
    if len(nodes_to_prune_set - nodes_set):
        raise ValueError("nodes_to_prune should be a subset of the nodes in the graph.")
    nodes_to_keep_set = nodes_set - nodes_to_prune_set
    pruned_edge_counts = {node: defaultdict(int) for node in nodes_to_keep_set}
    for node, adj_dict in edge_counts.items():
        if node in nodes_to_prune_set:
            continue  # We don't care about this node
        for neighbor, count in adj_dict.items():
            if neighbor in nodes_to_prune_set:
                continue  # We don't care about this node
            pruned_edge_counts[node][neighbor] = count
    pruned_edge_counts = {
        x: {y: z for (y, z) in d.items()} for (x, d) in pruned_edge_counts.items()
    }
    pruned_node_counts = {
        x: y for (x, y) in node_counts.items() if x in nodes_to_keep_set
    }
    pruned_graph = (pruned_edge_counts, pruned_node_counts)
    return pruned_graph


def test__prune_nodes_from_graph():
    graph = (
        {
            "1": {"2": 2, "3": 3},
            "2": {"1": 2},
            "3": {"1": 2},
        },
        {
            "1": 10,
            "2": 11,
            "3": 12,
        },
    )
    pruned_graph = prune_nodes_from_graph(
        graph=graph,
        nodes_to_prune=["1"],
    )
    assert pruned_graph == ({"2": {}, "3": {}}, {"2": 11, "3": 12})
