from nova_clone.algorithm._build_intbc_graph import *
from nova_clone.algorithm._call_clones_from_coresets import *
from nova_clone.algorithm._clone_calling_algorithm import *
from nova_clone.algorithm._clone_call_ops import *
from nova_clone.algorithm._compute_cellbc_vs_coreset_overlap_matrix import *
from nova_clone.algorithm._compute_coreset_co_assignment_matrix import *
from nova_clone.algorithm._find_maximal_coresets import *
from nova_clone.algorithm._keep_reliable_edges import *
from nova_clone.algorithm._make_coresets_disjoint import *
from nova_clone.algorithm._prune_coresets import *
from nova_clone.algorithm._prune_custom_intbcs import *
from nova_clone.algorithm._prune_likely_homoplastic_intbcs import *
from nova_clone.algorithm._prune_likely_sequencing_errors import *
from nova_clone.algorithm._prune_low_cellbc_count_intbcs import *
from nova_clone.algorithm._prune_low_umi_count_intbcs import *
from nova_clone.algorithm._types import *
from nova_clone.algorithm.cassiopeia._call_lineage_groups import *
from nova_clone.algorithm._recursive_utils import *
from nova_clone.algorithm._clone_calling_algorithm import _check_are_clones


def test_recursive_clone_calling_algorithm_empty_progenitor_set(monkeypatch):
    """
    If the clone's coreset intBC graph is already disconnected, then
    `infer_progenitor_intbc_set_by_popping` returns an empty set. Recursing
    with an empty progenitor set removed would call the algorithm on the exact
    same input, leading to infinite recursion. Instead, the clone should be
    retained without subclonal structure.
    """
    import nova_clone.algorithm._clone_calling_algorithm as clone_calling_module

    num_calls = [0]

    def _infer_empty_progenitor_intbc_set(subsetted_intbc_graph):
        num_calls[0] += 1
        return set()

    monkeypatch.setattr(
        clone_calling_module,
        "infer_progenitor_intbc_set_by_popping",
        _infer_empty_progenitor_intbc_set,
    )

    intbc_table = pd.DataFrame(
        data=[
            ["cellBC_1", "AAA", 40],
            ["cellBC_1", "BBB", 40],
            ["cellBC_1", "CCC", 40],
            ["cellBC_2", "AAA", 40],
            ["cellBC_2", "BBB", 40],
            ["cellBC_2", "CCC", 40],
            ["cellBC_3", "AAA", 40],
            ["cellBC_3", "BBB", 40],
            ["cellBC_3", "DDD", 40],
            ["cellBC_4", "AAA", 40],
            ["cellBC_4", "BBB", 40],
            ["cellBC_4", "DDD", 40],
        ],
        columns=["cellBC", "intBC", "UMI_count"],
    )

    clone_call_namedtuple = clone_calling_module.clone_calling_algorithm(
        intbc_table_per_sample={"sample1": intbc_table},
        min_intbc_cellbc_count=1,
        edit_distance_cutoff=0,
        min_cells_assigned=1,
        recursive=True,
        recursive_min_cells_assigned=2,
    )

    assert num_calls[0] == 1
    assert clone_call_namedtuple.recursive_clone_call == {"clone_0": None}
    assert clone_call_namedtuple.recursive_clone_call_integration_hierarchy == {
        "clone_0": ({"AAA", "BBB", "CCC", "DDD"}, None),
    }


def test_recursive_clone_calling_algorithm_cellbc_collision_across_samples():
    """
    CellBCs are only unique within a sample. Here, both samples use the same
    cellBCs but for cells of different clones. The recursive call on a clone
    must only include that clone's cells, and not cells from other samples
    that happen to share a cellBC with them.
    """
    s1_rows = []
    for cellbc, subclone_intbc in [
        ("cellBC_1", "CCC"),
        ("cellBC_2", "CCC"),
        ("cellBC_3", "DDD"),
        ("cellBC_4", "DDD"),
    ]:
        for intbc in ["AAA", "BBB", subclone_intbc]:
            s1_rows.append([cellbc, intbc, 40])
    s2_rows = [
        [cellbc, intbc, 40]
        for cellbc in ["cellBC_1", "cellBC_2", "cellBC_3", "cellBC_4"]
        for intbc in ["EEE", "FFF", "GGG"]
    ]
    columns = ["cellBC", "intBC", "UMI_count"]

    clone_call_namedtuple = clone_calling_algorithm(
        intbc_table_per_sample={
            "s1": pd.DataFrame(data=s1_rows, columns=columns),
            "s2": pd.DataFrame(data=s2_rows, columns=columns),
        },
        min_intbc_cellbc_count=1,
        edit_distance_cutoff=0,
        min_cells_assigned=1,
        recursive=True,
        recursive_min_cells_assigned=2,
    )
    clone_call = clone_call_namedtuple.clone_call
    _check_are_clones(
        clone_call=clone_call,
        expected_clones=[
            ["s1_cellBC_1", "s1_cellBC_2", "s1_cellBC_3", "s1_cellBC_4"],
            ["s2_cellBC_1", "s2_cellBC_2", "s2_cellBC_3", "s2_cellBC_4"],
        ],
    )
    s1_clone = next(iter(clone_call.cell_to_clones_mapping["s1_cellBC_1"]))
    _check_are_clones(
        clone_call=clone_call_namedtuple.recursive_clone_call[s1_clone].clone_call,
        expected_clones=[
            ["s1_cellBC_1", "s1_cellBC_2"],
            ["s1_cellBC_3", "s1_cellBC_4"],
        ],
    )
    assert clone_call_namedtuple.recursive_clone_call_integration_hierarchy[s1_clone] == (
        {"AAA", "BBB"},
        {
            "clone_0": ({"CCC"}, None),
            "clone_1": ({"DDD"}, None),
        },
    )
