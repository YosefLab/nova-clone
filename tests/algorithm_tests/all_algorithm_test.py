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
from nova_clone.algorithm._clone_calling_algorithm import _check_are_clones


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


def test_recursive_clone_call_when_clone_intbc_graph_starts_disconnected(monkeypatch):
    """
    Regression test for an infinite recursion.

    W1 and W2 are progenitor intBCs that are only detected in 15 of the 400
    cells of each subclone (e.g. due to epigenetic silencing). This suffices
    for them to bridge subclones L and S in the coreset, but they make up less
    than `min_intbc_fraction` of the clone's cells, so they are dropped from
    the clone's intBC set in phase 2. As a result, the intBC graph built for
    the clone in the recursive step is disconnected before any intBC is
    popped, i.e. the inferred progenitor intBC set is empty. Recursing with an
    empty progenitor intBC set removed calls the algorithm on the same input
    again, which recurses forever.
    """
    import nova_clone.algorithm._clone_calling_algorithm as clone_calling_module

    original_clone_calling_algorithm = clone_calling_module.clone_calling_algorithm
    max_depth = 10
    depth = [0]

    def depth_limited_clone_calling_algorithm(*args, **kwargs):
        depth[0] += 1
        try:
            if depth[0] > max_depth:
                raise RecursionError(
                    f"Recursive clone call exceeded depth {max_depth}."
                )
            return original_clone_calling_algorithm(*args, **kwargs)
        finally:
            depth[0] -= 1

    monkeypatch.setattr(
        clone_calling_module,
        "clone_calling_algorithm",
        depth_limited_clone_calling_algorithm,
    )

    rows = []
    for prefix, subclone_intbcs in [("l", ["L1", "L2", "L3"]), ("s", ["S1", "S2", "S3"])]:
        for i in range(400):
            intbcs = subclone_intbcs + (["W1", "W2"] if i < 15 else [])
            for intbc in intbcs:
                rows.append([f"{prefix}{i}", intbc, 40])
    intbc_table = pd.DataFrame(data=rows, columns=["cellBC", "intBC", "UMI_count"])

    res = clone_calling_module.clone_calling_algorithm(
        intbc_table_per_sample={"sample1": intbc_table},
        edit_distance_cutoff=0,
        recursive=True,
    )
    assert list(res.clone_call.clone_intbc_sets.values()) == [
        {"L1", "L2", "L3", "S1", "S2", "S3"}
    ]
    assert list(res.recursive_clone_call_integration_hierarchy.keys()) == ["clone_0"]
