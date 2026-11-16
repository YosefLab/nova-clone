"""
Module for calling clones once coresets have been identified.

The main function exposed by this module is `call_clones_from_coresets`.
"""

import time

import editdistance
import numpy as np
import pandas as pd

from ._compute_cellbc_vs_coreset_overlap_matrix import (
    compute_cellbc_vs_coreset_overlap_matrix,
)
from ._logging import tqdm
from ._prune_low_umi_count_intbcs import prune_low_umi_count_intbcs
from ._types import *
from ._utils import _cell_intbc_sets_from_intbc_tables


class CallClonesFromCoresetsReturnType(NamedTuple):
    """
    Object returned by the `call_clones_from_coresets` function.

    Attributes:
        clone_call: The clone call.
        total_time: Total time taken.
    """

    clone_call: CloneCallType
    total_time: float


def call_clones_from_coresets(
    intbc_table_per_sample: Dict[SampleNameType, IntBCTableType],
    min_intbc_umi_count: int,
    coresets: List[Coreset[str]],
    min_overlap_for_assigning_cellbc: int,
    min_intbc_counts_for_adding_intbc_to_clone_intbc_set: int,
    min_intbc_fraction_for_adding_to_clone_intbc_set: float,
    edit_distance_cutoff: int,
    custom_intbc_filtering_function: str,
    validate: bool = True,
) -> CallClonesFromCoresetsReturnType:
    """
    Call clones given the coresets characterizing each clone.

    Set ``validate=False`` when the caller has already validated the tables.
    """
    st = time.time()

    if validate:
        validate_intbc_table_per_sample(intbc_table_per_sample=intbc_table_per_sample)

    # Remember pre-prune cellBCs so cells that lose all intBCs to UMI filtering
    # still appear in the output mappings. Only the keys are needed here - avoid
    # building full cell <-> intBC sets twice.
    all_cellbc = {
        f"{sample_name}_{cellbc}"
        for sample_name, intbc_table in intbc_table_per_sample.items()
        for cellbc in intbc_table["cellBC"].unique()
    }
    intbc_table_per_sample = prune_low_umi_count_intbcs(
        intbc_table_per_sample=intbc_table_per_sample,
        min_intbc_umi_count=min_intbc_umi_count,
        validate=False,
    ).intbc_table_per_sample
    cell_intbc_sets = _cell_intbc_sets_from_intbc_tables(
        intbc_table_per_sample=intbc_table_per_sample,
    )

    cellbc_vs_coreset_overlap_matrix = compute_cellbc_vs_coreset_overlap_matrix(  # We MUST prune low-count UMIs before doing this! Else the background RNA will mess it up. In constrast, it's OK to not have discarded sequencing errors (bc they are not in the coresets anyway) nor homoplastic intBCs (because they are also not in the coresets anyway). Sequencing errors and homoplastic intBCs will be recovered de-novo.
        cellbc_intbc_sets=cell_intbc_sets,
        coresets=coresets,
    ).cellbc_vs_coreset_overlap_matrix

    cells = sorted(list(cell_intbc_sets.keys()))
    num_clones = cellbc_vs_coreset_overlap_matrix.shape[1]
    num_cells = cellbc_vs_coreset_overlap_matrix.shape[0]
    assert len(cells) == num_cells
    # Use a numpy view for assignment to avoid per-cell pandas Series overhead.
    overlap_matrix = cellbc_vs_coreset_overlap_matrix.to_numpy()

    # First compute clone_intbc_counts by using the confidently mapped cells.
    clone_intbc_counts = {
        clone_idx: defaultdict(int) for clone_idx in range(num_clones)
    }
    clone_mapping = {cell: {} for cell in cell_intbc_sets}
    clone_to_cells_mapping = {clone_idx: set() for clone_idx in range(num_clones)}
    clone_sizes = defaultdict(int)
    for cell_idx, cell in enumerate(tqdm(cells)):
        clones_assigned = list(
            np.where(overlap_matrix[cell_idx] >= min_overlap_for_assigning_cellbc)[0]
        )
        clone_mapping[cell] = set(clones_assigned)
        if len(clones_assigned) == 1:
            # Is confidently assigned, so use to compute clone intBC sets
            clone_idx = clones_assigned[0]
            clone_to_cells_mapping[clone_idx].add(cell)
            clone_sizes[clone_idx] += 1
            for intbc in cell_intbc_sets[cell]:
                clone_intbc_counts[clone_idx][intbc] += 1

    # Now it's time to figure out the intBC sets of each clone from their intBC
    # counts
    clone_intbc_sets = {}
    for clone_idx in tqdm(range(num_clones)):
        clone_size = clone_sizes[clone_idx]
        kept_intbcs = []
        intbc_counts = sorted(
            clone_intbc_counts[clone_idx].items(), key=lambda x: -x[1]
        )
        for intbc, counts in intbc_counts:
            if custom_intbc_filtering_function(intbc):
                # Is weird, so we discard it immediately
                continue
            else:
                # First see if it looks anything like another *kept* intBC
                ds = {
                    kept_intbc: editdistance.eval(intbc, kept_intbc)
                    for kept_intbc in kept_intbcs
                }
                close_kept_intbcs = [
                    kept_intbc
                    for (kept_intbc, d) in ds.items()
                    if d <= edit_distance_cutoff
                ]
                if len(close_kept_intbcs) == 0:
                    # If it meets the fraction & count cutoff, we keep it
                    intbc_fraction = counts / clone_size
                    if (
                        (
                            (
                                intbc_fraction
                                >= min_intbc_fraction_for_adding_to_clone_intbc_set
                            )
                        )
                        and counts
                        >= min_intbc_counts_for_adding_intbc_to_clone_intbc_set
                    ):
                        kept_intbcs.append(intbc)
        clone_intbc_sets[clone_idx] = set(kept_intbcs)

    # At this point, we have our clone intBC sets! It's time to compute the
    # cell intBC sets (simply by intersection)
    observed_cell_intbc_sets = cell_intbc_sets
    cell_intbc_sets = {}
    for cell, clones in tqdm(clone_mapping.items()):
        observed = observed_cell_intbc_sets[cell]
        if len(clones) != 1:
            # We did not confidently map the cell, so we just drop all intBCs
            cell_intbc_set = set()
        else:
            clone_idx = list(clones)[0]
            clone_intbc_set = clone_intbc_sets[clone_idx]
            cell_intbc_set = observed.intersection(clone_intbc_set)
        cell_intbc_sets[cell] = cell_intbc_set

    cell_to_clones_mapping = {
        x: set(["clone_" + str(y) for y in ys]) for (x, ys) in clone_mapping.items()
    }
    for cellbc in all_cellbc:
        if cellbc not in cell_to_clones_mapping:
            cell_to_clones_mapping[cellbc] = (
                set()
            )  # These lost all their intBCs due to low UMI counts
        if cellbc not in cell_intbc_sets:
            cell_intbc_sets[cellbc] = (
                set()
            )  # These lost all their intBCs due to low UMI counts
    clone_intbc_sets = {("clone_" + str(x)): y for (x, y) in clone_intbc_sets.items()}
    clone_to_cells_mapping = {
        ("clone_" + str(x)): y for (x, y) in clone_to_cells_mapping.items()
    }

    clone_call = CloneCallType(
        clone_to_cells_mapping=clone_to_cells_mapping,
        cell_to_clones_mapping=cell_to_clones_mapping,
        clone_intbc_sets=clone_intbc_sets,
        cell_intbc_sets=cell_intbc_sets,
    )

    total_time = time.time() - st
    res = CallClonesFromCoresetsReturnType(
        clone_call=clone_call,
        total_time=total_time,
    )
    return res


def test_call_clones_from_coresets():
    intbc_table = pd.DataFrame(
        [
            ["c1", "A", 5],
            ["c1", "B", 5],
            ["c2", "C", 5],
        ],
        columns=["cellBC", "intBC", "UMI_count"],
    )
    out = call_clones_from_coresets(
        intbc_table_per_sample={"s0": intbc_table},
        min_intbc_umi_count=1,
        coresets=[{"A", "B"}, {"C"}],
        min_overlap_for_assigning_cellbc=1,
        min_intbc_counts_for_adding_intbc_to_clone_intbc_set=1,
        min_intbc_fraction_for_adding_to_clone_intbc_set=0.0,
        edit_distance_cutoff=0,
        custom_intbc_filtering_function=lambda intbc: False,
    )
    assert out.total_time >= 0.0
    cc = out.clone_call
    c1, c2 = "s0_c1", "s0_c2"
    assert cc.clone_to_cells_mapping["clone_0"] == {c1}
    assert cc.clone_to_cells_mapping["clone_1"] == {c2}
    assert cc.clone_intbc_sets["clone_0"] == {"A", "B"}
    assert cc.clone_intbc_sets["clone_1"] == {"C"}
    assert cc.cell_intbc_sets[c1] == {"A", "B"}
    assert cc.cell_intbc_sets[c2] == {"C"}


def restrict_single_intbc_clones(
    clone_call: CloneCallType,
    coresets: List[Coreset[str]],
) -> CloneCallType:
    """
    Post-process single-intBC clones after phase 2.

    For each coreset with a single unique intBC of multiplicity >= 2, keep only
    cells whose cell intBC set is exactly that signature intBC, and restrict the
    clone intBC set to that signature. Multiplicity-1 coresets are left unchanged.
    """
    clone_to_cells_mapping = {
        clone: set(cells)
        for clone, cells in clone_call.clone_to_cells_mapping.items()
    }
    cell_to_clones_mapping = {
        cell: set(clones)
        for cell, clones in clone_call.cell_to_clones_mapping.items()
    }
    clone_intbc_sets = {
        clone: set(intbcs)
        for clone, intbcs in clone_call.clone_intbc_sets.items()
    }
    cell_intbc_sets = {
        cell: set(intbcs) for cell, intbcs in clone_call.cell_intbc_sets.items()
    }

    for coreset_idx, coreset in enumerate(coresets):
        unique_intbcs = set(coreset)
        if len(unique_intbcs) != 1:
            continue
        signature_intbc = next(iter(unique_intbcs))
        multiplicity = sum(1 for intbc in coreset if intbc == signature_intbc)
        if multiplicity < 2:
            continue

        clone = f"clone_{coreset_idx}"
        if clone not in clone_to_cells_mapping:
            continue

        signature_set = {signature_intbc}
        kept_cells = set()
        for cell in clone_to_cells_mapping[clone]:
            if cell_intbc_sets.get(cell, set()) == signature_set:
                kept_cells.add(cell)
            else:
                cell_to_clones_mapping[cell] = cell_to_clones_mapping.get(
                    cell, set()
                ) - {clone}
                if len(cell_to_clones_mapping[cell]) == 0:
                    cell_intbc_sets[cell] = set()
        clone_to_cells_mapping[clone] = kept_cells
        clone_intbc_sets[clone] = signature_set

    return CloneCallType(
        clone_to_cells_mapping=clone_to_cells_mapping,
        cell_to_clones_mapping=cell_to_clones_mapping,
        clone_intbc_sets=clone_intbc_sets,
        cell_intbc_sets=cell_intbc_sets,
    )


def test_restrict_single_intbc_clones():
    """
    Post phase-2 restriction for single-intBC clones.

    Example: signature Multiset([a,a]); after de-novo, clone intBC set is
    {a, z} with cells {a}, {a}, {a, z}. Post-process keeps only the {a} cells
    and sets the clone intBC set to {a}.

    This must fail if we only pin the clone intBC set during phase 2 (that
    would keep the {a, z} cell with a reduced set {a}). It fails without
    post-process. It passes once restrict_single_intbc_clones implements the
    keep-exact-signature-cells rule.
    """
    # Phase-2-like input via call_clones_from_coresets:
    # clone_0: Multiset([a,a]) single-intBC; cells a0/a1 have only a; a2 has a+z
    # clone_1: multiplicity-1 Multiset([c]) (collapsed multi-intBC); cells have c+e
    rows = [
        ["a0", "a", 10],
        ["a1", "a", 10],
        ["a2", "a", 10],
        ["a2", "z", 10],
    ]
    for i in range(5):
        rows.extend([[f"c{i}", "c", 10], [f"c{i}", "e", 10]])
    intbc_table = pd.DataFrame(rows, columns=["cellBC", "intBC", "UMI_count"])
    coresets = [
        Multiset(["a"]) + Multiset(["a"]),
        Multiset(["c"]),
    ]
    cc = call_clones_from_coresets(
        intbc_table_per_sample={"s0": intbc_table},
        min_intbc_umi_count=1,
        coresets=coresets,
        min_overlap_for_assigning_cellbc=1,
        min_intbc_counts_for_adding_intbc_to_clone_intbc_set=1,
        min_intbc_fraction_for_adding_to_clone_intbc_set=0.0,
        edit_distance_cutoff=0,
        custom_intbc_filtering_function=lambda intbc: False,
    ).clone_call

    # Sanity: phase 2 mapped all three a* cells and de-novo-expanded clone_0.
    assert cc.clone_to_cells_mapping["clone_0"] == {"s0_a0", "s0_a1", "s0_a2"}
    assert cc.clone_intbc_sets["clone_0"] == {"a", "z"}
    assert cc.cell_intbc_sets["s0_a0"] == {"a"}
    assert cc.cell_intbc_sets["s0_a1"] == {"a"}
    assert cc.cell_intbc_sets["s0_a2"] == {"a", "z"}

    cc = restrict_single_intbc_clones(clone_call=cc, coresets=coresets)

    # Single-intBC (multiplicity >= 2): drop cells whose intBC set is not exactly {a}.
    assert cc.clone_to_cells_mapping["clone_0"] == {"s0_a0", "s0_a1"}
    assert cc.clone_intbc_sets["clone_0"] == {"a"}
    assert cc.cell_to_clones_mapping["s0_a0"] == {"clone_0"}
    assert cc.cell_to_clones_mapping["s0_a1"] == {"clone_0"}
    assert cc.cell_to_clones_mapping["s0_a2"] == set()
    assert cc.cell_intbc_sets["s0_a0"] == {"a"}
    assert cc.cell_intbc_sets["s0_a1"] == {"a"}
    assert cc.cell_intbc_sets["s0_a2"] == set()
    # Multiplicity-1 coreset: unchanged de-novo expansion.
    assert cc.clone_intbc_sets["clone_1"] == {"c", "e"}
    assert "s0_c0" in cc.clone_to_cells_mapping["clone_1"]
