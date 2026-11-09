"""
Module for the clone calling algorithm.

The main function exposed by this module is the `clone_calling_algorithm`.
"""

import logging
import sys
import time
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import seaborn as sns

from ._build_intbc_graph import build_intbc_graph
from ._call_clones_from_coresets import call_clones_from_coresets
from ._compute_coreset_co_assignment_matrix import compute_coreset_co_assignment_matrix
from ._find_maximal_coresets import find_maximal_coresets
from ._keep_reliable_edges import keep_reliable_edges
from ._make_coresets_disjoint import *
from ._prune_coresets import prune_coresets
from ._prune_custom_intbcs import prune_custom_intbcs
from ._prune_likely_homoplastic_intbcs import prune_likely_homoplastic_intbcs
from ._prune_likely_sequencing_errors import prune_likely_sequencing_errors
from ._prune_low_cellbc_count_intbcs import prune_low_cellbc_count_intbcs
from ._prune_low_umi_count_intbcs import prune_low_umi_count_intbcs
from ._types import *
from ._utils import _cell_intbc_sets_from_intbc_tables
from ._recursive_utils import subset_intbc_graph, infer_progenitor_intbc_set_by_popping


def _init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)


_init_logger()


class CloneCallingAlgorithmReturnType(NamedTuple):
    """
    Object returned by `clone_calling_algorithm`

    Attributes:
        clone_call: The clone call
        recursive_clone_call: The recursive clone call.
        recursive_clone_call_integration_hierarchy: Describes the subclonal
            structure of each clone in a succint way, and hence allows easy
            parsing of the recursive_clone_call. E.g.:
            {
                "clone_0": ({"AAA", "BBB"},
                {
                    "clone_0": ({"CCC"},
                    None),
                    "clone_1": ({"DDD"},
                    None),
                }),
            }
        metadata: General metadata for debugging etc.
        total_time: Total time taken
    """

    clone_call: CloneCallType
    recursive_clone_call: Dict[CloneType, "CloneCallingAlgorithmReturnType"]
    recursive_clone_call_integration_hierarchy: Any = None
    total_time: Optional[float] = None
    metadata: Any = None


def _is_empty(intbc_table_per_sample_for_clone):
    total_numer_of_rows = sum(
        intbc_table.shape[0]
        for intbc_table in intbc_table_per_sample_for_clone.values()
    )
    return total_numer_of_rows == 0


def remove_empty_clones(clone_call: CloneCallType) -> CloneCallType:
    """
    Remove empty clones from the clone call.

    A clone is considered empty if it has no cells assigned to it, or if
    its intBC set is empty.
    """
    non_empty_clones = {
        clone
        for clone, cells in clone_call.clone_to_cells_mapping.items()
        if len(cells) > 0 and len(clone_call.clone_intbc_sets.get(clone, set())) > 0
    }

    clone_to_cells_mapping = {
        clone: cells
        for clone, cells in clone_call.clone_to_cells_mapping.items()
        if clone in non_empty_clones
    }
    clone_intbc_sets = {
        clone: intbcs
        for clone, intbcs in clone_call.clone_intbc_sets.items()
        if clone in non_empty_clones
    }
    cell_to_clones_mapping = {
        cellbc: {clone for clone in clones if clone in non_empty_clones}
        for cellbc, clones in clone_call.cell_to_clones_mapping.items()
    }

    return CloneCallType(
        clone_to_cells_mapping=clone_to_cells_mapping,
        cell_to_clones_mapping=cell_to_clones_mapping,
        clone_intbc_sets=clone_intbc_sets,
        cell_intbc_sets=clone_call.cell_intbc_sets,
    )


def test_remove_empty_clones():
    clone_call = CloneCallType(
        clone_to_cells_mapping={
            "clone_0": {"c1"},
            "clone_1": set(),
            "clone_2": {"c2"},
            "clone_3": {"c4"},
        },
        cell_to_clones_mapping={
            "c1": {"clone_0"},
            "c2": {"clone_2"},
            "c3": {"clone_1", "clone_2"},
            "c4": {"clone_3"},
        },
        clone_intbc_sets={
            "clone_0": {"A"},
            "clone_1": set(),
            "clone_2": {"C"},
            "clone_3": set(),
        },
        cell_intbc_sets={"c1": {"A"}, "c2": {"C"}, "c3": {"C"}, "c4": set()},
    )
    out = remove_empty_clones(clone_call)
    assert set(out.clone_to_cells_mapping.keys()) == {"clone_0", "clone_2"}
    assert set(out.clone_intbc_sets.keys()) == {"clone_0", "clone_2"}
    assert out.cell_to_clones_mapping["c3"] == {"clone_2"}
    assert out.cell_to_clones_mapping["c4"] == set()


def clone_calling_algorithm(
    ##### Public parameters (defaults should work well on most datasets) #####
    # Phase 1 parameters
    intbc_table_per_sample: Dict[str, pd.DataFrame],
    min_intbc_umi_count: int = 10,
    custom_intbc_filtering_function: Callable[[str], bool] = lambda intbc: False,
    min_intbc_cellbc_count: int = 10,
    edit_distance_cutoff: int = 4,
    # Phase 2 parameters
    min_cells_assigned: int = 10,
    min_intbc_fraction: float = 0.05,
    edit_distance_cutoff_phase_2: Optional[int] = None, # If `None`, defaults to `edit_distance_cutoff` value
    # Recursive step parameters
    recursive: bool = False,
    recursive_min_fraction_uniquely_mapped_cells_to_recurse: float = 0.7,
    recursive_min_cells_assigned: int = 10,
    # Single-intBC clone recovery
    recover_single_intbc_clones: bool = False,
    ##### Private parameters (we do not expect the user to ever have to change these) #####
    _sequencing_error_pruning_step__co_occurrence_cutoff: float = 0.5,
    _homoplastic_pruning_step__p_max_cutoff: float = 0.5,
    _reliable_edges_step__ratio_cutoff: float = 0.5,
    _preliminary_coreset_building_step__min_coreset_size: int = 2, # Private parameter, not exposed to the user. Used for the recursive call of the algorithm.
    _preliminary_coreset_building_step__min_overlap_for_merging_coresets: int = 2, # Private parameter, not exposed to the user. Used for the recursive call of the algorithm.
    _coreset_dealiasing_step__min_overlap_for_assigning_cellbc: int = 2, # Private parameter, not exposed to the user. Used for the recursive call of the algorithm.
    _coreset_dealiasing_step__co_occurrence_cutoff: float = 0.5, # Private parameter, not exposed to the user. Used for the recursive call of the algorithm.
    _coreset_disjointing_step__make_disjoint: bool = True, # Private parameter, not exposed to the user. Used for testing.
    _final_cell_assignment_step__min_overlap_for_assigning_cellbc: int = 1, # Private parameter, not exposed to the user. Used for the recursive call of the algorithm.
    _call_clones_from_coresets_step__min_intbc_counts_for_adding_intbc_to_clone_intbc_set: Optional[int] = None, # Private parameter, not exposed to the user. Used for testing.
) -> CloneCallingAlgorithmReturnType:
    """
    Args:
        intbc_table_per_sample: Each value should be a pandas DataFrame with the following columns:
            "cellBC" (str), "intBC" (str), "UMI_count" (int). For example, a row might be:
            ["AAACCCAAGAGCCTGA", "ACCTCATGATATTC", 45]
            indicating that the intBC "ACCTCATGATATTC" was observed 45 times in the cellBC "AAACCCAAGAGCCTGA".
            There should be one intBC table per sample; the sample name acts as the key of the dictionary.
        min_intbc_umi_count: IntBCs with less than this number of UMI_counts will be discarded by the algorithm.
            The default value is 10, which is effective at removing sequencing errors and background RNA and
            sequencing errors. For some datasets, such as the macsGESTALT which we analyze, we lower this
            threshold to two UMIs, as described in the main text. More generally, we recommend to set the UMI
            cutoff by considering a histogram of UMI counts across all (cellBC, intBC) pairs (rows in the input
            table) and use the leftmost mode (in case the count distribution is multi-modal) or a low quantile
            (otherwise).
        custom_intbc_filtering_function: IntBCs for which this function gives
            `True` will be discarded.
        min_intbc_cellbc_count: IntBCs appearing in less than this number of
            cellBCs will be discarded.
        edit_distance_cutoff: Minimum *allowed* edit distance. E.g. if you expect there to
            exist sequencing errors at edit distance 3 but not 4, please set this to 4.
            To be precise about how this is used: When building clonal signatures (phase 1 of
            the algorithm), we identify putative sequencing errors by checking for
            intBCs `x` frequently co-occurring with another (more frequent) intBC
            `y` at an edit distance <= edit_distance_cutoff. If so, we remove `x`.
            In the second phase of the algorithm, when we reconstruct
            the intBC set for each clone de novo, we require that the edit distance
            to any other more abundant intBC is >= edit_distance_cutoff.
            Note that our cutoff for phase 1 is extra stringent to be extra safe, since
            sequencing error in phase 1 have a bigger consequence (overmerging clones) than
            in phase 2 (just getting some aliased characters which will likely get dropped
            due to low frequency via `min_intbc_fraction` anyway). We allow setting the
            cutoff for phase 2 differently with `edit_distance_cutoff_phase_2`.
        min_cells_assigned: Coresets with less than this number of mapped cells
            will be dropped. If you wish to recover smaller clones, you can lower
            this, however, note that since all \intBCs are supported by at least
            min_intbc_cellbc_count cellBCs (by default 10, and we do not recommend
            changing this), all clones will have at least 10 mapped cellBCs.
        min_intbc_fraction:
            Following up on the above argument, we can also specify the
            minimum percentage of cells in the clone containing the intBC
            needed to include the intBC in the clone's intBC set. We set this
            to 5% by default to recover the intBC tail too. Set this to e.g.
            20% if you only wish to retain only high-frequency intBCs.
       edit_distance_cutoff_phase_2: See description of parameter edit_distance_cutoff.
            If `None` (default) this is set equal to edit_distance_cutoff. This should work
            well in most cases. Reasons for setting this value to something different than
            `edit_distance_cutoff` are (i) increase retention of intBCs after clone calling;
            note that while removing sequencing errors during the first phase (finding clonal
            signatures) is critical (or else we might overmerge clones), when building intBC
            sets de novo is it less critical because in the worst case, sequencing errors
            lead to "duplicate" characters which will likely get dropped due to low frequency
            via `min_intbc_fraction` anyway, and (ii) the user might want to curate the intBCs
            in each clone themselves, in which case they can set edit_distance_cutoff_phase_2=0
            to turn off this second sequencing error filter.
        recursive: If True, then we will recursively call the algorithm on each
            clone, thus resolving the subclonal structure derived by the
            hierarchy of integrations. When using this option, we recommend
            (and will enforce) using:
            min_intbc_fraction <= 0.01
        recursive_min_fraction_uniquely_mapped_cells_to_recurse: If recursive
            is True, then this determines the min fraction of cells that have
            to be uniquely mapped to a clone's subclones before we accept the
            subclonal decomposition of that clone. This thus serves as a
            stopping criteria. For example, a clone might have two resolvable
            subclones comprising 35% and 31% of the clone's cells respectively,
            with the remaining 34% unmapped (i.e. discarded). Hence, if this
            argument is set to 0.7 (i.e. 70%), then we would NOT accept this
            recursive decomposition and just retain the original clone. Instead,
            if the subclones comprised 35% and 35% of the clone's cells, then
            we would exactly meet the 70% threshold and accept the recursive
            decomposition.
        recursive_min_cells_assigned: If recursive
            is True, a lower bound on the size of the subclones we want to find.
            Subclones tend to be smaller than the parent clone, so it is a good
            idea to set this lower than the original
            min_cells_assigned.
        recover_single_intbc_clones: If `True`, single-intBC clones will also be
            recovered. Use this if your data contains single-intBC clones. There
            is an inherent tradeoff in trying to recover single-intBC clones: since
            these clones have weaker statistical support, unless you are certain
            that your data has negligible levels of homoplasy, you might end up
            accidentally overmerging clones. If evolving tracer marks are
            available, we recommend using these to confirm that the single-intBC
            clones are high quality, and if there is overmerging we recommend
            splitting into different clones based on evolving lineage marks.
        _sequencing_error_pruning_step__co_occurrence_cutoff: This establishes
            how frequently intBC `x` needs to co-occur with intBC `y` having a
            similar sequence (as determined by the edit_distance_cutoff above)
            to be considered a sequencing error. This frequency cutoff is as
            follows: p(y | x) >= co_occurrence_cutoff. Thus, using a smaller
            value for this argument detects more sequencing errors but also
            risks discarding intBCs which are not actually sequencing errors
            (i.e. higher recall, lower precision). The default value is 0.5;
            we do not recommend the user to change this value, hence we have
            made it a private parameter.
        _homoplastic_pruning_step__p_max_cutoff: This determines how homoplastic
            intBCs are found. Homoplastic intBCs appear in multiple clones and
            therefore may not co-occurr with any other intBC at high
            probability. This p_max_cutoff argument specifies that we consider
            an intBC `x` to be homoplastic if p_max_cutoff > max_y p(y | x).
            Thus, setting this argument to 0 bypasses this step. The default
            value is 0.5; we do not recommend the user to change this value,
            hence we have made it a private parameter.
        _reliable_edges_step__ratio_cutoff: We say that an edge in the intBC
            graph is "reliable" if we are very certain that there exists one
            and only one clone in which its two intBCs are integrations.
            Since the graph's been pruned from sequencing errors at this point,
            reliable pairs will have a large ratio. Here the ratio is defined
            as: ratio(x, y) = c(x, y) / min(c_x, c_y) = max { p(x|y), p(y|x) }
            The default value is 0.5; we do not recommend the user to change
            this value, hence we have made it a private parameter.
        _preliminary_coreset_building_step__min_coreset_size: The minimum size
            of an intBC set, all of which are pairwise reliably co-occurring,
            which implies that said set of intBCs exists in one and only one
            clone. Such sets are called "coresets". This is a private parameter
            not exposed to the user. Please do not use it. The only reason it is
            here is because it is relaxed in the recursive call of the algorithm.
        _preliminary_coreset_building_step__min_overlap_for_merging_coresets:
            Coresets also have the property that if two coresets A and B
            intersect in `min_overlap_for_merging_coresets`, then they
            represent the same clone and their union is a coreset for that
            clone. This is a private parameter not exposed to the user. Please
            do not use it. The only reason it is here is because it is relaxed in
            the recursive call of the algorithm.
        _coreset_dealiasing_step__min_overlap_for_assigning_cell: After we
            compute the preliminary coresets, it is possible that two of these
            refer to the same clone. Because of this, we map cellBCs to
            preliminary coresets to determine which coresets are aliased. To
            map a cellBC to a coreset, we require that this number of intBCs
            overlap with the coreset. This is a private parameter not exposed
            to the user. Please do not use it. The only reason it is here is
            because it is relaxed in the recursive call of the algorithm.
        _coreset_dealiasing_step__co_occurrence_cutoff: To identify aliased
            coresets (i.e. representing the same clone), we look at the cellBCs
            mapped to these two coresets. Let m_1 and m_2 be the number of
            cellBCs mapped to each, and m_12 the number of cellBCs mapped to
            both. If m_12 / min(m_1, m_2) >= co_occurrence_cutoff (and if
            min(m_1, m_2) >= min_cells_assigned below) we consider the coresets
            to be aliased. Of course, the co_occurrence_cutoff and
            min_cells_assigned should be chosen to exclude multiplets as the
            reason for co-occurrence.  This is a private parameter not exposed
            to the user. Please do not use it. The only reason it is here is
            for testing purposes.
        _coreset_disjointing_step__make_disjoint: Once all coresets have been
            found (which map one-to-one with clones), we can remove homoplastic
            intBCs from the coresets to create an even stronger signature for
            mapping cellBCs. This is a private parameter not exposed to the user. Please
            do not use it. The only reason it is here is for testing purposes.
        _final_cell_assignment_step__min_overlap_for_assigning_cellbc: To
            map a cellBC to a coreset, we require that this number of intBCs
            overlap with the coreset. This is a private parameter not exposed
            to the user. Please do not use it. The only reason it is here is
            because it is relaxed in the recursive call of the algorithm.
        _call_clones_from_coresets_step__min_intbc_counts_for_adding_intbc_to_clone_intbc_set:
            After mapping all the cells to that clone, we can calculate the
            *full* intBC set for each clone (which goes beyond the coreset,
            i.e. it includes homoplastic intBCs). This argument specifies the
            minimum number of cells in that clone containing the intBC which
            are needed to include the intBC in that clone's final intBC set.
            If `None` (default), this is set to `min_intbc_cellbc_count`.
            This is a private parameter not exposed to the user. Please do not
            use it. The only reason it is here is for testing purposes.

    Returns:
        A `results` NamedTuple of type CloneCallingAlgorithmReturnType, which contains
        the clone call in the field `clone_call`. The `clone_call` itself is also a
        NamedTuple; the most important fields are:
        - `results.clone_call.cell_to_clones_mapping (Dict[str, Set[str]])`:
            Dictionary mapping cellBCs to clone IDs. A cellBCs mapping to more than one
            clone is a multiplet, and a cellBC mapping to no clone is an empty cellBC or
            a dropped cellBC (e.g. due to low quality).
        - `results.clone_call.clone_to_cells_mapping (Dict[str, Set[str]])`:
            Dictionary clone IDs to cellBCs.
        - `results.clone_call.clone_intbc_sets (Dict[str, Set[str]])`:
            Dictionary mapping clone IDs to its set of IntBCs.
        - `results.clone_call.cell_intbc_sets (Dict[str, Set[str]])`:
            Dictionary mapping cellBCs to its set of IntBCs.
        NOTE: the cellBCs are prefixed with the sample name in the `results` (to avoid
        cellBC collision between different samples.)
    """
    if recursive:
        if min_intbc_fraction > 0.05:
            raise ValueError("When using recursive mode, we recommend (and will enforce) using: min_intbc_fraction <= 0.05. Otherwise, the subclonal intBCs will be discarded and little recursive structure will be found.")

    if edit_distance_cutoff_phase_2 is None:
        edit_distance_cutoff_phase_2 = edit_distance_cutoff

    if _call_clones_from_coresets_step__min_intbc_counts_for_adding_intbc_to_clone_intbc_set is None:
        _call_clones_from_coresets_step__min_intbc_counts_for_adding_intbc_to_clone_intbc_set = min_intbc_cellbc_count

    if recover_single_intbc_clones:
        # We set the min_coreset_size to size to 1.
        # This retains isolated vertices in the starting coreset family.
        _preliminary_coreset_building_step__min_coreset_size = 1
        # Also, if the user is running in single-intBC mode, the homoplasy pruning step should be
        # skipped, because it discards single-intBC clones. The user has acknowledged
        # the limitations of recovering clones with weaker statistical support, and will
        # check the post-QC plots closely.
        _homoplastic_pruning_step__p_max_cutoff = -1


    st = time.time()

    logger = logging.getLogger(__name__)
    print(
        "*********************************************************************************"
    )
    print(
        '***** Phase 1 of algorithm: Construct clonal signatures (a.k.a. "coresets") *****'
    )
    print(
        "*********************************************************************************"
    )

    # Validate arguments
    validate_intbc_table_per_sample(intbc_table_per_sample=intbc_table_per_sample)
    original_intbc_table_per_sample = deepcopy(
        intbc_table_per_sample
    )  # After identifying the coresets, we will use the original intBC tables to map cells to clones, call sequencing errors and homoplastic intBCs.
    for sample in intbc_table_per_sample:
        if "_" in sample:
            raise ValueError("The sample names should not contain underscores.")

    # Filter intBCs based on min_intbc_umi_count
    intbc_table_per_sample = prune_low_umi_count_intbcs(
        intbc_table_per_sample=intbc_table_per_sample,
        min_intbc_umi_count=min_intbc_umi_count,
    ).intbc_table_per_sample

    # Filter intBCs based on `custom_intbc_filtering_function`
    intbc_table_per_sample = prune_custom_intbcs(
        intbc_table_per_sample=intbc_table_per_sample,
        custom_intbc_filtering_function=custom_intbc_filtering_function,
    ).intbc_table_per_sample

    # Filter intBCs based on `min_intbc_cellbc_count`
    intbc_table_per_sample = prune_low_cellbc_count_intbcs(
        intbc_table_per_sample=intbc_table_per_sample,
        min_intbc_cellbc_count=min_intbc_cellbc_count,
    ).intbc_table_per_sample

    # Build the intBC graph.
    intbc_graph = build_intbc_graph(
        intbc_table_per_sample=intbc_table_per_sample,
    ).intbc_graph

    # Prune intBCs with small edit distance and high co-occurrence to some more
    # frequent intBC.
    # This gets rid of many (high-count) sequencing errors.
    prune_likely_sequencing_errors_return = prune_likely_sequencing_errors(
        intbc_graph=intbc_graph,
        edit_distance_cutoff=edit_distance_cutoff,  # noqa
        co_occurrence_cutoff=_sequencing_error_pruning_step__co_occurrence_cutoff,  # noqa
    )
    intbc_graph = prune_likely_sequencing_errors_return.intbc_graph
    pruned_sequencing_errors = (
        prune_likely_sequencing_errors_return.sequencing_errors_and_originals
    )

    # Prune intBCs with p_max too low.
    # This gets rid of highly homoplastic intBCs.
    intbc_graph = prune_likely_homoplastic_intbcs(
        intbc_graph=intbc_graph,
        p_max_cutoff=_homoplastic_pruning_step__p_max_cutoff,
    ).intbc_graph

    # Keep reliable edges only
    intbc_graph = keep_reliable_edges(
        intbc_graph=intbc_graph,
        ratio_cutoff=_reliable_edges_step__ratio_cutoff,
    ).intbc_graph

    # Infer preliminary coresets by merging cliques
    preliminary_coresets = find_maximal_coresets(
        graph=intbc_graph,
        min_coreset_size=_preliminary_coreset_building_step__min_coreset_size,  # noqa
        min_overlap_for_merging_coresets=_preliminary_coreset_building_step__min_overlap_for_merging_coresets,  # noqa
        recover_single_intbc_clones=recover_single_intbc_clones,
    ).maximal_coresets

    # Find aliased coresets; we first need to assign the cells to the coresets.
    logger.info("Going to compute_coreset_co_assignment_matrix ...")
    cellbc_intbc_sets = _cell_intbc_sets_from_intbc_tables(
        intbc_table_per_sample=intbc_table_per_sample,
    )
    preliminary_coreset_co_assignment_matrix = compute_coreset_co_assignment_matrix(
        cellbc_intbc_sets=cellbc_intbc_sets,
        coresets=preliminary_coresets,
        min_overlap_for_assigning_cellbc=_coreset_dealiasing_step__min_overlap_for_assigning_cellbc,
    ).coreset_co_assignment_matrix

    # Prune preliminary coresets that are aliased (so we end with one per clone)
    pruned_coresets = prune_coresets(
        coresets=preliminary_coresets,
        coreset_co_assignment_matrix=preliminary_coreset_co_assignment_matrix,
        co_occurrence_cutoff=_coreset_dealiasing_step__co_occurrence_cutoff,
        min_cells_assigned=min_cells_assigned,
    ).coresets

    if _coreset_disjointing_step__make_disjoint:
        disjoint_coresets = make_coresets_disjoint(
            coresets=pruned_coresets,
        ).coresets
    else:
        disjoint_coresets = pruned_coresets

    print("********************************************************************")
    print("***** Phase 2 of algorithm: Call clones from clonal signatures *****")
    print("********************************************************************")

    call_clones_from_coresets_return = call_clones_from_coresets(
        intbc_table_per_sample=original_intbc_table_per_sample,
        min_intbc_umi_count=min_intbc_umi_count,
        coresets=disjoint_coresets,
        min_overlap_for_assigning_cellbc=_final_cell_assignment_step__min_overlap_for_assigning_cellbc,
        min_intbc_counts_for_adding_intbc_to_clone_intbc_set=_call_clones_from_coresets_step__min_intbc_counts_for_adding_intbc_to_clone_intbc_set,
        min_intbc_fraction_for_adding_to_clone_intbc_set=min_intbc_fraction,
        min_edit_distance_for_adding_to_clone_intbc_set=edit_distance_cutoff_phase_2,
        custom_intbc_filtering_function=custom_intbc_filtering_function,
    )
    clone_call = call_clones_from_coresets_return.clone_call

    # Remove empty clones -- can happen in very extreme cases (low quality input data)
    # due to e.g. (1) the disjointing step, which causes a putative clonal signature
    # to drop from 2 intBCs to 1, leading to 0 cells assigned in phase 2, as well as
    # (2) the intBC quality filter in step 2 rejects even the intBCs in the clonal
    # signature, leading to a clone with empty intBC set.
    clone_call = remove_empty_clones(clone_call)

    print("********************************************************************")
    print("****** Recurse if requested to estimate integration hierarchy ******")
    print("********************************************************************")

    if recursive:
        recursive_clone_call = {}
        recursive_clone_call_integration_hierarchy = {}
        clones = list(clone_call.clone_to_cells_mapping.keys())
        for clone in clones:
            print(f"Recursively calling clones on clone: {clone} ...")

            cells_in_clone = sorted(list(clone_call.clone_to_cells_mapping[clone]))
            cells_in_clone = ["_".join(x.split("_")[1:]) for x in cells_in_clone]

            # Since we are going to be popping intBCs until the graph disconnects,
            # we better start with a connected graph. This is ensured by the coresets.
            # Specifically, we consider the pruned coresets (which are dealiased), and
            # not the disjointed ones (since the disjointed ones lose homoplastic intBCs).
            clone_id = int(clone[len("clone_") :])
            pruned_set_for_clone = set(pruned_coresets[clone_id])  # Cast to set bc of multiset case.
            subsetted_intbc_graph = subset_intbc_graph(
                intbc_graph=intbc_graph,
                nodes_to_keep=sorted(list(set(pruned_set_for_clone))),
            )
            inferred_progenitor_intbc_set = infer_progenitor_intbc_set_by_popping(
                subsetted_intbc_graph=subsetted_intbc_graph,
            )
            all_intbcs_popped = sorted(list(inferred_progenitor_intbc_set)) == sorted(list(pruned_set_for_clone))
            clone_intbc_set = clone_call.clone_intbc_sets[clone]
            if (all_intbcs_popped or len(inferred_progenitor_intbc_set.intersection(clone_intbc_set)) == 0):
                # Case 1 (all_intbcs_popped): There was no subclonal structure left.
                # Case 2: We found some progenitors, but they were all filtered out in Phase 2 of the algorithm
                # (e.g. border case where user set more aggressive edit distance thresholds for phase 2),
                # so we choose not to recurse.
                recursive_clone_call[clone] = None
                recursive_clone_call_integration_hierarchy[clone] = (
                    clone_intbc_set,  # Note that we use the clone intBC set, to make sure we include everything.
                    None,
                )
                logger.info(
                    "All intBCs popped when trying to identify progenitor intBC set. No subclonal structure found. Will not try to recurse."
                )
                continue
            # If we are here, it means we were able to disconnect the graph.
            # Therefore, unless the two (or more) components are aliased, we have a subclonal structure
            # which we are ready to uncover using the recursive call.
            logger.info(
                f"Putative progenitor intBC set found (will proceed to call clones recursively excluding these intBCs): {inferred_progenitor_intbc_set}"
            )

            # Phase 2: Remove progenitor intBC set, and call recursively.
            # Note that in the recursive call, we can just retain the intBCs
            # in the clone's intBC set. This ensures there are no more
            # sequencing errors nor homoplastic intBCs, cimplifying the
            # recursive clone call.
            # TODO: This might be very slow - use groupby instead to precompute it for all clones?
            intbc_table_per_sample_for_clone = {
                sample: intbc_table.drop(
                    intbc_table[
                        (intbc_table["intBC"].isin(inferred_progenitor_intbc_set))
                        | (~intbc_table["cellBC"].isin(cells_in_clone))
                    ].index
                )
                for sample, intbc_table in original_intbc_table_per_sample.items()
            }
            recursive_clone_call__for_this_clone = clone_calling_algorithm(
                intbc_table_per_sample=intbc_table_per_sample_for_clone,  # This no longer contains sequencing errors not homoplastic intBCs.
                min_intbc_umi_count=min_intbc_umi_count,  # OK (still need it to discard intra-clone background RNA)
                custom_intbc_filtering_function=custom_intbc_filtering_function,  # Doesn't matter (OK)
                min_intbc_cellbc_count=min_intbc_cellbc_count,  # OK? Can consider removing this filter at this point?
                edit_distance_cutoff=edit_distance_cutoff,  # Doesn't actually matter since there are no sequencing errors at this point
                _sequencing_error_pruning_step__co_occurrence_cutoff=_sequencing_error_pruning_step__co_occurrence_cutoff,  # Doesn't actually matter since there are no sequencing errors at this point
                _homoplastic_pruning_step__p_max_cutoff=-1,  # Technically not needed since recover_single_intbc_clones will force it to -1 anyway.
                _reliable_edges_step__ratio_cutoff=_reliable_edges_step__ratio_cutoff,  # OK (still need it)
                _preliminary_coreset_building_step__min_coreset_size=1,  # Since there is no homoplasy eny more, can set it to 1.
                _preliminary_coreset_building_step__min_overlap_for_merging_coresets=1,  # Basically, we are just looking for connected components at this point.
                _coreset_dealiasing_step__min_overlap_for_assigning_cellbc=1,  # Again, there is no homoplasy, so we can set it to 1.
                recover_single_intbc_clones=True,  # Since we allow for single-intBC clones.
                _coreset_dealiasing_step__co_occurrence_cutoff=_coreset_dealiasing_step__co_occurrence_cutoff,  # Technically, there should be no aliasing since that's how we found the progenitor_intbc_set.
                min_cells_assigned=recursive_min_cells_assigned,  # OK - we allow for a smaller value to recover smaller subclones.
                _coreset_disjointing_step__make_disjoint=_coreset_disjointing_step__make_disjoint,  # Doesn't matter - there should be no homoplasy at this point.
                _final_cell_assignment_step__min_overlap_for_assigning_cellbc=1,  # Since there is no homoplasy eny more, can set it to 1.
                _call_clones_from_coresets_step__min_intbc_counts_for_adding_intbc_to_clone_intbc_set=_call_clones_from_coresets_step__min_intbc_counts_for_adding_intbc_to_clone_intbc_set,  # OK? Can consider lowering this.
                min_intbc_fraction=min_intbc_fraction,  # OK.
                edit_distance_cutoff_phase_2=edit_distance_cutoff_phase_2,  # Doesn't actually matter since there are no sequencing errors at this point
                recursive=True,
                recursive_min_fraction_uniquely_mapped_cells_to_recurse=recursive_min_fraction_uniquely_mapped_cells_to_recurse,
                recursive_min_cells_assigned=recursive_min_cells_assigned,
            )
            # Check how many cells were mapped in the recursive clone call, to determine if we should retain the recursive split.
            num_subclones_found = len(
                recursive_clone_call__for_this_clone.clone_call.clone_to_cells_mapping
            )
            logger.info(
                f"Found {num_subclones_found} clones in recursive call, with intBC sets: {recursive_clone_call__for_this_clone.clone_call.clone_intbc_sets}"
            )
            uniquely_mapped_cells_in_recursive_clone_call = sum(
                [
                    len(cells_in_clone)
                    for cells_in_clone in recursive_clone_call__for_this_clone.clone_call.clone_to_cells_mapping.values()
                ]
            )
            fraction_uniquely_mapped_cells_in_recursive_clone_call = (
                uniquely_mapped_cells_in_recursive_clone_call / len(cells_in_clone)
            )
            logger.info(
                f"fraction_uniquely_mapped_cells_in_recursive_clone_call = {fraction_uniquely_mapped_cells_in_recursive_clone_call}"
            )
            if (
                len(
                    recursive_clone_call__for_this_clone.clone_call.clone_to_cells_mapping
                )
                == 1
            ):
                logger.info(
                    f"While is able to disconnect the coreset intBC graph, I was only able to find one subclone. This may be because either (1) the coresets were aliased, or (2) the subclonal structure was too small to be detected. Hence I will not recurse and just retain the original clone."
                )
                recursive_clone_call[clone] = None
                recursive_clone_call_integration_hierarchy[clone] = (
                    clone_intbc_set,  # Note that we use the clone intBC set, to make sure we include everything.
                    None,
                )
            elif (
                len(
                    recursive_clone_call__for_this_clone.clone_call.clone_to_cells_mapping
                )
                >= 2
            ):
                if (
                    fraction_uniquely_mapped_cells_in_recursive_clone_call
                    >= recursive_min_fraction_uniquely_mapped_cells_to_recurse
                ):
                    logger.info(
                        f"Recursive clone call found {num_subclones_found} subclones, and enough cells were uniquely mapped to these subclones ({fraction_uniquely_mapped_cells_in_recursive_clone_call}). Accepting the recursive decomposition."
                    )
                    recursive_clone_call[clone] = recursive_clone_call__for_this_clone
                    recursive_clone_call_integration_hierarchy[clone] = (
                        inferred_progenitor_intbc_set.intersection(clone_intbc_set),  # NOTE: The `inferred_progenitor_intbc_set` might have lost some intBCs in phase 2 of the algorithm due to user filters, so we retain only the intBCs that the user wants.
                        recursive_clone_call__for_this_clone.recursive_clone_call_integration_hierarchy,
                    )
                else:
                    logger.info(
                        f"Recursive clone call found {num_subclones_found} subclones, but not enough cells were uniquely mapped to these subclones ({fraction_uniquely_mapped_cells_in_recursive_clone_call}). Hence I will not recurse and just retain the original clone."
                    )
                    recursive_clone_call[clone] = None
                    recursive_clone_call_integration_hierarchy[clone] = (
                        clone_intbc_set,  # Note that we use the clone intBC set, to make sure we include everything.
                        None,
                    )
            else:
                # No clones found.
                logger.info(
                    f"No subclonal structure found (i.e.). Hence I will not recurse and just retain the original clone."
                )
                recursive_clone_call[clone] = None
                recursive_clone_call_integration_hierarchy[clone] = (
                    clone_intbc_set,  # Note that we use the clone intBC set, to make sure we include everything.
                    None,
                )
    else:
        recursive_clone_call = None
        recursive_clone_call_integration_hierarchy = None

    sanity_check_clone_call(clone_call)

    total_time = time.time() - st
    res = CloneCallingAlgorithmReturnType(
        clone_call=clone_call,
        recursive_clone_call=recursive_clone_call,
        recursive_clone_call_integration_hierarchy=recursive_clone_call_integration_hierarchy,
        total_time=total_time,
        metadata={
            "pruned_sequencing_errors": pruned_sequencing_errors,
        },
    )
    return res


def _get_clone_to_cellbcs(
    clone_call: CloneCallType,
) -> Dict[CloneType, List[CellBCType]]:
    """
    Return the set of cellBCs mapping (uniquely, i.e. eliminating multiplets)
    to each clone.
    """
    res = defaultdict(list)
    for cellbc, clones in clone_call.cell_to_clones_mapping.items():
        if len(clones) == 1:  # I.e. we ignore multiplets
            clone = list(clones)[0]
            res[clone].append(cellbc)
    return res


def _check_are_clones(
    clone_call: CloneCallType, expected_clones: List[List[CellBCType]]
) -> None:
    """
    Raise error if the clone_call does not have the .
    """
    clone_to_cellbcs = _get_clone_to_cellbcs(clone_call=clone_call)
    expected_clones = sorted([sorted(x) for x in expected_clones])
    actual_clones = sorted([sorted(x) for x in clone_to_cellbcs.values()])
    if expected_clones != actual_clones:
        raise ValueError(
            f"Expected clones:\n"
            f"{expected_clones}\n"
            f"but actual clones:\n"
            f"{actual_clones}"
        )


def test_clone_calling_algorithm_mini():
    """
    Just tests that the clone calling algorithm runs on a trivial case.

    In this test case, there are two clones characterized by the intBC sets
    ["AAA", "BBB", "CCC"] and ["DDD", "EEE", "FFF"] respectively. The first
    clone contains the cellBCs ["cellBC_1", "cellBC_2"] whereas the second
    clone contains the cellBCs ["cellBC_3", "cellBC_4", "cellBC_5"].

    Uniquely assigned cells should have ``cell_intbc_sets`` equal to the
    intersection of their observed intBCs with their clone's intBC set.
    """
    intbc_table = pd.DataFrame(
        data=[
            ["cellBC_1", "AAA", 40],
            ["cellBC_1", "BBB", 40],
            # ["cellBC_1", "CCC", 40], # Exclude this to make test more interesting.
            ["cellBC_2", "AAA", 40],
            ["cellBC_2", "BBB", 40],
            ["cellBC_2", "CCC", 40],
            ["cellBC_3", "DDD", 40],
            ["cellBC_3", "EEE", 40],
            ["cellBC_3", "FFF", 40],
            ["cellBC_4", "DDD", 40],
            ["cellBC_4", "EEE", 40],
            ["cellBC_4", "FFF", 40],
            ["cellBC_5", "DDD", 9],  # Low UMI count to make the test more interesting.
            ["cellBC_5", "EEE", 40],
            ["cellBC_5", "FFF", 40],
            ["cellBC_5", "FFH", 15], # Sequencing error.
        ],
        columns=["cellBC", "intBC", "UMI_count"],
    )

    clone_call = clone_calling_algorithm(
        intbc_table_per_sample={"sample1": intbc_table},
        min_intbc_cellbc_count=1,
        edit_distance_cutoff=1,
        min_cells_assigned=1,
        # Collapse 1-edit neighbors of an already-kept intBC (drops FFH vs FFF).
        edit_distance_cutoff_phase_2=2,
    ).clone_call
    _check_are_clones(
        clone_call=clone_call,
        expected_clones=[
            {"sample1_cellBC_1", "sample1_cellBC_2"},
            {"sample1_cellBC_3", "sample1_cellBC_4", "sample1_cellBC_5"},
        ],
    )
    assert clone_call.clone_intbc_sets["clone_0"] == set(["AAA", "BBB", "CCC"])
    assert clone_call.clone_intbc_sets["clone_1"] == set(["DDD", "EEE", "FFF"])
    assert clone_call.cell_to_clones_mapping["sample1_cellBC_1"] == set(["clone_0"])
    assert clone_call.cell_to_clones_mapping["sample1_cellBC_2"] == set(["clone_0"])
    assert clone_call.cell_to_clones_mapping["sample1_cellBC_3"] == set(["clone_1"])
    assert clone_call.cell_to_clones_mapping["sample1_cellBC_4"] == set(["clone_1"])
    assert clone_call.cell_to_clones_mapping["sample1_cellBC_5"] == set(["clone_1"])

    clone0_intbcs = {"AAA", "BBB", "CCC"}
    clone1_intbcs = {"DDD", "EEE", "FFF"}
    assert clone_call.cell_intbc_sets["sample1_cellBC_1"] == {"AAA", "BBB"}
    assert clone_call.cell_intbc_sets["sample1_cellBC_2"] == clone0_intbcs
    assert clone_call.cell_intbc_sets["sample1_cellBC_3"] == clone1_intbcs
    assert clone_call.cell_intbc_sets["sample1_cellBC_4"] == clone1_intbcs
    assert clone_call.cell_intbc_sets["sample1_cellBC_5"] == {"EEE", "FFF"}


def test_recursive_clone_calling_algorithm_mini():
    """
    Just tests that the resursive clone calling algorithm runs on a small case.

    Subclone structure is:
    
        root
         |
        A,B  (first round of integration)
         |
        /  \
       C    D  (second round of integration)
       |    |
      ABC  ABD
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(0)

    intbc_table = pd.DataFrame(
        data=[
            ["cellBC_1", "AAA", 40],
            ["cellBC_1", "BBB", 40],
            ["cellBC_1", "CCC", 40],
            ["cellBC_11", "AAA", 40],
            ["cellBC_11", "BBB", 40],
            ["cellBC_11", "CCC", 40],
            ["cellBC_2", "AAA", 40],
            ["cellBC_2", "BBB", 40],
            ["cellBC_2", "CCC", 40],
            ["cellBC_22", "AAA", 40],
            ["cellBC_22", "BBB", 40],
            ["cellBC_22", "CCC", 40],
            ["cellBC_3", "AAA", 40],
            ["cellBC_3", "BBB", 40],
            ["cellBC_3", "DDD", 40],
            ["cellBC_33", "AAA", 40],
            ["cellBC_33", "BBB", 40],
            ["cellBC_33", "DDD", 40],
            ["cellBC_4", "AAA", 40],
            ["cellBC_4", "BBB", 40],
            ["cellBC_4", "DDD", 40],
            ["cellBC_44", "AAA", 40],
            ["cellBC_44", "BBB", 40],
            ["cellBC_44", "DDD", 40],
            ["cellBC_5", "AAA", 40],  # Cell must be unmapped in the recursion
            ["cellBC_5", "BBB", 40],
            ["cellBC_6", "AAA", 40],  # Intra-clonal doublet.
            ["cellBC_6", "BBB", 40],
            ["cellBC_6", "CCC", 40],
            ["cellBC_6", "DDD", 40],
        ],
        columns=["cellBC", "intBC", "UMI_count"],
    )

    clone_call_namedtuple = clone_calling_algorithm(
        intbc_table_per_sample={"sample1": intbc_table},
        min_intbc_cellbc_count=1,
        edit_distance_cutoff=0,
        min_cells_assigned=1,
        recursive=True,
        recursive_min_cells_assigned=2,
    )
    clone_call = clone_call_namedtuple.clone_call
    recursive_clone_call = clone_call_namedtuple.recursive_clone_call
    _check_are_clones(
        clone_call=clone_call,
        expected_clones=[
            [
                "sample1_cellBC_1",
                "sample1_cellBC_11",
                "sample1_cellBC_2",
                "sample1_cellBC_22",
                "sample1_cellBC_3",
                "sample1_cellBC_33",
                "sample1_cellBC_4",
                "sample1_cellBC_44",
                "sample1_cellBC_5",
                "sample1_cellBC_6",
            ],
        ],
    )
    _check_are_clones(
        clone_call=recursive_clone_call["clone_0"].clone_call,
        expected_clones=[
            [
                "sample1_cellBC_1",
                "sample1_cellBC_11",
                "sample1_cellBC_2",
                "sample1_cellBC_22",
            ],
            [
                "sample1_cellBC_3",
                "sample1_cellBC_33",
                "sample1_cellBC_4",
                "sample1_cellBC_44",
            ],
        ],
    )
    print(
        f"clone_call_namedtuple.recursive_clone_call_integration_hierarchy = {clone_call_namedtuple.recursive_clone_call_integration_hierarchy}"
    )
    assert clone_call_namedtuple.recursive_clone_call_integration_hierarchy == {
        "clone_0": (
            {"AAA", "BBB"},
            {
                "clone_0": ({"CCC"}, None),
                "clone_1": ({"DDD"}, None),
            },
        ),
    }


def test_recursive_clone_calling_algorithm_larger():
    """
    Same as test_recursive_clone_calling_algorithm_mini but with many cells.

    Subclone structure is:

        root
         |
        A,B  (first round of integration)
         |
        /  \
       C    D  (second round of integration)
       |    |
      ABC  ABD
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(0)

    n_cell_copies = 10

    intbc_table = pd.DataFrame(
        data=[
            [f"cellBC_1_{i}", "AAA", 40] for i in range(n_cell_copies)
        ] + [
            [f"cellBC_1_{i}", "BBB", 40] for i in range(n_cell_copies)
        ] + [
            [f"cellBC_1_{i}", "CCC", 40] for i in range(n_cell_copies)
        ] + [
            [f"cellBC_11_{i}", "AAA", 40] for i in range(n_cell_copies)
        ] + [
            [f"cellBC_11_{i}", "BBB", 40] for i in range(n_cell_copies)
        ] + [
            [f"cellBC_11_{i}", "CCC", 40] for i in range(n_cell_copies)
        ] + [
            [f"cellBC_2_{i}", "AAA", 40] for i in range(n_cell_copies)
        ] + [
            [f"cellBC_2_{i}", "BBB", 40] for i in range(n_cell_copies)
        ] + [
            [f"cellBC_2_{i}", "CCC", 40] for i in range(n_cell_copies)
        ] + [
            [f"cellBC_22_{i}", "AAA", 40] for i in range(n_cell_copies)
        ] + [
            [f"cellBC_22_{i}", "BBB", 40] for i in range(n_cell_copies)
        ] + [
            [f"cellBC_22_{i}", "CCC", 40] for i in range(n_cell_copies)
        ] + [
            [f"cellBC_3_{i}", "AAA", 40] for i in range(n_cell_copies)
        ] + [
            [f"cellBC_3_{i}", "BBB", 40] for i in range(n_cell_copies)
        ] + [
            [f"cellBC_3_{i}", "DDD", 40] for i in range(n_cell_copies)
        ] + [
            [f"cellBC_33_{i}", "AAA", 40] for i in range(n_cell_copies)
        ] + [
            [f"cellBC_33_{i}", "BBB", 40] for i in range(n_cell_copies)
        ] + [
            [f"cellBC_33_{i}", "DDD", 40] for i in range(n_cell_copies)
        ] + [
            [f"cellBC_4_{i}", "AAA", 40] for i in range(n_cell_copies)
        ] + [
            [f"cellBC_4_{i}", "BBB", 40] for i in range(n_cell_copies)
        ] + [
            [f"cellBC_4_{i}", "DDD", 40] for i in range(n_cell_copies)
        ] + [
            [f"cellBC_44_{i}", "AAA", 40] for i in range(n_cell_copies)
        ] + [
            [f"cellBC_44_{i}", "BBB", 40] for i in range(n_cell_copies)
        ] + [
            [f"cellBC_44_{i}", "DDD", 40] for i in range(n_cell_copies)
        ] + [
            [f"cellBC_5_{i}", "AAA", 40] for i in range(n_cell_copies)
        ] + [
            [f"cellBC_5_{i}", "BBB", 40] for i in range(n_cell_copies)
        ] + [
            [f"cellBC_6_{i}", "AAA", 40] for i in range(n_cell_copies)
        ] + [
            [f"cellBC_6_{i}", "BBB", 40] for i in range(n_cell_copies)
        ] + [
            [f"cellBC_6_{i}", "CCC", 40] for i in range(n_cell_copies)
        ] + [
            [f"cellBC_6_{i}", "DDD", 40] for i in range(n_cell_copies)
        ],
        columns=["cellBC", "intBC", "UMI_count"],
    )

    clone_call_namedtuple = clone_calling_algorithm(
        intbc_table_per_sample={"sample1": intbc_table},
        edit_distance_cutoff=0,
        recursive=True,
    )
    clone_call = clone_call_namedtuple.clone_call
    recursive_clone_call = clone_call_namedtuple.recursive_clone_call
    _check_are_clones(
        clone_call=clone_call,
        expected_clones=[  # all cells
            [f"sample1_cellBC_1_{i}" for i in range(n_cell_copies)]
            + [f"sample1_cellBC_11_{i}" for i in range(n_cell_copies)]
            + [f"sample1_cellBC_2_{i}" for i in range(n_cell_copies)]
            + [f"sample1_cellBC_22_{i}" for i in range(n_cell_copies)]
            + [f"sample1_cellBC_3_{i}" for i in range(n_cell_copies)]
            + [f"sample1_cellBC_33_{i}" for i in range(n_cell_copies)]
            + [f"sample1_cellBC_4_{i}" for i in range(n_cell_copies)]
            + [f"sample1_cellBC_44_{i}" for i in range(n_cell_copies)]
            + [f"sample1_cellBC_5_{i}" for i in range(n_cell_copies)]
            + [f"sample1_cellBC_6_{i}" for i in range(n_cell_copies)]
        ],
    )
    _check_are_clones(
        clone_call=recursive_clone_call["clone_0"].clone_call,
        expected_clones=[[]
            + [f"sample1_cellBC_1_{i}" for i in range(n_cell_copies)]
            + [f"sample1_cellBC_11_{i}" for i in range(n_cell_copies)]
            + [f"sample1_cellBC_2_{i}" for i in range(n_cell_copies)]
            + [f"sample1_cellBC_22_{i}" for i in range(n_cell_copies)]
        ] + [ []
            + [f"sample1_cellBC_3_{i}" for i in range(n_cell_copies)]
            + [f"sample1_cellBC_33_{i}" for i in range(n_cell_copies)]
            + [f"sample1_cellBC_4_{i}" for i in range(n_cell_copies)]
            + [f"sample1_cellBC_44_{i}" for i in range(n_cell_copies)]
        ],
    )
    print(
        f"clone_call_namedtuple.recursive_clone_call_integration_hierarchy = {clone_call_namedtuple.recursive_clone_call_integration_hierarchy}"
    )
    assert clone_call_namedtuple.recursive_clone_call_integration_hierarchy == {
        "clone_0": (
            {"AAA", "BBB"},
            {
                "clone_0": ({"CCC"}, None),
                "clone_1": ({"DDD"}, None),
            },
        ),
    }


def test_recursive_clone_calling_algorithm_on_real_data():
    """
    Tests the recursive clone calling algorithm on real data.
    """
    res = clone_calling_algorithm(
        intbc_table_per_sample={
            "3B": pd.read_csv(
                "notebooks/test_data/3B_clone_11_intBC_table_per_sample.csv"
            )
        },
        recursive=True,
    )

    clone_to_cells_mapping = res.clone_call.clone_to_cells_mapping
    clone_intbc_sets = res.clone_call.clone_intbc_sets
    cell_intbc_sets = res.clone_call.cell_intbc_sets

    clone = "clone_0"

    cells_in_clone = clone_to_cells_mapping[clone]

    intbc_co_counts = pd.DataFrame(
        np.zeros(shape=(len(clone_intbc_sets[clone]), len(clone_intbc_sets[clone]))),
        index=sorted(list(clone_intbc_sets[clone])),
        columns=sorted(list(clone_intbc_sets[clone])),
    )

    for cell in cells_in_clone:
        cell_intbc_list = list(cell_intbc_sets[cell])
        for i in range(len(cell_intbc_list)):
            intbc_i = cell_intbc_list[i]
            for j in range(len(cell_intbc_list)):
                intbc_j = cell_intbc_list[j]
                intbc_co_counts.loc[intbc_i, intbc_j] += 0.5
                intbc_co_counts.loc[intbc_j, intbc_i] += 0.5

    g = sns.clustermap(intbc_co_counts)

    ### This would show the co-occurrence matrix with the rows and columns reordered
    ### according to the hierarchical clustering.
    # # Get the reordered row and column indices
    # row_order = g.dendrogram_row.reordered_ind
    # col_order = g.dendrogram_col.reordered_ind

    # # Reorder the matrix
    # reordered_matrix = intbc_co_counts.iloc[row_order, col_order]

    # # Print or use the reordered matrix
    # print(reordered_matrix)

    assert res.recursive_clone_call_integration_hierarchy == {
        "clone_0": (
            {
                "CCTATGGAAAATGT",
                "CGTCGTAATCGTCT",
                "GCCCCTAATATAAG",
                "GCTAGCGACGTTCG",
                "TTTGCCCTGCAAAC",
            },
            {
                "clone_0": (
                    {
                        "AAATGAGCATTACG",
                        "CGGTTCAACGAGTA",
                        "GTAGTGACTGACTG",
                        "TCTTCAGTTTTAAT",
                    },
                    None,
                ),
                "clone_1": (
                    {
                        "ACATTGAACATCTT",
                        "ACATTGGAGGCCGA",
                        "ACTATTAATCTGCT",
                        "GGGCCCCGCGCTTA",
                    },
                    None,
                ),
                "clone_2": (
                    {"ATTCCTGCTGTACG", "CTTCAACCATTTTG", "GTTGAGTGTCGTCT"},
                    None,
                ),
                "clone_3": (
                    {
                        "CAAAATCTGATAAA",
                        "CTATTTTCATTAAT",
                        "GCTACCTGCCAGCG",
                        "GTCCCCCAATCTGC",
                        "TTCATTTCTTACGA",
                    },
                    None,
                ),
            },
        )
    }


def test_clone_calling_algorithm_doublet():
    """
    We check that cassiopeia can find the doublet.
    """
    ncells_per_clone = 100
    intbc_table = pd.DataFrame(
        data=[[f"cellBC_{i}", "AAA", 40] for i in range(ncells_per_clone)]
        + [[f"cellBC_{i}", "BBB", 40] for i in range(ncells_per_clone)]
        + [[f"cellBC_{i}", "CCC", 40] for i in range(ncells_per_clone)]
        + [
            [f"cellBC_{i}", "DDD", 40]
            for i in range(ncells_per_clone, 2 * ncells_per_clone)
        ]
        + [
            [f"cellBC_{i}", "EEE", 40]
            for i in range(ncells_per_clone, 2 * ncells_per_clone)
        ]
        + [
            [f"cellBC_{i}", "FFF", 40]
            for i in range(ncells_per_clone, 2 * ncells_per_clone)
        ]
        + [
            ["cellBC_doublet", "AAA", 40],
            ["cellBC_doublet", "BBB", 40],
            ["cellBC_doublet", "CCC", 40],
            ["cellBC_doublet", "DDD", 40],
            ["cellBC_doublet", "EEE", 40],
            ["cellBC_doublet", "FFF", 40],
        ],
        columns=["cellBC", "intBC", "UMI_count"],
    )

    clone_call = clone_calling_algorithm(
        intbc_table_per_sample={"sample1": intbc_table},
        edit_distance_cutoff=0,  # Turn off sequencing error correction
    ).clone_call

    _check_are_clones(
        clone_call=clone_call,
        expected_clones=[
            [f"sample1_cellBC_{i}" for i in range(ncells_per_clone)],
            [
                f"sample1_cellBC_{i}"
                for i in range(ncells_per_clone, 2 * ncells_per_clone)
            ],
        ],
    )
    assert clone_call.cell_to_clones_mapping["sample1_cellBC_doublet"] == set(
        ["clone_0", "clone_1"]
    )


def test_clone_calling_algorithm_single_intBC_clones():
    """
    NovaClone can recover single intBC clones too, by running with
    `recover_single_intbc_clones=True`.
    """
    ncells_per_clone = 100
    intbc_table = pd.DataFrame(
        data=[[f"cellBC_{i}", "AAA", 40] for i in range(ncells_per_clone)]
        + [[f"cellBC_{i}", "BBB", 40] for i in range(ncells_per_clone)]
        + [[f"cellBC_{i}", "CCC", 40] for i in range(ncells_per_clone)]
        + [
            [f"cellBC_{i}", "DDD", 40]
            for i in range(ncells_per_clone, 2 * ncells_per_clone)
        ]
        + [
            [f"cellBC_{i}", "EEE", 40]
            for i in range(ncells_per_clone, 2 * ncells_per_clone)
        ]
        + [
            [f"cellBC_{i}", "FFF", 40]
            for i in range(ncells_per_clone, 2 * ncells_per_clone)
        ] + [
            [f"cellBC_{i}", "GGG", 40]
            for i in range(2 * ncells_per_clone, 3 * ncells_per_clone)
        ] + [
            [f"cellBC_{i}", "HHH", 40]
            for i in range(3 * ncells_per_clone, 4 * ncells_per_clone)
        ]
        + [
            ["cellBC_doublet", "AAA", 40],
            ["cellBC_doublet", "BBB", 40],
            ["cellBC_doublet", "CCC", 40],
            ["cellBC_doublet", "DDD", 40],
            ["cellBC_doublet", "EEE", 40],
            ["cellBC_doublet", "FFF", 40],
        ] + [
            ["cellBC_doublet2", "AAA", 40],
            ["cellBC_doublet2", "BBB", 40],
            ["cellBC_doublet2", "CCC", 40],
            ["cellBC_doublet2", "GGG", 40],
        ] + [
            ["cellBC_doublet3", "GGG", 40],
            ["cellBC_doublet3", "HHH", 40],
        ] + [
            ["cellBC_doublet4", "AAA", 40],
            ["cellBC_doublet4", "BBB", 40],
            ["cellBC_doublet4", "GGG", 40],
            ["cellBC_doublet4", "HHH", 40],
        ] + [
            ["empty_droplet", "AAA", 1],
            ["empty_droplet", "BBB", 1],
            ["empty_droplet", "GGG", 1],
        ],
        columns=["cellBC", "intBC", "UMI_count"],
    )

    clone_call = clone_calling_algorithm(
        intbc_table_per_sample={"sample1": intbc_table},
        edit_distance_cutoff=0,  # Turn off sequencing error correction
        recover_single_intbc_clones=True,
    ).clone_call

    _check_are_clones(
        clone_call=clone_call,
        expected_clones=[
            [f"sample1_cellBC_{i}" for i in range(ncells_per_clone)],
            [
                f"sample1_cellBC_{i}"
                for i in range(ncells_per_clone, 2 * ncells_per_clone)
            ],
            [
                f"sample1_cellBC_{i}"
                for i in range(2 * ncells_per_clone, 3 * ncells_per_clone)
            ],
            [
                f"sample1_cellBC_{i}"
                for i in range(3 * ncells_per_clone, 4 * ncells_per_clone)
            ],
        ],
    )
    assert clone_call.cell_to_clones_mapping["sample1_cellBC_doublet"] == set(
        ["clone_0", "clone_1"]
    )
    assert clone_call.cell_to_clones_mapping["sample1_cellBC_doublet2"] == set(
        ["clone_0", "clone_2"]
    )
    assert clone_call.cell_to_clones_mapping["sample1_cellBC_doublet3"] == set(
        ["clone_2", "clone_3"]
    )
    assert clone_call.cell_to_clones_mapping["sample1_cellBC_doublet4"] == set(
        ["clone_0", "clone_2", "clone_3"]
    )
    assert clone_call.cell_to_clones_mapping["sample1_empty_droplet"] == set(
        []
    )


def test_clone_calling_algorithm_single_intBC_clones_trivial():
    """
    NovaClone can recover single intBC clones too, by running with
    `recover_single_intbc_clones=True`. Just one clone with one intBC.
    """
    ncells_per_clone = 100
    intbc_table = pd.DataFrame(
        data=[[f"cellBC_{i}", "AAA", 40] for i in range(ncells_per_clone)],
        columns=["cellBC", "intBC", "UMI_count"],
    )

    clone_call = clone_calling_algorithm(
        intbc_table_per_sample={"sample1": intbc_table},
        edit_distance_cutoff=0,  # Turn off sequencing error correction
        recover_single_intbc_clones=True,
    ).clone_call

    _check_are_clones(
        clone_call=clone_call,
        expected_clones=[
            [f"sample1_cellBC_{i}" for i in range(ncells_per_clone)],
        ],
    )


def test_clone_calling_algorithm_single_intBC_clones__border_case():
    f"""
    The pruned intBC graph has:
    ["a", "b"]
    ["a", "c"]
    ["d"]
    After making the coresets disjoint, this leads to coresets ["b"], ["c"], ["d"];
    - With a _final_cell_assignment_step__min_overlap_for_assigning_cellbc=1,
    the three clones remain, all doublets correctly identified.
    - With a _final_cell_assignment_step__min_overlap_for_assigning_cellbc=2,
    this should cause the ["b"] and ["c"] coresets to get no cells mapped (and thus drop off),
    versus ["d"] does get them since it's a single-intBC clone. However, ["d"] gets `cellBC_doublet2`
    mapped since it only overlaps in 1 intBC with the pruned coresets ["b"], ["c"].
    """
    ncells_per_clone = 100
    intbc_table = pd.DataFrame(
        data=[[f"cellBC_{i}", "AAA", 40] for i in range(ncells_per_clone)]
        + [[f"cellBC_{i}", "BBB", 40] for i in range(ncells_per_clone)]
        + [
            [f"cellBC_{i}", "AAA", 40]
            for i in range(ncells_per_clone, 2 * ncells_per_clone)
        ]
        + [
            [f"cellBC_{i}", "CCC", 40]
            for i in range(ncells_per_clone, 2 * ncells_per_clone)
        ]
        + [
            [f"cellBC_{i}", "DDD", 40]
            for i in range(2 * ncells_per_clone, 3 * ncells_per_clone)
        ]
        + [
            ["cellBC_doublet", "AAA", 40],
            ["cellBC_doublet", "BBB", 40],
            ["cellBC_doublet", "CCC", 40],
        ] + [
            ["cellBC_doublet2", "AAA", 40],
            ["cellBC_doublet2", "BBB", 40],
            ["cellBC_doublet2", "CCC", 40],
            ["cellBC_doublet2", "DDD", 40],
        ] + [
            ["empty_droplet", "AAA", 1],
            ["empty_droplet", "BBB", 1],
            ["empty_droplet", "GGG", 1],
        ],
        columns=["cellBC", "intBC", "UMI_count"],
    )

    clone_call_1 = clone_calling_algorithm(
        intbc_table_per_sample={"sample1": intbc_table},
        edit_distance_cutoff=0,  # Turn off sequencing error correction
        recover_single_intbc_clones=True,
    ).clone_call

    _check_are_clones(
        clone_call=clone_call_1,
        expected_clones=[
            [f"sample1_cellBC_{i}" for i in range(ncells_per_clone)],
            [f"sample1_cellBC_{i}" for i in range(ncells_per_clone, 2 * ncells_per_clone)],
            [f"sample1_cellBC_{i}" for i in range(2 * ncells_per_clone, 3 * ncells_per_clone)],
        ],
    )

    clone_call_2 = clone_calling_algorithm(
        intbc_table_per_sample={"sample1": intbc_table},
        edit_distance_cutoff=0,  # Turn off sequencing error correction
        recover_single_intbc_clones=True,
        _final_cell_assignment_step__min_overlap_for_assigning_cellbc=2,
    ).clone_call

    _check_are_clones(
        clone_call=clone_call_2,
        expected_clones=[
            [f"sample1_cellBC_{i}" for i in range(2 * ncells_per_clone, 3 * ncells_per_clone)] + ["sample1_cellBC_doublet2"],
        ],
    )
