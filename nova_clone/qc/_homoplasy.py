from collections import defaultdict
from typing import Dict, Set

import numpy as np

from nova_clone.algorithm._types import CloneCallType, CloneType, IntBCType


def qc_homoplastic_intbcs(
    clone_call: CloneCallType,
    verbose: bool = True,
) -> Dict[IntBCType, Set[CloneType]]:
    """
    Returns the homoplastic intBCs and the clones in which they appear.
    """
    clone_intbc_sets = clone_call.clone_intbc_sets
    intbc_to_clones = defaultdict(set)
    for clone, intbc_set in clone_intbc_sets.items():
        for intbc in intbc_set:
            intbc_to_clones[intbc].add(clone)
    res = {
        intbc: clones for intbc, clones in intbc_to_clones.items() if len(clones) >= 2
    }

    # For the homoplastic intBCs, compute the abundance in each clone it appears.
    cell_intbc_sets = clone_call.cell_intbc_sets
    clone_to_cells_mapping = clone_call.clone_to_cells_mapping
    abundances_dict = {}
    numerator_dict = {}
    denominator_dict = {}
    for intbc, clones in res.items():
        abundances_dict[intbc] = {}
        numerator_dict[intbc] = {}
        denominator_dict[intbc] = {}
        for clone in clones:
            numerator_dict[intbc][clone] = np.sum(
                [
                    intbc in cell_intbc_sets[cell]
                    for cell in clone_to_cells_mapping[clone]
                ]
            )
            denominator_dict[intbc][clone] = len(clone_to_cells_mapping[clone])
            abundances_dict[intbc][clone] = (
                numerator_dict[intbc][clone] / denominator_dict[intbc][clone]
            )

    if verbose:
        print(f"******************************")
        print(f"***** Homoplasy Analysis *****")
        print(f"******************************")
        if len(res) == 0:
            print(f"No homoplastic intBCs found amongst the called clones.")
        for intbc, clones in res.items():
            output_str = (
                f"Homoplastic IntBC {intbc} shared by clones: {clones}. "
                "We suggest, at your own discretion, either (i) keeping these clones, (ii) merging these clones, "
                "(iii) dropping these clones. You may merge the clones using the nova_clone.algorithm.merge_clones "
                "function, or drop clones using the nova_clone.algorithm.drop_clones function ."
            )
            if any([len(clone_intbc_sets[clone]) <= 10 for clone in clones]):
                output_str += f" Clone intBC sets (for those with <= 10 intBCs):"
                for clone in clones:
                    if len(clone_intbc_sets[clone]) <= 10:
                        output_str += f"\n{clone}: {clone_intbc_sets[clone]}"
            output_str += f"\n\tAbundances: "
            for clone in clones:
                output_str += f"\n\t\t{clone}: {100 * abundances_dict[intbc][clone]:.0f}% = {numerator_dict[intbc][clone]} / {denominator_dict[intbc][clone]}"
            print(output_str)

    return res


def test_qc_homoplastic_intbcs():
    clone_call = CloneCallType(
        clone_to_cells_mapping={"clone_0": {"c0"}, "clone_1": {"c1"}},
        cell_to_clones_mapping={"c0": {"clone_0"}, "c1": {"clone_1"}},
        clone_intbc_sets={"clone_0": {"X", "A"}, "clone_1": {"X", "B"}},
        cell_intbc_sets={"c0": {"X", "A"}, "c1": {"X", "B"}},
    )
    res = qc_homoplastic_intbcs(clone_call=clone_call, verbose=False)
    assert res == {"X": {"clone_0", "clone_1"}}
