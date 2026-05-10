from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm

from nova_clone.algorithm._types import (
    CellBCType,
    CloneCallType,
    CloneType,
    IntBCType,
)

PLT_SAVEFIG_KWARGS = {
    "bbox_inches": "tight",
    "dpi": 300,
}


def qc_inter_clonal_kinship(
    clone_intbc_sets: Dict[CloneType, Set[IntBCType]],
    max_overlap: int = 1,
    verbose: bool = True,
) -> List[Tuple[CloneType, CloneType, Set[IntBCType]]]:
    """
    Reports pairs of clones with unusually high kinship.
    """
    print(f"*****************************************")
    print(f"***** Inter-Clonal Kinship Analysis *****")
    print(f"*****************************************")
    res = []
    for clone_1, intbc_set_1 in clone_intbc_sets.items():
        for clone_2, intbc_set_2 in clone_intbc_sets.items():
            if clone_1 >= clone_2:
                continue
            overlap_set = intbc_set_1.intersection(intbc_set_2)
            overlap = len(overlap_set)
            if overlap > max_overlap:
                res.append((clone_1, clone_2, overlap_set))
                print(
                    f"Clones {clone_1} and {clone_2} share {overlap} intBCs: {overlap_set}.\n"
                    f"This suggests that {clone_1} and {clone_2} might actually be the same clone.\n"
                    f"Consider merging these clones or excluding them from your downstream analysis."
                )
    return res


def test_qc_inter_clonal_kinship():
    clone_intbc_sets = {
        "A": {"a", "b", "c"},
        "B": {"b", "c", "d"},
        "C": {"c", "d", "e"},
    }
    res = qc_inter_clonal_kinship(
        clone_intbc_sets=clone_intbc_sets, max_overlap=1, verbose=False
    )
    assert res == [("A", "B", {"b", "c"}), ("B", "C", {"c", "d"})]
