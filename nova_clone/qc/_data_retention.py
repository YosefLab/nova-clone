from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

from nova_clone.algorithm._types import *

PLT_SAVEFIG_KWARGS = {
    "bbox_inches": "tight",
    "dpi": 300,
}


def qc_data_retention(
    intbc_table_per_sample: Dict[SampleNameType, IntBCTableType],
    clone_call: CloneCallType,
    fig_path: Optional[str] = None,
    plot: bool = True,
    only_kept_intbcs: bool = True,
):
    """
    Percentage of retained UMIs of successfully mapped cells.

    A cellBC is considered successfully mapped if it is assigned
    to exactly one clone.
    """
    print(f"***********************************")
    print(f"***** Data Retention Analysis *****")
    print(f"***********************************")
    successfully_mapped_cellbcs = sorted(
        [
            cellbc
            for cellbc, clones in clone_call.cell_to_clones_mapping.items()
            if len(clones) == 1
        ]
    )
    total_cellbcs = len(clone_call.cell_to_clones_mapping)
    assert len(successfully_mapped_cellbcs) > 0
    kept_umis = {cellbc: 0 for cellbc in successfully_mapped_cellbcs}
    discarded_umis = {cellbc: 0 for cellbc in successfully_mapped_cellbcs}
    for sample_name, intbc_table in intbc_table_per_sample.items():
        intbc_table = deepcopy(intbc_table)
        intbc_table["cellBC"] = sample_name + "_" + intbc_table["cellBC"]
        intbc_table = intbc_table[
            intbc_table["cellBC"].isin(successfully_mapped_cellbcs)
        ]
        for _, row in intbc_table.iterrows():
            row_dict = row.to_dict()
            cellbc = row_dict["cellBC"]
            intbc = row_dict["intBC"]
            umi_count = row_dict["UMI_count"]
            clones = list(clone_call.cell_to_clones_mapping[cellbc])
            assert (len(clones)) == 1
            clone = clones[0]
            set_to_check = clone_call.clone_intbc_sets[clone]
            if only_kept_intbcs:
                set_to_check = set_to_check.intersection(
                    clone_call.cell_intbc_sets[cellbc]
                )
            if intbc in set_to_check:
                kept_umis[cellbc] += umi_count
            else:
                discarded_umis[cellbc] += umi_count

    total_kept_umis = sum(kept_umis.values())
    total_discarded_umis = sum(discarded_umis.values())
    print(
        f"Percent of UMIs kept in successfully mapped cellBCs: {total_kept_umis / (total_kept_umis + total_discarded_umis):.2%} [higher is better]\n"
    )
    percent_kept_distro = [
        100 * kept_umis[cellbc] / (kept_umis[cellbc] + discarded_umis[cellbc])
        for cellbc in successfully_mapped_cellbcs
    ]
    plt.hist(
        percent_kept_distro,
        bins=50,
        range=(0, 100),
        color="blue",
        alpha=0.5,
    )
    plt.xlabel("Percent of UMIs kept in cellBC")
    plt.ylabel("Number of cells")
    plt.title(
        f"Percent of UMIs kept in successfully mapped cells\n"
        f"({len(successfully_mapped_cellbcs)} cellBCs out of {total_cellbcs} total cellBCs)"
    )
    if fig_path is not None:
        plt.savefig(
            fig_path,
            **PLT_SAVEFIG_KWARGS,
        )
    if plot:
        plt.show()
    plt.close()
    return kept_umis, discarded_umis


def test_qc_data_retention():
    intbc_table = pd.DataFrame(
        data=[
            ["cellBC_1", "AAA", 40],
            ["cellBC_1", "AAB", 5],
            ["cellBC_1", "BBB", 41],
            ["cellBC_1", "CCC", 42],
            ["cellBC_2", "AAA", 10],
            ["cellBC_2", "BBB", 20],
            ["cellBC_2", "BBZ", 2],
            ["cellBC_2", "ZZZ", 4],
            ["cellBC_3", "DDD", 40],
            ["cellBC_3", "EEE", 40],
            ["cellBC_3", "FFF", 40],
            ["cellBC_4", "DDD", 40],
            ["cellBC_4", "EEE", 40],
            ["cellBC_4", "FFF", 40],
            ["cellBC_5", "DDD", 40],
            ["cellBC_5", "EEE", 40],
            ["cellBC_5", "FFF", 40],
            ["cellBC_5", "GGG", 40],
            ["cellBC_6", "AAA", 30],  # an unmapped cell
            ["cellBC_7", "AAA", 40],  # a doublet
            ["cellBC_7", "DDD", 40],  # a doublet
            ["cellBC_7", "DDE", 1],  # a doublet
            ["cellBC_8", "ZZZ", 33],  # noise
        ],
        columns=["cellBC", "intBC", "UMI_count"],
    )
    clone_call = CloneCallType(
        cell_to_clones_mapping={
            "sample0_cellBC_1": {"clone_1"},
            "sample0_cellBC_2": {"clone_1"},
            "sample0_cellBC_3": {"clone_2"},
            "sample0_cellBC_4": {"clone_2"},
            "sample0_cellBC_5": {"clone_2"},
            "sample0_cellBC_6": {},
            "sample0_cellBC_7": {"clone_1", "clone_2"},
            "sample0_cellBC_8": {"clone_1"},
        },
        clone_intbc_sets={
            "clone_1": {"AAA", "BBB", "CCC"},
            "clone_2": {"DDD", "EEE", "FFF"},
        },
        cell_intbc_sets={
            "sample0_cellBC_1": {"AAA", "BBB", "CCC"},
            "sample0_cellBC_2": {"AAA", "BBB", "CCC"},
            "sample0_cellBC_3": {"DDD", "EEE", "FFF"},
            "sample0_cellBC_4": {"DDD", "EEE", "FFF"},
            "sample0_cellBC_5": {"DDD", "EEE", "FFF"},
            "sample0_cellBC_6": set(),
            "sample0_cellBC_7": {"AAA", "DDD"},
            "sample0_cellBC_8": {},
        },
        clone_to_cells_mapping=None,
    )
    total_kept_umis, total_discarded_umis = qc_data_retention(
        intbc_table_per_sample={"sample0": intbc_table},
        clone_call=clone_call,
        plot=False,
    )
    assert total_kept_umis == {
        "sample0_cellBC_1": 123,
        "sample0_cellBC_2": 30,
        "sample0_cellBC_3": 120,
        "sample0_cellBC_4": 120,
        "sample0_cellBC_5": 120,
        "sample0_cellBC_8": 0,
    }
    assert total_discarded_umis == {
        "sample0_cellBC_1": 5,
        "sample0_cellBC_2": 6,
        "sample0_cellBC_3": 0,
        "sample0_cellBC_4": 0,
        "sample0_cellBC_5": 40,
        "sample0_cellBC_8": 33,
    }
