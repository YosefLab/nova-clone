from copy import deepcopy

from cassiopeia.preprocess.pipeline import (
    call_lineage_groups as call_lineage_groups_cassiopeia,
)
from nova_clone.algorithm._prune_low_umi_count_intbcs import prune_low_umi_count_intbcs

from nova_clone.algorithm._prune_low_umi_count_intbcs import (
    prune_low_umi_count_intbcs,
)

from .._clone_calling_algorithm import _check_are_clones
from .._types import *


def _build_input_df(intbc_table_per_sample) -> pd.DataFrame:
    rows = []
    for sample_name, intbc_table in intbc_table_per_sample.items():
        for _, row in intbc_table.iterrows():
            row_dict = row.to_dict()
            res = {
                "cellBC": f"{sample_name}_{row_dict['cellBC']}",
                "UMI": row_dict["UMI_count"],
                "intBC": row_dict["intBC"],
                "allele": "A",
                "readCount": 1000,
            }
            rows.append(res)
    return pd.DataFrame(rows)


def call_lineage_groups(
    intbc_table_per_sample: Dict[SampleNameType, IntBCTableType],
    min_umi_per_cell: int = 10,
    min_avg_reads_per_umi: float = 2.0,
    min_cluster_prop: float = 0.005,
    min_intbc_thresh: float = 0.05,
    inter_doublet_threshold: float = 0.35,
    kinship_thresh: float = 0.25,
    plot: bool = False,
    min_intbc_umi_count: int = 0,  # Default value is 0 because it is not used in the cassiopeia code. This is expected to be part of user preprocessing.
) -> CloneCallType:
    """
    Wrapper around Cassiopeia's call_lineage_groups function.

    Assigns cells to their clonal populations.

    Performs multiple rounds of filtering and assigning to lineage groups:
        1. Iteratively generates putative lineage groups by forming intBC
        groups for each lineage group and then assigning cells based on how
        many intBCs they share with each intBC group (kinship).

        2. Refines these putative groups by removing non-informative intBCs
        and reassigning cells through kinship.

        3. Removes all inter-lineage doublets, defined as cells that have
        relatively equal kinship scores across multiple lineages and whose
        assignments are therefore ambigious.

        4. Finally, performs one more round of filtering non-informative intBCs
        and cellBCs with low UMI counts before returning a final table of
        lineage assignments, allele information, and read and umi counts for
        each sample.

    Args:
        intbc_table_per_sample: The intBC table per sample.
        min_umi_per_cell: The threshold specifying the minimum number of UMIs a
            cell needs in order to not be filtered during filtering
        min_avg_reads_per_umi: The threshold specifying the minimum coverage
            (i.e. average) reads per UMI in a cell needed in order for that
            cell not to be filtered during filtering
        min_cluster_prop: The minimum cluster size in the putative lineage
            assignment step, as a proportion of the number of cells
        min_intbc_thresh: The threshold specifying the minimum proportion of
            cells in a lineage group that need to have an intBC in order for it
            be retained during filtering. Also specifies the minimum proportion
            of cells that share an intBC with the most frequent intBC in
            forming putative lineage groups
        inter_doublet_threshold: The threshold specifying the minimum proportion
            of kinship a cell shares with its assigned lineage group out of all
            lineage groups for it to be retained during doublet filtering
        kinship_thresh: The threshold specifying the minimum proportion of
            intBCs shared between a cell and the intBC set of a lineage group
            needed to assign that cell to that lineage group in putative
            assignment
        plot: Indicates whether to generate plots. If so, they will be written to
            `output_directory_cassiopeia`.
        min_intbc_umi_count: The minimum number of UMIs for an intBC to be considered
            in a cellBC.

    Returns:
        The clone call, with fields `clone_to_cells_mapping, cell_to_clones_mapping, clone_intbc_sets, cell_intbc_sets`.
    """
    validate_intbc_table_per_sample(intbc_table_per_sample=intbc_table_per_sample)
    original_intbc_table_per_sample = deepcopy(intbc_table_per_sample)
    intbc_table_per_sample = deepcopy(intbc_table_per_sample)

    # Filter intBCs based on min_intbc_umi_count
    intbc_table_per_sample = prune_low_umi_count_intbcs(
        intbc_table_per_sample=intbc_table_per_sample,
        min_intbc_umi_count=min_intbc_umi_count,
    ).intbc_table_per_sample

    input_df = _build_input_df(intbc_table_per_sample)
    lineage_groups = call_lineage_groups_cassiopeia(
        input_df=input_df,
        output_directory="output_directory_cassiopeia",
        min_umi_per_cell=min_umi_per_cell,
        min_avg_reads_per_umi=min_avg_reads_per_umi,
        min_cluster_prop=min_cluster_prop,
        min_intbc_thresh=min_intbc_thresh,
        inter_doublet_threshold=inter_doublet_threshold,
        kinship_thresh=kinship_thresh,
        plot=plot,
    )

    cell_to_clones_mapping = {}
    cell_intbc_sets = {}
    clone_intbc_sets = defaultdict(set)
    for _, row in lineage_groups.iterrows():
        row = row.to_dict()
        clone = "clone_" + str(row["lineageGrp"])
        cell = str(row["cellBC"])
        if cell not in cell_to_clones_mapping:
            cell_to_clones_mapping[cell] = set()
        cell_to_clones_mapping[cell].add(clone)
        clone_intbc_sets[clone].add(row["intBC"])
        if cell not in cell_intbc_sets:
            cell_intbc_sets[cell] = set()
        cell_intbc_sets[cell].add(row["intBC"])
    # Add cellBCs that were not assigned to any clone
    for sample, intbc_table in original_intbc_table_per_sample.items():
        for _, row in intbc_table.iterrows():
            row = row.to_dict()
            cellbc = f"{sample}_{row['cellBC']}"
            if cellbc not in cell_to_clones_mapping:
                cell_to_clones_mapping[cellbc] = set()
    clone_to_cells_mapping = defaultdict(set)
    for cell, clones in cell_to_clones_mapping.items():
        for clone in clones:
            clone_to_cells_mapping[clone].add(cell)
    clone_intbc_sets = {x: y for (x, y) in clone_intbc_sets.items()}
    clone_call = CloneCallType(
        cell_to_clones_mapping=cell_to_clones_mapping,
        clone_to_cells_mapping=clone_to_cells_mapping,
        clone_intbc_sets=clone_intbc_sets,
        cell_intbc_sets=cell_intbc_sets,
    )
    return clone_call


def test_call_lineage_groups():
    """
    Just tests that the clone calling algorithm runs on a trivial case.

    In this test case, there are two clones characterized by the intBC sets
    ["AAA", "BBB", "CCC"] and ["DDD", "EEE", "FFF"] respectively. The first
    clone contains the cellBCs ["cellBC_1", "cellBC_2"] whereas the second
    clone contains the cellBCs ["cellBC_3", "cellBC_4"].
    """
    intbc_table = pd.DataFrame(
        data=[
            ["cellBC_1", "AAA", 40],
            ["cellBC_1", "BBB", 40],
            ["cellBC_1", "CCC", 40],
            ["cellBC_2", "AAA", 40],
            ["cellBC_2", "BBB", 40],
            ["cellBC_2", "CCC", 40],
            ["cellBC_3", "DDD", 40],
            ["cellBC_3", "EEE", 40],
            ["cellBC_3", "FFF", 40],
            ["cellBC_4", "DDD", 40],
            ["cellBC_4", "EEE", 40],
            ["cellBC_4", "FFF", 40],
            ["cellBC_5", "DDD", 40],
            ["cellBC_5", "EEE", 40],
            ["cellBC_5", "FFF", 40],
        ],
        columns=["cellBC", "intBC", "UMI_count"],
    )

    clone_call = call_lineage_groups(
        intbc_table_per_sample={"sample1": intbc_table},
        min_umi_per_cell=1,
        min_avg_reads_per_umi=2.0,
        min_cluster_prop=0.005,
        min_intbc_thresh=0.05,
        inter_doublet_threshold=0.35,
        kinship_thresh=0.25,
        plot=False,
    )

    _check_are_clones(
        clone_call=clone_call,
        expected_clones=[
            ["sample1_cellBC_1", "sample1_cellBC_2"],
            ["sample1_cellBC_3", "sample1_cellBC_4", "sample1_cellBC_5"],
        ],
    )
    assert clone_call.clone_intbc_sets["clone_2"] == set(["AAA", "BBB", "CCC"])
    assert clone_call.clone_intbc_sets["clone_1"] == set(["DDD", "EEE", "FFF"])
    assert clone_call.cell_to_clones_mapping["sample1_cellBC_1"] == set(["clone_2"])
    assert clone_call.cell_to_clones_mapping["sample1_cellBC_2"] == set(["clone_2"])
    assert clone_call.cell_to_clones_mapping["sample1_cellBC_3"] == set(["clone_1"])
    assert clone_call.cell_to_clones_mapping["sample1_cellBC_4"] == set(["clone_1"])
    assert clone_call.cell_to_clones_mapping["sample1_cellBC_5"] == set(["clone_1"])


def test_call_lineage_groups_doublet():
    """
    We check that cassiopeia can find the doublet.
    """
    ncells_per_clone = 20
    intbc_table = pd.DataFrame(
        data=[[f"cellBC_{i}", "AAA", 40] for i in range(ncells_per_clone)]
        + [[f"cellBC_{i}", "BBB", 40] for i in range(ncells_per_clone)]
        + [[f"cellBC_{i}", "CCC", 40] for i in range(ncells_per_clone)]
        + [
            [f"cellBC_{i}", "DDD", 40]
            for i in range(
                ncells_per_clone, 2 * ncells_per_clone + 1
            )  # One more cell to ensure it is the first clone
        ]
        + [
            [f"cellBC_{i}", "EEE", 40]
            for i in range(
                ncells_per_clone, 2 * ncells_per_clone + 1
            )  # One more cell to ensure it is the first clone
        ]
        + [
            [f"cellBC_{i}", "FFF", 40]
            for i in range(
                ncells_per_clone, 2 * ncells_per_clone + 1
            )  # One more cell to ensure it is the first clone
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

    clone_call = call_lineage_groups(
        intbc_table_per_sample={"sample1": intbc_table},
        min_umi_per_cell=1,
        min_avg_reads_per_umi=2.0,
        min_cluster_prop=0.005,
        min_intbc_thresh=0.05,
        inter_doublet_threshold=0.55,  # With default value 0.35, the doublet is not detected! Indeed, it get's kinship 0.5 for both clones.
        kinship_thresh=0.25,
        plot=False,
    )

    _check_are_clones(
        clone_call=clone_call,
        expected_clones=[
            [f"sample1_cellBC_{i}" for i in range(ncells_per_clone)],
            [
                f"sample1_cellBC_{i}"
                for i in range(ncells_per_clone, 2 * ncells_per_clone + 1)
            ],
        ],
    )
    assert clone_call.clone_intbc_sets["clone_2"] == set(["AAA", "BBB", "CCC"])
    assert clone_call.clone_intbc_sets["clone_1"] == set(["DDD", "EEE", "FFF"])
    # assert(clone_call.cell_to_clones_mapping["sample1_cellBC_doublet"] == set(["clone_0", "clone_1"]))  # Cassiopeia code does not return doublet identity.
