from typing import Set

from ._types import *


def _cell_intbc_sets_with_counts__for_intbc_table(
    intbc_table: IntBCTableType,
) -> Dict[CellBCType, Dict[IntBCType, UMICountType]]:
    """
    Get the intBC sets corresponding to each cell, w/#UMIs.
    """
    validate_intbc_table(intbc_table)
    cell_intbc_sets_with_counts = {}
    for cellbc, cell_df in intbc_table.groupby(
        "cellBC"
    ):  # Not using tqdm bc its fast (<10 sec)
        # Lets see what intBCs this cell has
        intbc_and_umi_count = list(zip(list(cell_df.intBC), list(cell_df.UMI_count)))
        cell_intbc_sets_with_counts[cellbc] = dict(intbc_and_umi_count)
    return cell_intbc_sets_with_counts


def _cell_intbc_set_from_intbc_table(
    intbc_table: IntBCTableType,
) -> Dict[str, Set[str]]:
    """
    Get the intBC sets corresponding to each cell.
    """
    validate_intbc_table(intbc_table)
    cell_intbc_sets: Dict[str, Set[str]] = {}
    # Iterate raw column values to avoid pandas groupby/row overhead.
    cellbcs = intbc_table["cellBC"].to_numpy()
    intbcs = intbc_table["intBC"].to_numpy()
    for cellbc, intbc in zip(cellbcs, intbcs):
        cell_intbc_sets.setdefault(cellbc, set()).add(intbc)
    return cell_intbc_sets


def _cell_intbc_sets_from_intbc_tables(
    intbc_table_per_sample: Dict[SampleNameType, IntBCTableType],
) -> Dict[str, Set[str]]:
    """
    Get the intBC sets corresponding to each cell.
    """
    res: Dict[str, Set[str]] = {}
    for sample_name, intbc_table in intbc_table_per_sample.items():
        validate_intbc_table(intbc_table)
        prefix = f"{sample_name}_"
        cellbcs = intbc_table["cellBC"].to_numpy()
        intbcs = intbc_table["intBC"].to_numpy()
        for cellbc, intbc in zip(cellbcs, intbcs):
            res.setdefault(prefix + cellbc, set()).add(intbc)
    return res


def test__cell_intbc_sets_from_intbc_tables():
    intbc_table_1 = pd.DataFrame(
        data=[
            ["cellBC_1", "AAA", 40],
            ["cellBC_1", "BBB", 10],
            ["cellBC_2", "AAA", 40],
            ["cellBC_2", "CCC", 40],
        ],
        columns=["cellBC", "intBC", "UMI_count"],
    )
    intbc_table_2 = pd.DataFrame(
        data=[
            ["cellBC_1", "DDD", 20],
            ["cellBC_1", "EEE", 5],
            ["cellBC_3", "FFF", 40],
        ],
        columns=["cellBC", "intBC", "UMI_count"],
    )
    res = _cell_intbc_sets_from_intbc_tables(
        intbc_table_per_sample={
            "sample1": intbc_table_1,
            "sample2": intbc_table_2,
        },
    )
    assert res == {
        "sample1_cellBC_1": {"AAA", "BBB"},
        "sample1_cellBC_2": {"AAA", "CCC"},
        "sample2_cellBC_1": {"DDD", "EEE"},
        "sample2_cellBC_3": {"FFF"},
    }
