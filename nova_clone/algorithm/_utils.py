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
    cell_intbc_sets_with_counts = _cell_intbc_sets_with_counts__for_intbc_table(
        intbc_table=intbc_table,
    )
    cell_intbc_sets = {
        cell: {intbc for (intbc, counts) in intbcs_with_counts.items()}
        for (cell, intbcs_with_counts) in cell_intbc_sets_with_counts.items()
    }
    return cell_intbc_sets


def _cell_intbc_sets_from_intbc_tables(
    intbc_table_per_sample: Dict[SampleNameType, IntBCTableType],
) -> Dict[str, Set[str]]:
    """
    Get the intBC sets corresponding to each cell.
    """
    res = {}
    for sample_name, intbc_table in intbc_table_per_sample.items():
        cell_intbc_sets = _cell_intbc_set_from_intbc_table(
            intbc_table=intbc_table,
        )
        for x, y in cell_intbc_sets.items():
            assert x not in res
            res[f"{sample_name}_" + x] = y
    return res
