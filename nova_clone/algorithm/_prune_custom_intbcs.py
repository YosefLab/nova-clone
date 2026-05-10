"""
This module implements filtering of intBCs based on a custom function.

The main function exposed by this module is `prune_custom_intbcs`.
"""

import logging
import sys
import time
from typing import Callable

from ._types import *


def _init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)


_init_logger()


class PruneCustomIntBCsReturnType(NamedTuple):
    """
    Object returned by the prune_custom_intbcs function.

    Attributes:
        intbc_table_per_sample: The filtered `intbc_table_per_sample`,
            containing only the high-quality intBCs meeting the
            `min_intbc_umi_count`
        fraction_discarded_intbcs: What fraction of intBCs (i.e. rows in the
            intBC tables) were discarded by the
            `custom_intbc_filtering_function`.
        total_time: Total time taken.
    """

    intbc_table_per_sample: Dict[SampleNameType, IntBCTableType]
    fraction_discarded_intbcs: float
    total_time: float


def prune_custom_intbcs(
    intbc_table_per_sample: Dict[SampleNameType, IntBCTableType],
    custom_intbc_filtering_function: Callable[[str], bool],
) -> PruneCustomIntBCsReturnType:
    """
    Prunes away all intBCs for which `custom_intbc_filtering_function` is True.

    The purpose of this function is to get rid of extraneous intBC sequences,
    e.g. sequences that are too short or contain ambiguous bases such as 'N'.
    """
    st = time.time()
    logger = logging.getLogger(__name__)
    logger.info("Going to prune_custom_intbcs ...")
    validate_intbc_table_per_sample(intbc_table_per_sample)
    new_intbc_table_per_sample = {}
    for sample_name, intbc_table in intbc_table_per_sample.items():
        validate_intbc_table(intbc_table)
        mask = ~intbc_table["intBC"].apply(custom_intbc_filtering_function)
        new_intbc_table = intbc_table.loc[mask].reset_index(drop=True)
        validate_intbc_table(new_intbc_table)
        new_intbc_table_per_sample[sample_name] = new_intbc_table
    denominator = sum(
        intbc_table.shape[0] for intbc_table in intbc_table_per_sample.values()
    )
    numerator = denominator - sum(
        intbc_table.shape[0] for intbc_table in new_intbc_table_per_sample.values()
    )
    fraction_discarded_intbcs = numerator / (denominator + 1e-9)
    validate_intbc_table_per_sample(new_intbc_table_per_sample)
    total_time = time.time() - st
    res = PruneCustomIntBCsReturnType(
        intbc_table_per_sample=new_intbc_table_per_sample,
        fraction_discarded_intbcs=fraction_discarded_intbcs,
        total_time=total_time,
    )
    logger.info(
        f"Pruned {100.0 * fraction_discarded_intbcs:.1f}% = {numerator} / {denominator} rows from the intBC tables. [Lower is better.]"
    )
    logger.info(f"Done with prune_custom_intbcs! Total time: {total_time}")
    return res


def test_prune_custom_intbcs():
    intbc_table_1 = pd.DataFrame(
        data=[
            ["cellBC_1", "AAA", 40],
            ["cellBC_1", "BBB", 40],
            ["cellBC_2", "AAA", 40],
            ["cellBC_2", "BBB", 40],
            ["cellBC_2", "CCC", 40],
            ["cellBC_3", "DDD", 40],
            ["cellBC_3", "EEE", 40],
            ["cellBC_3", "AA", 40],
            ["cellBC_3", "NNN", 40],
        ],
        columns=["cellBC", "intBC", "UMI_count"],
    )
    intbc_table_2 = pd.DataFrame(
        data=[
            ["cellBC_1", "NNN", 40],
            ["cellBC_1", "A", 40],
            ["cellBC_1", "FFF", 40],
            ["cellBC_2", "GGG", 40],
        ],
        columns=["cellBC", "intBC", "UMI_count"],
    )
    res = prune_custom_intbcs(
        intbc_table_per_sample={
            "sample1": intbc_table_1,
            "sample2": intbc_table_2,
        },
        custom_intbc_filtering_function=lambda intbc: len(intbc) <= 2 or "N" in intbc,
    )
    filtered_intbc_table_per_sample = res.intbc_table_per_sample
    expected_intbc_table_per_sample = {
        "sample1": pd.DataFrame(
            data=[
                ["cellBC_1", "AAA", 40],
                ["cellBC_1", "BBB", 40],
                ["cellBC_2", "AAA", 40],
                ["cellBC_2", "BBB", 40],
                ["cellBC_2", "CCC", 40],
                ["cellBC_3", "DDD", 40],
                ["cellBC_3", "EEE", 40],
            ],
            columns=["cellBC", "intBC", "UMI_count"],
        ),
        "sample2": pd.DataFrame(
            data=[
                ["cellBC_1", "FFF", 40],
                ["cellBC_2", "GGG", 40],
            ],
            columns=["cellBC", "intBC", "UMI_count"],
        ),
    }
    assert (
        filtered_intbc_table_per_sample.keys() == expected_intbc_table_per_sample.keys()
    )
    for sample in filtered_intbc_table_per_sample.keys():
        pd.testing.assert_frame_equal(
            filtered_intbc_table_per_sample[sample],
            expected_intbc_table_per_sample[sample],
        )
