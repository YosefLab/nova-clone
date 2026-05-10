"""
This module implements filtering of intBCs based on UMI counts.

The main function exposed by this module is `prune_low_umi_count_intbcs`.
"""

import logging
import sys
import time

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


class PruneLowUMICountIntBCsReturnType(NamedTuple):
    """
    Object returned by the prune_low_umi_count_intbcs function.

    Attributes:
        intbc_table_per_sample: The filtered `intbc_table_per_sample`,
            containing only the high-quality intBCs meeting the
            `min_intbc_umi_count`
        fraction_discarded_intbcs: What fraction of intBCs (i.e. rows in the
            intBC tables) were discarded by the `min_intbc_umi_count`
            threshold.
        total_time: Total time taken.
    """

    intbc_table_per_sample: Dict[SampleNameType, IntBCTableType]
    fraction_discarded_intbcs: float
    total_time: float


def prune_low_umi_count_intbcs(
    intbc_table_per_sample: Dict[SampleNameType, IntBCTableType],
    min_intbc_umi_count: int,
) -> PruneLowUMICountIntBCsReturnType:
    """
    Prunes away all intBCs with low UMI counts.

    This is effective at getting rid of background RNA contaminants and many
    (but not all) sequencing errors.
    """
    st = time.time()
    logger = logging.getLogger(__name__)
    logger.info("Going to prune_low_umi_count_intbcs ...")
    validate_intbc_table_per_sample(intbc_table_per_sample)
    new_intbc_table_per_sample = {}
    for sample_name, intbc_table in intbc_table_per_sample.items():
        new_intbc_table = intbc_table[
            intbc_table.UMI_count >= min_intbc_umi_count
        ].reset_index(drop=True)
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
    res = PruneLowUMICountIntBCsReturnType(
        intbc_table_per_sample=new_intbc_table_per_sample,
        fraction_discarded_intbcs=fraction_discarded_intbcs,
        total_time=total_time,
    )
    logger.info(
        f"Pruned {100.0 * fraction_discarded_intbcs:.1f}% = {numerator} / {denominator} rows from the intBC tables. [Lower is better.]"
    )
    logger.info(f"Done with prune_low_umi_count_intbcs! Total time: {total_time}")
    return res


def test_prune_low_umi_count_intbcs():
    intbc_table_1 = pd.DataFrame(
        data=[
            ["cellBC_1", "AAA", 40],
            ["cellBC_1", "AAB", 2],  # "Sequencing error"
            ["cellBC_1", "BBB", 40],
            ["cellBC_1", "CCC", 9],  # Low counts
            ["cellBC_2", "AAA", 40],
            ["cellBC_2", "BBB", 40],
            ["cellBC_2", "CCC", 40],
            ["cellBC_3", "DDD", 40],
            ["cellBC_3", "EEE", 40],
            ["cellBC_3", "FFF", 40],
        ],
        columns=["cellBC", "intBC", "UMI_count"],
    )
    intbc_table_2 = pd.DataFrame(
        data=[
            ["cellBC_1", "DDD", 9],
            ["cellBC_1", "EEE", 10],
            ["cellBC_1", "FFF", 40],
            ["cellBC_2", "GGG", 40],
            ["cellBC_3", "N", 2],  # Will get dropped by umi counts
        ],
        columns=["cellBC", "intBC", "UMI_count"],
    )
    res = prune_low_umi_count_intbcs(
        intbc_table_per_sample={
            "sample1": intbc_table_1,
            "sample2": intbc_table_2,
        },
        min_intbc_umi_count=10,
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
                ["cellBC_3", "FFF", 40],
            ],
            columns=["cellBC", "intBC", "UMI_count"],
        ),
        "sample2": pd.DataFrame(
            data=[
                ["cellBC_1", "EEE", 10],
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
