from typing import Dict, Optional

import pandas as pd

from nova_clone.algorithm import clone_calling_algorithm
from nova_clone.algorithm._types import *

from ._clone_intbc_sets import qc_clone_intbc_sets
from ._compare_clone_calls import (
    qc_compare_clone_calls__align_clones,
    qc_compare_clone_calls__detect_undermerging,
    qc_compare_clone_calls__overlap_matrix,
)
from ._data_retention import qc_data_retention
from ._homoplasy import qc_homoplastic_intbcs
from ._inter_clonal_kinship import qc_inter_clonal_kinship
from ._intra_clonal_kinship import qc_intra_clonal_kinship
from ._multiplet_rate import qc_multiplet_rate, qc_multiplet_rate_per_clone
from ._number_of_clones import qc_number_of_clones
from ._pre_qc import (
    pre_qc_edit_distance_distribution,
    pre_qc_intbc_frequencies,
    pre_qc_intbcs_per_cellbc,
    pre_qc_max_umi_count_per_cellbc_quantiles,
    pre_qc_umi_per_intbc_distribution,
)


def pre_qc_complete_analysis(
    intbc_table_per_sample: Dict[CloneType, IntBCTableType],
    min_intbc_umi_count: int = 10,
    show_plot: bool = True,
    fig_paths: bool = True,
):
    pre_qc_max_umi_count_per_cellbc_quantiles(
        intbc_table_per_sample=intbc_table_per_sample,
        show_plot=show_plot,
        fig_path="pre_qc_max_umi_count_per_cellbc_quantiles" if fig_paths else None,
    )
    pre_qc_umi_per_intbc_distribution(
        intbc_table_per_sample=intbc_table_per_sample,
        show_plot=show_plot,
        fig_path="pre_qc_umi_per_intbc_distribution" if fig_paths else None,
    )
    pre_qc_intbcs_per_cellbc(
        intbc_table_per_sample=intbc_table_per_sample,
        min_intbc_umi_count=min_intbc_umi_count,
        show_plot=show_plot,
        fig_path="pre_qc_intbcs_per_cellbc" if fig_paths else None,
    )
    pre_qc_intbc_frequencies(
        intbc_table_per_sample=intbc_table_per_sample,
        min_intbc_umi_count=min_intbc_umi_count,
        show_plot=show_plot,
        fig_path="pre_qc_intbc_frequencies" if fig_paths else None,
    )
    pre_qc_edit_distance_distribution(
        intbc_table_per_sample=intbc_table_per_sample,
        show_plot=show_plot,
        fig_path="pre_qc_edit_distance_distribution_no_filtering" if fig_paths else None,
    )
    pre_qc_edit_distance_distribution(
        intbc_table_per_sample=intbc_table_per_sample,
        show_plot=show_plot,
        min_intbc_umi_count=min_intbc_umi_count,
        fig_path=f"pre_qc_edit_distance_distribution_min_intbc_umi_count_{min_intbc_umi_count}" if fig_paths else None,
    )


def qc_complete_analysis(
    intbc_table_per_sample: Dict[SampleNameType, IntBCTableType],
    clone_call: CloneCallType,
    show_plot: bool = True,
    fig_paths: bool = True,
    do_qc_homoplastic_intbcs: bool = True,
    do_qc_multiplet_rate_per_clone: bool = True,
    do_qc_intra_clonal_kinship: bool = True,
    do_qc_data_retention: bool = True,
    do_qc_clone_intbc_sets: bool = True,
):
    qc_multiplet_rate(
        clone_call.cell_to_clones_mapping,
        verbose=True,
    )
    qc_number_of_clones(
        clone_call=clone_call,
        show_plot=show_plot,
        fig_path="qc_number_of_clones" if fig_paths else None,
    )
    if do_qc_multiplet_rate_per_clone:
        qc_multiplet_rate_per_clone(
            clone_call.cell_to_clones_mapping,
            show_plot=show_plot,
            plot_path="qc_multiplet_rate_per_clone" if fig_paths else None,
        )
    if do_qc_homoplastic_intbcs:
        qc_homoplastic_intbcs(
            clone_call=clone_call,
        )
    if do_qc_intra_clonal_kinship:
        qc_intra_clonal_kinship(
            clone_to_cells_mapping=clone_call.clone_to_cells_mapping,
            cell_to_clones_mapping=clone_call.cell_to_clones_mapping,
            cell_intbc_sets=clone_call.cell_intbc_sets,
            fig_path="qc_intra_clonal_kinship" if fig_paths else None,
            subsample_ncells_per_clone=200,
            figsize=(10, 20),
            plot=show_plot,
        )
    qc_inter_clonal_kinship(
        clone_intbc_sets=clone_call.clone_intbc_sets,
    )
    if do_qc_data_retention:
        qc_data_retention(
            intbc_table_per_sample=intbc_table_per_sample,
            clone_call=clone_call,
            fig_path="qc_data_retention" if fig_paths else None,
            plot=show_plot,
        )
    if do_qc_clone_intbc_sets:
        qc_clone_intbc_sets(
            intbc_table_per_sample=intbc_table_per_sample,
            cell_to_clones_mapping=clone_call.cell_to_clones_mapping,
            clone_intbc_sets=clone_call.clone_intbc_sets,
            show_plot=show_plot,
            min_intbc_umi_count=1,
            fig_path="qc_clone_intbc_sets" if fig_paths else None,
        )


def test_qc():
    """
    We check that the QC functions run without errors.
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

    pre_qc_complete_analysis(
        intbc_table_per_sample={"sample1": intbc_table},
        show_plot=False,
        fig_paths=False,
    )

    clone_call = clone_calling_algorithm(
        intbc_table_per_sample={"sample1": intbc_table},
        edit_distance_cutoff=0,  # Turn off sequencing error correction
    ).clone_call

    qc_complete_analysis(
        intbc_table_per_sample={"sample1": intbc_table},
        clone_call=clone_call,
        show_plot=False,
        fig_paths=False,
    )
