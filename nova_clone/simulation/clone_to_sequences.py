import os
import pickle
import random
import sys
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import scipy.stats as stats
current_script_dir = os.path.dirname(os.path.realpath(__file__))
print(f"{current_script_dir}/..")
sys.path.append(f"{current_script_dir}/..")
import simulation.creating_tree as creating_tree
from nova_clone.algorithm._types import (
    CellBCType,
    CloneType,
    IntBCTableType,
    IntBCType,
)


def get_number_of_cells_per_clone(
    clones_all_superclones: List[str],
    lognorm_dist_cell_per_clone_params: Dict[str, float],
    min_cells_number_per_clone: int = 15,
) -> List[int]:
    """
    Get the number of cells per clone from a lognorm distribution with a minimum number of cells per clone

    Args:
        clones_all_superclones: list of all clones
        lognorm_dist_cell_per_clone_params: dictionary of parameters for lognorm distribution of number of cells per clone
        min_cells_number_per_clone: minimum number of cells per clone

    Returns:
        num_cells_per_clone_list: list of number of cells per clone
    """
    num_cells_per_clone_list = stats.lognorm.rvs(
        s=lognorm_dist_cell_per_clone_params["sigma"],  # sigma
        loc=lognorm_dist_cell_per_clone_params["loc"],  # shifts the distribution
        scale=lognorm_dist_cell_per_clone_params["scale"],  # np.exp(mean)
        size=len(clones_all_superclones),
    )
    while (num_cells_per_clone_list < min_cells_number_per_clone).any():
        under_min_values = num_cells_per_clone_list < min_cells_number_per_clone
        num_cells_per_clone_list[under_min_values] = stats.lognorm.rvs(
            s=lognorm_dist_cell_per_clone_params["sigma"],
            loc=lognorm_dist_cell_per_clone_params["loc"],
            scale=lognorm_dist_cell_per_clone_params["scale"],
            size=sum(under_min_values),
        )
    num_cells_per_clone_list = [int(round(count)) for count in num_cells_per_clone_list]
    return num_cells_per_clone_list


def cells_to_clone(
    num_cells_per_clone: List[int],
    all_clones_clone_intBC: Dict[CloneType, List[IntBCType]],
) -> Tuple[
    Dict[CloneType, List[CellBCType]],
    Dict[CellBCType, List[IntBCType]],
    Dict[CellBCType, List[CloneType]],
]:
    """
    Assign cells to each clone

    Args:
        num_cells_per_clone: list of number of cells per clone
        all_clones_clone_intBC: dictionary of intBCs per clone

    Returns:
        clone_cells: dictionary of cells names per clone
        cell_intBCs: dictionary of intBCs per cell
        cell_clone: dictionary of clones per cell
    """
    clone_cells: Dict[CloneType, List[CellBCType]] = {}
    cell_intBCs: Dict[CellBCType, List[IntBCType]] = {}
    cell_clone: Dict[CellBCType, List[CloneType]] = {}
    current_index_cell = 0
    for clone_name, count in zip(all_clones_clone_intBC.keys(), num_cells_per_clone):
        cells_in_clone = [
            "cell_" + str(i)
            for i in range(current_index_cell, current_index_cell + count)
        ]
        clone_cells[clone_name] = cells_in_clone
        for cell_name in cells_in_clone:
            cell_intBCs[cell_name] = all_clones_clone_intBC[clone_name]
            cell_clone[cell_name] = [clone_name]
        current_index_cell += count
    return clone_cells, cell_intBCs, cell_clone


def get_full_number_of_intBCs(
    cell_intBCs: Dict[CellBCType, List[IntBCType]],
    beta_dist_UMIs_params: Dict[str, List[float]],
    seed: int | None = None,
) -> Dict[CellBCType, List[IntBCType]]:
    """
    Assign number of UMIs to each intBC in each cell

    Args:
        cell_intBCs: dictionary of intBCs per cell
        beta_dist_UMIs_params: dictionary of parameters for beta distribution of UMI number per intBC per cell

    Returns:
        full_cell_intBCs: dictionary of intBCs with full copies of UMIs per intBC per cell
    """
    full_cell_intBCs: Dict[CellBCType, List[IntBCType]] = {}
    for cell, intBCs in cell_intBCs.items():
        multi_index = 0
        if seed:
            np.random.seed(seed)
        UMI_num_perCell = (
            (
                beta_dist_UMIs_params["loc"]
                + beta_dist_UMIs_params["scale"]
                * np.random.beta(
                    a=beta_dist_UMIs_params["alpha"],
                    b=beta_dist_UMIs_params["beta"],
                    size=len(cell_intBCs[cell]),
                )
            )
            .round()
            .astype("int32")
        )
        expanded_sequences: List[str] = []
        for seq in intBCs:
            number_of_UMIs = UMI_num_perCell[multi_index]
            expanded_sequences.extend([seq] * number_of_UMIs)
            multi_index += 1
        full_cell_intBCs[cell] = expanded_sequences
    return full_cell_intBCs


def insert_seq_error_substitution(
    full_cell_intBCs: Dict[CellBCType, List[IntBCType]],
    error_rate: float = 0.0001,
    seed: int | None = None,
) -> Dict[CellBCType, List[IntBCType]]:
    """
    Insert sequencing errors (substitution errrors only) in the intBCs

    Args:
        full_cell_intBCs: dictionary of intBCs per cell
        error_rate: error rate for substitution error

    Returns:
        modified_intBCs: dictionary of intBCs with substitution errors
    """
    if seed:
        random.seed(seed)
    modified_intBCs: Dict[CellBCType, List[IntBCType]] = {}
    for cell, sequences in full_cell_intBCs.items():
        modified_sequences: List[str] = []
        for seq in sequences:
            seq = list(seq)
            for i in range(len(seq)):
                if random.random() < error_rate:
                    original_base = seq[i]
                    new_base = random.choice(
                        [b for b in ["A", "T", "C", "G"] if b != original_base]
                    )
                    seq[i] = new_base
            modified_sequences.append("".join(seq))
        modified_intBCs[cell] = modified_sequences
    return modified_intBCs


def create_multiplets(
    full_cell_intBCs: Dict[CellBCType, List[IntBCType]],
    clone_cell: Dict[CellBCType, List[CloneType]] = {},
    multiplet_rate: float = 0.05,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Create multiplets by combining cells intBCs and clones

    Args:
        full_cell_intBCs: dictionary of intBCs per cell
        clone_cell: dictionary of cells and their clones
        multiplet_rate: ratio of multiplets to total number of cells

    Returns:
        full_cell_intBCs: dictionary of cells and their intBCs, updated with the multiplets
        clone_cell: dictionary of cells and their clones, updated with the multiplets
    """
    clone_cell_lst=list(clone_cell.keys())
    random.shuffle(clone_cell_lst)
    multiplet_cell_list = list()
    for cell in clone_cell_lst:
        if random.random() < multiplet_rate:
            multiplet_cell_list.append(cell)
        else:
            if len(multiplet_cell_list) > 0:
                multiplet_cell_list.append(cell)
                clone_cell['_'.join(multiplet_cell_list)] = list(set([clone for cell in multiplet_cell_list for clone in clone_cell[cell]]))
                full_cell_intBCs['_'.join(multiplet_cell_list)] = [bc for cell in multiplet_cell_list for bc in full_cell_intBCs[cell]]
                clone_cell = {k: v for k, v in clone_cell.items() if k not in multiplet_cell_list}
                full_cell_intBCs = {k: v for k, v in full_cell_intBCs.items() if k not in multiplet_cell_list}
                multiplet_cell_list = list()
    return full_cell_intBCs, clone_cell


def create_missing_sequences(
    counts: IntBCTableType, missing_data_rate: float = 0.05, seed: int | None = None
) -> IntBCTableType:
    """
    Create missing intBCs by deleting part of the counts

    Args:
        counts: cellBC, intBC and UMI_counts
        missing_data_rate: rate of missing data

    Returns:
        counts data frame with missing data
    """
    if seed:
        random.seed(seed)
    missing_data = counts.copy()
    missing_data["UMI_count"] = missing_data["UMI_count"].apply(
        lambda x: x if random.random() > missing_data_rate else 0
    )
    return missing_data


def create_counts_table(
    final_intBCs_cells: Dict[CellBCType, List[IntBCType]]
) -> pd.DataFrame:
    """
    Create a counts table from the final dictionary of intBCs per cell

    Args:
        final_intBCs_cells: dictionary of intBCs per cell

    Returns:
        dataframe of counts table, columns are cellBC, intBC and UMI_counts
    """
    data = []
    for cell, sequences in final_intBCs_cells.items():
        sequence_counts = {seq: sequences.count(seq) for seq in set(sequences)}
        for seq, count in sequence_counts.items():
            data.append({"cellBC": cell, "intBC": seq, "UMI_count": count})
    return pd.DataFrame(data)


def run_synthetic_data_creation(
    number_superclones: int,
    beta_dist_UMIs_params: Dict[str, float] = {
        "alpha": 2.183894307165218,
        "beta": 6862288.44110363,
        "loc": 9.083437389474298,
        "scale": 128414251.97702634,
    },
    lognorm_dist_cell_per_clone_params: Dict[str, float] = {
        "sigma": 1.1823347397246888,
        "loc": 14.463437982337393,
        "scale": 62.06989618615775,
    },
    size_of_barcodes: int = 14,
    seq_error_sub_rate: float = 0.0001,
    missing_data_rate: float = 0.05,
    multiplet_rate: float = 0.05,
    tree_depth_range_min: int = 3,
    tree_depth_range_max: int = 5,
    alpha: float = -1,
    decay_speed: float = 1,
    min_cells_number_per_clone: int = 15,
    percentage_of_barcodes_silenced_total: float = 0.1,
    min_barcodes_per_generation: int = 10,
    minP: float = 0.1,
    maxP: float = 0.9,
) -> None:
    """
    Run the synthetic data creation pipeline

    Args:
        number_superclones: number of superclones to create
        beta_dist_UMIs_params: dictionary of parameters for beta distribution of UMI number per intBC per cell
        lognorm_dist_cell_per_clone_params: dictionary of parameters for lognorm distribution of number of cells per clone
        output_path: path to save the output files
        size_of_barcodes: size of barcodes
        seq_error_sub_rate: sequencing error rate for substitution errors
        missing_data_rate: rate of missing data
        multiplet_rate: ratio of multiplets to total number of cells
        tree_depth_range_min: minimum depth of the tree
        tree_depth_range_max: maximum depth of the tree
        alpha: alpha parameter for the function of integration barcodes number
        decay_speed: decay speed parameter for the function of integration barcodes number
        # generation_time: generation time parameter for the function of integration barcodes number
        min_cells_number_per_clone: minimum number of cells per clone
        min_barcodes_per_clone: minimum number of barcodes per clone
    """
    if tree_depth_range_min == 1:
        raise ValueError("tree_depth_range_min should be greater than 1")
    clone_hierarchy_all_superclones: Dict[str, List[str]] = {}
    clone_barcodes_all_superclones: Dict[str, List[str]] = {}
    all_clones_barcodes: Dict[str, List[str]] = {}
    each_subclone_barcodes: Dict[str, List[str]] = {}
    all_subtree_combinations: List[str] = []

    depth_tree_list = np.random.randint(
        low=tree_depth_range_min, high=tree_depth_range_max, size=number_superclones
    )
    total_barcodes_per_superclone = min_barcodes_per_generation * tree_depth_range_max

    for i in range(number_superclones):
        new_superclone = creating_tree.SuperClone(
            superclone_number=i,
            number_of_generations=depth_tree_list[i],
            decay_speed=decay_speed,
            alpha=alpha,
            size_of_barcodes=size_of_barcodes,
            percentage_of_barcodes_silenced_total=percentage_of_barcodes_silenced_total,
            total_barcodes_per_superclone=total_barcodes_per_superclone,
            minP=minP,
            maxP=maxP,
        )
        tree = new_superclone.generate_tree()
        creating_tree.print_clone_tree(tree)
        clone_hierarchy, clone_barcodes, clones, all_clone_barcodes, subtree_combinations = (
            creating_tree.extract_tree_hierarchy_clones_barcodes(tree)
        )
        each_subclone_barcodes = each_subclone_barcodes | clone_barcodes
        all_clones_barcodes["superclone_" + str(i)] = list(all_clone_barcodes)
        clone_hierarchy_all_superclones["superclone_" + str(i)] = clone_hierarchy
        clone_barcodes_all_superclones = clone_barcodes_all_superclones | clone_barcodes
        all_subtree_combinations.extend(subtree_combinations)

    leaves_tree_clones = [key for superclone_dict in clone_hierarchy_all_superclones.values() for key, value in superclone_dict.items() if value == []]

    num_cells_per_clone_list = get_number_of_cells_per_clone(
        clones_all_superclones=leaves_tree_clones,
        lognorm_dist_cell_per_clone_params=lognorm_dist_cell_per_clone_params,
        min_cells_number_per_clone=min_cells_number_per_clone,
    )
    clone_barcodes_all_superclones_leaves = {key:val for key,val in clone_barcodes_all_superclones.items() if key in leaves_tree_clones}
    clone_cells_dict, cell_intBCs_dict, cell_clone_dict = cells_to_clone(
        num_cells_per_clone=num_cells_per_clone_list, 
        all_clones_clone_intBC = clone_barcodes_all_superclones_leaves,
    )
    full_number_UMI_dict = get_full_number_of_intBCs(
        cell_intBCs_dict, beta_dist_UMIs_params
    )
    with_seq_errors = insert_seq_error_substitution(
        full_number_UMI_dict, error_rate=seq_error_sub_rate
    )
    cell_clone_dict_copy = cell_clone_dict.copy()
    with_multiplets, final_cell_clone = create_multiplets(
        with_seq_errors, cell_clone_dict_copy, multiplet_rate=multiplet_rate
    )
    counts_df = create_counts_table(with_multiplets)
    counts_missing_data = create_missing_sequences(
        counts_df, missing_data_rate=missing_data_rate
    )

    results = {
        'clone_hierarchy': clone_hierarchy_all_superclones,
        'subtree_combinations': all_subtree_combinations,
        "clone_cells_dict": clone_cells_dict,
        "cell_intBCs_dict": cell_intBCs_dict,
        "full_UMI_dict": full_number_UMI_dict,
        "with_seq_errors": with_seq_errors,
        "clones_barcodes": all_clones_barcodes,
        "each_subclone_barcodes": each_subclone_barcodes,
        "with_multiplets": with_multiplets,
        "final_cell_clone": final_cell_clone,
        "counts_df": counts_df,
        "counts_missing_data": counts_missing_data,
    }
    return results

def test_run_synthetic_data_creation_smoketest():
    test_run_synthetic=run_synthetic_data_creation(
        number_superclones=1,
        beta_dist_UMIs_params={
            "alpha": 2.183894307165218,
            "beta": 6862288.44110363,
            "loc": 9.083437389474298,
            "scale": 128414251.97702634,
        },
        lognorm_dist_cell_per_clone_params={
            "sigma": 1.1823347397246888,
            "loc": 14.463437982337393,
            "scale": 62.06989618615775,
        },
        size_of_barcodes=10,
        seq_error_sub_rate=0.0001,
        missing_data_rate=0.05,
        multiplet_rate=0.05,
        tree_depth_range_min=3,
        tree_depth_range_max=4,
        alpha=-1,
        decay_speed=1,
        min_cells_number_per_clone=2,
        percentage_of_barcodes_silenced_total=0.1,
        min_barcodes_per_generation=10,
        minP=1,
        maxP=1,
    )
