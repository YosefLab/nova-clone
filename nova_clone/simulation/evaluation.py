import os
import random
import sys
import numpy as np
current_script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{current_script_dir}/..")
import pickle
import time
import nova_clone.simulation.clone_to_sequences as clone_to_sequences
from nova_clone import clone_calling_algorithm
from nova_clone.algorithm.cassiopeia._call_lineage_groups import call_lineage_groups
from nova_clone.simulation import benchmarking
from nova_clone.simulation import _caching
from nova_clone.algorithm._types import CloneCallType
from nova_clone.simulation.jaccard_hierarchical_by_intBC import clone_calling_jaccard_hierarchical_by_intBC
from nova_clone.simulation.HDBSCAN_method_adjusted import clone_calling_pearson_hdbscan_by_intBC

@_caching.caching_decorator()
def run_synthetic_data_creation_for_eval(
    number_superclones,
    tree_depth_range_min,
    tree_depth_range_max,
    alpha,
    decay_speed,
    size_of_barcodes,
    seq_error_sub_rate,
    missing_data_rate,
    multiplet_rate,
    min_barcodes_per_generation,
    min_cells_number_per_clone,
    seed,
):
    time_start = time.time()
    synthetic_data = clone_to_sequences.run_synthetic_data_creation(
        number_superclones=number_superclones,
        size_of_barcodes=size_of_barcodes,
        seq_error_sub_rate=seq_error_sub_rate,
        missing_data_rate=missing_data_rate,
        multiplet_rate=multiplet_rate,
        tree_depth_range_min=tree_depth_range_min,
        tree_depth_range_max=tree_depth_range_max,
        alpha=alpha,
        decay_speed=decay_speed,
        min_barcodes_per_generation=min_barcodes_per_generation,
        min_cells_number_per_clone=min_cells_number_per_clone,
    )
    synthetic_data = {k: v for k, v in synthetic_data.items() if k in ['final_cell_clone', 'clones_barcodes', 'each_subclone_barcodes', 'counts_missing_data','subtree_combinations','clone_cells_dict']}
    time_end = time.time()
    synthetic_data['time_synthetic'] = time_end - time_start
    print(f"time for creating synthetic data: {time_end - time_start}")
    return synthetic_data


@_caching.caching_decorator(exclude_args=["synthetic_data_results"])
def run_model_for_eval(
    model_name, 
    synthetic_data_results, 
    number_superclones,
    tree_depth_range_min,
    tree_depth_range_max,
    alpha,
    decay_speed,
    size_of_barcodes,
    seq_error_sub_rate,
    missing_data_rate,
    multiplet_rate,
    min_barcodes_per_generation,
    min_cells_number_per_clone,
    seed,
):
    time_start = time.time()
    with open(synthetic_data_results.cache_path, "rb") as f:
        synthetic_data = pickle.load(f)
    if model_name == "clone_calling_algorithm":
        clone_calling_results = clone_calling_algorithm(
            {"sample0": synthetic_data["counts_missing_data"]},
            recursive=True,
            min_intbc_fraction=0,
            edit_distance_cutoff=2,
        )
        recursive_clone_calling_results = clone_calling_results.recursive_clone_call
        clone_calling_results = clone_calling_results.clone_call
    elif model_name == "cassiopeia":
        clone_calling_results = call_lineage_groups(
            intbc_table_per_sample={"sample0": synthetic_data["counts_missing_data"]},
            min_intbc_umi_count=10,
        )
        recursive_clone_calling_results = None
    elif model_name == "one_large_clone__2025_05_12_v2":
        cellbcs = sorted(list(set(synthetic_data["counts_missing_data"].cellBC)))
        cellbcs = ["sample0_" + cellbc for cellbc in cellbcs]
        clone_calling_results = CloneCallType(
            cell_to_clones_mapping={cellbc: {"clone_1"} for cellbc in cellbcs},
            clone_to_cells_mapping={"clone_1": set(cellbcs)},
            clone_intbc_sets={"clone_1": set()},
            cell_intbc_sets={cellbc: set() for cellbc in cellbcs},
        )
        recursive_clone_calling_results = None
    elif model_name == "empty_clone_call__2025_05_12_v1":
        cellbcs = sorted(list(set(synthetic_data["counts_missing_data"].cellBC)))
        cellbcs = ["sample0_" + cellbc for cellbc in cellbcs]
        clone_calling_results = CloneCallType(
            cell_to_clones_mapping={cellbc: set() for cellbc in cellbcs},
            clone_to_cells_mapping={},
            clone_intbc_sets={cellbc: set() for cellbc in cellbcs},
            cell_intbc_sets={cellbc: set() for cellbc in cellbcs},
        )
        recursive_clone_calling_results = None
    elif model_name == "jaccard_hierarchical":
        clone_calling_results = clone_calling_jaccard_hierarchical_by_intBC(
            input_df=synthetic_data["counts_missing_data"],
            umi_threshold=10, #min_intbc_umi_count
            ratio_dendrogram=0.9, # thats currently the default value of this algorithm
            min_ibtbc_per_clone=2, #_preliminary_coreset_building_step__min_coreset_size
            min_cells_per_clone=10, #min_cells_assigned
            min_cells_with_intBC=10 #min_intbc_cellbc_count=10
        )
        clone_calling_results = CloneCallType(
            cell_to_clones_mapping=clone_calling_results["cell_to_clones_mapping"],
            clone_to_cells_mapping={},
            clone_intbc_sets=clone_calling_results['clone_intbc_sets'],
            cell_intbc_sets={},
        )
        recursive_clone_calling_results = None
    elif model_name == "pearson_hdbscan":
        clone_calling_results = clone_calling_pearson_hdbscan_by_intBC(
            df=synthetic_data["counts_missing_data"],
            min_UMIs=10,  # min_intbc_umi_count
            min_ibtbc_per_clone=2, #_preliminary_coreset_building_step__min_coreset_size
            min_cell_cluster_size=30, #min_cells_assigned
            min_cells_with_intBC=10, #min_intbc_cellbc_count=10
            min_samples=30
        )
        clone_calling_results = CloneCallType(
            cell_to_clones_mapping=clone_calling_results["cell_to_clones_mapping"],
            clone_to_cells_mapping={},
            clone_intbc_sets=clone_calling_results['clone_to_intBCs'],
            cell_intbc_sets={},
        )
        recursive_clone_calling_results = None
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    time_end = time.time()
    clone_calling_results = {'clone_calling_results': clone_calling_results, 'time_model': time_end - time_start, 'recursive_clone_calling_results': recursive_clone_calling_results}
    print(f"time for running model {model_name}: {time_end - time_start}")
    return clone_calling_results

@_caching.caching_decorator(exclude_args=['synthetic_data_results','clone_calling_results_results'])
def run_eval(
    model_name,
    synthetic_data_results,
    clone_calling_results_results,
    number_superclones,
    tree_depth_range_min,
    tree_depth_range_max,
    alpha,
    decay_speed,
    size_of_barcodes,
    seq_error_sub_rate,
    missing_data_rate,
    multiplet_rate,
    min_barcodes_per_generation,
    min_cells_number_per_clone,
    seed,
):
    time_start = time.time()
    with open(synthetic_data_results.cache_path, "rb") as f:
        synthetic_data = pickle.load(f)
    with open(clone_calling_results_results.cache_path, "rb") as f:
        clone_calling_results_all = pickle.load(f)
        time_model = clone_calling_results_all["time_model"]
        clone_calling_results = clone_calling_results_all["clone_calling_results"]
    eval_results = benchmarking.evaluation_results(
        cell_clone_dict_input=synthetic_data["final_cell_clone"],
        cells_clone_dict_output=clone_calling_results.cell_to_clones_mapping,
        clone_barcode_dict_input=synthetic_data["each_subclone_barcodes"],
        clone_barcode_dict_output=clone_calling_results.clone_intbc_sets,
        subtree_combinations=synthetic_data["subtree_combinations"],
        clone_to_cells_dict=synthetic_data['clone_cells_dict'],
        recursive_clone_calling_results=clone_calling_results_all['recursive_clone_calling_results']
    )
    time_end = time.time()
    eval_results["time"] = {'time_eval': time_end - time_start, 'time_model': time_model, 'time_synthetic': synthetic_data["time_synthetic"]}
    return eval_results

def check_missing_files(
    model_name,
    number_superclones,
    tree_depth_range_min,
    tree_depth_range_max,
    alpha,
    decay_speed,
    regime_args,
    seed,
):

    np.random.seed(seed)
    random.seed(seed)

    synthetic_data_results = run_synthetic_data_creation_for_eval(
        number_superclones=number_superclones,
        tree_depth_range_min=tree_depth_range_min,
        tree_depth_range_max=tree_depth_range_max,
        alpha=alpha,
        decay_speed=decay_speed,
        size_of_barcodes=regime_args["size_of_barcodes"],
        seq_error_sub_rate=regime_args["seq_error_sub_rate"],
        missing_data_rate=regime_args["missing_data_rate"],
        multiplet_rate=regime_args["multiplet_rate"],
        min_barcodes_per_generation=regime_args["min_barcodes_per_generation"],
        min_cells_number_per_clone=regime_args["min_cells_number_per_clone"],
        seed=seed
    )

    clone_calling_results_results = run_model_for_eval(
        model_name, 
        synthetic_data_results, 
        number_superclones=number_superclones,
        tree_depth_range_min=tree_depth_range_min,
        tree_depth_range_max=tree_depth_range_max,
        alpha=alpha,
        decay_speed=decay_speed,
        size_of_barcodes=regime_args["size_of_barcodes"],
        seq_error_sub_rate=regime_args["seq_error_sub_rate"],
        missing_data_rate=regime_args["missing_data_rate"],
        multiplet_rate=regime_args["multiplet_rate"],
        min_barcodes_per_generation=regime_args["min_barcodes_per_generation"],
        min_cells_number_per_clone=regime_args["min_cells_number_per_clone"],
        seed=seed
    )

    evaluation_results = run_eval(
        model_name,
        synthetic_data_results,
        clone_calling_results_results,
        number_superclones=number_superclones,
        tree_depth_range_min=tree_depth_range_min,
        tree_depth_range_max=tree_depth_range_max,
        alpha=alpha,
        decay_speed=decay_speed,
        size_of_barcodes=regime_args["size_of_barcodes"],
        seq_error_sub_rate=regime_args["seq_error_sub_rate"],
        missing_data_rate=regime_args["missing_data_rate"],
        multiplet_rate=regime_args["multiplet_rate"],
        min_barcodes_per_generation=regime_args["min_barcodes_per_generation"],
        min_cells_number_per_clone=regime_args["min_cells_number_per_clone"],
        seed=seed,
    )
    return evaluation_results


def eval_function(
    model_name,
    metric_name,
    number_superclones: int = 10,
    tree_depth_range_min: int = 3,
    tree_depth_range_max: int = 5,
    alpha: int = -1,
    decay_speed: int = 1,
    regime_args={
        "size_of_barcodes": 14,
        "seq_error_sub_rate": 0.0001,
        "missing_data_rate": 0.05,
        "multiplet_rate": 0.05,
        "min_barcodes_per_generation": 10,
        "min_cells_number_per_clone": 15,
    },
    repetition: int = 0,
):
    seed = repetition

    evaluation_results = check_missing_files(
        model_name=model_name,
        number_superclones=number_superclones,
        tree_depth_range_min=tree_depth_range_min,
        tree_depth_range_max=tree_depth_range_max,
        alpha=alpha,
        decay_speed=decay_speed,
        regime_args=regime_args,
        seed=seed,
    )

    with open(evaluation_results.cache_path, "rb") as f:
        eval_results = pickle.load(f)

    if metric_name.startswith("clone"):
        clonal_level = 'clone'
    elif metric_name.startswith("subclone"):
        clonal_level = 'subclone'
    elif 'time' in metric_name:
        metric_value = eval_results['time'][metric_name]
    else:
        raise ValueError(f"Unknown clonal level name: {metric_name}")
    
    if 'doublets' in metric_name:
        if "jaccard-similarity" in metric_name:
            metric_value = eval_results[clonal_level]["doublets"]["jaccard-similarity"]
        else:
            metric_value = eval_results[clonal_level]["doublets"]["similarity_matices"][
                metric_name.split("_")[2]
            ]
    elif "subclonal_cells_recovery_rate" in metric_name: 
        metric_value = eval_results[clonal_level]["subclonal_cells_recovery_rate"]
    elif 'cells' in metric_name:
        if 'ARI' in metric_name:
            metric_value = (
            np.mean(eval_results[clonal_level]["results_cells_summary"]['ARI_score'])
        )
        else:
            assert(len(eval_results[clonal_level]["results_cells_summary"][metric_name.split("_")[2]]) > 0)
            metric_value = (
                np.mean(eval_results[clonal_level]["results_cells_summary"][metric_name.split("_")[2]])
            )
    elif 'barcodes' in metric_name:
        if 'weighted' in metric_name:
            assert(len(eval_results[clonal_level]["results_barcodes_summary"][metric_name.split("_")[2]]) > 0)
            metric_value = (
                np.mean(eval_results[clonal_level]["results_barcodes_summary_weighted"][metric_name.split("_")[2]])
            )
        
        else:
            if 'ARI' in metric_name:
                metric_value = (
                    np.mean(eval_results[clonal_level]["results_barcodes_summary"]['ARI_score'])
                )
            else:
                assert(len(eval_results[clonal_level]["results_barcodes_summary"][metric_name.split("_")[2]]) > 0)
                metric_value = (
                    np.mean(eval_results[clonal_level]["results_barcodes_summary"][metric_name.split("_")[2]])
                )       
    
    print(f"{metric_name}: {metric_value}")
    return metric_value


def test_evaluation_smoketest():
    eval_function(
        model_name="clone_calling_algorithm",
        metric_name="clone_barcodes_F1",
        number_superclones=1,
        tree_depth_range_min=3,
        tree_depth_range_max=4,
        alpha=-1,
        decay_speed=1,
        regime_args={
            "size_of_barcodes": 5,
            "seq_error_sub_rate": 0.0001,
            "missing_data_rate": 0.05,
            "multiplet_rate": 0.05,
            "min_cells_number_per_clone": 2,
            "min_barcodes_per_generation": 10,
        },
        repetition=0,
    )