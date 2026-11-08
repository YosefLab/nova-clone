"""
Run with:
$ time python -m nova_clone.simulation.slice_benchmark_clone_calling
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from nova_clone.simulation.evaluation import eval_function as eval_function_clone_calling
from nova_clone.simulation.slice_benchmark import slice_benchmark


def slice_benchmark_clone_calling():
    slice_benchmark(
        model_names=["clone_calling_algorithm", "cassiopeia", "jaccard_hierarchical", "pearson_hdbscan", "one_large_clone__2025_05_12_v2", "empty_clone_call__2025_05_12_v1"], 
        metric_names=["clone_cells_accuracy",'clone_cells_recall','clone_cells_precision','clone_cells_F1','clone_cells_ARI_score',
                      "subclone_cells_accuracy",'subclone_cells_recall','subclone_cells_precision','subclone_cells_F1',
                      "clone_doublets_recall",'clone_doublets_precision','clone_doublets_F1',
                      "clone_barcodes_recall", "clone_barcodes_precision", "clone_barcodes_F1", "clone_barcodes_accuracy",'clone_barcodes_ARI_score',
                      "subclone_barcodes_recall", "subclone_barcodes_precision", "subclone_barcodes_F1",
                      "clone_barcodes_recall_weighted", "clone_barcodes_precision_weighted", "clone_barcodes_F1_weighted",
                      "subclone_barcodes_recall_weighted", "subclone_barcodes_precision_weighted", "subclone_barcodes_F1_weighted",
                      'time_model'],
        eval_function=eval_function_clone_calling,
        context_names=[],
        context_values=[{}],
        knobs=[
            "size_of_barcodes",
            "seq_error_sub_rate",
            "missing_data_rate",
            "multiplet_rate",
            "homoplasy_sigma",
        ],
        defaults={
            "size_of_barcodes": 14,
            "seq_error_sub_rate": 0.0001,
            "missing_data_rate": 0.05,
            "multiplet_rate": 0.05,
            "homoplasy_sigma": 0.0,
        },
        ranges={
            "size_of_barcodes": [4, 8, 10, 12, 14], # [4, 8, 10, 12, 14],
            "seq_error_sub_rate": [0.0001, 0.001, 0.01, 0.05], # [0.0001, 0.001, 0.01, 0.05],  
            "missing_data_rate": [0.01, 0.05, 0.1, 0.2, 0.3], # [0.01, 0.05, 0.1, 0.2, 0.3],  
            "multiplet_rate": [0.01, 0.05, 0.1, 0.2, 0.4], # [0.01, 0.05, 0.1, 0.2, 0.4],
            "homoplasy_sigma": [0, 0.3, 0.6, 1, 1.5], # [0, 0.3, 0.6, 1, 1.5],
        },
        repetitions=range(10),
        plot_model_params=None,
        metric_display_names=None,
        knob_display_names=None,
        metric_lower_is_better=None,
        log=False,
        use_tqdm=True,
        aggregation="mean",
        use_ranks=False,
        plot_type="heatmap",
        cmap="YlGnBu",
        fmt=None,
        figsize=None,
        ylim=None,
        outdir="slice_benchmark_clone_calling_figures",
        num_processes=8
    )


if __name__ == "__main__":
    slice_benchmark_clone_calling()
