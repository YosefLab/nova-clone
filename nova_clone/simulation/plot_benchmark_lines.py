"""
Line-plot version of the slice_benchmark_clone_calling heatmaps:
one row of panels (one per noise knob), one line per method, mean +/- SEM over repetitions.

Reads the cached run_eval results directly (no recomputation).

Run with (from nova-clone/):
$ python -m nova_clone.simulation.plot_benchmark_lines
"""
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

# editable text in Illustrator
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 13
import numpy as np
import pandas as pd

CACHE_DIR = "/home/projects/amit/danari/LineageTracing/nova-clone/_cache/run_eval"
OUTDIR = "/home/projects/amit/danari/LineageTracing/nova-clone/slice_benchmark_clone_calling_figures"

FIXED = dict(number_superclones=10, tree_depth_range_min=3, tree_depth_range_max=5, alpha=-1, decay_speed=1)
# defaults and ranges must be written exactly as in slice_benchmark_clone_calling.py (0 and 0.0 are different cache paths)
DEFAULTS = {
    "size_of_barcodes": 14,
    "seq_error_sub_rate": 0.0001,
    "missing_data_rate": 0.05,
    "multiplet_rate": 0.05,
    "homoplasy_sigma": 0.0,
}
RANGES = {
    "size_of_barcodes": [4, 8, 10, 12, 14],
    "missing_data_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "multiplet_rate": [0.01, 0.05, 0.1, 0.2, 0.4],
    "seq_error_sub_rate": [0.0001, 0.001, 0.01, 0.05],
    "homoplasy_sigma": [0, 0.3, 0.6, 1, 1.5],
}
KNOB_LABELS = {
    "size_of_barcodes": "intBC length (nt)",
    "missing_data_rate": "Missing data rate",
    "multiplet_rate": "Multiplet rate",
    "seq_error_sub_rate": "Sequencing error rate",
    "homoplasy_sigma": "Homoplasy (library copy-number σ)",
}
# model_name -> (display name, color, linestyle). Deep palette: crimson / navy / teal / ochre; trivial baselines in grey.
# Methods often tie (e.g. at 1.0), so each gets its own dash pattern (and a small horizontal dodge, below);
# a later dashed line leaves the earlier one visible in its gaps.
MODELS = {
    "clone_calling_algorithm": ("NovaClone", "#C1272D", "-"),
    "cassiopeia": ("Cassiopeia", "#1F3B73", (0, (4, 4))),
    "jaccard_hierarchical": ("Jaccard hierarchical", "#2A9D8F", (0, (1, 1.5))),
    "pearson_hdbscan": ("Pearson HDBSCAN", "#E9A23B", (0, (4, 1.5, 1, 1.5))),
    "one_large_clone__2025_05_12_v2": ("One large clone", "#7F7F7F", "--"),
    "empty_clone_call__2025_05_12_v1": ("Empty call", "#BBBBBB", ":"),
}
LINE_KW = dict(marker="o", ms=4, lw=1.6)
BASELINE_LINE_KW = {
    "one_large_clone__2025_05_12_v2": dict(marker="o", ms=3.5, lw=1.6),
    "empty_clone_call__2025_05_12_v1": dict(marker="o", ms=3.5, lw=1.6),
}
REPETITIONS = range(10)
DODGE_PT = 2.5  # horizontal spacing between methods, in points (0 = no dodge)


def cache_path(model_name, regime, seed):
    args = {**FIXED, **regime}
    # same order as the run_eval signature in evaluation.py
    order = ["number_superclones", "tree_depth_range_min", "tree_depth_range_max", "alpha", "decay_speed",
             "size_of_barcodes", "seq_error_sub_rate", "missing_data_rate", "multiplet_rate",
             "homoplasy_sigma"]
    parts = [f"model_name_{model_name}"] + [f"{k}_{args[k]}" for k in order] + [f"seed_{seed}"]
    return os.path.join(CACHE_DIR, *parts, "result.pickle")


def get_metric(eval_results, metric_name):
    """Same extraction as evaluation.eval_function for *_cells_*, *_barcodes_* and time_* metrics."""
    if metric_name.startswith("time"):
        return eval_results["time"][metric_name]
    level, kind, stat = metric_name.split("_")[:3]
    key = f"results_{kind}_summary" + ("_weighted" if metric_name.endswith("weighted") else "")
    return np.mean(eval_results[level][key]["ARI_score" if stat == "ARI" else stat])


def load_results(metric_names):
    """Long dataframe with one row per (knob, knob_value, model, repetition, metric)."""
    rows = []
    n_missing = 0
    for knob, values in RANGES.items():
        for value in values:
            regime = {**DEFAULTS, knob: value}
            for model_name in MODELS:
                for seed in REPETITIONS:
                    path = cache_path(model_name, regime, seed)
                    if not os.path.exists(path):
                        n_missing += 1
                        continue
                    with open(path, "rb") as f:
                        res = pickle.load(f)
                    for metric_name in metric_names:
                        rows.append([knob, value, model_name, seed, metric_name, get_metric(res, metric_name)])
    print(f"loaded {len(rows) // len(metric_names)} cached results, {n_missing} missing (not run yet)")
    return pd.DataFrame(rows, columns=["knob", "knob_value", "model_name", "repetition", "metric_name", "value"])


def plot_metric_lines(df, rows, title=None, outpath=None, log_y=False, models=MODELS):
    """
    Grid of line plots: one row per metric, one column per knob (columns share the x axis).
    rows: list of (metric_name, y label).
    """
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, len(RANGES), figsize=(4.2 * len(RANGES), 3.0 * n_rows + 1.0),
                             sharex="col", sharey="row", squeeze=False)
    for r, (metric_name, metric_label) in enumerate(rows):
        for ax, knob in zip(axes[r], RANGES):
            _plot_panel(fig, ax, df[(df.metric_name == metric_name) & (df.knob == knob)], knob, log_y, models,
                        last_row=r == n_rows - 1)
        axes[r, 0].set_ylabel(metric_label)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # legend above the top right panel, so it does not take a column of its own.
    # it is left out of tight_layout (otherwise the columns get pushed apart), and space is reserved at the top
    legend = axes[0, -1].legend(handles, labels, loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=3,
                                frameon=False, fontsize=10, handlelength=2.5, columnspacing=1.0)
    legend.set_in_layout(False)
    fig_height = fig.get_size_inches()[1]
    if title:
        fig.suptitle(title, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig_height))
    if outpath:
        fig.savefig(outpath, bbox_inches="tight", dpi=300)
    return fig


def _plot_panel(fig, ax, sub, knob, log_y, models, last_row):
        for i, (model_name, (label, color, ls)) in enumerate(models.items()):
            s = sub[sub.model_name == model_name].groupby("knob_value")["value"]
            mean, sem = s.mean(), s.sem()
            # dodge: shift each method a few points sideways so tied values stay visible (baselines stay in place)
            shift = 0 if model_name in BASELINE_LINE_KW else DODGE_PT * (i - (len(models) - 1) / 2)
            dodge = transforms.offset_copy(ax.transData, fig=fig, x=shift, units="points")
            ax.plot(mean.index, mean.values, color=color, linestyle=ls, label=label, transform=dodge, **BASELINE_LINE_KW.get(model_name, LINE_KW))
            ax.fill_between(mean.index, mean - sem, mean + sem, color=color, alpha=0.15, lw=0, transform=dodge)
        # x limits are set by hand: the dodge transform breaks matplotlib's automatic limits
        x_min, x_max = min(RANGES[knob]), max(RANGES[knob])
        if knob == "seq_error_sub_rate":
            ax.set_xscale("log")
            ax.set_xlim(x_min / 1.5, x_max * 1.5)
        else:
            padding = 0.05 * (x_max - x_min)
            ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_xticks(RANGES[knob])
        ax.minorticks_off()
        if last_row:
            ax.set_xticklabels([f"{v:g}" for v in RANGES[knob]], rotation=45, ha="right", rotation_mode="anchor")
            ax.set_xlabel(KNOB_LABELS[knob])
        if log_y:
            ax.set_yscale("log")
        else:
            ax.set_ylim(-0.02, 1.02)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", lw=0.4, alpha=0.4)


# metric_name -> (y label, figure title)
PAPER_FIGURES = {
    "subclone_barcodes_F1": ("F1", "Benchmarking barcode F1 under simulated noise"),
    "subclone_barcodes_precision": ("Precision", "Benchmarking barcode precision under simulated noise"),
    "subclone_barcodes_recall": ("Recall", "Benchmarking barcode recall under simulated noise"),
    "subclone_cells_accuracy": ("Accuracy", "Benchmarking cell-to-clone accuracy under simulated noise"),
    "subclone_cells_F1": ("F1", "Benchmarking cell-to-clone F1 under simulated noise"),
    "subclone_cells_precision": ("Precision", "Benchmarking cell-to-clone precision under simulated noise"),
    "subclone_cells_recall": ("Recall", "Benchmarking cell-to-clone recall under simulated noise"),
    "time_model": ("Run time (s)", "Benchmarking run time under simulated noise"),
    # weighted barcode variants
    "subclone_barcodes_F1_weighted": ("F1 (weighted)", "Benchmarking weighted barcode F1 under simulated noise"),
    "subclone_barcodes_precision_weighted": ("Precision (weighted)", "Benchmarking weighted barcode precision under simulated noise"),
    "subclone_barcodes_recall_weighted": ("Recall (weighted)", "Benchmarking weighted barcode recall under simulated noise"),
}
# trivial baselines take no time, so they are left out of the run time plot
BASELINES = ["one_large_clone__2025_05_12_v2", "empty_clone_call__2025_05_12_v1"]
REAL_MODELS = {k: v for k, v in MODELS.items() if k not in BASELINES}


if __name__ == "__main__":
    outdir = os.path.join(OUTDIR, "lineplots")
    os.makedirs(outdir, exist_ok=True)
    df = load_results(list(PAPER_FIGURES))
    df.to_csv(os.path.join(outdir, "benchmark_results_long.csv"), index=False)
    # one figure per metric
    for metric, (label, title) in PAPER_FIGURES.items():
        is_time = metric.startswith("time")
        fig = plot_metric_lines(
            df,
            rows=[(metric, label)],
            title=title,
            outpath=os.path.join(outdir, f"lines_{metric}.pdf"),
            log_y=is_time,
            models=REAL_MODELS if is_time else MODELS,
        )
        fig.savefig(os.path.join(outdir, f"lines_{metric}.png"), bbox_inches="tight", dpi=200)
        plt.close(fig)

    # combined figures: one row per metric
    COMBINED = {
        "barcodes": ("Benchmarking barcode recognition under simulated noise",
                     [f"subclone_barcodes_{s}{w}" for s in ["F1", "precision", "recall"] for w in ["", "_weighted"]]),
        "cells": ("Benchmarking cell-to-clone assignment under simulated noise",
                  [f"subclone_cells_{s}" for s in ["accuracy", "F1", "precision", "recall"]]),
        "runtime": ("Benchmarking run time under simulated noise", ["time_model"]),
    }
    for name, (title, metrics) in COMBINED.items():
        is_time = name == "runtime"
        fig = plot_metric_lines(
            df,
            rows=[(m, PAPER_FIGURES[m][0]) for m in metrics],
            title=title,
            outpath=os.path.join(outdir, f"combined_{name}.pdf"),
            log_y=is_time,
            models=REAL_MODELS if is_time else MODELS,
        )
        fig.savefig(os.path.join(outdir, f"combined_{name}.png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
