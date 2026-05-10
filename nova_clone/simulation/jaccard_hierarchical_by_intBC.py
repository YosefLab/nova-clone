import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

def clone_calling_jaccard_hierarchical_by_intBC(
        input_df: pd.DataFrame, 
        umi_threshold: int = 10, 
        min_ibtbc_per_clone: int = 2,
        ratio_dendrogram: float = 0.9,
        min_cells_per_clone: int = 10,
        method_linkage: str = 'complete',
        min_cells_with_intBC: int = 10
    ) -> dict:
    
    # Pre-filtering:
    # Filter out cells with low UMI counts
    df_filtered = input_df[input_df['UMI_count'] >= umi_threshold].copy()
    # Filter out intBCs that appear in too few cells
    intBC_counts = df_filtered['intBC'].value_counts()
    valid_intBCs = intBC_counts[intBC_counts >= min_cells_with_intBC].index
    df_filtered = df_filtered[df_filtered['intBC'].isin(valid_intBCs)]

    # Binary intBC × cell matrix (each intBC = row, each cell = column)
    intBC_cell_matrix = pd.crosstab(df_filtered['intBC'], df_filtered['cellBC'])

    # Compute Jaccard distance between intBCs
    distance_matrix = pairwise_distances(intBC_cell_matrix.to_numpy(), metric='jaccard')

    # Hierarchical clustering
    Z = linkage(distance_matrix, method=method_linkage)
    plt.figure(figsize=(10, 7))
    dendrogram(Z, color_threshold=np.max(Z[:, 2])*ratio_dendrogram)
    plt.show()

    # Cut the tree to form clusters
    clone_labels = fcluster(Z, t=np.max(Z[:, 2])*ratio_dendrogram, criterion='distance')
    counts = Counter(clone_labels)
    clone_labels = np.array([
        cl if counts[cl] > min_ibtbc_per_clone else None  # filter clones with too few intBCs
        for cl in clone_labels
    ])

    # Map intBC to clone
    intBCs = intBC_cell_matrix.index.tolist()
    intBC_to_clone = {
        intbc: f"clone_{cl}" for intbc, cl in zip(intBCs, clone_labels) if cl is not None
    }

    # Group intBCs per clone
    clone_to_intBCs = defaultdict(set)
    for intbc, clone in intBC_to_clone.items():
        clone_to_intBCs[clone].add(intbc)

    # For each cell, assign to the clone with most overlapping intBCs
    cell_to_clone = {}
    for cell in df_filtered['cellBC'].unique():
        cell_intBCs = set(df_filtered[df_filtered['cellBC'] == cell]['intBC'])
        best_clone = None
        best_overlap = 0
        for clone, clone_bcs in clone_to_intBCs.items():
            overlap = len(cell_intBCs & clone_bcs)
            if overlap > best_overlap:
                best_overlap = overlap
                best_clone = clone
        if best_clone is not None and best_overlap > 0:
            cell_to_clone['sample0_' + cell] = best_clone

    # Filter clones with too few cells
    cell_clone_counts = Counter(cell_to_clone.values())
    cell_to_clone = {cell: set([clone]) for cell, clone in cell_to_clone.items() if cell_clone_counts[clone] >= min_cells_per_clone}

    return {
        "cell_to_clones_mapping": cell_to_clone,
        "clone_intbc_sets": dict(clone_to_intBCs)
    }
