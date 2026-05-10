import pandas as pd
import numpy as np
import hdbscan

def clone_calling_pearson_hdbscan_by_intBC(
        df: pd.DataFrame, 
        min_UMIs:int = 10, 
        min_ibtbc_per_clone:int = 2, 
        min_cell_cluster_size: int = 30,
        min_cells_with_intBC:int = 10,
        min_samples: int = 30
    ) -> dict:

    # Pre-filtering:
    # Filter out cells with low UMI counts
    df_filtered = df[df['UMI_count'] >= min_UMIs].copy()
    # Filter out intBCs that appear in too few cells
    intBC_counts = df_filtered['intBC'].value_counts()
    valid_intBCs = intBC_counts[intBC_counts >= min_cells_with_intBC].index
    df_filtered = df_filtered[df_filtered['intBC'].isin(valid_intBCs)]

    # cellBC × intBC UMI matrix
    umi_mat = df_filtered.pivot_table(index="cellBC", columns="intBC", values="UMI_count", aggfunc="sum", fill_value=0)

    # Pearson correlation between intBCs
    intBC_mat = umi_mat.T  # now rows are intBCs
    corr = np.corrcoef(intBC_mat)

    # cluster intBCs using HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_ibtbc_per_clone, min_samples=min_samples)
    clusterer.fit(corr)
    cluster_labels = clusterer.labels_
    intBCs = intBC_mat.index.tolist()
    cluster_df = pd.DataFrame({
        "intBC": intBCs,
        "cluster": cluster_labels
    })

    clone_to_intBCs = {}
    for cluster_id in np.unique(cluster_labels):
        if cluster_id == -1:  # noise
            continue
        clone_to_intBCs['clone_'+str(cluster_id)] = cluster_df.loc[cluster_df["cluster"] == cluster_id, "intBC"].tolist()

    # Assign each cell to best clone by intersection of intBCs
    cell_to_clone = {}
    for cell, row in umi_mat.iterrows():
        cell_intBCs = set(row[row > 0].index)
        best_score = 0
        best_clone = None
        for clone_id, intBC_list in clone_to_intBCs.items():
            overlap = len(cell_intBCs.intersection(set(intBC_list)))
            if overlap > best_score:
                best_score = overlap
                best_clone = clone_id
        if best_clone is not None:
            cell_to_clone[cell] = best_clone

    # remove clones with less than min_cell_cluster_size cells
    clone_counts = pd.Series(cell_to_clone).value_counts()
    valid_clones = clone_counts[clone_counts >= min_cell_cluster_size].index.tolist()
    cell_to_clone = {'sample0_'+cell: set([clone]) for cell, clone in cell_to_clone.items() if clone in valid_clones}
    clone_to_intBCs = {clone: set(intBCs) for clone, intBCs in clone_to_intBCs.items() if clone in valid_clones}

    return {
            'cell_to_clones_mapping':cell_to_clone, 
            'clone_to_intBCs':clone_to_intBCs
            }
