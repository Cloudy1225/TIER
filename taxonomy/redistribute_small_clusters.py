import argparse
import os

import joblib
import numpy as np
from sklearn.preprocessing import normalize

def redistribute_small_clusters(
    z, cluster_indices, cluster_centers, min_n_samples_per_cluster=5,
        save_path="redistribute_small_clusters.joblib", verbose=True
):
    """
    Redistribute samples in clusters with fewer than min_n_samples_per_cluster samples.
    Each such sample will be reassigned to the nearest large cluster (by cosine similarity to center).
    After redistribution, remaining clusters will be reindexed to [0, n_clusters_left).
    The new cluster centers are simply the original centers of the kept clusters (newly assigned outliers do not affect the centers).

    Args:
        z: ndarray (n_samples, n_features), sample embeddings.
        cluster_indices: ndarray (n_samples,), cluster index for each sample.
        cluster_centers: ndarray (n_clusters, n_features), cluster centers (should be L2-normalized).
        min_n_samples_per_cluster: int, minimum number of samples required for a cluster to be considered large.
        save_path: str or None, if provided will save/load the redistributed result.
        verbose: bool, print progress.

    Returns:
        new_cluster_indices: ndarray (n_samples,), the adjusted and reindexed cluster indices.
        new_cluster_centers: ndarray (n_clusters_left, n_features), centers for reindexed clusters (unchanged from original).
    """
    # If save_path exists, load and return result
    if save_path is not None and os.path.exists(save_path):
        if verbose:
            print(f"Found existing redistribution result at {save_path}, loading...")
        new_cluster_indices, new_cluster_centers = joblib.load(save_path)
        if verbose:
            print(f"Loaded new_cluster_indices shape: {new_cluster_indices.shape}, new_cluster_centers shape: {new_cluster_centers.shape}")
        return new_cluster_indices, new_cluster_centers

    n_samples = z.shape[0]
    n_clusters = cluster_centers.shape[0]
    # Count samples per cluster
    counts = np.bincount(cluster_indices, minlength=n_clusters)
    # Identify small and large clusters
    small_clusters = np.where(counts < min_n_samples_per_cluster)[0]
    large_clusters = np.where(counts >= min_n_samples_per_cluster)[0]
    if verbose:
        print(f"Found {len(small_clusters)} small clusters, {len(large_clusters)} large clusters.")

    # L2 normalize large cluster centers (for cosine similarity)
    large_centers = normalize(cluster_centers[large_clusters])

    # Copy cluster indices for in-place modification
    new_cluster_indices = cluster_indices.copy()
    z_norm = normalize(z)

    # For each small cluster, redistribute its samples
    for small_c in small_clusters:
        idxs = np.where(cluster_indices == small_c)[0]
        if len(idxs) > 0:
            sim_matrix = np.dot(z_norm[idxs], large_centers.T)  # shape (m, k)
            nearest_idx = np.argmax(sim_matrix, axis=1)  # shape (m,)
            nearest_clusters = large_clusters[nearest_idx]  # shape (m,)
            new_cluster_indices[idxs] = nearest_clusters
            if verbose:
                print(f"Reassigned {len(idxs)} samples from small cluster {small_c} to {nearest_clusters}.")

    # Only keep clusters which now have at least min_n_samples_per_cluster samples originally
    kept_clusters = large_clusters
    if verbose:
        print(f"Kept {len(kept_clusters)} clusters after redistribution.")

    # Remap old cluster indices to [0, n_clusters_left)
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_clusters)}
    new_cluster_indices = np.array([old_to_new[idx] for idx in new_cluster_indices])

    # New cluster centers: just the original large cluster centers, in new order
    new_cluster_centers = cluster_centers[kept_clusters]

    if verbose:
        print(f"Final cluster indices are reindexed to [0, {len(kept_clusters)}), new_cluster_centers shape: {new_cluster_centers.shape}")

    # Save result if save_path is provided
    if save_path is not None:
        joblib.dump((new_cluster_indices, new_cluster_centers), save_path)
        if verbose:
            print(f"(new_cluster_indices, new_cluster_centers) saved to {save_path}.")

    return new_cluster_indices, new_cluster_centers

if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'citeseer', 'pubmed', 'wikics',
                                 'photo', 'computer', 'history', 'arxiv', 'products',
                                 'cornell', 'wisconsin', 'texas', 'washington'])
    parser.add_argument('--n_finest_clusters', type=int, default=64, help="Number of finest clusters.")
    parser.add_argument('--min_n_samples_per_cluster', type=int, default=10, help="Minimum number of samples required for a cluster.")
    args = parser.parse_args()

    dataset = args.dataset
    n_finest_clusters = args.n_finest_clusters
    min_n_samples_per_cluster = args.min_n_samples_per_cluster

    ebd_path = f'../pretrain/embeddings/{dataset}.joblib'
    z = joblib.load(ebd_path)

    save_dir = f'./outputs/{dataset}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    initial_save_path = f'{save_dir}/{n_finest_clusters}/initial_finest_clustering.joblib'
    redistributed_save_path = f'{save_dir}/{n_finest_clusters}/redistribute_small_clusters.joblib'

    cluster_indices, cluster_centers = joblib.load(initial_save_path)
    new_cluster_indices, new_cluster_centers = redistribute_small_clusters(
        z, cluster_indices, cluster_centers, min_n_samples_per_cluster, verbose=True, save_path=redistributed_save_path
    )
    print(f"(new_cluster_indices, new_cluster_centers) are available at {redistributed_save_path}.")