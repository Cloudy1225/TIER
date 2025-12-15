import numpy as np


LLM = {
    'DeepSeek': {
        'model': 'deepseek-chat',
        'base_url': 'https://api.deepseek.com',
        'api_key': 'DEEPSEEK_API_KEY',
    },
    'OpenAI': {
        'model': 'gpt-4o-mini',
        'base_url': 'https://api.openai.com/v1',
        'api_key': "OPENAI_API_KEY",
    },
}


def compute_cluster_centers(z, cluster_indices, n_clusters=None):
    """
    Compute the mean vector (center) of each cluster given sample embeddings and cluster assignments.

    Parameters:
        z (np.ndarray): Array of shape (n_samples, n_features) containing sample embeddings.
        cluster_indices (np.ndarray): Array of shape (n_samples,) containing the cluster index (int) for each sample.
        n_clusters (int, optional): Total number of clusters. If None, it will be inferred from cluster_indices.

    Returns:
        np.ndarray: Array of shape (n_clusters, n_features) containing the mean embedding for each cluster.
    """
    # If number of clusters not provided, infer it from the maximum index
    if n_clusters is None:
        n_clusters = np.max(cluster_indices) + 1

    # Count number of samples assigned to each cluster
    cluster_counts = np.bincount(cluster_indices, minlength=n_clusters)

    # Sum embeddings for each cluster across all dimensions
    # Each row corresponds to a cluster, each column to a dimension
    cluster_sums = np.vstack([
        np.bincount(cluster_indices, weights=z[:, dim], minlength=n_clusters)
        for dim in range(z.shape[1])
    ]).T  # Transpose to shape (n_clusters, n_features)

    # Divide summed vectors by the number of samples per cluster to get the mean
    cluster_centers = cluster_sums / cluster_counts[:, None]  # Broadcasting division

    return cluster_centers
