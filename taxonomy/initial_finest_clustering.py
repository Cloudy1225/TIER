import argparse
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import joblib


def initial_finest_clustering(z, n_finest_clusters, save_path="initial_finest_clustering.joblib", verbose=True):
    """
    Perform KMeans clustering to obtain the finest clusters and save/load the results.

    Args:
        z: ndarray of shape (n_samples, n_features), the sample embeddings.
        n_finest_clusters: int, number of clusters for the finest granularity.
        save_path: str, path to save/load the clustering results.
        verbose: bool, whether to print progress logs.

    Returns:
        cluster_indices: ndarray of shape (n_samples,), predicted cluster index for each sample.
        cluster_centers: ndarray of shape (n_finest_clusters, n_features), cluster centers after normalization.
    """
    # If clustering result already exists, load and return
    if os.path.exists(save_path):
        if verbose:
            print(f"Found existing clustering result at {save_path}, loading...")
        cluster_indices, cluster_centers = joblib.load(save_path)
        if verbose:
            print(f"Loaded cluster_indices shape: {cluster_indices.shape}, cluster_centers shape: {cluster_centers.shape}")
        return cluster_indices, cluster_centers

    if verbose:
        print(f"Performing KMeans clustering on {z.shape[0]} samples to obtain {n_finest_clusters} finest clusters (L2 normalization).")
    z_norm = normalize(z)
    kmeans = KMeans(n_clusters=n_finest_clusters, random_state=0)
    cluster_indices = kmeans.fit_predict(z_norm)
    cluster_centers = kmeans.cluster_centers_

    if verbose:
        print(f"Clustering finished. First 10 cluster indices: {cluster_indices[:10]}, cluster_centers shape: {cluster_centers.shape}")
        print(f"Saving clustering result to {save_path}...")
    joblib.dump((cluster_indices, cluster_centers), save_path)
    return cluster_indices, cluster_centers


if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='computer',
                        choices=['cora', 'citeseer', 'pubmed', 'wikics',
                                 'photo', 'computer', 'history', 'arxiv', 'products',
                                 'cornell', 'wisconsin', 'texas', 'washington'])
    parser.add_argument('--n_finest_clusters', type=int, default=512, help="Number of finest clusters.")
    args = parser.parse_args()

    dataset = args.dataset
    ebd_path = f'../pretrain/embeddings/{dataset}.joblib'
    z = joblib.load(ebd_path)

    n_finest_clusters = args.n_finest_clusters
    save_dir = f'./outputs/{dataset}/{n_finest_clusters}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = f'{save_dir}/initial_finest_clustering.joblib'

    cluster_indices, cluster_centers = initial_finest_clustering(z, n_finest_clusters, save_path, verbose=True)
    print(f"(cluster_indices, cluster_centers) are available at {save_path}.")

    # print(f"Performing KMeans clustering on {z.shape[0]} samples to obtain {n_finest_clusters} finest clusters.")
    # z_norm = normalize(z)
    # kmeans = KMeans(n_clusters=n_finest_clusters, random_state=0)
    # cluster_indices = kmeans.fit_predict(z_norm)
    # print("First 10 cluster indices:", cluster_indices[:10])
    #
    # print(f"Clustering finished. Saving results to {save_path} ...")
    # joblib.dump(cluster_indices, save_path)

    # dataset = args.dataset
    # encoder_name = 'roberta'
    # re_split = 0
    # data = load_graph_dataset_for_gnn(dataset_name=dataset,
    #                                   device='cpu',
    #                                   path_prefix='..',
    #                                   emb_model=encoder_name if len(encoder_name) else "shallow",
    #                                   re_split=0)
    # x, edge_index, y = data.x, data.edge_index, data.y
    #
    #
    # n_finest_clusters = 256
    # save_path = "finest_clustering_results.joblib"
    # cluster_indices, cluster_centers = finest_clustering(z, n_finest_clusters, save_path=save_path, verbose=True)
    # print("First 10 cluster indices:", cluster_indices[:10])
    # print("Cluster centers shape:", cluster_centers.shape)
