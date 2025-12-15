import argparse
import os
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from dataloader import load_graph_dataset

from initial_finest_clustering import initial_finest_clustering
from split_low_cohesion_clusters import split_low_cohesion_clusters_with_llm
from merge_similar_clusters import merge_similar_clusters_with_llm
from redistribute_small_clusters import redistribute_small_clusters
from summarize_clusters import summarize_clusters_with_llm
from reassign_outliers import reassign_outliers_with_llm


def clustering_pipeline(
    z, raw_texts, save_dir, n_finest_clusters,
    cohesion_threshold=0.75,
    n_selected_texts_split=20,
    sim_threshold=0.85,
    n_selected_texts_merge=10,
    min_n_samples_per_cluster=5,
    n_selected_texts_summarize=10,
    outlier_ratio=0.05,
    n_nearest_clusters=3,
    verbose=True,
):
    # Step 1: initial finest clustering
    finest_path = os.path.join(save_dir, "initial_finest_clustering.joblib")
    if os.path.exists(finest_path):
        if verbose: print(f"Loading {finest_path}")
        cluster_indices, cluster_centers = joblib.load(finest_path)
    else:
        cluster_indices, cluster_centers = initial_finest_clustering(
            z, n_finest_clusters, save_path=finest_path, verbose=verbose
        )

    # Step 2: split low cohesion clusters
    split_path = os.path.join(save_dir, "split_low_cohesion_clusters.joblib")
    if os.path.exists(split_path):
        if verbose: print(f"Loading {split_path}")
        cluster_indices, cluster_centers = joblib.load(split_path)
    else:
        split_llm_decisions_path = os.path.join(save_dir, "split_llm_decisions.jsonl")
        cluster_indices, cluster_centers = split_low_cohesion_clusters_with_llm(
            z, raw_texts, cluster_indices, cluster_centers,
            cohesion_threshold=cohesion_threshold,
            n_selected_texts=n_selected_texts_split,
            save_path=split_llm_decisions_path,
            verbose=verbose
        )
        joblib.dump((cluster_indices, cluster_centers), split_path)

    # Step 3: merge similar clusters
    merge_path = os.path.join(save_dir, "merge_similar_clusters.joblib")
    if os.path.exists(merge_path):
        if verbose: print(f"Loading {merge_path}")
        cluster_indices, cluster_centers = joblib.load(merge_path)
    else:
        merge_llm_decisions_path = os.path.join(save_dir, "merge_llm_decisions.jsonl")
        cluster_indices, cluster_centers = merge_similar_clusters_with_llm(
            z, raw_texts, cluster_indices, cluster_centers,
            similar_pairs=None,
            sim_threshold=sim_threshold,
            n_selected_texts=n_selected_texts_merge,
            save_path=merge_llm_decisions_path,
            verbose=verbose
        )
        joblib.dump((cluster_indices, cluster_centers), merge_path)

    # Step 4: redistribute small clusters
    redistrib_path = os.path.join(save_dir, "redistribute_small_clusters.joblib")
    if os.path.exists(redistrib_path):
        if verbose: print(f"Loading {redistrib_path}")
        cluster_indices, cluster_centers = joblib.load(redistrib_path)
    else:
        cluster_indices, cluster_centers = redistribute_small_clusters(
            z, cluster_indices, cluster_centers,
            min_n_samples_per_cluster=min_n_samples_per_cluster,
            save_path=redistrib_path,
            verbose=verbose
        )

    # Step 5: summarize clusters
    cluster_summaries_path = os.path.join(save_dir, "cluster_summaries.jsonl")
    cluster_llm_summaries = summarize_clusters_with_llm(
        z, raw_texts, cluster_indices, cluster_centers,
        n_selected_texts=n_selected_texts_summarize,
        save_path=cluster_summaries_path,
        verbose=verbose
    )

    # Step 6: reassign outliers with LLM
    outlier_llm_decisions_path = os.path.join(save_dir, "outlier_llm_decisions.jsonl")
    outlier_joblib_path = os.path.join(save_dir, "reassign_outliers_with_llm.joblib")
    if os.path.exists(outlier_joblib_path):
        if verbose: print(f"Loading {outlier_joblib_path}")
        cluster_indices, cluster_centers = joblib.load(outlier_joblib_path)
    else:
        cluster_indices, cluster_centers = reassign_outliers_with_llm(
            z, raw_texts, cluster_indices, cluster_centers, cluster_llm_summaries,
            outlier_ratio=outlier_ratio,
            n_nearest_clusters=n_nearest_clusters,
            save_path=outlier_llm_decisions_path,
            verbose=verbose
        )
        joblib.dump((cluster_indices, cluster_centers), outlier_joblib_path)

    if verbose:
        print("Clustering pipeline finished.")
        print(f"Final clusters: {np.unique(cluster_indices).size}, centers shape: {cluster_centers.shape}")

    return cluster_indices, cluster_centers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'citeseer', 'pubmed', 'wikics',
                                 'photo', 'computer', 'history', 'arxiv', 'products',
                                 'cornell', 'wisconsin', 'texas', 'washington'])
    # (7, 64), (6, 64), (3, 16, 64), (10, 128), (12, 64, 256), (10, 128, 512), (12, 64, 256), (40, 128, 512, 2048)
    parser.add_argument('--n_clusters_list', type=str, default="7, 64")
    parser.add_argument('--cohesion_threshold', type=float, default=0.75,
                        help="Cohesion threshold for splitting clusters.")
    parser.add_argument('--n_selected_texts_split', type=int, default=20,
                        help="Number of texts for LLM splitting.")
    parser.add_argument('--sim_threshold', type=float, default=0.85,
                        help="Similarity threshold for merging clusters.")
    parser.add_argument('--n_selected_texts_merge', type=int, default=10,
                        help="Number of texts for LLM merging.")
    parser.add_argument('--min_n_samples_per_cluster', type=int, default=10,
                        help="Minimum samples per cluster for redistribution.")
    parser.add_argument('--n_selected_texts_summarize', type=int, default=10,
                        help="Number of texts for LLM summarization.")
    parser.add_argument('--outlier_ratio', type=float, default=0.05,
                        help="Outlier ratio for LLM outlier reassignment.")
    parser.add_argument('--n_nearest_clusters', type=int, default=3,
                        help="Number of nearest clusters for outlier reassignment.")
    parser.add_argument('--use_llm', type=int, default=1,
                        help="Whether to use LLM-powered clustering refinement.")
    args = parser.parse_args()

    verbose = True
    dataset = args.dataset
    n_clusters_list = [int(n_clusters.strip()) for n_clusters in args.n_clusters_list.split(",")]
    n_finest_clusters = n_clusters_list[-1]

    if dataset in ['cora', 'citeseer', 'pubmed', 'wikics']:
        args.cohesion_threshold = 0.75
        args.sim_threshold = 0.9
        args.outlier_ratio = 0.05

    if dataset in ['photo', 'history']:
        args.cohesion_threshold = 0.75
        args.sim_threshold = 0.9
        args.outlier_ratio = 0.05

    if dataset in ['computer', 'arxiv']:
        args.cohesion_threshold = 0.75
        args.sim_threshold = 0.95
        args.outlier_ratio = 0.05

    data = load_graph_dataset(dataset_name=dataset,
                              device='cpu',
                              path_prefix='..',
                              re_split=0)
    raw_texts, y, label_name = np.array(data.raw_texts), data.y.numpy(), data.label_name

    ebd_path = os.path.join('..', 'pretrain', 'embeddings', f'{dataset}.joblib')
    z = joblib.load(ebd_path)

    save_dir = os.path.join('outputs', dataset, str(n_finest_clusters))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, "conf.txt"), "a") as f:
        f.write(str(args)+'\n')

    n_samples = z.shape[0]
    n_layers = len(n_clusters_list)
    # To store each sample's cluster index at each layer (from coarsest to finest)
    sample_cluster_indices = np.zeros((n_samples, n_layers), dtype=np.int32)
    # To store cluster centers at each layer
    layer_centers = [None] * n_layers
    # To store parent cluster index for each cluster at each layer (except coarsest)
    cluster_hierarchy = [None] * (n_layers - 1)

    # Step 1: Start from the finest layer
    current_points = z
    for l in reversed(range(n_layers)):
        n_clusters = n_clusters_list[l]
        if verbose:
            if l == n_layers - 1:
                print(
                    f"Layer L{l + 1}: KMeans clustering to {n_clusters} clusters on {current_points.shape[0]} samples.")
            else:
                print(
                    f"Layer L{l + 1}: KMeans clustering to {n_clusters} clusters on {current_points.shape[0]} cluster centers from layer L{l + 2}.")
        if args.use_llm and l == n_layers - 1:
            indices, centers = clustering_pipeline(
                z, raw_texts, save_dir, n_finest_clusters,
                cohesion_threshold=args.cohesion_threshold,
                n_selected_texts_split=args.n_selected_texts_split,
                sim_threshold=args.sim_threshold,
                n_selected_texts_merge=args.n_selected_texts_merge,
                min_n_samples_per_cluster=args.min_n_samples_per_cluster,
                n_selected_texts_summarize=args.n_selected_texts_summarize,
                outlier_ratio=args.outlier_ratio,
                n_nearest_clusters=args.n_nearest_clusters,
                verbose=verbose
            )
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            indices = kmeans.fit_predict(normalize(current_points))
            centers = kmeans.cluster_centers_
        layer_centers[l] = centers
        if l == n_layers - 1:
            # Finest layer, assign sample -> cluster
            sample_cluster_indices[:, l] = indices
        else:
            # Parent mapping: cluster in layer l+1 -> cluster in layer l
            cluster_hierarchy[l] = indices
            # For each sample, propagate up the hierarchy
            sample_cluster_indices[:, l] = indices[sample_cluster_indices[:, l + 1]]
        current_points = centers  # Next: cluster on these centers

    # cluster_hierarchy: each entry shape (n_clusters_next_layer,), value is parent cluster index in this layer
    cluster_hierarchy = [np.array(parents) for parents in cluster_hierarchy]

    # Compute shortest path length between finest clusters
    # n_finest_clusters = n_clusters_list[-1]
    n_finest_clusters = np.unique(sample_cluster_indices[:, -1]).shape[0]
    # Build hierarchical path for each finest cluster (from finest to coarsest)
    cluster_paths = []
    for c in range(n_finest_clusters):
        path = [c]
        parent = c
        for l in reversed(range(n_layers - 1)):
            parent = cluster_hierarchy[l][parent]
            path.append(parent)
        cluster_paths.append(path[::-1])  # from coarsest to finest
    cluster_paths = np.array(cluster_paths)  # shape (n_finest_clusters, n_layers)
    # For every pair of finest clusters, find the first layer where they differ
    min_layer_dist = []
    for i in range(n_finest_clusters):
        for j in range(i+1, n_finest_clusters):
            diff_layer = np.argmax(cluster_paths[i] != cluster_paths[j])
            # Shortest path: 2 * (n_layers - diff_layer)
            min_layer_dist.append(2 * (n_layers - diff_layer))
    min_layer_dist = np.array(min_layer_dist)

    if verbose:
        print("Done. Outputs:")
        print("sample_cluster_indices.shape:", sample_cluster_indices.shape)
        print("cluster_hierarchy layers:", [c.shape for c in cluster_hierarchy])
        print("min_layer_dist shape:", min_layer_dist.shape)

    save_path = os.path.join(save_dir, f'hierarchical_clustering_'
                                       f'{"_".join([str(n_clusters) for n_clusters in n_clusters_list])}.joblib')
    joblib.dump((sample_cluster_indices, cluster_hierarchy, min_layer_dist), save_path)
    print(min_layer_dist.max())
