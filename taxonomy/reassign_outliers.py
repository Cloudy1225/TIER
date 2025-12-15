import json
import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils import compute_cluster_centers

from utils import LLM

from openai import OpenAI

def llm_reassign_cluster(sample_text, summaries,
                         api_key, base_url="https://api.deepseek.com", model="deepseek-chat") -> int:
    """
    Given a sample's text and a list of 3 cluster summaries (list of dicts: {"label": ..., "summary": ...}),
    use DeepSeek/OpenAI API to decide which cluster a document best belongs to.
    Returns the cluster number (1, 2, 3, ...).
    """
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Format cluster summaries
    cluster_blocks = []
    for i, summary in enumerate(summaries):
        label = summary.get("label", "")
        summ = summary.get("summary", "")
        cluster_blocks.append(f"Cluster {i+1}:\nLabel: {label}\nSummary: {summ}\n")
    clusters_text = "\n".join(cluster_blocks)

    user_prompt = (
        "Here is a document and a list of cluster summaries:\n\n"
        f"Document:\n{sample_text.strip()}\n\n"
        f"{clusters_text}\n"
        "Decide which cluster the document best belongs to. "
        "Only return the number of the best matching cluster (e.g., 1, 2, 3, ...)."
    )

    messages = [
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )

        reply = response.choices[0].message.content.strip()
        # Extract the integer from the reply
        reply = int(reply)
        if reply >= len(summaries):
            return 1
        return int(reply)
    except Exception:
        # If LLM returns something unexpected, fallback to 1
        return 1


def load_outlier_decisions_from_file(save_path):
    """Load existing outlier decisions from file (JSONL: one dict per line)."""
    if not os.path.exists(save_path):
        return {}
    decisions = {}
    with open(save_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                key = make_decision_key(
                    record["sample_index"],
                    tuple(record["nearest_clusters"])
                )
                decisions[key] = record
    return decisions


def append_outlier_decision_to_file(save_path, decision_detail):
    """Append a single outlier decision to file (JSONL: one dict per line)."""
    with open(save_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(decision_detail, ensure_ascii=False) + "\n")


def make_decision_key(sample_index, nearest_clusters):
    """Key for a unique outlier/nearest-3-cluster assignment."""
    return f"{sample_index}_" + "_".join([str(x) for x in nearest_clusters])


def reassign_outliers_with_llm(z, raw_texts, cluster_indices,
                               cluster_centers, cluster_llm_summaries,
                               outlier_ratio=0.05, n_nearest_clusters=3,
                               save_path="outlier_llm_decisions.jsonl", verbose=True, llm='DeepSeek'):
    """
    For each cluster, select the `outlier_ratio` samples farthest from the center (by cosine similarity).
    For each such sample, find its nearest `n_nearest_clusters` cluster centers (by cosine similarity),
    and use LLM to decide which of those `n_nearest_clusters` clusters it fits best.
    After each LLM decision, update cluster_indices accordingly.
    Save detailed LLM decision info for each reassigned sample.

    Args:
        z: ndarray (n_samples, n_features), sample embeddings.
        raw_texts: ndarray (n_samples,), text description for each sample.
        cluster_indices: ndarray (n_samples,), cluster index for each sample.
        cluster_centers: ndarray (n_clusters, n_features), cluster centers.
        cluster_llm_summaries: list of dicts, each {"label": ..., "summary": ...} for each cluster.
        outlier_ratio: float, outlier ratio.
        n_nearest_clusters: int, number of nearest clusters.
        save_path: str, file path to save checkpointed LLM decision results.
        verbose: bool

    Returns:
        new_cluster_indices: ndarray, cluster assignment after reassignment.
        new_cluster_centers: ndarray, cluster centers after reassignment.
    """
    n_clusters = cluster_centers.shape[0]
    new_cluster_indices = np.copy(cluster_indices)
    outlier_decisions = []

    # Load existing decisions from checkpoint file
    decision_cache = load_outlier_decisions_from_file(save_path)
    if verbose:
        print(f"Loaded {len(decision_cache)} previous outlier decisions from {save_path}")

    for c in range(n_clusters):
        idxs = np.where(cluster_indices == c)[0]
        if len(idxs) == 0:
            continue
        # Compute cosine similarity for all samples in the cluster to the center
        sims = cosine_similarity(z[idxs], cluster_centers[c:c + 1]).squeeze()
        # Sort ascending (farthest first)
        outlier_count = max(1, int(np.ceil(outlier_ratio * len(idxs))))
        outlier_idx_sort = np.argsort(sims)[:outlier_count]
        outlier_sample_idxs = idxs[outlier_idx_sort]
        for sample_idx in outlier_sample_idxs:
            # Compute similarity to all cluster centers
            center_sims = cosine_similarity(z[sample_idx:sample_idx+1], cluster_centers).squeeze()
            # Get indices of top nearest clusters (highest similarity)
            nearest = np.argsort(-center_sims)[:n_nearest_clusters]
            summaries = [cluster_llm_summaries[k] for k in nearest]
            sample_text = raw_texts[sample_idx]

            # Compose lookup key for this assignment
            key = make_decision_key(sample_idx, nearest)
            if key in decision_cache:
                # Use previous decision
                decision_detail = decision_cache[key]
                assigned_cluster = int(decision_detail["assigned_cluster"])
                if verbose:
                    print(f"[Cached] Sample {sample_idx}, nearest {nearest}, assign to {assigned_cluster}")
            else:
                # Query LLM and record result
                decision = llm_reassign_cluster(sample_text, summaries, **LLM[llm])
                assigned_cluster = int(nearest[decision - 1])
                prev_cluster = int(new_cluster_indices[sample_idx])
                outlier_score = float(sims[np.where(idxs == sample_idx)[0][0]])
                decision_detail = {
                    "sample_index": int(sample_idx),
                    "sample_text": sample_text,
                    "original_cluster": prev_cluster,
                    "outlier_scores_in_original": outlier_score,
                    "nearest_clusters": [int(x) for x in nearest],
                    "nearest_llm_summaries": summaries,
                    "llm_decision": int(decision),
                    "assigned_cluster": assigned_cluster
                }
                append_outlier_decision_to_file(save_path, decision_detail)
                if verbose:
                    print(
                        f"[New] Sample {sample_idx} (orig cluster {prev_cluster}) "
                        f"outlier sim {outlier_score:.4f} "
                        f"nearest clusters {nearest}, "
                        f"LLM decision: {decision}, assign to cluster {assigned_cluster}"
                    )

            # Update cluster assignment and add to result list
            new_cluster_indices[sample_idx] = assigned_cluster
            outlier_decisions.append(decision_detail)

    new_cluster_centers = compute_cluster_centers(z, cluster_indices, n_clusters)

    return new_cluster_indices, new_cluster_centers

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    n = 100
    d = 8
    k = 6
    z = np.random.randn(n, d)
    cluster_indices = np.random.choice(k, size=n)
    cluster_centers = np.random.randn(k, d)
    raw_texts = np.array([f"Sample text {i}" for i in range(n)])
    # Mock LLM cluster summaries
    cluster_llm_summaries = [{"label": f"Label_{i}", "summary": f"Summary for cluster {i}"} for i in range(k)]
    new_cluster_indices, new_cluster_centers = reassign_outliers_with_llm(
        z, raw_texts, cluster_indices, cluster_centers, cluster_llm_summaries,
        save_path="outlier_llm_decisions.jsonl",
        outlier_ratio=0.5,
        n_nearest_clusters=5,
        verbose=True
    )
    print("Updated cluster indices:", new_cluster_indices)