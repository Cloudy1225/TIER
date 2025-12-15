import json
import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

from utils import LLM


def cluster_coherence(embeddings, cluster_center):
    """
    Compute the mean cosine similarity between samples and their center.
    """
    sim_matrix = cosine_similarity(embeddings, cluster_center)
    return float(np.mean(sim_matrix))


from openai import OpenAI

def llm_should_split(texts, api_key, base_url="https://api.deepseek.com", model="deepseek-chat") -> int:
    """
    Use DeepSeek/OpenAI API to determine the number of subclusters for a cluster of texts.
    Returns an integer >= 1.
    """
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Format documents list
    text_block = '\n'.join(f'{i+1}. {text.strip()}' for i, text in enumerate(texts))

    user_prompt = (
        "Here are some documents currently in one cluster:\n\n"
        f"{text_block}\n\n"
        "Please decide whether they should stay in one cluster, or be split into multiple subclusters based on different topics. "
        "Only respond with a single number: the number of subclusters needed. If no split is needed, respond with 1."
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
        return int(reply)
    except Exception:
        # If LLM returns something unexpected, fallback to 1
        return 1


def _load_llm_split_decisions(save_path):
    """Load previous split LLM decisions from JSONL file into a dict."""
    if not os.path.exists(save_path):
        return {}
    decisions = {}
    with open(save_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    key = record.get("cluster_id")
                    if key is not None:
                        decisions[key] = record
                except Exception as e:
                    print(f"Warning: failed to load line: {line.strip()} ({e})")
    return decisions


def _append_llm_split_decision(save_path, record):
    """Append a single split decision record to file."""
    with open(save_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_low_cohesion_clusters_with_llm(
        z, raw_texts, cluster_indices, cluster_centers,
        cohesion_threshold=0.75, n_selected_texts=20,
        save_path="split_llm_decisions.jsonl", verbose=True, llm='DeepSeek'):
    """
    For each cluster, compute the mean cosine similarity between samples and their center.
    For clusters with mean similarity less than the threshold, randomly select `n_selected_texts` samples' texts,
    and pass them to an LLM to decide whether to split the cluster (and into how many).
    If split is required, use KMeans to further cluster the samples in that cluster.
    Return updated cluster_indices and cluster_centers, with new indices from 0.

    Decision checkpointing: Each LLM split decision is saved to a file, and loaded on resume.

    Args:
        z: ndarray (n_samples, n_features), sample embeddings.
        raw_texts: ndarray (n_samples,), text description for each sample.
        cluster_indices: ndarray (n_samples,), cluster index for each sample.
        cluster_centers: ndarray (n_clusters, n_features), cluster centers.
        cohesion_threshold: float, threshold for cluster cohesion.
        n_selected_texts: int, number of texts to select for each cluster.
        save_path: str, path to JSONL file for LLM split decisions.
        verbose: bool

    Returns:
        new_cluster_indices: ndarray (n_samples,), after possible splitting and reindex.
        new_cluster_centers: ndarray (n_final_clusters, n_features)
    """
    n_clusters = cluster_centers.shape[0]
    new_cluster_indices = np.zeros_like(cluster_indices, dtype=int)
    new_centers = []
    cur_cluster_id = 0

    # Load previous LLM split decisions
    llm_split_decisions = _load_llm_split_decisions(save_path)
    if verbose and llm_split_decisions:
        print(f"Loaded {len(llm_split_decisions)} previous LLM split decisions from {save_path}")

    for c in range(n_clusters):
        idxs = np.where(cluster_indices == c)[0]
        if len(idxs) == 0:
            continue
        center = cluster_centers[c]
        # Compute the mean cosine similarity between samples and their center.
        mean_sim = cluster_coherence(z[idxs], cluster_centers[c:c+1])
        if verbose:
            print(f"Cluster {c}: mean cohesion={mean_sim:.4f}, size={len(idxs)}")
        # If cohesion is high enough, keep the cluster as is
        if mean_sim >= cohesion_threshold or len(idxs) <= 1:
            new_cluster_indices[idxs] = cur_cluster_id
            new_centers.append(center)
            cur_cluster_id += 1
        else:
            # Randomly texts from the cluster
            rng = np.random.default_rng(seed=0)
            chosen = rng.choice(idxs, size=min(n_selected_texts, len(idxs)), replace=False)
            texts = list(raw_texts[chosen])

            # Check if LLM decision exists for this cluster
            if str(c) in llm_split_decisions:
                n_split = llm_split_decisions[str(c)]["n_split"]
                if verbose:
                    print(f"[Cached] Cluster {c} flagged for split, LLM suggests {n_split} clusters.")
            else:  # Ask LLM for split decision
                n_split = llm_should_split(texts, **LLM[llm])
                # Save LLM decision
                split_record = {
                    "cluster_id": str(c),
                    "mean_cohesion": float(mean_sim),
                    "sampled_texts": texts,
                    "n_split": int(n_split)
                }
                _append_llm_split_decision(save_path, split_record)
                if verbose:
                    print(f"[New] Cluster {c} flagged for split, LLM suggests {n_split} clusters.")

            if n_split <= 1 or len(idxs) < n_split:
                # Cannot split, keep as is
                new_cluster_indices[idxs] = cur_cluster_id
                new_centers.append(center)
                cur_cluster_id += 1
            else:
                # Further split the cluster using KMeans
                sub_embeddings = z[idxs]
                sub_kmeans = KMeans(n_clusters=n_split, random_state=0)
                sub_labels = sub_kmeans.fit_predict(sub_embeddings)
                for sub_c in range(n_split):
                    sub_idxs = idxs[sub_labels == sub_c]
                    if len(sub_idxs) == 0:
                        continue
                    new_cluster_indices[sub_idxs] = cur_cluster_id
                    new_centers.append(sub_kmeans.cluster_centers_[sub_c])
                    cur_cluster_id += 1

    # new cluster centers
    new_cluster_centers = np.stack(new_centers, axis=0)
    # Remap new_cluster_indices to be consecutive from 0
    unique, inverse = np.unique(new_cluster_indices, return_inverse=True)
    new_cluster_indices = inverse

    if verbose:
        print(f"Final number of clusters: {new_cluster_centers.shape[0]}")
    return new_cluster_indices, new_cluster_centers


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    n = 100
    d = 8
    k = 6
    z = np.random.randn(n, d)
    cluster_indices = np.random.choice(k, size=n)
    cluster_centers = normalize(np.random.randn(k, d))
    raw_texts = np.array([f"Sample text {i}" for i in range(n)])
    new_cluster_indices, new_centers = split_low_cohesion_clusters_with_llm(
        z, raw_texts, cluster_indices, cluster_centers,
        cohesion_threshold=0.75, verbose=True,
        save_path="split_llm_decisions.jsonl"
    )
    print("Old cluster indices:", cluster_indices)
    print("New cluster indices:", new_cluster_indices)
    print("New centers shape:", new_centers.shape)