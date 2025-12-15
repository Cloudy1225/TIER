import json
import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from utils import LLM

from openai import OpenAI

def llm_should_merge(texts_a, texts_b, api_key, base_url="https://api.deepseek.com", model="deepseek-chat") -> bool:
    """
    Use DeepSeek/OpenAI API to determine if two clusters should be merged.
    Returns True if they should be merged, False otherwise.
    """
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Format documents
    text_block_a = '\n'.join(f'{i+1}. {text.strip()}' for i, text in enumerate(texts_a))
    text_block_b = '\n'.join(f'{i+1}. {text.strip()}' for i, text in enumerate(texts_b))

    user_prompt = (
        "Here are two clusters of documents:\n\n"
        f"Cluster A:\n{text_block_a}\n\n"
        f"Cluster B:\n{text_block_b}\n\n"
        "Determine whether the two clusters are about the same or highly similar topic and should be merged. "
        "Return only 1 if they should be merged, or 0 if they should remain separate."
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
        # Extract the integer from the reply, interpret 1 as True, 0 as False
        return int(reply) == 1
    except Exception:
        # If LLM returns something unexpected, fallback to False
        return False


def _make_merge_key(root_i, root_j):
    """Generate a unique key for a merge decision based on sorted cluster roots."""
    return f"{min(root_i, root_j)}_{max(root_i, root_j)}"


def _load_merge_decisions(save_path):
    """Load previous merge LLM decisions from JSONL file into a dict."""
    if not os.path.exists(save_path):
        return {}
    decisions = {}
    with open(save_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    key = record.get("merge_key")
                    if key is not None:
                        decisions[key] = record
                except Exception as e:
                    print(f"Warning: failed to load line: {line.strip()} ({e})")
    return decisions


def _append_merge_decision(save_path, record):
    """Append a single merge decision record to file."""
    with open(save_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def find_similar_cluster_pairs(cluster_centers, sim_threshold=0.85):
    """
    Find all unique pairs of clusters whose center cosine similarity > sim_threshold.
    Returns in descending similarity order.

    Args:
        cluster_centers: ndarray (n_clusters, n_features), cluster centers.
        sim_threshold: float, similarity threshold.

    Returns:
        similar_pairs: List of tuples (i, j, sim), where i < j and sim > sim_threshold,
                       sorted by sim descending.
    """
    # similarity matrix
    sim_matrix = cosine_similarity(cluster_centers)  # shape (n_clusters, n_clusters)
    n_clusters = sim_matrix.shape[0]
    # Only upper triangle (i < j)
    triu_indices = np.triu_indices(n_clusters, k=1)
    sim_vals = sim_matrix[triu_indices]
    mask = sim_vals > sim_threshold
    i_indices = triu_indices[0][mask]
    j_indices = triu_indices[1][mask]
    sim_selected = sim_vals[mask]
    # Sort by similarity descending
    sort_idx = np.argsort(-sim_selected)
    similar_pairs = list(zip(i_indices[sort_idx], j_indices[sort_idx], sim_selected[sort_idx]))
    return similar_pairs


def merge_similar_clusters_with_llm(
        z, raw_texts, cluster_indices, cluster_centers,
        similar_pairs=None, sim_threshold=0.85, n_selected_texts=10,
        save_path="merge_llm_decisions.jsonl", verbose=True, llm='DeepSeek'):
    """
    Merge similar clusters whose center cosine similarity > sim_threshold.
    For each candidate pair, before merging, randomly select `n_selected_texts` samples from each cluster,
    pass their texts to an LLM (via llm_should_merge), and merge only if LLM returns True.
    Only recalculate similarity if either cluster has already been merged.
    Each LLM decision is checkpointed to file, and loaded on resume.

    Args:
        z: ndarray (n_samples, n_features), sample embeddings.
        raw_texts: ndarray (n_samples,), text description for each sample.
        cluster_indices: ndarray (n_samples,), cluster index for each sample.
        cluster_centers: ndarray (n_clusters, n_features), initial cluster centers.
        similar_pairs: List of tuples (i, j, sim), sorted descending by sim.
        sim_threshold: float, similarity threshold.
        n_selected_texts: int, number of texts to select from each cluster.
        save_path: str, path to save merge decisions.
        verbose: bool

    Returns:
        new_cluster_indices: ndarray (n_samples,), after merge and reindex.
        merged_cluster_centers: ndarray (n_final_clusters, n_features)
    """
    n_clusters = cluster_centers.shape[0]
    # Map from original cluster id to current cluster id (for union-find)
    cluster_id_map = {i: i for i in range(n_clusters)}
    # Each cluster's members: index set
    cluster_members = [set(np.where(cluster_indices == i)[0]) for i in range(n_clusters)]
    # Each cluster's current center
    cluster_centers_current = cluster_centers.copy()
    # Each cluster's status: active or merged
    active = [True] * n_clusters
    merged = [False] * n_clusters

    # Load previous merge decisions
    merge_decisions = _load_merge_decisions(save_path)
    if verbose and merge_decisions:
        print(f"Loaded {len(merge_decisions)} previous merge LLM decisions from {save_path}")

    def get_root(cid):
        """Trace and find the current representative cluster id."""
        while cluster_id_map[cid] != cid:
            cid = cluster_id_map[cid]
        return cid

    # Candidate similar pairs to merge
    if similar_pairs is None:
        similar_pairs = find_similar_cluster_pairs(cluster_centers, sim_threshold)

    for idx, (i, j, orig_sim) in enumerate(similar_pairs):
        root_i = get_root(i)
        root_j = get_root(j)
        # If already merged, skip
        if root_i == root_j:
            continue

        # If either cluster has been merged, recalculate similarity; else use initial similarity
        need_recalc = merged[root_i] or merged[root_j]
        if need_recalc:
            ci = cluster_centers_current[root_i]
            cj = cluster_centers_current[root_j]
            ci_norm = ci / (np.linalg.norm(ci) + 1e-8)
            cj_norm = cj / (np.linalg.norm(cj) + 1e-8)
            sim_to_use = float(np.dot(ci_norm, cj_norm))
        else:
            sim_to_use = orig_sim

        if sim_to_use < sim_threshold:
            if verbose:
                print(f"Skip merging {root_i} and {root_j}: sim {sim_to_use:.4f} < {sim_threshold}")
            continue

        merge_key = _make_merge_key(root_i, root_j)
        # Only use current root ids for key!
        if merge_key in merge_decisions:
            should_merge = merge_decisions[merge_key]["should_merge"]
            if verbose:
                print(f"[Cached] Merge {root_i}, {root_j}: {should_merge}")
        else:
            # Randomly select sample texts from each cluster
            idxs_i = list(cluster_members[root_i])
            idxs_j = list(cluster_members[root_j])
            rng = np.random.default_rng(seed=0)
            chosen_i = rng.choice(idxs_i, size=min(n_selected_texts, len(idxs_i)), replace=False) if len(idxs_i) > 0 else []
            rng = np.random.default_rng(seed=0)
            chosen_j = rng.choice(idxs_j, size=min(n_selected_texts, len(idxs_j)), replace=False) if len(idxs_j) > 0 else []
            texts_a = list(raw_texts[chosen_i]) if len(chosen_i) > 0 else []
            texts_b = list(raw_texts[chosen_j]) if len(chosen_j) > 0 else []

            # Ask LLM for merge decision
            should_merge = llm_should_merge(texts_a, texts_b, **LLM[llm])
            decision_record = {
                "merge_key": merge_key,
                "root_i": int(root_i),
                "root_j": int(root_j),
                "i_texts": texts_a,
                "j_texts": texts_b,
                "sim": float(sim_to_use),
                "should_merge": bool(should_merge),
            }
            _append_merge_decision(save_path, decision_record)
            merge_decisions[merge_key] = decision_record
            if verbose:
                print(f"[New] Merge {root_i}, {root_j}: {should_merge}")

        if not should_merge:
            if verbose:
                print(f"LLM decided not to merge clusters {root_i} and {root_j}.")
            continue

        # Always merge higher id into lower id for determinism
        if root_j < root_i:
            root_i, root_j = root_j, root_i

        # Update members and center
        merged_members = cluster_members[root_i] | cluster_members[root_j]
        merged_embeddings = z[list(merged_members)]
        merged_center = np.mean(merged_embeddings, axis=0, keepdims=True)[0]
        cluster_members[root_i] = merged_members
        cluster_centers_current[root_i] = merged_center
        # Mark root_j as inactive
        active[root_j] = False
        cluster_id_map[root_j] = root_i
        merged[root_i] = True
        merged[root_j] = True
        if verbose:
            print(f"Merged clusters {root_i} and {root_j} (sim={sim_to_use:.4f}), new size: {len(merged_members)}")

    # Map each sample to its final root cluster
    final_roots = {}
    final_centers = []
    cur_idx = 0
    # Assign new cluster indices from 0 to n_final_clusters-1
    for old_id, is_active in enumerate(active):
        if is_active and len(cluster_members[old_id]) > 0:
            final_roots[old_id] = cur_idx
            final_centers.append(cluster_centers_current[old_id])
            cur_idx += 1

    # For each sample, assign the new cluster index
    new_cluster_indices = np.zeros_like(cluster_indices)
    for i, orig_c in enumerate(cluster_indices):
        root_c = get_root(orig_c)
        new_cluster_indices[i] = final_roots[root_c]

    merged_cluster_centers = np.stack(final_centers, axis=0)
    if verbose:
        print(f"Final number of clusters: {merged_cluster_centers.shape[0]}")
    return new_cluster_indices, merged_cluster_centers

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    n = 40
    d = 8
    k = 6
    z = np.random.randn(n, d)
    cluster_indices = np.random.choice(k, size=n)
    cluster_centers = normalize(np.random.randn(k, d))
    raw_texts = np.array([f"Sample text {i}" for i in range(n)])
    similar_pairs = [(1, 2, 0.99), (2, 3, 0.97), (0, 4, 0.96), (4, 5, 0.95)]
    new_cluster_indices, merged_centers = merge_similar_clusters_with_llm(
        z, raw_texts, cluster_indices, cluster_centers, similar_pairs, sim_threshold=0.85, verbose=True
    )
    print("Old cluster indices:", cluster_indices)
    print("New cluster indices:", new_cluster_indices)
    print("Merged centers shape:", merged_centers.shape)