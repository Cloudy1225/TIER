import json
import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils import LLM

from openai import OpenAI

def llm_label_and_summary(texts, api_key, base_url="https://api.deepseek.com", model="deepseek-chat") -> dict[str, str]:
    """
    Use DeepSeek/OpenAI API to generate a label and summary for a cluster of texts.
    Given a list of sample texts from a cluster, calls the LLM to generate a label and summary.
    Returns a dict: {"label": "...", "summary": "..."}
    """
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Format documents list
    text_block = '\n'.join(f'{i+1}. {text.strip()}' for i, text in enumerate(texts))

    user_prompt = (
        "Here are some documents from the same cluster:\n\n"
        f"{text_block}\n\n"
        "Please:\n"
        "- Generate a short topic label (2–5 words)\n"
        "- Summarize the common theme in 1–2 sentences\n\n"
        "Output format (in JSON):\n"
        "{\n"
        '  "label": "[label]",\n'
        '  "summary": "[summary]"\n'
        "}\n"
    )

    messages = [
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={'type': 'json_object'}
        )
    except Exception as e:
        print(e)
        print(user_prompt)

    return json.loads(response.choices[0].message.content)


def load_existing_cluster_results(save_path):
    """
    Load existing cluster results from a JSONL file into a dict keyed by cluster index.
    """
    if not os.path.exists(save_path):
        return {}
    results = {}
    with open(save_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    cluster_id = record.get("cluster_id")
                    if cluster_id is not None:
                        results[cluster_id] = record
                except Exception as e:
                    print(f"Warning: failed to load line: {line.strip()} ({e})")
    return results


def summarize_clusters_with_llm(
        z, raw_texts, cluster_indices, cluster_centers,
        n_selected_texts=20, save_path="cluster_summaries.jsonl", verbose=True, llm='DeepSeek'):
    """
    For each cluster, select the `n_selected_texts` samples closest to the center,
    pass their texts to an LLM to get a label and summary,
    and save the results as a list of JSON objects (one per cluster).

    Args:
        z: ndarray (n_samples, n_features), sample embeddings.
        raw_texts: ndarray (n_samples,), text description for each sample.
        cluster_indices: ndarray (n_samples,), cluster index for each sample.
        cluster_centers: ndarray (n_clusters, n_features), cluster centers.
        n_selected_texts: int, number of texts to select for each cluster.
        save_path: str, where to save the resulting list of dicts as JSONL.
        verbose: bool

    Returns:
        cluster_summaries: list of dicts, each with 'label' and 'summary' for each cluster.
    """
    n_clusters = cluster_centers.shape[0]
    cluster_summaries = []
    existing = load_existing_cluster_results(save_path)
    already_done = set(existing.keys())

    if verbose and already_done:
        print(f"Loaded {len(already_done)} already processed clusters from {save_path}")

    for c in range(n_clusters):
        if c in already_done:
            if verbose:
                print(f"[Cached] Cluster {c}: {existing[c]}")
            cluster_summaries.append(existing[c])
            continue
        idxs = np.where(cluster_indices == c)[0]
        if len(idxs) == 0:
            continue

        # Compute cosine similarity for all samples in the cluster to the center
        sims = cosine_similarity(z[idxs], cluster_centers[c:c + 1]).squeeze()
        # Get indices of closest samples (highest similarity)
        top_k = min(n_selected_texts, len(idxs))
        top_idx_indices = np.argsort(-sims)[:top_k]
        chosen_idxs = idxs[top_idx_indices]
        chosen_texts = list(raw_texts[chosen_idxs])
        # Call LLM to get label and summary
        result = {'cluster_id': c}
        result.update(llm_label_and_summary(chosen_texts, **LLM[llm]))
        if verbose:
            print(f"[New] Cluster {c}: {result}")
        cluster_summaries.append(result)
        # Save result after each LLM call
        with open(save_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    return cluster_summaries


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
    results = summarize_clusters_with_llm(z, raw_texts, cluster_indices, cluster_centers,
                                          n_selected_texts=20, save_path="cluster_summaries.jsonl", verbose=True)
    print("Results:", results)
