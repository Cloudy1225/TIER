# Learning Hierarchical Knowledge in Text-Rich Networks with Taxonomy-Informed Representation Learning

This repository contains the official implementation of the paper:

> **Learning Hierarchical Knowledge in Text-Rich Networks with Taxonomy-Informed Representation Learning**  
> *Yunhui Liu, Yongchao Liu, Yinfeng Chen, Chuntao Hong, Tao Zheng, Tieke He*  
> Nanjing University & Ant Group, China  
> KDD 2026

TIER is a taxonomy-informed representation learning framework for Text-Rich Networks (TRNs). It constructs a hierarchical taxonomy in a fully automatic manner using LLM-empowered clustering, and injects it into node representation learning via a novel Cophenetic Correlation Coefficient (CCC) based regularization.



## 📁 Project Structure

```
TIER
│  embedding.py          # LM-based node embedding generation
│  dataloader.py         # Main data loading utilities
│  gnns.py               # GNN model architecture
│  train.py              # TIER model training and evaluation
│  run.sh                # Example execution script
│
├─datasets/              # Place to unzip raw datasets from HuggingFace or Google Drive
│
├─pretrain/              # Similarity-Guided Contrastive Learning (SGCL) stage
│  ├─pretrain.py         # Entry point for pretraining
│  ├─model.py            # GNN encoder and projection head
│  ├─gnn.py              # GNN backbone modules
│  ├─dataloader.py       # Pretraining-specific dataloader
│  ├─transform.py        # Graph augmentations
│  ├─utils.py            # Pretraining helpers
│  ├─cluster_metrics.py  # Evaluation of clustering quality
│  ├─pretrain.conf.yaml  # Hyperparameter config
│  └─ckpts/              # Pretrained model checkpoints
│
└─taxonomy/              # Hierarchical Taxonomy Construction
├─full_clustering_pipeline.py       # End-to-end taxonomy pipeline
├─initial_finest_clustering.py      # Initial KMeans clustering
├─split_low_cohesion_clusters.py    # LLM-based cluster splitting
├─merge_similar_clusters.py         # LLM-based cluster merging
├─redistribute_small_clusters.py    # Reassigning small or noisy clusters
├─reassign_outliers.py              # Outlier handling with LLM
├─summarize_clusters.py             # LLM-based summarization
└─outputs/                          # Saved taxonomy results (per dataset)
```

---



## 🚀 Getting Started

### 1. Prepare Datasets

Download datasets from one of the following sources and unzip to `datasets/`:

- [Google Drive](https://drive.google.com/file/d/14GmRVwhP1pUD_OIhoJU3oATZWTnklhPG/view)
- [HuggingFace](https://huggingface.co/datasets/xxwu/LLMNodeBed/tree/main)

Then, generate RoBERTa-based node embeddings:

```bash
python embedding.py --dataset=citeseer --encoder_name=roberta
````

---



### 2. Run with Precomputed Taxonomy

This uses the pre-built hierarchical taxonomy (e.g., in `taxonomy/outputs`):

```bash
python train.py --dataset citeseer --n_layers 2 --hidden_dim 128 --dropout 0.7 --residual_conn 0 --batch_norm 0 --n_clusters_list "6, 64" --lamda 2

```

---



### 3. Build Your Own Taxonomy

#### (a) Similarity-Guided Contrastive Pretraining (SGCL)

```bash
cd pretrain
python pretrain.py --dataset citeseer
# or to load a provided checkpoint:
python pretrain.py --dataset citeseer --load_ckpt 1
```

#### (b) LLM-Powered Hierarchical Clustering

```bash
cd taxonomy
python full_clustering_pipeline.py --dataset citeseer
```

This will perform:

* Initial KMeans clustering
* LLM-based refinement (splitting, merging, outlier reassignment)
* Multi-level hierarchy construction
* Natural language summarization

Results saved under `taxonomy/outputs/{dataset}/{dim}/`.

---



### 4. Taxonomy-Informed Representation Learning

Once the taxonomy is built, run the final training with CCC regularization:

```bash
python train.py --dataset citeseer \
  --n_layers 2 --hidden_dim 128 --dropout 0.7 \
  --residual_conn 0 --batch_norm 0 \
  --n_clusters_list "6, 64" --lamda 2
```

`--lamda` controls the weight of taxonomy regularization (CCC loss).

---



## ✨ Highlights

* **SGCL Pretraining**: Jointly encodes text and structure for clustering-friendly embeddings.
* **LLM-Based Refinement**: Uses LLM to split/merge clusters, reassign outliers, and generate summaries.
* **Hierarchical Regularization**: CCC loss aligns representation space with constructed taxonomy tree.
* **Efficient**: Outperforms LLM-based methods while being more efficient in memory and time.

---



## 🙏 Acknowledgements

We thank the authors of [LLMNodeBed](https://github.com/WxxShirley/LLMNodeBed) for dataset and evaluation tools.

