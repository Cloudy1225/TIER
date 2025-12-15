import argparse
import os
import time

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data

from utils import get_training_config, get_logger, set_seed
from torch_geometric.loader import NeighborLoader
from model import SCL
from transform import GSSLTransform
from cluster_metrics import label_metrics
from sklearn.cluster import KMeans

from gnn import GNNEncoder
from dataloader import load_graph_dataset_for_gnn


def main():
    parser = argparse.ArgumentParser(description='Pretraining')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'citeseer', 'pubmed', 'wikics',
                                 'photo', 'computer', 'history', 'arxiv', 'products',
                                 'cornell', 'wisconsin', 'texas', 'washington'])
    # Encoder
    #  - Note that we set default "LM" as roberta, and default "LLM" as Mistral-7B
    parser.add_argument("--encoder_name", type=str, default="roberta",
                        choices=["", "shallow", "LM", "LLM", "e5-large", "SentenceBert", "MiniLM", "roberta", "Qwen-3B",
                                 "Mistral-7B", "Qwen-7B", "Llama-8B"])

    # GNN configuration
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--residual_conn", type=int, default=0)
    parser.add_argument("--jump_knowledge", type=int, default=0)
    parser.add_argument("--batch_norm", type=int, default=0)

    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--ckpt_dir', type=str, default='ckpts')
    parser.add_argument('--load_ckpt', type=int, default=1)
    parser.add_argument('--runs', type=int, default=5)

    # Train setting
    parser.add_argument("--re_split", type=int, default=0)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    conf = get_training_config(args.dataset, config_path='pretrain.conf.yaml')
    conf = dict(args.__dict__, **conf)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logger = get_logger(os.path.join(args.log_dir, f'{args.dataset}.log'))
    logger.info(str(conf))

    """Load Dataset"""
    set_seed(args.seed)

    data = load_graph_dataset_for_gnn(dataset_name=args.dataset,
                                      device='cpu',
                                      path_prefix='..',
                                      emb_model=args.encoder_name if len(args.encoder_name) else "shallow",
                                      re_split=args.re_split)

    x, edge_index, y = data.x, data.edge_index, data.y
    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask

    masked_y = y.clone()
    masked_y[~train_mask] = -1
    data = Data(x=x, edge_index=edge_index, y=masked_y)

    """Create Model"""
    set_seed(args.seed)
    mini_batch_training = (args.dataset in
                           ('photo', 'computer', 'history', 'arxiv', 'products'))
    gnn_type = 'SAGE' if mini_batch_training else 'GCN'
    transform1 = GSSLTransform(p_feat_mask=conf['p_fm1'], p_edge_drop=conf['p_ed1'], )
    transform2 = GSSLTransform(p_feat_mask=conf['p_fm2'], p_edge_drop=conf['p_ed2'], )

    encoder = GNNEncoder(input_dim=data.x.shape[1],
                         hidden_dim=conf['hidden_dim'],
                         output_dim=conf['hidden_dim'],
                         n_layers=conf['n_layers'],
                         gnn_type=gnn_type,
                         dropout=conf['dropout'],
                         use_softmax=False,
                         batch_norm=conf['batch_norm'],
                         residual_conn=conf['residual_conn'],
                         jump_knowledge=conf['jump_knowledge']).to(device)

    model = SCL(encoder=encoder, transform1=transform1, transform2=transform2, gamma=conf['gamma'],
                use_nei=args.dataset not in ('cornell', 'wisconsin', 'texas', 'washington')).to(device)

    """Training"""
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    best_ckpt_path = os.path.join(args.ckpt_dir, f"{args.dataset}_snapshot.pt")
    if args.load_ckpt and os.path.exists(best_ckpt_path):
        print("Load checkpoint...")
        # model.load_state_dict(torch.load(best_ckpt_path, weights_only=False))
    else:
        set_seed(args.seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=conf['lr'], weight_decay=conf['wd'])
        best_loss = float('inf')
        patience_counter = 0

        print("Start training...")
        start_time = time.time()
        epoch_times = []
        if mini_batch_training:
            train_data = NeighborLoader(data, input_nodes=None,
                                        num_neighbors=[conf['fan_out']] * conf['n_layers'],
                                        batch_size=conf['batch_size'], shuffle=True)
            train = model.train_batch
        else:
            train_data = data.to(device)
            train = model.train_full
        for epoch in range(1, conf['epochs'] + 1):
            t0 = time.time()
            loss = train(train_data, optimizer, epoch)
            epoch_times.append(time.time() - t0)
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= conf['patience']:
                    print(f"Early stopping at epoch {epoch}")
                    break
            torch.save(model.state_dict(), best_ckpt_path)
        total_time = time.time() - start_time
        avg_time = sum(epoch_times) / len(epoch_times) * 1000  # ms
        mem_reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 2 if torch.cuda.is_available() else 0
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

        logger.info(
            f"Training done in {total_time:.2f}s, avg per epoch: {avg_time:.2f}ms, "
            f"GPU mem peak: {mem_reserved:.1f}MB, total params: {total_params:.3f}M")

    model.load_state_dict(torch.load(best_ckpt_path, weights_only=False))

    """Inference"""
    set_seed(args.seed)
    try:
        z = model.infer_full(data.to(device))
    except RuntimeError:
        print("[Warning] Full-batch inference failed. Falling back to mini-batch inference.")
        full_loader = NeighborLoader(data, input_nodes=None,
                                     num_neighbors=[-1] * conf['n_layers'],
                                     batch_size=conf['batch_size'] * 4, shuffle=False)
        z = model.infer_batch(full_loader)

    ebd_dir = './embeddings'
    if not os.path.exists(ebd_dir):
        os.makedirs(ebd_dir)
    import joblib
    joblib.dump(z.cpu().numpy(), filename=f'{ebd_dir}/{args.dataset}.joblib')

    """Clustering"""
    z = nn.functional.normalize(z, p=2, dim=1)
    z = z.cpu().numpy()
    y = y.cpu().numpy()

    set_seed(args.seed)
    all_scores = []
    for run in range(args.runs):
        kmeans = KMeans(n_clusters=int(y.max()) + 1, random_state=args.seed + run)
        pred = kmeans.fit_predict(z)

        result = label_metrics(y, pred, metrics=('NMI', 'ARI', 'ACC', 'F1'))
        scores = [100 * r for r in result]
        all_scores.append(scores)
        logger.info(f'NMI={scores[0]:.2f}, ARI={scores[1]:.2f}, ACC={scores[2]:.2f}, F1={scores[3]:.2f}')

        try:
            kmeans = KMeans(n_clusters=int(y.max()) + 1, random_state=args.seed + run)
            pred = kmeans.fit_predict(z[train_mask])
            result = label_metrics(y[train_mask], pred, metrics=('NMI', 'ARI', 'ACC', 'F1'))
        except:
            # A Bug triggered when using the history dataset.
            kmeans = KMeans(n_clusters=int(y.max()), random_state=args.seed + run)
            pred = kmeans.fit_predict(z[train_mask])
            result = label_metrics(y[train_mask], pred, metrics=('NMI', 'ARI', 'ACC', 'F1'))
        scores = [100 * r for r in result]
        logger.info(f'=======Train: NMI={scores[0]:.2f}, ARI={scores[1]:.2f}, ACC={scores[2]:.2f}, F1={scores[3]:.2f}')

    all_scores = np.array(all_scores)
    mean = all_scores.mean(axis=0)
    std = all_scores.std(axis=0)

    print("=== Final Results ===")
    logger.info(
        f"NMI={mean[0]:.2f}+-{std[0]:.2f}, ARI={mean[1]:.2f}+-{std[1]:.2f}, "
        f"ACC={mean[2]:.2f}+-{std[2]:.2f}, F1={mean[3]:.2f}+-{std[3]:.2f}")
    logger.info('')


if __name__ == '__main__':
    main()