from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.inits import reset
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import subgraph
from tqdm import tqdm

from utils import filter_kwargs


def compute_similarity_mask(y, edge_index=None):
    """
    Computes a binary node similarity mask.

    The similarity matrix is used to indicate which node pairs are likely
    to belong to the same class. The mask is built using the following rules:

    1. If both nodes are labeled and belong to the same class -> similarity = 1
    2. If two nodes are neighbors, and at least
       one of them is unlabeled -> similarity = 1
       (Labeled neighbor pairs are ignored in this step)
    3. Every node is always similar to itself (diagonal = 1)

    Args:
        y (Tensor): Label vector of shape (n,), with <0 indicating unlabeled nodes.
        edge_index (Tensor): Edge list of shape (2, E), where E is the number of edges.

    Returns:
        Tensor: A (n, n) binary similarity matrix.
    """
    # 1. Initialize mask: same class & both nodes are labeled
    y1 = y.view(-1, 1)  # shape (n, 1)
    y2 = y.view(1, -1)  # shape (1, n)
    labeled_mask = (y1 >= 0) & (y2 >= 0)
    same_class = (y1 == y2)
    binary_mask = (labeled_mask & same_class)

    if edge_index is not None:
        # 2. Incorporate node-neighbor pairs — only include if at least one node is unlabeled
        src, dst = edge_index  # shape (2, E)

        y_src = y[src]
        y_dst = y[dst]

        # Keep only edges where NOT (both nodes labeled)
        keep_edges = ~((y_src >= 0) & (y_dst >= 0))

        valid_src = src[keep_edges]
        valid_dst = dst[keep_edges]

        # Set similarity for valid edges (symmetrically)
        binary_mask[valid_src, valid_dst] = True
        binary_mask[valid_dst, valid_src] = True

    # 3. Set self-similarity to 1
    binary_mask.fill_diagonal_(True)

    return binary_mask


class SCL(torch.nn.Module):
    r"""
    The SCL (Similarity-guided Contrastive Learning) model.

    Args:
        encoder (torch.nn.Module): The encoder shared across both views.
        transform1 (torch_geometric.transforms.BaseTransform): The 1-st graph view transformation.
        transform2 (torch_geometric.transforms.BaseTransform): The 2-nd graph view transformation.
        gamma (float): Weight for the negative term (default: :obj:`1.0`).
        use_nei (bool): Whether to use node-neighbor pairs (default: :obj:`use_nei`).
    """

    def __init__(self, encoder: Module, transform1: BaseTransform, transform2: BaseTransform,
                 gamma: float = 1.0, use_nei=True):
        super(SCL, self).__init__()
        self.encoder = encoder
        self.transform1 = transform1
        self.transform2 = transform2
        self.gamma = gamma

        self.use_nei = use_nei
        self._cached_mask = None

    def reset_parameters(self):
        reset(self.encoder)

    def embed(self, *args, **kwargs) -> Tensor:
        r"""Computes node embeddings."""
        return self.encoder(*args, **filter_kwargs(self.encoder.forward, kwargs))

    def forward(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        r"""Generates embeddings from two graph augmentations."""
        data1 = self.transform1(*args, **kwargs)
        data2 = self.transform2(*args, **kwargs)
        z1 = self.encoder(**data1)
        z2 = self.encoder(**data2)
        return z1, z2

    def loss(self, z1: Tensor, z2: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        Computes the total loss and its individual components based on two-view embeddings.

        Args:
            z1 (torch.Tensor): The first node embeddings.
            z1 (torch.Tensor): The second node features.
            mask (torch.Tensor): binary pairwise similarity mask.
        """
        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)

        S = z1 @ z2.T

        loss_pos = -S[mask].mean()
        loss_neg = (S[~mask]).pow(2).mean()

        loss = loss_pos + self.gamma * loss_neg
        return loss, loss_pos, loss_neg

    def train_full(self, data: Data, optimizer: torch.optim.Optimizer, epoch: int, verbose: bool = True) -> float:
        """
        Runs one epoch of full-batch training.

        Args:
            data (torch_geometric.data.Data): The full graph data.
            optimizer (torch.optim.Optimizer): The optimizer.
            epoch (int): Current epoch.
            verbose (bool, optional): If True, prints training progress. (default: :obj:`True`)

        Returns:
            Loss value of the epoch.
        """
        self.train()
        optimizer.zero_grad()
        z1, z2 = self(data)
        if self._cached_mask is None:
            if self.use_nei:
                self._cached_mask = compute_similarity_mask(data.y, data.edge_index)
            else:
                self._cached_mask = compute_similarity_mask(data.y)

        loss, pos, neg = self.loss(z1, z2, self._cached_mask)
        loss.backward()
        optimizer.step()
        if verbose:
            print(f"Epoch: {epoch:02d} Loss: {loss:.4f} POS: {pos:.4f} NEG: {neg:.4f}")
        return float(loss)

    def train_batch(self,
                    loader: NeighborLoader,
                    optimizer: torch.optim.Optimizer,
                    epoch: int, verbose: bool = True) -> float:
        """Runs one epoch of mini-batch training.

        Args:
            loader (torch_geometric.loader.NeighborLoader): The mini-batch loader.
            optimizer (torch.optim.Optimizer): The optimizer.
            epoch (int): Current epoch.
            verbose (bool, optional): If True, prints training progress. (default: :obj:`True`)

        Returns:
            Average loss value of the epoch.
        """
        self.train()
        if loader.input_nodes is None:
            num_nodes = loader.data.num_nodes
        else:
            num_nodes = loader.input_nodes.size(0)
        if verbose:
            pbar = tqdm(total=num_nodes)
            pbar.set_description(f'Epoch {epoch:02d}')
        total_loss = total_pos = total_neg = 0
        device = next(self.parameters()).device
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            z1, z2 = self(batch)
            z1 = z1[:batch.batch_size]
            z2 = z2[:batch.batch_size]
            if self.use_nei:
                input_nodes = torch.arange(batch.batch_size, device=device)
                input_edge_index, _ = subgraph(input_nodes, batch.edge_index, edge_attr=None, relabel_nodes=False)
                mask= compute_similarity_mask(batch.y[:batch.batch_size], input_edge_index)
            else:
                mask = compute_similarity_mask(batch.y[:batch.batch_size])

            loss, pos, neg = self.loss(z1, z2, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); total_pos += pos.item(); total_neg += neg.item()
            if verbose:
                pbar.update(batch.batch_size)
        loss = total_loss / len(loader); pos = total_pos / len(loader); neg = total_neg / len(loader)
        if verbose:
            pbar.close()
            print(f"Epoch: {epoch:02d} Loss: {loss:.4f} POS: {pos:.4f} NEG: {neg:.4f}")
        return float(loss)

    @torch.no_grad()
    def infer_full(self, data: Data) -> Tensor:
        r""""Full-batch inference.

        Args:
            data (torch_geometric.data.Data): The input full graph data.

        Returns:
            Node embeddings.
        """
        self.eval()
        return self.embed(**data)

    @torch.no_grad()
    def infer_batch(self, loader: NeighborLoader, verbose: bool = True) -> Tensor:
        r"""Mini-batch inference step.

        Args:
            loader (torch_geometric.loader.NeighborLoader): Mini-batch loader.
            verbose (bool): Print progress bar or not. (default: :obj:`True`)

        Returns:
            Node embeddings.
        """
        self.eval()
        all_z = []
        device = next(self.parameters()).device
        if verbose:
            pbar = tqdm(total=loader.data.num_nodes)
            pbar.set_description("Inference stage")
        for batch in loader:
            batch = batch.to(device)
            z = self.embed(**batch)
            all_z.append(z[:batch.batch_size])
            if verbose:
                pbar.update(batch.batch_size)
        if verbose:
            pbar.close()
        return torch.cat(all_z, dim=0)
