# Copyright 2026, IBM Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unsupervised GraphSAGE baseline.

Implements mean-aggregator GraphSAGE (Hamilton et al., 2017) in an
unsupervised setting using a graph-context (DeepWalk-style) loss.

Two backends are provided:
1. PyTorch  — full trainable GraphSAGE with unsupervised negative-sampling loss.
2. NumPy    — spectral mean-aggregation fallback (no training required).

Reference: Hamilton, W., Ying, R., & Leskovec, J. (2017).
           Inductive Representation Learning on Large Graphs. NeurIPS.
"""

import logging
import numpy as np
import networkx as nx
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch implementation
# ──────────────────────────────────────────────────────────────────────────────

if _TORCH:

    class _SAGELayer(nn.Module):
        """Single mean-aggregator GraphSAGE layer."""

        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            # Weight for [h_self || h_agg]
            self.W = nn.Linear(in_dim * 2, out_dim, bias=True)
            nn.init.xavier_uniform_(self.W.weight)

        def forward(self, H: "torch.Tensor", adj_sp: "torch.Tensor") -> "torch.Tensor":
            # adj_sp : (N, N) sparse or dense row-normalised adjacency
            H_agg = torch.mm(adj_sp, H)              # mean aggregation
            H_cat = torch.cat([H, H_agg], dim=1)     # self || neighbours
            H_out = F.relu(self.W(H_cat))
            return F.normalize(H_out, p=2, dim=1)    # L2 normalise per node

    class _GraphSAGETorch(nn.Module):
        """Stacked mean-aggregator GraphSAGE."""

        def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 2):
            super().__init__()
            dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
            self.layers = nn.ModuleList(
                [_SAGELayer(dims[i], dims[i + 1]) for i in range(n_layers)]
            )

        def forward(self, H: "torch.Tensor", adj_sp: "torch.Tensor") -> "torch.Tensor":
            for layer in self.layers:
                H = layer(H, adj_sp)
            return H

    def _run_graphsage_torch(
        G: nx.Graph,
        nodes: List,
        dimensions: int,
        hidden_dim: int,
        n_layers: int,
        epochs: int,
        lr: float,
        neg_samples: int,
        seed: int,
        device: str = "cpu",
    ) -> np.ndarray:
        """Train unsupervised GraphSAGE with PyTorch."""
        torch.manual_seed(seed)
        np.random.seed(seed)

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
            if device == "auto" else torch.device(device)

        N = len(nodes)
        node_idx = {n: i for i, n in enumerate(nodes)}

        # ── Build row-normalised adjacency (dense) ─────────────────────────
        A = nx.to_numpy_array(G, nodelist=nodes, dtype=np.float32)
        D_inv = np.diag(1.0 / (A.sum(axis=1) + 1e-12))
        A_hat = (D_inv @ A).astype(np.float32)
        adj_t = torch.from_numpy(A_hat).to(dev)

        # ── Initial features: normalised degree + random projection ────────
        deg = np.array([G.degree(n) for n in nodes], dtype=np.float32)
        deg_norm = (deg - deg.mean()) / (deg.std() + 1e-12)

        in_dim = max(8, min(64, N // 4))
        rng = np.random.default_rng(seed)
        R = rng.standard_normal((N, in_dim)).astype(np.float32)
        # Smooth R over 1 hop so initial features carry graph structure
        H0_raw = A_hat @ R + 0.3 * deg_norm[:, None] * R
        H0 = (H0_raw / (np.linalg.norm(H0_raw, axis=1, keepdims=True) + 1e-12))
        H_t = torch.from_numpy(H0.astype(np.float32)).to(dev)

        # ── Model & optimiser ──────────────────────────────────────────────
        model = _GraphSAGETorch(in_dim, hidden_dim, dimensions, n_layers).to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # ── Build positive pairs from edges ────────────────────────────────
        pos_pairs = [(node_idx[u], node_idx[v])
                     for u, v in G.edges()
                     if u in node_idx and v in node_idx]
        if len(pos_pairs) == 0:
            # Degenerate graph — return random embedding
            return rng.standard_normal((N, dimensions)).astype(np.float32)

        pos_u = torch.tensor([p[0] for p in pos_pairs], dtype=torch.long, device=dev)
        pos_v = torch.tensor([p[1] for p in pos_pairs], dtype=torch.long, device=dev)

        # ── Training loop with early stopping ──────────────────────────────
        model.train()
        best_loss = float('inf')
        patience_counter = 0
        patience = 15

        for epoch in range(epochs):
            optimizer.zero_grad()
            Z = model(H_t, adj_t)            # (N, dim) L2-normalised

            # Positive loss: want high similarity for edge pairs
            pos_sim = (Z[pos_u] * Z[pos_v]).sum(dim=1)
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_sim, torch.ones_like(pos_sim)
            )

            # Negative loss: random non-edge pairs (limit samples for speed)
            num_neg = min(len(pos_pairs) * neg_samples, len(pos_pairs) * 5)
            neg_u_idx = torch.randint(0, N, (num_neg,), device=dev)
            neg_v_idx = torch.randint(0, N, (num_neg,), device=dev)
            # Filter self-loops (best-effort)
            valid = neg_u_idx != neg_v_idx
            neg_u_idx, neg_v_idx = neg_u_idx[valid], neg_v_idx[valid]

            neg_sim = (Z[neg_u_idx] * Z[neg_v_idx]).sum(dim=1)
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_sim, torch.zeros_like(neg_sim)
            )

            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()

            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        model.eval()
        with torch.no_grad():
            Z_final = model(H_t, adj_t).cpu().numpy()
        return Z_final.astype(np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# NumPy fallback (spectral mean aggregation — no training)
# ──────────────────────────────────────────────────────────────────────────────

def _run_graphsage_numpy(
    G: nx.Graph,
    nodes: List,
    dimensions: int,
    n_layers: int,
    seed: int,
) -> np.ndarray:
    """
    Spectral mean-aggregation GraphSAGE fallback (no PyTorch required).

    Applies n_layers rounds of symmetric mean aggregation on random
    initial features, then compresses to *dimensions* via truncated SVD.
    This approximates the linear (no non-linearity) limit of GraphSAGE.
    """
    N = len(nodes)
    A = nx.to_numpy_array(G, nodelist=nodes, dtype=np.float64)

    # Symmetric normalisation A_hat = D^{-1/2} A D^{-1/2}
    deg = A.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    A_hat = d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]

    # Random initial features (graph-smoothed)
    rng = np.random.default_rng(seed)
    in_dim = max(16, min(128, N // 2))
    H = rng.standard_normal((N, in_dim))

    for _ in range(n_layers):
        H_agg = A_hat @ H
        H = np.concatenate([H, H_agg], axis=1)  # mean-concat (as in the paper)
        # Column-wise L2 normalise
        norms = np.linalg.norm(H, axis=0, keepdims=True)
        H = H / (norms + 1e-12)

    # Reduce to desired dimensionality with truncated SVD
    dims = min(dimensions, H.shape[1], N - 1)
    try:
        from scipy.sparse.linalg import ArpackError, svds
        U, s, _ = svds(H, k=dims)
        # svds returns ascending singular values — reverse to descending
        idx = np.argsort(s)[::-1]
        emb = U[:, idx] * s[idx]
    except (ArpackError, np.linalg.LinAlgError, ValueError) as exc:
        # Truncated SVD needs k < min(H.shape) and does not always converge.
        # The dense fallback is exact and returns the same quantity, so this is a
        # genuine alternative route rather than a degraded result -- but it is
        # O(N*d^2), so it is worth being able to see in a log.
        logger.debug("svds(k=%d) failed (%s); using dense SVD instead", dims, exc)
        U, s, _ = np.linalg.svd(H, full_matrices=False)
        emb = U[:, :dims] * s[:dims]

    return emb.astype(np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def run_graphsage(
    graph: nx.Graph,
    nodes: List,
    dimensions: int = 64,
    hidden_dim: int = 128,
    n_layers: int = 2,
    epochs: int = 50,
    lr: float = 0.01,
    neg_samples: int = 5,
    seed: int = 42,
    device: str = "cpu",
) -> np.ndarray:
    """
    Unsupervised GraphSAGE embedding.

    Uses the PyTorch backend when available; falls back to spectral mean
    aggregation otherwise.

    Parameters
    ----------
    graph : nx.Graph
    nodes : list
        Canonical node ordering; embedding rows correspond to these nodes.
    dimensions : int
        Output embedding dimensionality.
    hidden_dim : int
        Hidden layer width (PyTorch backend only).
    n_layers : int
        Number of aggregation layers.
    epochs : int
        Training epochs (PyTorch backend only).
    lr : float
        Learning rate (PyTorch backend only).
    neg_samples : int
        Number of negative samples per positive edge (PyTorch backend only).
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray  (len(nodes) × dimensions)
    """
    if _TORCH:
        try:
            logger.info("GraphSAGE: using PyTorch backend")
            return _run_graphsage_torch(
                G=graph, nodes=nodes, dimensions=dimensions,
                hidden_dim=hidden_dim, n_layers=n_layers,
                epochs=epochs, lr=lr, neg_samples=neg_samples, seed=seed,
                device=device,
            )
        except Exception as e:
            logger.warning(f"GraphSAGE PyTorch backend failed ({e}); falling back to NumPy")

    logger.info("GraphSAGE: using NumPy spectral-aggregation fallback")
    return _run_graphsage_numpy(
        G=graph, nodes=nodes, dimensions=dimensions,
        n_layers=n_layers, seed=seed,
    )
