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
QuVINE GAT Variants: controlled downstream probes for quantum/classical diffusion features.

This module replaces the earlier GCN-MF pathway with a cleaner GAT pathway:
    feature_builder(G) -> X_variant -> same GAT encoder -> embeddings / logits

Important design choice:
    The GAT is NOT quantum. Quantum content enters only through the input
    representation, e.g. QCal-Heat/QCal-Poly features calibrated to CTQW/DTQW
    targets, or direct CTQW/DTQW embeddings computed elsewhere.

Supported input variants:
    raw_structural       : scalable structural node features
    provided             : user-provided features
    rwr                  : classical random-walk-with-restart diffusion features
    heat_fixed           : classical heat kernel features with user-provided t
    poly_fixed           : classical polynomial features with user-provided coeffs
    heat_qcal_ctqw       : heat kernel calibrated to CTQW targets
    poly_qcal_ctqw       : polynomial filter calibrated to CTQW targets
    heat_qcal_rwr        : heat kernel calibrated to RWR/classical-walk targets
    poly_qcal_rwr        : polynomial filter calibrated to RWR/classical-walk targets
    direct_ctqw          : direct CTQW embedding/features provided by user
    direct_dtqw          : direct DTQW embedding/features provided by user

Dependencies beyond the previous GCN-MF file:
    Required: PyTorch, NumPy, SciPy, NetworkX

    Optional: scikit-learn only for StandardScaler if
    normalize_structural_features=True; code falls back to internal
    standardization when sklearn is absent.

No PyTorch Geometric is required. This file implements a sparse edge-index GAT
using only PyTorch index_add/scatter_reduce operations.
"""
from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import networkx as nx

# Import PyTorch utilities for better error handling
try:
    from ..utils.torch_utils import (
        check_torch_available,
        TorchNotAvailableError,
        get_device
    )
    TORCH_UTILS_AVAILABLE = True
except ImportError:
    TORCH_UTILS_AVAILABLE = False
    
    # Fallback implementations
    def check_torch_available():
        try:
            import torch
            return True
        except ImportError:
            return False
    
    class TorchNotAvailableError(ImportError):
        pass
    
    def get_device():
        return "cpu"

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, "torch.Tensor"]


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class GATConfig:
    hidden_dim: int = 64
    output_dim: int = 128
    num_layers: int = 2
    heads: int = 4
    dropout: float = 0.5
    attention_dropout: float = 0.2
    negative_slope: float = 0.2
    residual: bool = True


@dataclass
class TrainConfig:
    epochs: int = 200
    lr: float = 0.005
    weight_decay: float = 5e-4
    patience: int = 25
    edge_batch_size: int = 8192
    val_edge_fraction: float = 0.1
    random_state: int = 42
    device: str = "cpu"
    verbose: bool = False


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def require_torch() -> None:
    """
    Check if PyTorch is available and raise informative error if not.
    
    Raises:
        TorchNotAvailableError: If PyTorch is not installed
    """
    if not TORCH_AVAILABLE:
        raise TorchNotAvailableError(
            "PyTorch is required for GAT embedding generation.\n"
            "Install PyTorch with: pip install torch\n"
            "See installation guide: https://pytorch.org/get-started/locally/\n"
            "For CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def get_nodelist(G: nx.Graph, nodelist: Optional[Sequence] = None) -> List:
    """Return the node order to use, validating that it belongs to ``G``.

    Every consumer of this ordering indexes ``G`` by these ids -- degree views,
    Laplacian construction, per-node metric dicts -- so an id that is not in the
    graph has no meaningful feature row. Rejecting it here names the offending
    ids; letting it through produced either a silently zeroed feature column or
    an inscrutable numpy shape error several frames deeper.

    Raises:
        ValueError: if ``nodelist`` contains ids absent from ``G``, or duplicates.
    """
    if nodelist is None:
        return list(G.nodes())
    nodes = list(nodelist)
    missing = [v for v in nodes if v not in G]
    if missing:
        shown = missing[:5]
        raise ValueError(
            f"nodelist contains {len(missing)} node(s) not present in the graph: "
            f"{shown}{' ...' if len(missing) > len(shown) else ''}. "
            f"The graph has {G.number_of_nodes()} nodes."
        )
    if len(set(nodes)) != len(nodes):
        dupes = sorted({v for v in nodes if nodes.count(v) > 1}, key=repr)[:5]
        raise ValueError(
            f"nodelist contains duplicate node ids, e.g. {dupes}; "
            f"the ordering must be a permutation of the nodes it covers."
        )
    return nodes


def as_numpy(X: ArrayLike) -> np.ndarray:
    if TORCH_AVAILABLE and isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy()
    return np.asarray(X)


def row_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def standardize_columns(X: np.ndarray, train_mask: Optional[np.ndarray] = None, eps: float = 1e-12) -> np.ndarray:
    """
    Standardize columns with optional train-only statistics.
    
    Args:
        X: Feature matrix [N, d]
        train_mask: Boolean mask for training nodes [N]. If None, uses all nodes.
        eps: Small constant for numerical stability
    
    Returns:
        Standardized features [N, d]
    """
    if train_mask is None:
        # Fallback to global standardization (for unsupervised tasks)
        mu = np.nanmean(X, axis=0, keepdims=True)
        sd = np.nanstd(X, axis=0, keepdims=True)
    else:
        # Use train nodes only for statistics (prevents test leakage)
        X_train = X[train_mask]
        mu = np.nanmean(X_train, axis=0, keepdims=True)
        sd = np.nanstd(X_train, axis=0, keepdims=True)
    
    Xs = (X - mu) / np.maximum(sd, eps)
    return np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)


def scipy_sparse_to_torch_coo(A: sp.spmatrix, device: str = "cpu") -> "torch.Tensor":
    require_torch()
    A = A.tocoo().astype(np.float32)
    idx = np.vstack([A.row, A.col])
    indices = torch.from_numpy(idx).long().to(device)
    values = torch.from_numpy(A.data).float().to(device)
    return torch.sparse_coo_tensor(indices, values, A.shape, device=device).coalesce()


def edge_index_from_graph(
    G: nx.Graph,
    nodelist: Sequence,
    add_self_loops: bool = True,
    device: str = "cpu",
) -> "torch.Tensor":
    """Return directed edge_index [2, E_dir], including both directions for undirected G."""
    require_torch()
    node_to_idx = {node: i for i, node in enumerate(nodelist)}
    rows: List[int] = []
    cols: List[int] = []

    for u, v in G.edges():
        if u not in node_to_idx or v not in node_to_idx:
            continue
        i, j = node_to_idx[u], node_to_idx[v]
        rows.extend([i, j])
        cols.extend([j, i])

    if add_self_loops:
        n = len(nodelist)
        rows.extend(range(n))
        cols.extend(range(n))

    edge_index = torch.tensor([rows, cols], dtype=torch.long, device=device)
    return edge_index


def build_normalized_laplacian(
    G: nx.Graph,
    nodelist: Sequence,
    normalized: bool = True,
    weight: Optional[str] = "weight",
) -> sp.csr_matrix:
    if normalized:
        return nx.normalized_laplacian_matrix(G, nodelist=nodelist, weight=weight).astype(float).tocsr()
    return nx.laplacian_matrix(G, nodelist=nodelist, weight=weight).astype(float).tocsr()


# -----------------------------------------------------------------------------
# Scalable base features
# -----------------------------------------------------------------------------

def build_structural_features(
    G: nx.Graph,
    nodelist: Optional[Sequence] = None,
    train_mask: Optional[np.ndarray] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Scalable structural node features for graphs without attributes.
    
    Args:
        G: NetworkX graph
        nodelist: Ordered list of nodes (default: G.nodes())
        train_mask: Boolean mask [N] indicating training nodes.
                   If None, uses all nodes (acceptable for transductive tasks).
        normalize: Whether to standardize features
    
    Returns:
        Feature matrix [N, 8] with columns:
        [degree, log_degree, clustering, core_number, pagerank,
         log_triangles, avg_neighbor_degree, local_degree_fraction]
    
    Note:
        For transductive node classification, train_mask can be None since
        the full graph structure is known. For inductive tasks or to prevent
        test leakage, always provide train_mask.

    Degenerate graphs:
        Any structural column that networkx cannot compute for this graph class
        (clustering and triangles on multigraphs, k-core with self-loops,
        pagerank that fails to converge) falls back to zeros -- pagerank to the
        uniform distribution -- and logs the reason at DEBUG. The feature matrix
        always has the documented width; it is never partially built.
    """
    nodelist = get_nodelist(G, nodelist)
    n = len(nodelist)
    if n == 0:
        return np.zeros((0, 8), dtype=np.float32)

    deg = dict(G.degree())
    deg_arr = np.array([deg.get(v, 0.0) for v in nodelist], dtype=np.float64)

    # Each structural column below degrades independently. networkx raises
    # NetworkXNotImplemented for metrics undefined on the given graph class
    # (multigraphs, directed graphs, self-loops) and PowerIterationFailedConvergence
    # when pagerank does not settle; those are expected inputs here, not errors, so
    # the column falls back to a documented sentinel and the reason is logged at
    # debug rather than raised or discarded.
    try:
        clustering = nx.clustering(G)
        clust_arr = np.array([clustering.get(v, 0.0) for v in nodelist], dtype=np.float64)
    except nx.NetworkXException as exc:
        logger.debug("clustering coefficients unavailable, using zeros: %s", exc)
        clust_arr = np.zeros(n, dtype=np.float64)

    try:
        core = nx.core_number(G)
        core_arr = np.array([core.get(v, 0.0) for v in nodelist], dtype=np.float64)
    except nx.NetworkXException as exc:
        logger.debug("k-core numbers unavailable, using zeros: %s", exc)
        core_arr = np.zeros(n, dtype=np.float64)

    try:
        pr = nx.pagerank(G, alpha=0.85, max_iter=200, tol=1e-6)
        pr_arr = np.array([pr.get(v, 0.0) for v in nodelist], dtype=np.float64)
    except nx.NetworkXException as exc:
        logger.debug("PageRank unavailable, using zeros: %s", exc)
        pr_arr = np.zeros(n, dtype=np.float64)

    try:
        tri = nx.triangles(G)
        tri_arr = np.array([tri.get(v, 0.0) for v in nodelist], dtype=np.float64)
    except nx.NetworkXException as exc:
        logger.debug("triangle counts unavailable, using zeros: %s", exc)
        tri_arr = np.zeros(n, dtype=np.float64)

    try:
        avg_nbr_deg = nx.average_neighbor_degree(G)
        avg_nbr_deg_arr = np.array([avg_nbr_deg.get(v, 0.0) for v in nodelist], dtype=np.float64)
    except nx.NetworkXException as exc:
        logger.debug("average neighbour degrees unavailable, using zeros: %s", exc)
        avg_nbr_deg_arr = np.zeros(n, dtype=np.float64)

    max_deg = max(float(deg_arr.max()), 1.0)
    X = np.column_stack([
        deg_arr,
        np.log1p(deg_arr),
        clust_arr,
        core_arr,
        pr_arr,
        np.log1p(tri_arr),
        avg_nbr_deg_arr,
        deg_arr / max_deg,
    ]).astype(np.float32)

    # Normalize using train statistics only (prevents test leakage)
    return standardize_columns(X, train_mask=train_mask).astype(np.float32) if normalize else X.astype(np.float32)


def align_features(
    features: ArrayLike,
    nodelist: Sequence,
    feature_nodes: Optional[Sequence] = None,
) -> np.ndarray:
    """Align feature matrix to nodelist.

    If feature_nodes is None, features are assumed already ordered as nodelist.
    """
    X = as_numpy(features).astype(np.float32)
    if feature_nodes is None:
        if X.shape[0] != len(nodelist):
            raise ValueError(f"features has {X.shape[0]} rows but graph has {len(nodelist)} nodes")
        return X

    if len(feature_nodes) != X.shape[0]:
        raise ValueError("feature_nodes length must match features rows")
    src = {node: i for i, node in enumerate(feature_nodes)}
    missing = [node for node in nodelist if node not in src]
    if missing:
        raise ValueError(f"features missing {len(missing)} graph nodes; first missing={missing[0]!r}")
    return X[[src[node] for node in nodelist]].astype(np.float32)


# -----------------------------------------------------------------------------
# Classical and quantum-calibrated feature filters
# -----------------------------------------------------------------------------

def apply_heat_filter(L: sp.spmatrix, X: np.ndarray, t: float) -> np.ndarray:
    if t < 0:
        raise ValueError("heat-kernel time t must be non-negative")
    return np.asarray(spla.expm_multiply((-float(t)) * L, X), dtype=np.float32)


def apply_polynomial_filter(L: sp.spmatrix, X: np.ndarray, coeffs: Sequence[float]) -> np.ndarray:
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if coeffs.ndim != 1:
        raise ValueError("coeffs must be a 1D array")
    Z = coeffs[0] * X
    V = X.copy()
    for k in range(1, len(coeffs)):
        V = L @ V
        Z = Z + coeffs[k] * V
    return np.asarray(Z, dtype=np.float32)


def apply_rwr_filter(
    G: nx.Graph,
    X: np.ndarray,
    nodelist: Sequence,
    alpha: float = 0.15,
    steps: int = 50,
    tol: float = 1e-6,
) -> np.ndarray:
    """Random walk with restart / PPR-style feature diffusion.

    Iteration: Z_{k+1} = alpha X + (1-alpha) P Z_k,
    where P = D^{-1} A is row-stochastic. This is classical, not quantum.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    A = nx.adjacency_matrix(G, nodelist=nodelist).astype(float).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv_deg = np.zeros_like(deg, dtype=float)
    mask = deg > 0
    inv_deg[mask] = 1.0 / deg[mask]
    P = sp.diags(inv_deg) @ A

    Z = X.copy().astype(np.float32)
    X0 = X.astype(np.float32)
    for _ in range(int(steps)):
        Z_next = alpha * X0 + (1.0 - alpha) * (P @ Z)
        diff = np.linalg.norm(Z_next - Z) / max(np.linalg.norm(Z), 1e-12)
        Z = np.asarray(Z_next, dtype=np.float32)
        if diff < tol:
            break
    return Z


def _validate_q_target(item: Mapping, node_to_idx: Mapping) -> Tuple[List, int, np.ndarray]:
    for key in ("nodes", "center", "pQ"):
        if key not in item:
            raise KeyError(f"q_target missing key {key!r}")
    nodes = list(item["nodes"])
    center = item["center"]
    p = np.asarray(item["pQ"], dtype=np.float64)
    if len(nodes) != len(p):
        raise ValueError("q_target length mismatch: len(nodes) != len(pQ)")
    if center not in node_to_idx:
        raise ValueError("q_target center not in graph node_to_idx")
    if any(node not in node_to_idx for node in nodes):
        raise ValueError("q_target contains nodes not present in graph node_to_idx")
    p = np.maximum(p, 0.0)
    s = p.sum()
    if s <= 0:
        raise ValueError("q_target pQ has zero mass after clipping negatives")
    return nodes, center, p / s


def calibrate_heat_kernel(
    L: sp.spmatrix,
    targets: Sequence[Mapping],
    node_to_idx: Mapping,
    t_grid: Optional[np.ndarray] = None,
    loss: str = "l2",
) -> Tuple[float, float]:
    """Fit heat time to target distributions.

    targets can be CTQW targets, DTQW targets, or classical RWR targets. The
    routine is agnostic: it simply matches target probabilities on sampled node
    sets.
    """
    if t_grid is None:
        t_grid = np.logspace(-2, 2, 40)
    if len(targets) == 0:
        raise ValueError("targets is empty; cannot calibrate heat kernel")

    n = L.shape[0]
    best_loss = np.inf
    best_t: Optional[float] = None

    for t in t_grid:
        total = 0.0
        for item in targets:
            nodes, center, p = _validate_q_target(item, node_to_idx)
            x = np.zeros(n, dtype=np.float64)
            x[node_to_idx[center]] = 1.0
            y = spla.expm_multiply((-float(t)) * L, x)
            idx = [node_to_idx[v] for v in nodes]
            q = np.maximum(np.asarray(y)[idx], 0.0)
            qsum = q.sum()
            if qsum <= 0:
                continue
            q = q / qsum
            if loss == "l2":
                total += float(np.sum((q - p) ** 2))
            elif loss == "kl":
                eps = 1e-12
                total += float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))
            else:
                raise ValueError("loss must be 'l2' or 'kl'")
        if total < best_loss:
            best_loss = total
            best_t = float(t)

    if best_t is None:
        raise RuntimeError("failed to calibrate heat kernel")
    return best_loss, best_t


def calibrate_polynomial_filter(
    L: sp.spmatrix,
    targets: Sequence[Mapping],
    node_to_idx: Mapping,
    K: int = 4,
    ridge: float = 1e-5,
) -> np.ndarray:
    """Fit monomial polynomial coefficients to target distributions.

    Critical bug fix relative to the earlier code: we do NOT column-normalize
    the polynomial basis during fitting unless that same normalization is also
    used at application time. Ridge handles conditioning.
    """
    if K < 0:
        raise ValueError("K must be non-negative")
    if len(targets) == 0:
        raise ValueError("targets is empty; cannot calibrate polynomial filter")

    n = L.shape[0]
    AtA = np.zeros((K + 1, K + 1), dtype=np.float64)
    Atb = np.zeros(K + 1, dtype=np.float64)

    for item in targets:
        nodes, center, p = _validate_q_target(item, node_to_idx)
        idx = [node_to_idx[v] for v in nodes]
        x = np.zeros(n, dtype=np.float64)
        x[node_to_idx[center]] = 1.0

        basis = []
        v = x.copy()
        basis.append(np.asarray(v)[idx])
        for _ in range(1, K + 1):
            v = L @ v
            basis.append(np.asarray(v)[idx])

        Phi = np.stack(basis, axis=1)  # no column normalization
        AtA += Phi.T @ Phi
        Atb += Phi.T @ p

    AtA += float(ridge) * np.eye(K + 1)
    try:
        coeffs = np.linalg.solve(AtA, Atb)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(AtA, Atb, rcond=None)[0]
    return np.asarray(coeffs, dtype=np.float64)


def heat_taylor_coeffs(t: float, K: int) -> np.ndarray:
    """Classical fixed polynomial baseline approximating exp(-tL)."""
    return np.asarray([((-t) ** k) / math.factorial(k) for k in range(K + 1)], dtype=np.float64)


# -----------------------------------------------------------------------------
# Direct CTQW / DTQW feature adapters
# -----------------------------------------------------------------------------

def get_direct_walk_features(
    direct_features: Union[np.ndarray, Mapping[str, np.ndarray]],
    key: str,
    nodelist: Sequence,
    feature_nodes: Optional[Sequence] = None,
) -> np.ndarray:
    """Fetch direct CTQW/DTQW features computed by the SGNS/walk pipeline.

    This function intentionally does not recompute CTQW/DTQW. Your attached
    pipeline trains SGNS embeddings from corpora produced by BaseWalker and then
    stores embeddings in an EmbeddingStore; pass the resulting array here.
    """
    if isinstance(direct_features, Mapping):
        if key not in direct_features:
            raise KeyError(f"direct_features mapping does not contain key {key!r}")
        X = direct_features[key]
    else:
        X = direct_features
    return align_features(X, nodelist=nodelist, feature_nodes=feature_nodes)


# -----------------------------------------------------------------------------
# GAT implementation without PyTorch Geometric
# -----------------------------------------------------------------------------

class EdgeIndexGATLayer(nn.Module):
    """Sparse edge-index GAT layer using only PyTorch.

    This layer computes attention over incoming neighbors for each destination
    node. edge_index[0] = source, edge_index[1] = destination.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.5,
        attention_dropout: float = 0.2,
        negative_slope: float = 0.2,
        concat: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.negative_slope = negative_slope
        self.residual = residual

        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))

        if residual:
            target_dim = heads * out_dim if concat else out_dim
            self.res_proj = nn.Linear(in_dim, target_dim, bias=False) if in_dim != target_dim else nn.Identity()
        else:
            self.res_proj = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)
        if isinstance(self.res_proj, nn.Linear):
            nn.init.xavier_uniform_(self.res_proj.weight)

    def forward(self, x: "torch.Tensor", edge_index: "torch.Tensor") -> "torch.Tensor":
        n = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        h = self.lin(x).view(n, self.heads, self.out_dim)
        h_src = h[src]
        h_dst = h[dst]

        e = (h_src * self.att_src).sum(dim=-1) + (h_dst * self.att_dst).sum(dim=-1)
        e = F.leaky_relu(e, negative_slope=self.negative_slope)

        # Segment softmax over incoming edges per destination node.
        E, H = e.shape
        max_per_dst = torch.full((n, H), -torch.inf, device=x.device, dtype=e.dtype)
        max_per_dst.scatter_reduce_(0, dst[:, None].expand(E, H), e, reduce="amax", include_self=True)
        exp_e = torch.exp(e - max_per_dst[dst])
        denom = torch.zeros((n, H), device=x.device, dtype=e.dtype)
        denom.index_add_(0, dst, exp_e)
        alpha = exp_e / (denom[dst] + 1e-16)
        alpha = F.dropout(alpha, p=self.attention_dropout, training=self.training)

        msg = h_src * alpha.unsqueeze(-1)
        out = torch.zeros((n, self.heads, self.out_dim), device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, msg)

        if self.concat:
            out = out.reshape(n, self.heads * self.out_dim)
        else:
            out = out.mean(dim=1)

        out = out + self.bias
        if self.res_proj is not None:
            out = out + self.res_proj(x)
        return out


class GATEncoder(nn.Module):
    """GAT encoder returning node embeddings."""
    def __init__(self, input_dim: int, config: GATConfig):
        super().__init__()
        if config.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.config = config
        self.layers = nn.ModuleList()

        if config.num_layers == 1:
            self.layers.append(
                EdgeIndexGATLayer(
                    input_dim,
                    config.output_dim,
                    heads=1,
                    concat=False,
                    dropout=config.dropout,
                    attention_dropout=config.attention_dropout,
                    negative_slope=config.negative_slope,
                    residual=config.residual,
                )
            )
        else:
            head_dim = max(1, config.hidden_dim // config.heads)
            self.layers.append(
                EdgeIndexGATLayer(
                    input_dim,
                    head_dim,
                    heads=config.heads,
                    concat=True,
                    dropout=config.dropout,
                    attention_dropout=config.attention_dropout,
                    negative_slope=config.negative_slope,
                    residual=config.residual,
                )
            )
            hidden_actual = head_dim * config.heads
            for _ in range(config.num_layers - 2):
                self.layers.append(
                    EdgeIndexGATLayer(
                        hidden_actual,
                        head_dim,
                        heads=config.heads,
                        concat=True,
                        dropout=config.dropout,
                        attention_dropout=config.attention_dropout,
                        negative_slope=config.negative_slope,
                        residual=config.residual,
                    )
                )
            self.layers.append(
                EdgeIndexGATLayer(
                    hidden_actual,
                    config.output_dim,
                    heads=1,
                    concat=False,
                    dropout=config.dropout,
                    attention_dropout=config.attention_dropout,
                    negative_slope=config.negative_slope,
                    residual=config.residual,
                )
            )

    def forward(self, x: "torch.Tensor", edge_index: "torch.Tensor") -> "torch.Tensor":
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index)
            if i < len(self.layers) - 1:
                h = F.elu(h)
                h = F.dropout(h, p=self.config.dropout, training=self.training)
        return h


class GATNodeClassifier(nn.Module):
    """GAT encoder plus linear classifier."""
    def __init__(self, input_dim: int, num_classes: int, config: GATConfig):
        super().__init__()
        self.encoder = GATEncoder(input_dim, config)
        self.classifier = nn.Linear(config.output_dim, num_classes)

    def forward(self, x: "torch.Tensor", edge_index: "torch.Tensor") -> "torch.Tensor":
        z = self.encoder(x, edge_index)
        return self.classifier(z)

    def get_embeddings(self, x: "torch.Tensor", edge_index: "torch.Tensor") -> "torch.Tensor":
        return self.encoder(x, edge_index)


# -----------------------------------------------------------------------------
# Link-reconstruction training for unsupervised embeddings
# -----------------------------------------------------------------------------

def _indexed_edges(G: nx.Graph, nodelist: Sequence) -> List[Tuple[int, int]]:
    node_to_idx = {node: i for i, node in enumerate(nodelist)}
    edges = []
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            i, j = node_to_idx[u], node_to_idx[v]
            if i != j:
                edges.append((min(i, j), max(i, j)))
    return sorted(set(edges))


def sample_negative_edges(
    n_nodes: int,
    existing_edges: set,
    num_samples: int,
    rng: np.random.Generator,
    max_attempt_factor: int = 50,
) -> List[Tuple[int, int]]:
    neg = set()
    max_attempts = max(num_samples * max_attempt_factor, 1000)
    attempts = 0
    while len(neg) < num_samples and attempts < max_attempts:
        i = int(rng.integers(0, n_nodes))
        j = int(rng.integers(0, n_nodes))
        attempts += 1
        if i == j:
            continue
        e = (min(i, j), max(i, j))
        if e in existing_edges or e in neg:
            continue
        neg.add(e)
    if len(neg) < num_samples:
        logger.warning("sampled only %d/%d negative edges", len(neg), num_samples)
    return list(neg)


def split_edges(
    edges: Sequence[Tuple[int, int]],
    val_fraction: float,
    seed: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    edges = list(edges)
    if len(edges) == 0:
        raise ValueError("graph must contain at least one edge")
    perm = rng.permutation(len(edges))
    n_val = int(round(len(edges) * val_fraction))
    n_val = min(max(n_val, 1 if len(edges) > 10 else 0), len(edges) - 1)
    val_idx = set(perm[:n_val])
    train_edges = [e for i, e in enumerate(edges) if i not in val_idx]
    val_edges = [e for i, e in enumerate(edges) if i in val_idx]
    return train_edges, val_edges


def edge_scores_dot(z: "torch.Tensor", edge_index_2xn: "torch.Tensor") -> "torch.Tensor":
    return (z[edge_index_2xn[0]] * z[edge_index_2xn[1]]).sum(dim=1)


def make_edge_batch(
    pos_edges: Sequence[Tuple[int, int]],
    existing_edges: set,
    n_nodes: int,
    batch_size: int,
    rng: np.random.Generator,
    device: str,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    n_pos_avail = len(pos_edges)
    n_pos = min(max(1, batch_size // 2), n_pos_avail)
    pos_idx = rng.choice(n_pos_avail, size=n_pos, replace=n_pos > n_pos_avail)
    pos = [pos_edges[int(i)] for i in pos_idx]
    neg = sample_negative_edges(n_nodes, existing_edges, n_pos, rng)
    edges = pos + neg
    labels = [1.0] * len(pos) + [0.0] * len(neg)
    if len(edges) == 0:
        raise RuntimeError("empty edge batch")
    edge_t = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    label_t = torch.tensor(labels, dtype=torch.float32, device=device)
    return edge_t, label_t


def train_gat_link_reconstruction(
    G: nx.Graph,
    X: np.ndarray,
    nodelist: Optional[Sequence] = None,
    gat_config: Optional[GATConfig] = None,
    train_config: Optional[TrainConfig] = None,
    train_graph_for_message_passing: Optional[nx.Graph] = None,
) -> Tuple[np.ndarray, Dict]:
    """Train a GAT encoder with edge reconstruction and return embeddings."""
    require_torch()
    gat_config = gat_config or GATConfig(output_dim=X.shape[1])
    train_config = train_config or TrainConfig()
    set_seed(train_config.random_state)

    device = train_config.device
    nodelist = get_nodelist(G, nodelist)
    n = len(nodelist)
    X = align_features(X, nodelist=nodelist)
    x_t = torch.from_numpy(X.astype(np.float32)).to(device)

    G_msg = train_graph_for_message_passing if train_graph_for_message_passing is not None else G
    edge_index = edge_index_from_graph(G_msg, nodelist=nodelist, add_self_loops=True, device=device)

    all_edges = _indexed_edges(G, nodelist)
    train_edges, val_edges = split_edges(all_edges, train_config.val_edge_fraction, train_config.random_state)
    existing = set(all_edges)

    model = GATEncoder(input_dim=X.shape[1], config=gat_config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)

    rng = np.random.default_rng(train_config.random_state)
    best_state = None
    best_val = float("inf")
    patience_left = train_config.patience
    history = {"train_loss": [], "val_loss": [], "best_epoch": 0}

    for epoch in range(train_config.epochs):
        model.train()
        opt.zero_grad()
        z = model(x_t, edge_index)
        batch_edges, batch_labels = make_edge_batch(
            train_edges, existing, n, train_config.edge_batch_size, rng, device
        )
        scores = edge_scores_dot(z, batch_edges)
        loss = F.binary_cross_entropy_with_logits(scores, batch_labels)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            z_val = model(x_t, edge_index)
            if len(val_edges) > 0:
                val_neg = sample_negative_edges(n, existing, len(val_edges), rng)
                val_batch = torch.tensor(val_edges + val_neg, dtype=torch.long, device=device).t().contiguous()
                val_labels = torch.tensor([1.0] * len(val_edges) + [0.0] * len(val_neg), dtype=torch.float32, device=device)
                val_scores = edge_scores_dot(z_val, val_batch)
                val_loss = F.binary_cross_entropy_with_logits(val_scores, val_labels).item()
            else:
                val_loss = loss.item()

        history["train_loss"].append(float(loss.item()))
        history["val_loss"].append(float(val_loss))

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch
            patience_left = train_config.patience
        else:
            patience_left -= 1

        if train_config.verbose and (epoch + 1) % 25 == 0:
            logger.info("GAT epoch %d/%d train=%.4f val=%.4f", epoch + 1, train_config.epochs, loss.item(), val_loss)
        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        z = model(x_t, edge_index).detach().cpu().numpy().astype(np.float32)

    meta = {
        "training_mode": "link_reconstruction",
        "best_val_loss": best_val,
        "best_epoch": history["best_epoch"],
        "history": history,
        "embedding_shape": z.shape,
    }
    return z, meta


# -----------------------------------------------------------------------------
# Node classification training
# -----------------------------------------------------------------------------

def train_gat_node_classifier(
    G: nx.Graph,
    X: np.ndarray,
    y: ArrayLike,
    train_mask: ArrayLike,
    val_mask: ArrayLike,
    nodelist: Optional[Sequence] = None,
    gat_config: Optional[GATConfig] = None,
    train_config: Optional[TrainConfig] = None,
    test_mask: Optional[ArrayLike] = None,
) -> Tuple[np.ndarray, Dict]:
    """Train GAT directly for node classification and return embeddings plus metadata."""
    require_torch()
    gat_config = gat_config or GATConfig(output_dim=64)
    train_config = train_config or TrainConfig()
    set_seed(train_config.random_state)

    device = train_config.device
    nodelist = get_nodelist(G, nodelist)
    X = align_features(X, nodelist=nodelist)
    x_t = torch.from_numpy(X.astype(np.float32)).to(device)
    y_t = torch.as_tensor(as_numpy(y), dtype=torch.long, device=device)
    train_mask_t = torch.as_tensor(as_numpy(train_mask), dtype=torch.bool, device=device)
    val_mask_t = torch.as_tensor(as_numpy(val_mask), dtype=torch.bool, device=device)
    test_mask_t = torch.as_tensor(as_numpy(test_mask), dtype=torch.bool, device=device) if test_mask is not None else None

    if y_t.shape[0] != X.shape[0]:
        raise ValueError("y length must match number of nodes")
    num_classes = int(y_t.max().item()) + 1
    edge_index = edge_index_from_graph(G, nodelist=nodelist, add_self_loops=True, device=device)

    model = GATNodeClassifier(input_dim=X.shape[1], num_classes=num_classes, config=gat_config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)

    best_state = None
    best_val_acc = -1.0
    patience_left = train_config.patience
    history = {"train_loss": [], "val_acc": [], "best_epoch": 0}

    for epoch in range(train_config.epochs):
        model.train()
        opt.zero_grad()
        logits = model(x_t, edge_index)
        loss = F.cross_entropy(logits[train_mask_t], y_t[train_mask_t])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(x_t, edge_index)
            pred = logits.argmax(dim=1)
            val_acc = (pred[val_mask_t] == y_t[val_mask_t]).float().mean().item()

        history["train_loss"].append(float(loss.item()))
        history["val_acc"].append(float(val_acc))

        if val_acc > best_val_acc + 1e-8:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch
            patience_left = train_config.patience
        else:
            patience_left -= 1
        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        embeddings = model.get_embeddings(x_t, edge_index).detach().cpu().numpy().astype(np.float32)
        logits = model(x_t, edge_index)
        pred = logits.argmax(dim=1)
        meta = {"best_val_acc": best_val_acc, "best_epoch": history["best_epoch"], "history": history}
        if test_mask_t is not None:
            meta["test_acc"] = float((pred[test_mask_t] == y_t[test_mask_t]).float().mean().item())
    return embeddings, meta


# -----------------------------------------------------------------------------
# Variant feature factory
# -----------------------------------------------------------------------------

def build_gat_input_features(
    G: nx.Graph,
    variant: str,
    nodelist: Optional[Sequence] = None,
    train_mask: Optional[np.ndarray] = None,
    base_features: Optional[ArrayLike] = None,
    base_feature_nodes: Optional[Sequence] = None,
    direct_features: Optional[Union[np.ndarray, Mapping[str, np.ndarray]]] = None,
    direct_feature_nodes: Optional[Sequence] = None,
    ctqw_targets: Optional[Sequence[Mapping]] = None,
    dtqw_targets: Optional[Sequence[Mapping]] = None,
    rwr_targets: Optional[Sequence[Mapping]] = None,
    heat_t: Optional[float] = None,
    poly_coeffs: Optional[Sequence[float]] = None,
    poly_K: int = 4,
    poly_ridge: float = 1e-5,
    rwr_alpha: float = 0.15,
    rwr_steps: int = 50,
    normalize_laplacian: bool = True,
    t_grid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict]:
    """Construct input features for a GAT variant with optional train/test separation.
    
    Args:
        G: NetworkX graph
        variant: Feature variant name
        nodelist: Ordered list of nodes
        train_mask: Boolean mask [N] for training nodes. If None, uses all nodes
                   (acceptable for transductive tasks). Prevents test leakage.
        base_features: Pre-computed base features (if provided, assumed pre-normalized)
        ... (other args as before)
    
    Returns:
        (X, meta): Feature matrix and metadata dict
    """
    nodelist = get_nodelist(G, nodelist)
    node_to_idx = {node: i for i, node in enumerate(nodelist)}

    if base_features is None:
        # Use train_mask to prevent test leakage in normalization
        X0 = build_structural_features(G, nodelist=nodelist, train_mask=train_mask, normalize=True)
        base_kind = "structural"
    else:
        X0 = align_features(base_features, nodelist=nodelist, feature_nodes=base_feature_nodes).astype(np.float32)
        base_kind = "provided"

    meta: Dict = {"variant": variant, "base_features": base_kind, "n_nodes": len(nodelist), "input_dim_raw": X0.shape[1]}

    variant = variant.lower()
    L = None
    if variant not in {"raw", "raw_structural", "provided", "direct_ctqw", "direct_dtqw"}:
        L = build_normalized_laplacian(G, nodelist=nodelist, normalized=normalize_laplacian)
        meta["laplacian_normalized"] = normalize_laplacian

    if variant in {"raw", "raw_structural", "provided"}:
        X = X0

    elif variant == "rwr":
        X = apply_rwr_filter(G, X0, nodelist=nodelist, alpha=rwr_alpha, steps=rwr_steps)
        meta.update({"rwr_alpha": rwr_alpha, "rwr_steps": rwr_steps})

    elif variant == "heat_fixed":
        if heat_t is None:
            raise ValueError("heat_fixed requires heat_t")
        X = apply_heat_filter(L, X0, heat_t)
        meta["heat_t"] = float(heat_t)

    elif variant == "poly_fixed":
        if poly_coeffs is None:
            poly_coeffs = heat_taylor_coeffs(t=1.0 if heat_t is None else float(heat_t), K=poly_K)
            meta["poly_coeff_source"] = "heat_taylor"
        X = apply_polynomial_filter(L, X0, poly_coeffs)
        meta["poly_coeffs"] = np.asarray(poly_coeffs).tolist()

    elif variant == "heat_qcal_ctqw":
        if ctqw_targets is None:
            raise ValueError("heat_qcal_ctqw requires ctqw_targets")
        loss, t_star = calibrate_heat_kernel(L, ctqw_targets, node_to_idx, t_grid=t_grid)
        X = apply_heat_filter(L, X0, t_star)
        meta.update({"target": "ctqw", "heat_t": t_star, "calibration_loss": loss})

    elif variant == "poly_qcal_ctqw":
        if ctqw_targets is None:
            raise ValueError("poly_qcal_ctqw requires ctqw_targets")
        coeffs = calibrate_polynomial_filter(L, ctqw_targets, node_to_idx, K=poly_K, ridge=poly_ridge)
        X = apply_polynomial_filter(L, X0, coeffs)
        meta.update({"target": "ctqw", "poly_coeffs": coeffs.tolist(), "poly_K": poly_K})

    elif variant == "heat_qcal_dtqw":
        if dtqw_targets is None:
            raise ValueError("heat_qcal_dtqw requires dtqw_targets")
        loss, t_star = calibrate_heat_kernel(L, dtqw_targets, node_to_idx, t_grid=t_grid)
        X = apply_heat_filter(L, X0, t_star)
        meta.update({"target": "dtqw", "heat_t": t_star, "calibration_loss": loss})

    elif variant == "poly_qcal_dtqw":
        if dtqw_targets is None:
            raise ValueError("poly_qcal_dtqw requires dtqw_targets")
        coeffs = calibrate_polynomial_filter(L, dtqw_targets, node_to_idx, K=poly_K, ridge=poly_ridge)
        X = apply_polynomial_filter(L, X0, coeffs)
        meta.update({"target": "dtqw", "poly_coeffs": coeffs.tolist(), "poly_K": poly_K})

    elif variant == "heat_qcal_rwr":
        if rwr_targets is None:
            raise ValueError("heat_qcal_rwr requires rwr_targets")
        loss, t_star = calibrate_heat_kernel(L, rwr_targets, node_to_idx, t_grid=t_grid)
        X = apply_heat_filter(L, X0, t_star)
        meta.update({"target": "rwr", "heat_t": t_star, "calibration_loss": loss})

    elif variant == "poly_qcal_rwr":
        if rwr_targets is None:
            raise ValueError("poly_qcal_rwr requires rwr_targets")
        coeffs = calibrate_polynomial_filter(L, rwr_targets, node_to_idx, K=poly_K, ridge=poly_ridge)
        X = apply_polynomial_filter(L, X0, coeffs)
        meta.update({"target": "rwr", "poly_coeffs": coeffs.tolist(), "poly_K": poly_K})

    elif variant == "direct_ctqw":
        if direct_features is None:
            raise ValueError("direct_ctqw requires direct_features")
        X = get_direct_walk_features(direct_features, key="ctqw", nodelist=nodelist, feature_nodes=direct_feature_nodes)
        meta["direct_key"] = "ctqw"

    elif variant == "direct_dtqw":
        if direct_features is None:
            raise ValueError("direct_dtqw requires direct_features")
        X = get_direct_walk_features(direct_features, key="dtqw", nodelist=nodelist, feature_nodes=direct_feature_nodes)
        meta["direct_key"] = "dtqw"

    else:
        raise ValueError(
            "Unknown variant. Expected one of: raw, rwr, heat_fixed, poly_fixed, "
            "heat_qcal_ctqw, poly_qcal_ctqw, heat_qcal_dtqw, poly_qcal_dtqw, "
            "heat_qcal_rwr, poly_qcal_rwr, direct_ctqw, direct_dtqw."
        )

    X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    meta["input_dim"] = int(X.shape[1])
    return X, meta


# -----------------------------------------------------------------------------
# High-level wrappers
# -----------------------------------------------------------------------------

def generate_gat_embedding(
    G: nx.Graph,
    variant: str = "raw",
    nodelist: Optional[Sequence] = None,
    base_features: Optional[ArrayLike] = None,
    base_feature_nodes: Optional[Sequence] = None,
    direct_features: Optional[Union[np.ndarray, Mapping[str, np.ndarray]]] = None,
    direct_feature_nodes: Optional[Sequence] = None,
    ctqw_targets: Optional[Sequence[Mapping]] = None,
    dtqw_targets: Optional[Sequence[Mapping]] = None,
    rwr_targets: Optional[Sequence[Mapping]] = None,
    heat_t: Optional[float] = None,
    poly_coeffs: Optional[Sequence[float]] = None,
    poly_K: int = 4,
    poly_ridge: float = 1e-5,
    rwr_alpha: float = 0.15,
    rwr_steps: int = 50,
    normalize_laplacian: bool = True,
    t_grid: Optional[np.ndarray] = None,
    gat_config: Optional[GATConfig] = None,
    train_config: Optional[TrainConfig] = None,
    train_graph_for_message_passing: Optional[nx.Graph] = None,
) -> Tuple[np.ndarray, Dict]:
    """Generate unsupervised GAT embeddings for one feature variant."""
    nodelist = get_nodelist(G, nodelist)
    X, feature_meta = build_gat_input_features(
        G=G,
        variant=variant,
        nodelist=nodelist,
        base_features=base_features,
        base_feature_nodes=base_feature_nodes,
        direct_features=direct_features,
        direct_feature_nodes=direct_feature_nodes,
        ctqw_targets=ctqw_targets,
        dtqw_targets=dtqw_targets,
        rwr_targets=rwr_targets,
        heat_t=heat_t,
        poly_coeffs=poly_coeffs,
        poly_K=poly_K,
        poly_ridge=poly_ridge,
        rwr_alpha=rwr_alpha,
        rwr_steps=rwr_steps,
        normalize_laplacian=normalize_laplacian,
        t_grid=t_grid,
    )
    if gat_config is None:
        gat_config = GATConfig(output_dim=128)
    Z, train_meta = train_gat_link_reconstruction(
        G=G,
        X=X,
        nodelist=nodelist,
        gat_config=gat_config,
        train_config=train_config or TrainConfig(),
        train_graph_for_message_passing=train_graph_for_message_passing,
    )
    meta = {**feature_meta, **train_meta, "model": "GAT"}
    return Z, meta


def generate_multiple_gat_embeddings(
    G: nx.Graph,
    variants: Sequence[str],
    **kwargs,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
    """Generate embeddings for several variants using identical kwargs/configs."""
    embeddings: Dict[str, np.ndarray] = {}
    metadata: Dict[str, Dict] = {}
    for variant in variants:
        Z, meta = generate_gat_embedding(G, variant=variant, **kwargs)
        embeddings[variant] = Z
        metadata[variant] = meta

    return embeddings, metadata


# ============================================================================
# Standardized Method Name Mapping (Phase 1.2)
# ============================================================================

def generate_gat_embedding_by_method_name(
    G: nx.Graph,
    method_name: str,
    embedding_dim: int = 128,
    nodelist: Optional[Sequence] = None,
    base_features: Optional[ArrayLike] = None,
    ctqw_targets: Optional[Sequence[Mapping]] = None,
    dtqw_targets: Optional[Sequence[Mapping]] = None,
    rwr_targets: Optional[Sequence[Mapping]] = None,
    heat_t: Optional[float] = None,
    poly_K: int = 4,
    rwr_alpha: float = 0.15,
    gat_config: Optional[GATConfig] = None,
    train_config: Optional[TrainConfig] = None,
    **kwargs
) -> np.ndarray:
    """
    Generate GAT embedding using standardized method names.
    
    This function maps the 12 standardized GAT method names to the existing
    variant system in generate_gat_embedding.
    
    Supported methods:
        - gat_baseline: Raw structural features
        - gat_heat: Heat kernel filter only
        - gat_poly: Polynomial filter only
        - gat_rwr: RWR walk only
        - gat_ctqw: CTQW walk only (direct features)
        - gat_dtqw: DTQW walk only (direct features)
        - gat_rwr_heat: RWR + heat filter
        - gat_rwr_poly: RWR + polynomial filter
        - gat_ctqw_heat: CTQW + heat filter (quantum calibrated)
        - gat_ctqw_poly: CTQW + polynomial filter (quantum calibrated)
        - gat_dtqw_heat: DTQW + heat filter (quantum calibrated)
        - gat_dtqw_poly: DTQW + polynomial filter (quantum calibrated)
    
    Args:
        G: NetworkX graph
        method_name: Standardized method name (e.g., 'gat_baseline', 'gat_ctqw_heat')
        embedding_dim: Output embedding dimension
        nodelist: Ordered list of nodes
        base_features: Pre-computed base features
        ctqw_targets: CTQW calibration targets
        dtqw_targets: DTQW calibration targets
        rwr_targets: RWR calibration targets
        heat_t: Heat kernel time parameter (for fixed variants)
        poly_K: Polynomial degree
        rwr_alpha: RWR restart probability
        gat_config: GAT model configuration
        train_config: Training configuration
        **kwargs: Additional arguments
    
    Returns:
        Node embeddings [N, embedding_dim]
    
    Example:
        >>> G = nx.karate_club_graph()
        >>> emb = generate_gat_embedding_by_method_name(G, 'gat_baseline', embedding_dim=64)
        >>> print(emb.shape)  # (34, 64)
    
    Raises:
        ValueError: If method_name is not recognized
    """
    # Map standardized names to internal variant names
    method_to_variant = {
        'gat_baseline': 'raw',
        'gat_heat': 'heat_fixed',
        'gat_poly': 'poly_fixed',
        'gat_rwr': 'rwr',
        'gat_ctqw': 'direct_ctqw',
        'gat_dtqw': 'direct_dtqw',
        'gat_rwr_heat': 'heat_qcal_rwr',
        'gat_rwr_poly': 'poly_qcal_rwr',
        'gat_ctqw_heat': 'heat_qcal_ctqw',
        'gat_ctqw_poly': 'poly_qcal_ctqw',
        'gat_dtqw_heat': 'heat_qcal_dtqw',
        'gat_dtqw_poly': 'poly_qcal_dtqw',
    }
    
    if method_name not in method_to_variant:
        raise ValueError(
            f"Unknown GAT method: {method_name}. "
            f"Expected one of: {list(method_to_variant.keys())}"
        )
    
    variant = method_to_variant[method_name]
    
    # Set up GAT config with desired output dimension
    if gat_config is None:
        gat_config = GATConfig(output_dim=embedding_dim)
    else:
        # Override output_dim if provided
        gat_config.output_dim = embedding_dim
    
    # Call the existing generate_gat_embedding function
    Z, meta = generate_gat_embedding(
        G=G,
        variant=variant,
        nodelist=nodelist,
        base_features=base_features,
        ctqw_targets=ctqw_targets,
        dtqw_targets=dtqw_targets,
        rwr_targets=rwr_targets,
        heat_t=heat_t,
        poly_K=poly_K,
        rwr_alpha=rwr_alpha,
        gat_config=gat_config,
        train_config=train_config,
        **kwargs
    )
    
    return Z


# -----------------------------------------------------------------------------
# Convenience variant names for QuVINE experiments
# -----------------------------------------------------------------------------

RECOMMENDED_GAT_VARIANTS = [
    "raw",
    "rwr",
    "heat_fixed",
    "poly_fixed",
    "heat_qcal_ctqw",
    "poly_qcal_ctqw",
    "direct_ctqw",
    "direct_dtqw",
]


if __name__ == "__main__":  # smoke test
    logging.basicConfig(level=logging.INFO)
    G = nx.karate_club_graph()
    cfg = GATConfig(hidden_dim=32, output_dim=16, heads=2, num_layers=2, dropout=0.2)
    tc = TrainConfig(epochs=5, edge_batch_size=128, verbose=True)
    Z, meta = generate_gat_embedding(G, variant="raw", gat_config=cfg, train_config=tc)
    print(Z.shape, meta["variant"])
