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
QuVINE GraphGPS variants using PyTorch Geometric.

Purpose
-------
This module treats GraphGPS as a classical downstream graph learner.
It does not claim GraphGPS is quantum. The scientific question is whether
quantum-walk-calibrated or direct quantum-walk features add useful signal
relative to raw and classical-diffusion features under the same GraphGPS model.

Supported feature variants
--------------------------
raw
rwr
heat_fixed
poly_fixed
heat_qcal_ctqw
poly_qcal_ctqw
heat_qcal_dtqw
poly_qcal_dtqw
heat_qcal_rwr
poly_qcal_rwr
direct_ctqw
direct_dtqw


Notes
-----
- This module uses PyG's GPSConv.
- For link prediction, pass train_graph_for_message_passing so GraphGPS does
  not see validation/test edges during message passing.
- Direct CTQW/DTQW features are expected as precomputed arrays, e.g. from your
  SGNS pipeline. This module does not recompute DTQW/CTQW walks internally.
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

if TYPE_CHECKING:
    PyGData = Any
else:
    PyGData = Any

logger = logging.getLogger(__name__)

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

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv, GCNConv, GINConv, GINEConv, GPSConv, SAGEConv
    from torch_geometric.utils import negative_sampling

    PYG_AVAILABLE = True
except ImportError:  # pragma: no cover
    Data = None
    GATConv = None
    GCNConv = None
    GPSConv = None
    SAGEConv = None
    negative_sampling = None
    PYG_AVAILABLE = False


@dataclass
class GraphGPSConfig:
    """Hyperparameters for the GraphGPS encoder."""

    hidden_dim: int = 64
    output_dim: int = 128
    num_layers: int = 2
    heads: int = 4
    dropout: float = 0.2
    attn_dropout: float = 0.2
    local_gnn: str = "gcn"  # Default to GCN (works without edge attributes)
    attn_type: str = "multihead"
    use_layer_norm: bool = True
    activation: str = "relu"
    lap_pe_dim: int = 0
    standardize_features: bool = True


@dataclass
class TrainConfig:
    """Training config for unsupervised link reconstruction or node classification."""

    task: str = "link_reconstruction"
    epochs: int = 200
    lr: float = 5e-3
    weight_decay: float = 5e-4
    patience: int = 30
    edge_batch_size: int = 8192
    val_edge_fraction: float = 0.1
    device: str = "cpu"
    random_state: int = 42
    verbose: bool = False


SUPPORTED_GRAPHGPS_VARIANTS = (
    "raw",
    "rwr",
    "heat_fixed",
    "poly_fixed",
    "heat_qcal_ctqw",
    "poly_qcal_ctqw",
    "heat_qcal_dtqw",
    "poly_qcal_dtqw",
    "heat_qcal_rwr",
    "poly_qcal_rwr",
    "direct_ctqw",
    "direct_dtqw",
)


def _check_deps() -> None:
    """
    Check if required dependencies are available and raise informative errors if not.
    
    Raises:
        TorchNotAvailableError: If PyTorch is not installed
        ImportError: If PyTorch Geometric is not installed
    """
    if not TORCH_AVAILABLE:
        raise TorchNotAvailableError(
            "PyTorch is required for GraphGPS embedding generation.\n"
            "Install PyTorch with: pip install torch\n"
            "See installation guide: https://pytorch.org/get-started/locally/\n"
            "For CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )
    if not PYG_AVAILABLE:
        # Delegate to require_module so this reads like every other missing-extra
        # message in QuVINE -- naming the [quvine] extra and the exact pip command
        # -- instead of sending the user off to install torch-geometric by hand.
        # It raises QuvineDependencyError, which is an ImportError subclass, so
        # callers already catching ImportError here are unaffected.
        from qbiocode.apps.quvine._deps import require_module

        require_module("torch_geometric", feature="GraphGPS transformer embeddings")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    if TORCH_AVAILABLE and torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _select_device(device: str = "cpu") -> Any:
    _check_deps()
    if torch is None:
        raise ImportError("PyTorch is required. Install torch first.")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _stable_nodelist(G: nx.Graph, nodelist: Optional[Sequence] = None) -> List:
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


def as_numpy(x: Any) -> np.ndarray:
    if TORCH_AVAILABLE and torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def row_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norm, eps)


def standardize_columns(
    X: np.ndarray,
    train_mask: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Standardize columns with optional train-only statistics.

    If train_mask is provided, only those rows are used to compute mean/std.
    This mirrors the leakage fix used in gat.py.
    """
    X = np.asarray(X, dtype=np.float32)
    if train_mask is None:
        mu = np.nanmean(X, axis=0, keepdims=True)
        sd = np.nanstd(X, axis=0, keepdims=True)
    else:
        train_mask = np.asarray(train_mask, dtype=bool)
        if train_mask.shape[0] != X.shape[0]:
            raise ValueError("train_mask length must match number of rows in X")
        if not np.any(train_mask):
            raise ValueError("train_mask must contain at least one True entry")
        X_train = X[train_mask]
        mu = np.nanmean(X_train, axis=0, keepdims=True)
        sd = np.nanstd(X_train, axis=0, keepdims=True)
    Xs = (X - mu) / np.maximum(sd, eps)
    return np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def align_features(
    features,
    nodelist: Sequence,
    feature_nodes: Optional[Sequence] = None,
) -> np.ndarray:
    """Align feature matrix to nodelist."""
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


def graph_to_edge_index(
    G: nx.Graph,
    nodelist: Optional[Sequence] = None,
    add_reverse_edges: bool = True,
    add_self_loops: bool = False,
    device: Optional[Any] = None,
) -> Any:
    """Convert a NetworkX graph to PyG edge_index with stable node ordering."""
    _check_deps()
    nodes = _stable_nodelist(G, nodelist)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    edges: List[Tuple[int, int]] = []

    for u, v in G.edges():
        if u not in node_to_idx or v not in node_to_idx:
            continue
        i, j = node_to_idx[u], node_to_idx[v]
        edges.append((i, j))
        if add_reverse_edges and i != j:
            edges.append((j, i))

    if add_self_loops:
        edges.extend((i, i) for i in range(len(nodes)))

    if torch is None:
        raise ImportError("PyTorch is required. Install torch first.")
    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    return edge_index


def get_laplacian(
    G: nx.Graph,
    nodelist: Optional[Sequence] = None,
    normalized: bool = True,
) -> sp.csr_matrix:
    nodes = _stable_nodelist(G, nodelist)
    if normalized:
        L = nx.normalized_laplacian_matrix(G, nodelist=nodes).astype(float).tocsr()
    else:
        L = nx.laplacian_matrix(G, nodelist=nodes).astype(float).tocsr()
    return L


def make_base_features(
    G: nx.Graph,
    nodelist: Optional[Sequence] = None,
    embedding_dim: int = 128,
    features: Optional[np.ndarray] = None,
    feature_nodes: Optional[Sequence] = None,
    feature_mode: str = "structural",
    train_mask: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> np.ndarray:
    """
    Create base node features used by all feature variants.

    feature_mode:
        - 'provided': require features.
        - 'random': row-normalized random features.
        - 'structural': scalable local structural features padded/projected to embedding_dim.

    Degenerate graphs:
        Any structural column that networkx cannot compute for this graph class
        (clustering and triangles on multigraphs, k-core with self-loops,
        pagerank that fails to converge) falls back to zeros -- pagerank to the
        uniform distribution -- and logs the reason at DEBUG. The feature matrix
        always has the documented width; it is never partially built.
    """
    nodes = _stable_nodelist(G, nodelist)
    n = len(nodes)

    if features is not None:
        X = align_features(features, nodelist=nodes, feature_nodes=feature_nodes)
        if X.shape[0] != n:
            raise ValueError(f"features has {X.shape[0]} rows but graph has {n} nodes.")
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if feature_mode == "provided":
        raise ValueError("feature_mode='provided' requires features != None")

    rng = np.random.default_rng(random_state)

    if feature_mode == "random":
        X = rng.normal(size=(n, embedding_dim)).astype(np.float32)
        return row_normalize(X)

    if feature_mode != "structural":
        raise ValueError("feature_mode must be one of {'provided','random','structural'}")

    deg = np.array([G.degree(node) for node in nodes], dtype=np.float32)
    log_deg = np.log1p(deg)

    # Each structural column below degrades independently. networkx raises
    # NetworkXNotImplemented for metrics undefined on the given graph class
    # (multigraphs, directed graphs, self-loops) and PowerIterationFailedConvergence
    # when pagerank does not settle; those are expected inputs here, not errors, so
    # the column falls back to a documented sentinel and the reason is logged at
    # debug rather than raised or discarded.
    try:
        clustering_dict = nx.clustering(G)
        clustering = np.array([clustering_dict.get(node, 0.0) for node in nodes], dtype=np.float32)
    except nx.NetworkXException as exc:
        logger.debug("clustering coefficients unavailable, using zeros: %s", exc)
        clustering = np.zeros(n, dtype=np.float32)

    try:
        core_dict = nx.core_number(G)
        core = np.array([core_dict.get(node, 0.0) for node in nodes], dtype=np.float32)
    except nx.NetworkXException as exc:
        logger.debug("k-core numbers unavailable, using zeros: %s", exc)
        core = np.zeros(n, dtype=np.float32)

    try:
        pr_dict = nx.pagerank(G, alpha=0.85, max_iter=200, tol=1e-6)
        pagerank = np.array([pr_dict.get(node, 0.0) for node in nodes], dtype=np.float32)
    except nx.NetworkXException as exc:
        logger.debug("PageRank unavailable, using the uniform distribution: %s", exc)
        pagerank = np.ones(n, dtype=np.float32) / max(n, 1)

    try:
        tri_dict = nx.triangles(G)
        triangles = np.array([tri_dict.get(node, 0.0) for node in nodes], dtype=np.float32)
    except nx.NetworkXException as exc:
        logger.debug("triangle counts unavailable, using zeros: %s", exc)
        triangles = np.zeros(n, dtype=np.float32)

    try:
        avg_nbr_deg_dict = nx.average_neighbor_degree(G)
        avg_nbr_deg = np.array([avg_nbr_deg_dict.get(node, 0.0) for node in nodes], dtype=np.float32)
    except nx.NetworkXException as exc:
        logger.debug("average neighbour degrees unavailable, using zeros: %s", exc)
        avg_nbr_deg = np.zeros(n, dtype=np.float32)

    max_deg = max(float(deg.max()) if deg.size else 0.0, 1.0)
    X0 = np.vstack(
        [
            deg,
            log_deg,
            clustering,
            core,
            pagerank,
            np.log1p(triangles),
            avg_nbr_deg,
            deg / max_deg,
        ]
    ).T.astype(np.float32)
    X0 = standardize_columns(X0, train_mask=train_mask)

    if embedding_dim <= X0.shape[1]:
        result = X0[:, :embedding_dim].astype(np.float32)
    else:
        R = rng.normal(size=(X0.shape[1], embedding_dim)).astype(np.float32) / np.sqrt(X0.shape[1])
        X = X0 @ R
        result = row_normalize(X).astype(np.float32)
    
    # Ensure we never return None or empty array
    if result is None or result.size == 0:
        raise ValueError(
            f"make_base_features produced invalid output: "
            f"result={'None' if result is None else 'empty array'}"
        )
    
    return result


def apply_heat_filter(L: sp.spmatrix, X: np.ndarray, t: float) -> np.ndarray:
    if t < 0:
        raise ValueError("heat-kernel time t must be non-negative")
    return np.asarray(spla.expm_multiply(-float(t) * L, np.asarray(X, dtype=np.float32))).astype(np.float32)


def apply_polynomial_filter(L: sp.spmatrix, X: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if coeffs.ndim != 1:
        raise ValueError("coeffs must be a 1D array")
    X = np.asarray(X, dtype=np.float32)
    Z = coeffs[0] * X
    V = X.copy()
    for _k in range(1, len(coeffs)):
        V = L @ V
        Z = Z + coeffs[_k] * V
    return np.asarray(Z, dtype=np.float32)


def apply_rwr_filter(
    G: nx.Graph,
    X: np.ndarray,
    nodelist: Sequence,
    alpha: float = 0.15,
    steps: int = 50,
    tol: float = 1e-6,
) -> np.ndarray:
    """Random walk with restart / PPR-style feature diffusion."""
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    A = nx.adjacency_matrix(G, nodelist=nodelist).astype(float).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv_deg = np.zeros_like(deg, dtype=float)
    mask = deg > 0
    inv_deg[mask] = 1.0 / deg[mask]
    P = sp.diags(inv_deg) @ A

    Z = np.asarray(X, dtype=np.float32).copy()
    X0 = np.asarray(X, dtype=np.float32)
    for _ in range(int(steps)):
        Z_next = alpha * X0 + (1.0 - alpha) * (P @ Z)
        diff = float(np.linalg.norm(Z_next - Z)) / max(float(np.linalg.norm(Z)), 1e-12)
        Z = np.asarray(Z_next, dtype=np.float32)
        if diff < tol:
            break
    return Z


def _validate_targets(targets: List[Dict], node_to_idx: Mapping) -> List[Dict]:
    valid: List[Dict] = []
    for item in targets:
        nodes = list(item["nodes"])
        center = item["center"]
        p = np.asarray(item.get("pQ", item.get("p", None)), dtype=np.float64)
        if center not in node_to_idx:
            continue
        if any(node not in node_to_idx for node in nodes):
            continue
        if len(nodes) != len(p):
            raise ValueError("Each target must satisfy len(nodes) == len(pQ).")
        p = np.maximum(p, 0.0)
        s = p.sum()
        if s <= 0:
            continue
        valid.append({"nodes": nodes, "center": center, "pQ": p / s})
    if not valid:
        raise ValueError("No valid calibration targets after validation.")
    return valid


def calibrate_heat_kernel(
    L: sp.spmatrix,
    targets: List[Dict],
    node_to_idx: Mapping,
    t_grid: Optional[np.ndarray] = None,
    loss: str = "l2",
) -> Tuple[float, float]:
    """Fit heat time t to target distributions by local distribution matching."""
    if t_grid is None:
        t_grid = np.logspace(-2, 2, 40)
    targets = _validate_targets(targets, node_to_idx)
    n = L.shape[0]

    best_loss = np.inf
    best_t = float(t_grid[0])

    for t in t_grid:
        total = 0.0
        for item in targets:
            x = np.zeros(n, dtype=np.float32)
            x[node_to_idx[item["center"]]] = 1.0
            idx = [node_to_idx[node] for node in item["nodes"]]
            y = spla.expm_multiply(-float(t) * L, x)
            yS = np.maximum(np.asarray(y)[idx], 0.0)
            s = yS.sum()
            pT = yS / s if s > 0 else np.ones_like(yS) / len(yS)
            pQ = item["pQ"]
            if loss == "l2":
                total += float(np.sum((pT - pQ) ** 2))
            elif loss == "kl":
                eps = 1e-12
                total += float(np.sum(pQ * (np.log(pQ + eps) - np.log(pT + eps))))
            else:
                raise ValueError("loss must be 'l2' or 'kl'")
        if total < best_loss:
            best_loss = total
            best_t = float(t)

    return best_loss, best_t


def calibrate_polynomial_filter(
    L: sp.spmatrix,
    targets: List[Dict],
    node_to_idx: Mapping,
    K: int = 4,
    ridge: float = 1e-5,
) -> np.ndarray:
    """
    Fit monomial Laplacian polynomial coefficients to target distributions.

    Important fix: no column normalization is applied during fitting, so the
    fitted basis matches the deployed polynomial basis.
    """
    targets = _validate_targets(targets, node_to_idx)
    n = L.shape[0]
    AtA = np.zeros((K + 1, K + 1), dtype=np.float64)
    Atb = np.zeros(K + 1, dtype=np.float64)

    for item in targets:
        x = np.zeros(n, dtype=np.float64)
        x[node_to_idx[item["center"]]] = 1.0
        idx = [node_to_idx[node] for node in item["nodes"]]

        basis = []
        v = x
        basis.append(np.asarray(v)[idx])
        for _ in range(1, K + 1):
            v = np.asarray(L.dot(v), dtype=np.float64)
            basis.append(np.asarray(v)[idx])

        Phi = np.stack(basis, axis=1)
        b = item["pQ"]
        AtA += Phi.T @ Phi
        Atb += Phi.T @ b

    AtA += ridge * np.eye(K + 1)
    try:
        coeffs = np.linalg.solve(AtA, Atb)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(AtA, Atb, rcond=None)[0]
    return coeffs.astype(np.float64)


def build_rwr_targets_from_templates(
    G: nx.Graph,
    templates: List[Dict],
    nodelist: Optional[Sequence] = None,
    alpha: float = 0.15,
    steps: int = 50,
    tol: float = 1e-10,
) -> List[Dict]:
    """Build classical RWR/PPR target distributions matching CTQW target supports."""
    nodes = _stable_nodelist(G, nodelist)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    A = nx.adjacency_matrix(G, nodelist=nodes).astype(float).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv_deg = np.zeros_like(deg, dtype=float)
    mask = deg > 0
    inv_deg[mask] = 1.0 / deg[mask]
    P = sp.diags(inv_deg) @ A
    PT = P.T.tocsr()

    targets: List[Dict] = []
    for item in templates:
        center = item["center"]
        support = list(item["nodes"])
        if center not in node_to_idx or any(s not in node_to_idx for s in support):
            continue
        e = np.zeros(n, dtype=float)
        e[node_to_idx[center]] = 1.0
        p = e.copy()
        for _ in range(steps):
            p_new = alpha * e + (1.0 - alpha) * (PT @ p)
            if np.linalg.norm(p_new - p, ord=1) < tol:
                p = p_new
                break
            p = p_new
        idx = [node_to_idx[s] for s in support]
        pS = np.maximum(p[idx], 0.0)
        s = pS.sum()
        pS = pS / s if s > 0 else np.ones(len(idx), dtype=float) / len(idx)
        targets.append({"nodes": support, "center": center, "pQ": pS})
    if not targets:
        raise ValueError("Could not create any RWR targets from templates.")
    return targets


def fixed_heat_time_grid() -> np.ndarray:
    return np.logspace(-2, 2, 40)


def heat_taylor_coeffs(t: float = 1.0, K: int = 4) -> np.ndarray:
    from math import factorial

    return np.array([((-t) ** k) / factorial(k) for k in range(K + 1)], dtype=np.float64)


def align_direct_features(
    Z: Union[np.ndarray, Mapping],
    graph_nodes: Sequence,
    feature_nodes: Optional[Sequence] = None,
) -> np.ndarray:
    """Align a direct feature matrix/mapping to graph_nodes order."""
    return align_features(Z, nodelist=graph_nodes, feature_nodes=feature_nodes)


def build_graphgps_input_features(
    G: nx.Graph,
    variant: str,
    nodelist: Optional[Sequence] = None,
    train_mask: Optional[np.ndarray] = None,
    base_features: Optional[np.ndarray] = None,
    base_feature_nodes: Optional[Sequence] = None,
    embedding_dim: int = 128,
    feature_mode: str = "structural",
    ctqw_targets: Optional[List[Dict]] = None,
    dtqw_targets: Optional[List[Dict]] = None,
    rwr_targets: Optional[List[Dict]] = None,
    direct_features: Optional[Mapping[str, Union[np.ndarray, Mapping]]] = None,
    direct_feature_nodes: Optional[Sequence] = None,
    normalize_laplacian: bool = True,
    heat_t: float = 1.0,
    poly_K: int = 4,
    poly_ridge: float = 1e-5,
    rwr_alpha: float = 0.15,
    rwr_steps: int = 50,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict]:
    """Build one GraphGPS input feature matrix and metadata."""
    variant = variant.lower()
    if variant not in SUPPORTED_GRAPHGPS_VARIANTS:
        raise ValueError(f"Unsupported variant {variant!r}. Valid: {SUPPORTED_GRAPHGPS_VARIANTS}")

    def _clean_features(arr: np.ndarray) -> np.ndarray:
        return np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    nodes = _stable_nodelist(G, nodelist)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    L = get_laplacian(G, nodes, normalized=normalize_laplacian)

    meta: Dict = {
        "variant": variant,
        "n_nodes": len(nodes),
        "normalize_laplacian": normalize_laplacian,
    }

    if variant.startswith("direct_"):
        if direct_features is None:
            raise ValueError(f"variant={variant} requires direct_features dict")
        key = "ctqw" if variant == "direct_ctqw" else "dtqw"
        if key not in direct_features:
            raise ValueError(f"direct_features must contain key {key!r}")
        meta["direct_key"] = key
        return _clean_features(
            align_direct_features(direct_features[key], nodes, direct_feature_nodes)
        ), meta

    X0 = make_base_features(
        G,
        nodelist=nodes,
        embedding_dim=embedding_dim,
        features=base_features,
        feature_nodes=base_feature_nodes,
        feature_mode=feature_mode,
        train_mask=train_mask,
        random_state=random_state,
    )

    if variant == "raw":
        return X0.astype(np.float32), meta

    if variant == "rwr":
        meta.update({"rwr_alpha": rwr_alpha, "rwr_steps": rwr_steps})
        return _clean_features(
            apply_rwr_filter(G, X0, nodelist=nodes, alpha=rwr_alpha, steps=rwr_steps)
        ), meta

    if variant == "heat_fixed":
        meta["heat_t"] = heat_t
        return _clean_features(apply_heat_filter(L, X0, heat_t)), meta

    if variant == "poly_fixed":
        coeffs = heat_taylor_coeffs(t=heat_t, K=poly_K)
        meta.update(
            {
                "poly_coeffs": coeffs.tolist(),
                "poly_K": poly_K,
                "heat_t_for_taylor": heat_t,
            }
        )
        return _clean_features(apply_polynomial_filter(L, X0, coeffs)), meta

    if variant.endswith("_ctqw"):
        targets = ctqw_targets
        target_source = "ctqw"
    elif variant.endswith("_dtqw"):
        targets = dtqw_targets
        target_source = "dtqw"
    elif variant.endswith("_rwr"):
        target_source = "rwr"
        if rwr_targets is not None:
            targets = rwr_targets
        else:
            templates = ctqw_targets if ctqw_targets is not None else dtqw_targets
            if templates is None:
                raise ValueError("RWR calibration requires rwr_targets or ctqw/dtqw templates.")
            targets = build_rwr_targets_from_templates(
                G, templates, nodelist=nodes, alpha=rwr_alpha, steps=rwr_steps
            )
            meta.update({"rwr_targets_generated_from_templates": True})
    else:
        raise RuntimeError(f"Unhandled variant: {variant}")

    if targets is None:
        raise ValueError(f"variant={variant} requires {target_source}_targets")

    if variant.startswith("heat_qcal"):
        loss, t_star = calibrate_heat_kernel(
            L, targets, node_to_idx, t_grid=fixed_heat_time_grid(), loss="l2"
        )
        meta.update(
            {
                "target_source": target_source,
                "heat_t_star": t_star,
                "calibration_loss": loss,
            }
        )
        return _clean_features(apply_heat_filter(L, X0, t_star)), meta

    if variant.startswith("poly_qcal"):
        coeffs = calibrate_polynomial_filter(
            L, targets, node_to_idx, K=poly_K, ridge=poly_ridge
        )
        meta.update(
            {
                "target_source": target_source,
                "poly_coeffs": coeffs.tolist(),
                "poly_K": poly_K,
                "poly_ridge": poly_ridge,
            }
        )
        return _clean_features(apply_polynomial_filter(L, X0, coeffs)), meta

    raise RuntimeError(f"Unhandled variant: {variant}")


class PyGGraphGPS(nn.Module if nn is not None else object):
    """Node-level GraphGPS encoder."""

    def __init__(self, input_dim: int, config: GraphGPSConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(input_dim, config.hidden_dim)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(config.num_layers):
            local_conv = self._make_local_conv(config)
            # Try to pass attn_dropout, but handle if PyG version doesn't support it
            try:
                gps_layer = GPSConv(
                    channels=config.hidden_dim,
                    conv=local_conv,
                    heads=config.heads,
                    dropout=config.dropout,
                    attn_dropout=config.attn_dropout,
                    act=config.activation,
                    attn_type=config.attn_type,
                )
            except TypeError:
                # Fallback for older PyG versions without attn_dropout
                gps_layer = GPSConv(
                    channels=config.hidden_dim,
                    conv=local_conv,
                    heads=config.heads,
                    dropout=config.dropout,
                    act=config.activation,
                    attn_type=config.attn_type,
                )
            self.layers.append(gps_layer)
            self.norms.append(
                nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity()
            )

        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)

    @staticmethod
    def _make_local_conv(config: GraphGPSConfig):
        kind = config.local_gnn.lower()
        if kind == "gcn":
            return GCNConv(config.hidden_dim, config.hidden_dim)
        if kind == "sage":
            return SAGEConv(config.hidden_dim, config.hidden_dim)
        if kind == "gat":
            return GATConv(
                config.hidden_dim,
                config.hidden_dim,
                heads=1,
                concat=False,
                dropout=config.dropout,
            )
        if kind == "gin":
            # GIN (Graph Isomorphism Network) - uses MLP for aggregation
            from torch import nn as torch_nn
            mlp = torch_nn.Sequential(
                torch_nn.Linear(config.hidden_dim, config.hidden_dim),
                torch_nn.ReLU(),
                torch_nn.Linear(config.hidden_dim, config.hidden_dim)
            )
            return GINConv(mlp, train_eps=True)
        if kind == "gine":
            # GINE (GIN with Edge features) - similar to GIN but edge-aware
            from torch import nn as torch_nn
            mlp = torch_nn.Sequential(
                torch_nn.Linear(config.hidden_dim, config.hidden_dim),
                torch_nn.ReLU(),
                torch_nn.Linear(config.hidden_dim, config.hidden_dim)
            )
            return GINEConv(mlp, train_eps=True)
        if kind == "none":
            return None
        raise ValueError("local_gnn must be one of {'gcn','sage','gat','gin','gine','none'}")

    def forward(
        self,
        x: Any,
        edge_index: Any,
        batch: Optional[Any] = None,
    ):
        h = self.input_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            h = layer(h, edge_index, batch=batch)
            h = norm(h)
        z = self.output_proj(h)
        return z


def _append_laplacian_pe(
    G: nx.Graph,
    X: np.ndarray,
    nodelist: Sequence,
    pe_dim: int,
    normalized: bool = True,
    train_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if pe_dim <= 0:
        return X
    n = len(nodelist)
    if n <= 2:
        return np.concatenate([X, np.zeros((n, pe_dim), dtype=np.float32)], axis=1)
    L = get_laplacian(G, nodelist, normalized=normalized)
    k = min(pe_dim + 1, n - 1)
    try:
        vals, vecs = spla.eigsh(L, k=k, which="SM", tol=1)
        idx = np.argsort(vals)
        vecs = np.real(vecs[:, idx])
        pe = vecs[:, 1 : pe_dim + 1]
        if pe.shape[1] < pe_dim:
            pe = np.pad(pe, ((0, 0), (0, pe_dim - pe.shape[1])), mode="constant")
        for j in range(pe.shape[1]):
            if np.abs(pe[:, j]).max() > 0:
                if pe[np.argmax(np.abs(pe[:, j])), j] < 0:
                    pe[:, j] *= -1
        pe = standardize_columns(pe.astype(np.float32), train_mask=train_mask)
    except (spla.ArpackError, np.linalg.LinAlgError, ValueError) as exc:
        # ARPACK does not always find k smallest eigenpairs of a near-singular or
        # disconnected Laplacian (ArpackNoConvergence subclasses ArpackError).
        # Positional encodings are an additive feature block, so an all-zero block
        # keeps the output width contractual instead of failing the embedding.
        logger.debug("Laplacian positional encoding unavailable, using zeros: %s", exc)
        pe = np.zeros((n, pe_dim), dtype=np.float32)
    return np.concatenate([X, pe], axis=1).astype(np.float32)


def build_pyg_data(
    G: nx.Graph,
    X: np.ndarray,
    nodelist: Optional[Sequence] = None,
    gps_config: Optional[GraphGPSConfig] = None,
    train_graph_for_message_passing: Optional[nx.Graph] = None,
    train_mask: Optional[np.ndarray] = None,
) -> PyGData:
    _check_deps()
    
    # Validate input features
    if X is None:
        raise ValueError("Feature matrix X cannot be None")
    if not isinstance(X, np.ndarray):
        raise TypeError(f"Feature matrix X must be numpy array, got {type(X)}")
    if X.size == 0:
        raise ValueError("Feature matrix X cannot be empty")
    
    nodes = _stable_nodelist(G, nodelist)
    graph_mp = train_graph_for_message_passing if train_graph_for_message_passing is not None else G
    edge_index = graph_to_edge_index(
        graph_mp,
        nodelist=nodes,
        add_reverse_edges=True,
        add_self_loops=False,
    )

    X_use = np.asarray(X, dtype=np.float32)
    if gps_config is not None and gps_config.lap_pe_dim > 0:
        X_use = _append_laplacian_pe(
            G,
            X_use,
            nodes,
            gps_config.lap_pe_dim,
            normalized=True,
            train_mask=train_mask,
        )
    if gps_config is not None and gps_config.standardize_features:
        X_use = standardize_columns(X_use, train_mask=train_mask)

    X_use = np.nan_to_num(X_use, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if Data is None or torch is None:
        raise ImportError("PyTorch Geometric and PyTorch are required.")
    data = Data(x=torch.from_numpy(X_use).float(), edge_index=edge_index)
    data.batch = torch.zeros(X_use.shape[0], dtype=torch.long)
    return data


def _split_edges_for_reconstruction(
    G: nx.Graph,
    nodelist: Sequence,
    val_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    node_to_idx = {node: i for i, node in enumerate(nodelist)}
    edges = np.array(
        [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges() if u in node_to_idx and v in node_to_idx],
        dtype=np.int64,
    )
    if edges.size == 0:
        raise ValueError("GraphGPS link reconstruction needs at least one edge.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(edges))
    n_val = int(round(val_fraction * len(edges)))
    n_val = min(max(n_val, 1 if len(edges) > 10 else 0), len(edges) - 1)
    val_edges = edges[perm[:n_val]]
    train_edges = edges[perm[n_val:]]
    return train_edges, val_edges


def _sample_edge_batch(
    pos_edges: np.ndarray,
    num_nodes: int,
    batch_size: int,
    device,
) -> Tuple[Any, Any]:
    if torch is None or negative_sampling is None:
        raise ImportError("PyTorch Geometric and PyTorch are required.")
    n_pos = min(len(pos_edges), max(1, batch_size // 2))
    idx = torch.randint(0, len(pos_edges), (n_pos,), device=device)
    pos = torch.as_tensor(pos_edges, dtype=torch.long, device=device)[idx].t().contiguous()
    neg = negative_sampling(
        edge_index=pos,
        num_nodes=num_nodes,
        num_neg_samples=n_pos,
        method="sparse",
    ).to(device)
    edge_label_index = torch.cat([pos, neg], dim=1)
    labels = torch.cat(
        [torch.ones(pos.size(1), device=device), torch.zeros(neg.size(1), device=device)]
    )
    return edge_label_index, labels


def dot_decode(z: Any, edge_label_index: Any) -> Any:
    src, dst = edge_label_index[0], edge_label_index[1]
    return (z[src] * z[dst]).sum(dim=-1)


def train_graphgps_link_reconstruction(
    model: PyGGraphGPS,
    data: PyGData,
    G_for_edges: nx.Graph,
    nodelist: Sequence,
    train_config: TrainConfig,
) -> Dict:
    if torch is None or F is None:
        raise ImportError("PyTorch is required. Install torch first.")
    if torch is None or F is None or nn is None:
        raise ImportError("PyTorch is required. Install torch first.")
    device = _select_device(train_config.device)
    model = model.to(device)
    data = data.to(device)

    train_edges, val_edges = _split_edges_for_reconstruction(
        G_for_edges, nodelist, train_config.val_edge_fraction, train_config.random_state
    )
    opt = torch.optim.Adam(
        model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay
    )
    best_state = None
    best_val = np.inf
    patience_left = train_config.patience
    history = {"loss": [], "val_loss": [], "best_epoch": 0, "best_val_loss": np.inf}

    for epoch in range(train_config.epochs):
        model.train()
        opt.zero_grad()
        z = model(data.x, data.edge_index, batch=data.batch)
        edge_label_index, labels = _sample_edge_batch(
            train_edges, data.num_nodes, train_config.edge_batch_size, device
        )
        logits = dot_decode(z, edge_label_index)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            z = model(data.x, data.edge_index, batch=data.batch)
            if len(val_edges) > 0:
                pos = torch.as_tensor(val_edges, dtype=torch.long, device=device).t().contiguous()
                neg = negative_sampling(
                    edge_index=pos,
                    num_nodes=data.num_nodes,
                    num_neg_samples=pos.size(1),
                    method="sparse",
                ).to(device)
                edge_label_index = torch.cat([pos, neg], dim=1)
                labels = torch.cat(
                    [torch.ones(pos.size(1), device=device), torch.zeros(neg.size(1), device=device)]
                )
                val_loss = F.binary_cross_entropy_with_logits(
                    dot_decode(z, edge_label_index), labels
                )
            else:
                val_loss = loss

        loss_v = float(loss.item())
        val_v = float(val_loss.item())
        history["loss"].append(loss_v)
        history["val_loss"].append(val_v)

        if val_v < best_val - 1e-6:
            best_val = val_v
            best_state = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch
            history["best_val_loss"] = best_val
            patience_left = train_config.patience
        else:
            patience_left -= 1

        if train_config.verbose and (epoch + 1) % 25 == 0:
            logger.info("GraphGPS epoch %d loss %.4f val %.4f", epoch + 1, loss_v, val_v)
        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def train_graphgps_node_classifier(
    model: PyGGraphGPS,
    data: "Data",
    y: Union[np.ndarray, Any],
    train_mask: Union[np.ndarray, Any],
    val_mask: Union[np.ndarray, Any],
    num_classes: int,
    train_config: TrainConfig,
) -> Tuple[Any, Dict]:
    device = _select_device(train_config.device)
    model = model.to(device)
    data = data.to(device)
    y_t = torch.as_tensor(y, dtype=torch.long, device=device)
    train_mask_t = torch.as_tensor(train_mask, dtype=torch.bool, device=device)
    val_mask_t = torch.as_tensor(val_mask, dtype=torch.bool, device=device)

    clf = nn.Linear(model.config.output_dim, num_classes).to(device)
    opt = torch.optim.Adam(
        list(model.parameters()) + list(clf.parameters()),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    best_state = None
    best_val = np.inf
    patience_left = train_config.patience
    history = {"loss": [], "val_loss": [], "best_epoch": 0, "best_val_loss": np.inf}

    for epoch in range(train_config.epochs):
        model.train()
        clf.train()
        opt.zero_grad()
        z = model(data.x, data.edge_index, batch=data.batch)
        logits = clf(z)
        loss = F.cross_entropy(logits[train_mask_t], y_t[train_mask_t])
        loss.backward()
        opt.step()

        model.eval()
        clf.eval()
        with torch.no_grad():
            z = model(data.x, data.edge_index, batch=data.batch)
            logits = clf(z)
            val_loss = F.cross_entropy(logits[val_mask_t], y_t[val_mask_t])

        loss_v = float(loss.item())
        val_v = float(val_loss.item())
        history["loss"].append(loss_v)
        history["val_loss"].append(val_v)
        if val_v < best_val - 1e-6:
            best_val = val_v
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "clf": copy.deepcopy(clf.state_dict()),
            }
            history["best_epoch"] = epoch
            history["best_val_loss"] = best_val
            patience_left = train_config.patience
        else:
            patience_left -= 1
        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        clf.load_state_dict(best_state["clf"])
    return clf, history


def generate_graphgps_embedding(
    G: nx.Graph,
    variant: str,
    nodelist: Optional[Sequence] = None,
    train_mask: Optional[Union[np.ndarray, Any]] = None,
    base_features: Optional[np.ndarray] = None,
    base_feature_nodes: Optional[Sequence] = None,
    embedding_dim: int = 128,
    feature_mode: str = "structural",
    ctqw_targets: Optional[List[Dict]] = None,
    dtqw_targets: Optional[List[Dict]] = None,
    rwr_targets: Optional[List[Dict]] = None,
    direct_features: Optional[Mapping[str, Union[np.ndarray, Mapping]]] = None,
    direct_feature_nodes: Optional[Sequence] = None,
    train_graph_for_message_passing: Optional[nx.Graph] = None,
    train_graph_for_edges: Optional[nx.Graph] = None,
    y: Optional[Union[np.ndarray, Any]] = None,
    val_mask: Optional[Union[np.ndarray, Any]] = None,
    num_classes: Optional[int] = None,
    gps_config: Optional[GraphGPSConfig] = None,
    train_config: Optional[TrainConfig] = None,
    normalize_laplacian: bool = True,
    heat_t: float = 1.0,
    poly_K: int = 4,
    poly_ridge: float = 1e-5,
    rwr_alpha: float = 0.15,
    rwr_steps: int = 50,
) -> Tuple[np.ndarray, Dict]:
    """
    Generate embeddings using one feature variant followed by GraphGPS.

    If train_config.task == 'none', GraphGPS is untrained and returns initial forward pass.
    """
    _check_deps()
    train_config = train_config or TrainConfig()
    set_seed(train_config.random_state)

    nodes = _stable_nodelist(G, nodelist)
    gps_config = gps_config or GraphGPSConfig(output_dim=embedding_dim)
    train_mask_np = None if train_mask is None else np.asarray(as_numpy(train_mask), dtype=bool)

    X, feature_meta = build_graphgps_input_features(
        G,
        variant=variant,
        nodelist=nodes,
        train_mask=train_mask_np,
        base_features=base_features,
        base_feature_nodes=base_feature_nodes,
        embedding_dim=embedding_dim,
        feature_mode=feature_mode,
        ctqw_targets=ctqw_targets,
        dtqw_targets=dtqw_targets,
        rwr_targets=rwr_targets,
        direct_features=direct_features,
        direct_feature_nodes=direct_feature_nodes,
        normalize_laplacian=normalize_laplacian,
        heat_t=heat_t,
        poly_K=poly_K,
        poly_ridge=poly_ridge,
        rwr_alpha=rwr_alpha,
        rwr_steps=rwr_steps,
        random_state=train_config.random_state,
    )
    
    # Validate that features were successfully generated
    if X is None:
        raise ValueError(
            f"Feature generation failed for variant '{variant}'. "
            f"build_graphgps_input_features returned None."
        )
    if not isinstance(X, np.ndarray):
        raise TypeError(
            f"Expected numpy array from build_graphgps_input_features, got {type(X)}"
        )
    if X.size == 0:
        raise ValueError(
            f"Feature generation produced empty array for variant '{variant}'"
        )

    data = build_pyg_data(
        G,
        X,
        nodelist=nodes,
        gps_config=gps_config,
        train_graph_for_message_passing=train_graph_for_message_passing,
        train_mask=train_mask_np,
    )
    input_dim = int(data.x.shape[1])
    model = PyGGraphGPS(input_dim=input_dim, config=gps_config)

    metadata: Dict = {
        "variant": variant,
        "feature_meta": feature_meta,
        "gps_config": gps_config.__dict__,
        "train_config": train_config.__dict__,
    }

    if train_config.task == "link_reconstruction":
        edge_graph = (
            train_graph_for_edges
            if train_graph_for_edges is not None
            else (train_graph_for_message_passing or G)
        )
        history = train_graphgps_link_reconstruction(model, data, edge_graph, nodes, train_config)
        metadata["training_history"] = history
    elif train_config.task == "node_classification":
        if y is None or train_mask is None or val_mask is None or num_classes is None:
            raise ValueError("node_classification requires y, train_mask, val_mask, and num_classes")
        clf, history = train_graphgps_node_classifier(
            model, data, y, train_mask, val_mask, num_classes, train_config
        )
        metadata["training_history"] = history
        metadata["classifier"] = "linear"
    elif train_config.task == "none":
        metadata["training_history"] = None
    else:
        raise ValueError(
            "train_config.task must be 'link_reconstruction', 'node_classification', or 'none'"
        )

    device = _select_device(train_config.device)
    model = model.to(device)
    data = data.to(device)
    model.eval()
    with torch.no_grad():
        Z = model(data.x, data.edge_index, batch=data.batch).detach().cpu().numpy().astype(np.float32)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    metadata.update({"embedding_shape": tuple(Z.shape), "model": "GraphGPS"})
    return Z, metadata


def generate_multiple_graphgps_embeddings(
    G: nx.Graph,
    variants: Sequence[str],
    **kwargs,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
    embeddings: Dict[str, np.ndarray] = {}
    metadata: Dict[str, Dict] = {}
    for variant in variants:
        Z, meta = generate_graphgps_embedding(G, variant=variant, **kwargs)
        embeddings[variant] = Z
        metadata[variant] = meta

    return embeddings, metadata


# ============================================================================
# Standardized Method Name Mapping (Phase 1.3)
# ============================================================================

def generate_graphgps_embedding_by_method_name(
    G: nx.Graph,
    method_name: str,
    embedding_dim: int = 128,
    nodelist: Optional[Sequence] = None,
    base_features: Optional[np.ndarray] = None,
    ctqw_targets: Optional[List[Dict]] = None,
    dtqw_targets: Optional[List[Dict]] = None,
    rwr_targets: Optional[List[Dict]] = None,
    direct_features: Optional[Mapping[str, Union[np.ndarray, Mapping]]] = None,
    heat_t: float = 1.0,
    poly_K: int = 4,
    rwr_alpha: float = 0.15,
    gps_config: Optional[GraphGPSConfig] = None,
    train_config: Optional[TrainConfig] = None,
    **kwargs
) -> np.ndarray:
    """
    Generate GraphGPS embedding using standardized method names.
    
    This function maps the 12 standardized GraphGPS method names to the existing
    variant system in generate_graphgps_embedding.
    
    Supported methods:
        - graphgps_baseline: Raw structural features
        - graphgps_heat: Heat kernel filter only
        - graphgps_poly: Polynomial filter only
        - graphgps_rwr: RWR walk only
        - graphgps_ctqw: CTQW walk only (direct features)
        - graphgps_dtqw: DTQW walk only (direct features)
        - graphgps_rwr_heat: RWR + heat filter
        - graphgps_rwr_poly: RWR + polynomial filter
        - graphgps_ctqw_heat: CTQW + heat filter (quantum calibrated)
        - graphgps_ctqw_poly: CTQW + polynomial filter (quantum calibrated)
        - graphgps_dtqw_heat: DTQW + heat filter (quantum calibrated)
        - graphgps_dtqw_poly: DTQW + polynomial filter (quantum calibrated)
    
    Args:
        G: NetworkX graph
        method_name: Standardized method name (e.g., 'graphgps_baseline', 'graphgps_ctqw_heat')
        embedding_dim: Output embedding dimension
        nodelist: Ordered list of nodes
        base_features: Pre-computed base features
        ctqw_targets: CTQW calibration targets
        dtqw_targets: DTQW calibration targets
        rwr_targets: RWR calibration targets
        direct_features: Direct walk features (for direct_ctqw/dtqw variants)
        heat_t: Heat kernel time parameter (for fixed variants)
        poly_K: Polynomial degree
        rwr_alpha: RWR restart probability
        gps_config: GraphGPS model configuration
        train_config: Training configuration
        **kwargs: Additional arguments
    
    Returns:
        Node embeddings [N, embedding_dim]
    
    Example:
        >>> G = nx.karate_club_graph()
        >>> emb = generate_graphgps_embedding_by_method_name(G, 'graphgps_baseline', embedding_dim=64)
        >>> print(emb.shape)  # (34, 64)
    
    Raises:
        ValueError: If method_name is not recognized
    """
    # Map standardized names to internal variant names
    method_to_variant = {
        'graphgps_baseline': 'raw',
        'graphgps_heat': 'heat_fixed',
        'graphgps_poly': 'poly_fixed',
        'graphgps_rwr': 'rwr',
        'graphgps_ctqw': 'direct_ctqw',
        'graphgps_dtqw': 'direct_dtqw',
        'graphgps_rwr_heat': 'heat_qcal_rwr',
        'graphgps_rwr_poly': 'poly_qcal_rwr',
        'graphgps_ctqw_heat': 'heat_qcal_ctqw',
        'graphgps_ctqw_poly': 'poly_qcal_ctqw',
        'graphgps_dtqw_heat': 'heat_qcal_dtqw',
        'graphgps_dtqw_poly': 'poly_qcal_dtqw',
    }
    
    if method_name not in method_to_variant:
        raise ValueError(
            f"Unknown GraphGPS method: {method_name}. "
            f"Expected one of: {list(method_to_variant.keys())}"
        )
    
    variant = method_to_variant[method_name]
    
    # Set up GraphGPS config with desired output dimension
    if gps_config is None:
        gps_config = GraphGPSConfig(output_dim=embedding_dim)
    else:
        # Override output_dim if provided
        gps_config.output_dim = embedding_dim
    
    # Set default train_config for unsupervised embedding generation
    if train_config is None:
        train_config = TrainConfig(task='link_reconstruction')
    
    # Call the existing generate_graphgps_embedding function
    Z, meta = generate_graphgps_embedding(
        G=G,
        variant=variant,
        nodelist=nodelist,
        base_features=base_features,
        embedding_dim=embedding_dim,
        ctqw_targets=ctqw_targets,
        dtqw_targets=dtqw_targets,
        rwr_targets=rwr_targets,
        direct_features=direct_features,
        heat_t=heat_t,
        poly_K=poly_K,
        rwr_alpha=rwr_alpha,
        gps_config=gps_config,
        train_config=train_config,
        **kwargs
    )
    
    return Z

