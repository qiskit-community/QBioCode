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
QuVINE Quantum Filters: Quantum-Calibrated Graph Diffusion

This module implements quantum-calibrated graph diffusion for QuVINE, which uses
local quantum walk statistics from sampled subnetworks to calibrate parameters of
global graph diffusion operators.

Key Innovation:
- Run quantum walks on small subgraphs (scalable)
- Fit global diffusion parameters to match quantum behavior
- Apply calibrated classical operator to full graph (efficient)

Reference:
- QuVINE notebook: notebooks/Q-Caliber_Quantum_Calibrated_Graph_Diffusion.ipynb
- Hiperwalk: https://hiperwalk.org/

Author: QuVINE Team
"""

import numpy as np
import scipy.sparse as sp
from typing import cast
import scipy.sparse.linalg as spla
from scipy.linalg import expm
import networkx as nx
from typing import List, Dict, Tuple, Optional, Union
import logging
from qbiocode.apps.quvine.data.subgraph import expand_neighborhood
logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

def get_laplacian(G, normalize=True, weight="weight"):
    nodelist = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodelist)}

    if normalize:
        L = nx.normalized_laplacian_matrix(
            G, nodelist=nodelist, weight=weight
        ).astype(float).tocsr()
    else:
        L = nx.laplacian_matrix(
            G, nodelist=nodelist, weight=weight
        ).astype(float).tocsr()

    return L, nodelist, node_to_idx

def _normalize_laplacian(L):
    """
    Normalize a Laplacian matrix: L_norm = D^{-1/2} L D^{-1/2}
    
    Args:
        L: Laplacian matrix (sparse or dense)
    
    Returns:
        Normalized Laplacian matrix (same type as input)
    """
    if sp.issparse(L):
        # Sparse case
        N = L.shape[0]
        # Get diagonal (degree) values
        L_csr = sp.csr_matrix(L)
        d = np.array([L_csr[i, i] for i in range(N)])
        # Compute D^{-1/2}, handling zeros
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
        D_inv_sqrt = sp.diags(d_inv_sqrt, format='csr')
        # Normalize: D^{-1/2} L D^{-1/2}
        L_norm = D_inv_sqrt @ L_csr @ D_inv_sqrt
        return L_norm
    else:
        # Dense case
        L_dense = np.asarray(L)
        d = np.diag(L_dense)
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L_norm = D_inv_sqrt @ L_dense @ D_inv_sqrt
        return L_norm


def _expm_multiply_restricted(
    L: Union[np.ndarray, sp.spmatrix],
    t: float,
    x: np.ndarray,
    nodes: List[int],
    node_to_idx: Dict[int, int]
) -> np.ndarray:
    """
    Compute exp(-t*L) @ x and extract values at specified nodes.
    
    Uses scipy's expm_multiply for efficiency with sparse matrices.
    
    Args:
        L: Laplacian matrix
        t: Time parameter
        x: Initial vector
        nodes: List of node IDs to extract
        node_to_idx: Mapping from node IDs to matrix indices
    
    Returns:
        Array of values at specified nodes
    """
    # Compute exp(-t*L) @ x
    if sp.issparse(L):
        y = spla.expm_multiply(-t * L, x)
    else:
        y = expm(-t * L) @ x
    
    # Extract values at specified nodes
    node_indices = [node_to_idx[n] for n in nodes]
    return np.asarray(y)[node_indices]


def get_ego_net_nodes_quvine(G, center, k=2, max_nodes=None):
    """
    Get ego-net nodes using QuVINE's expand_neighborhood function.
    
    Args:
        G: NetworkX graph
        center: Center node
        k: Hop radius
        max_nodes: Maximum number of nodes (optional truncation)
    
    Returns:
        List of nodes in the ego-net
    """
    # Use QuVINE's expand_neighborhood function
    nodes = expand_neighborhood(G, roots={center}, radius=k)
    nodes = list(nodes)
    
    if max_nodes is not None and len(nodes) > max_nodes:
        # Keep center + top-degree neighbors first
        nodes_sorted = sorted(nodes, key=lambda n: G.degree[n], reverse=True)
        if center in nodes_sorted:
            nodes_sorted.remove(center)
        nodes = [center] + nodes_sorted[:max_nodes-1]
    
    return nodes


def importance_sample_centers(G, m, exclude=set(), bins=6, seed=0):
    """
    Sample m centers without replacement, stratified by degree bins.
    
    Args:
        G: NetworkX graph
        m: Number of centers to sample
        exclude: Set of nodes to exclude
        bins: Number of degree bins for stratification
        seed: Random seed
    
    Returns:
        List of sampled center nodes
    """
    rng = np.random.default_rng(seed)
    nodes = np.array([n for n in G.nodes() if n not in exclude])
    degrees = np.array([G.degree[n] for n in nodes])

    qs = np.quantile(degrees, np.linspace(0, 1, bins+1))
    strata = []
    for i in range(bins):
        lo, hi = qs[i], qs[i+1]
        mask = (degrees >= lo) & (degrees <= hi) if i == bins-1 else (degrees >= lo) & (degrees < hi)
        strata.append(nodes[mask])

    per = int(np.ceil(m / bins))
    chosen = []
    for s in strata:
        if len(chosen) >= m:
            break
        if len(s) == 0:
            continue
        k = min(per, m - len(chosen), len(s))
        chosen.extend(rng.choice(s, size=k, replace=False).tolist())

    if len(chosen) < m:
        remaining = np.array([n for n in nodes if n not in set(chosen)])
        k = m - len(chosen)
        if len(remaining) > 0:
            chosen.extend(rng.choice(remaining, size=min(k, len(remaining)), replace=False).tolist())

    return chosen[:m]

def _generate_subnets(G: nx.Graph, 
                    seed_nodes: List[int], 
                    max_nodes: int=60,
                    k: int = 2) -> List[nx.Graph]:
    subnets = []
    for s in seed_nodes: 
        subnets.append(get_ego_net_nodes_quvine(G, s, 
                                                k=k, 
                                                max_nodes=max_nodes))
        
    extra_centers = importance_sample_centers(G=G, 
                                            m=15, 
                                            exclude=set(seed_nodes))

# ============================================================================
# Calibration Functions
# ============================================================================

def calibrate_heat_kernel(
    L: Union[np.ndarray, sp.spmatrix],
    q_targets: List[Dict],
    t_grid: np.ndarray,
    node_to_idx: Dict[int, int],
    loss: str = 'l2'
) -> Tuple[float, float]:
    """
    Calibrate heat kernel time parameter by matching quantum walk targets.
    
    Fits the parameter t in g_t(L) = exp(-t*L) by minimizing the loss between
    heat kernel diffusion and quantum walk probability distributions on subnetworks.
    
    Args:
        L: Laplacian matrix of the full graph
        q_targets: List of quantum walk target distributions, each containing:
            - 'nodes': List of node IDs in subnetwork
            - 'center': Center node ID
            - 'pQ': Quantum walk probability distribution
        t_grid: Grid of time values to search over
        node_to_idx: Dictionary mapping node IDs to matrix indices
        loss: Loss function ('l2' or 'kl')
    
    Returns:
        Tuple of (best_loss, best_t)
    
    Example:
        >>> L = nx.laplacian_matrix(G).astype(float)
        >>> q_targets = [{'nodes': [0,1,2], 'center': 0, 'pQ': np.array([0.5, 0.3, 0.2])}]
        >>> t_grid = np.linspace(0.1, 5.0, 20)
        >>> node_to_idx = {i: i for i in range(G.number_of_nodes())}
        >>> loss_val, t_star = calibrate_heat_kernel(L, q_targets, t_grid, node_to_idx)
    """
    best_loss, best_t = np.inf, None
    N = L.shape[0]
    
    logger.info(f"Calibrating heat kernel over {len(t_grid)} time values...")
    
    for t in t_grid:
        tot = 0.0
        for item in q_targets:
            nodes = item['nodes']
            center = item['center']
            pQ = item['pQ']
            
            # Initialize with delta at center
            x = np.zeros(N)
            x[node_to_idx[center]] = 1.0
            
            # Compute heat kernel diffusion restricted to subnetwork
            yS = _expm_multiply_restricted(L, t, x, nodes, node_to_idx)
            yS = np.maximum(yS, 0)  # Ensure non-negative
            pT = yS / yS.sum() if yS.sum() > 0 else yS
            
            # Compute loss
            if loss == 'l2':
                tot += np.sum((pT - pQ) ** 2)
            elif loss == 'kl':
                eps = 1e-12
                tot += np.sum(pQ * (np.log(pQ + eps) - np.log(pT + eps)))
            else:
                raise ValueError(f"Unknown loss function: {loss}")
        
        if tot < best_loss:
            best_loss, best_t = tot, t
    
    if best_t is None:
        raise ValueError("No valid time parameter found in t_grid")
    
    logger.info(f"Best heat kernel time: t={best_t:.4f}, loss={best_loss:.6f}")
    return best_loss, best_t


def calibrate_polynomial_filter(
    L: Union[np.ndarray, sp.spmatrix],
    q_targets: List[Dict],
    node_to_idx: Dict[int, int],
    K: int = 4,
    ridge: float = 1e-6
) -> np.ndarray:
    """
    Calibrate polynomial filter coefficients by least squares.
    
    Fits coefficients {a_k} in g(L) = sum_{k=0}^K a_k * L^k by minimizing
    the squared error between polynomial filter output and quantum walk targets.
    
    Args:
        L: Laplacian matrix of the full graph
        q_targets: List of quantum walk target distributions (same format as calibrate_heat_kernel)
        node_to_idx: Dictionary mapping node IDs to matrix indices
        K: Polynomial degree (number of terms - 1)
        ridge: Ridge regularization parameter
    
    Returns:
        Array of polynomial coefficients [a_0, a_1, ..., a_K]
    
    Example:
        >>> coeffs = calibrate_polynomial_filter(L, q_targets, node_to_idx, K=4)
        >>> print(f"Polynomial coefficients: {coeffs}")
    """
    AtA = np.zeros((K + 1, K + 1))
    Atb = np.zeros(K + 1)
    N = L.shape[0]
    
    logger.info(f"Calibrating polynomial filter with degree K={K}...")
    
    for item in q_targets:
        nodes = item['nodes']
        center = item['center']
        pQ = item['pQ']
        
        # Initialize with delta at center
        x = np.zeros(N)
        x[node_to_idx[center]] = 1.0
        
        # Build basis: [x, L@x, L^2@x, ..., L^K@x]
        basis = []
        v = x.copy()
        node_indices = [node_to_idx[n] for n in nodes]
        basis.append(v[node_indices])
        
        for _ in range(1, K + 1):
            # Use @ (matmul) for both sparse and dense L.
            # scipy 1.14+ returns csr_array whose * operator is element-wise,
            # not matrix-vector; @ is always matrix multiplication.
            v = np.asarray(L @ v).ravel()
            basis.append(v[node_indices])
        
        # Stack basis vectors: Phi is |S| x (K+1)
        Phi = np.stack(basis, axis=1)
        #Phi = Phi / (np.linalg.norm(Phi, axis=0, keepdims=True) + 1e-12)  # Normalize columns
        b = pQ
        
        # Accumulate normal equations: A^T A and A^T b
        AtA += Phi.T @ Phi
        Atb += Phi.T @ b
    
    # Add ridge regularization
    AtA += ridge * np.eye(K + 1)
    
    # Solve for coefficients
    try:
        coeffs = np.linalg.solve(AtA, Atb)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(AtA, Atb, rcond=None)[0]
    
    # Validate coefficients - check if all are near zero
    if np.max(np.abs(coeffs)) < 1e-10:
        logger.warning("Degenerate polynomial coefficients (all near zero), using fallback")
        # Fallback: simple heat-like decay
        coeffs = np.array([1.0] + [0.5 ** (k+1) for k in range(K)])
    
    logger.info(f"Polynomial coefficients: {coeffs}")
    return coeffs


# ============================================================================
# Filter Application Functions
# ============================================================================

def apply_heat_filter(
    L: Union[np.ndarray, sp.spmatrix],
    X: np.ndarray,
    t: float
) -> np.ndarray:
    """
    Apply heat kernel filter: Z = exp(-t*L) @ X
    
    Args:
        L: Laplacian matrix
        X: Node features [N, d]
        t: Time parameter
    
    Returns:
        Filtered features [N, d]
    """
    logger.info(f"Applying heat kernel filter with t={t:.4f}...")
    
    if sp.issparse(L):
        # Use sparse matrix exponential
        Z = spla.expm_multiply(-t * L, X)
    else:
        # Dense case
        Z = expm(-t * L) @ X
    
    return np.asarray(Z)


def apply_polynomial_filter(
    L: Union[np.ndarray, sp.spmatrix],
    X: np.ndarray,
    coeffs: np.ndarray
) -> np.ndarray:
    """
    Apply polynomial filter: Z = sum_{k=0}^K a_k * L^k @ X
    
    Uses Horner's method for efficient computation.
    
    Args:
        L: Laplacian matrix
        X: Node features [N, d]
        coeffs: Polynomial coefficients [a_0, a_1, ..., a_K]
    
    Returns:
        Filtered features [N, d]
    """
    logger.info(f"Applying polynomial filter with {len(coeffs)} coefficients...")
    
    Z = coeffs[0] * X
    V = X.copy()
    
    for k in range(1, len(coeffs)):
        V = L @ V
        Z = Z + coeffs[k] * V
    
    return np.asarray(Z)


# ============================================================================
# Embedding Generation Functions
# ============================================================================

def generate_quvine_heat_embedding(
    G: nx.Graph,
    q_targets: List[Dict],
    t_grid: Optional[np.ndarray] = None,
    embedding_dim: int = 128,
    use_features: bool = False,
    features: Optional[np.ndarray] = None,
    normalize: bool = True,
    random_state: int = 42
) -> np.ndarray:
    """
    Generate QuVINE heat kernel embeddings.
    
    Workflow:
    1. Calibrate heat kernel time parameter using quantum walk targets
    2. Apply calibrated heat kernel to node features
    3. Return filtered features as embeddings
    
    Args:
        G: NetworkX graph
        q_targets: List of quantum walk target distributions
        t_grid: Grid of time values to search (default: np.linspace(0.1, 5.0, 20))
        embedding_dim: Embedding dimension (used if generating random features)
        use_features: Whether to use provided features or generate random ones
        features: Node features [N, d] (optional)
        normalize: Whether to normalize Laplacian
        random_state: Random seed
    
    Returns:
        Node embeddings [N, embedding_dim]
    """
    np.random.seed(random_state)
    N = G.number_of_nodes()
    
    # Get Laplacian
    L, nodelist, node_to_idx = get_laplacian(G, normalize=True)
    
    # Default time grid
    if t_grid is None:
        t_grid = np.logspace(-2, 2, 40)
    
    # Calibrate heat kernel
    _, t_star = calibrate_heat_kernel(L, q_targets, t_grid, node_to_idx, loss='l2')
    
    # Generate or use features
    if use_features and features is not None:
        X = features
    else:
        rng = np.random.default_rng(random_state)
        X = rng.normal(size=(N, embedding_dim)) 
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norm, 1e-12)
    
    # Apply heat kernel filter
    Z = apply_heat_filter(L, X, t_star)
    
    return Z


def generate_quvine_poly_embedding(
    G: nx.Graph,
    q_targets: List[Dict],
    K: int = 4,
    ridge: float = 1e-6,
    embedding_dim: int = 128,
    use_features: bool = False,
    features: Optional[np.ndarray] = None,
    normalize: bool = True,
    random_state: int = 42
) -> np.ndarray:
    """
    Generate QuVINE polynomial filter embeddings.
    
    Workflow:
    1. Calibrate polynomial filter coefficients using quantum walk targets
    2. Apply calibrated polynomial filter to node features
    3. Return filtered features as embeddings
    
    Args:
        G: NetworkX graph
        q_targets: List of quantum walk target distributions
        K: Polynomial degree
        ridge: Ridge regularization parameter
        embedding_dim: Embedding dimension (used if generating random features)
        use_features: Whether to use provided features or generate random ones
        features: Node features [N, d] (optional)
        normalize: Whether to normalize Laplacian
        random_state: Random seed
    
    Returns:
        Node embeddings [N, embedding_dim]
    """
    np.random.seed(random_state)
    N = G.number_of_nodes()
    
    # Get Laplacian - unpack the tuple!
    L, nodelist, node_to_idx = get_laplacian(G, normalize=True)
    
    # Calibrate polynomial filter
    coeffs = calibrate_polynomial_filter(L, q_targets, node_to_idx, K=K, ridge=ridge)
    
    # Generate or use features
    if use_features and features is not None:
        X = features
    else:
        X = np.random.randn(N, embedding_dim)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)  # Normalize rows
    
    # Apply polynomial filter
    Z = apply_polynomial_filter(L, X, coeffs)
    
    return Z


def generate_baseline_filter_embedding(
    G: nx.Graph,
    filter_type: str = 'heat',
    t: float = 1.0,
    K: int = 4,
    embedding_dim: int = 128,
    use_features: bool = False,
    features: Optional[np.ndarray] = None,
    normalize: bool = True,
    random_state: int = 42
) -> np.ndarray:
    """
    Generate baseline graph filter embeddings (without quantum calibration).
    
    This serves as a baseline to compare against QuVINE methods.
    
    Args:
        G: NetworkX graph
        filter_type: Type of filter ('heat' or 'poly')
        t: Time parameter for heat kernel (if filter_type='heat')
        K: Polynomial degree (if filter_type='poly')
        embedding_dim: Embedding dimension
        use_features: Whether to use provided features or generate random ones
        features: Node features [N, d] (optional)
        normalize: Whether to normalize Laplacian
        random_state: Random seed
    
    Returns:
        Node embeddings [N, embedding_dim]
    """
    np.random.seed(random_state)
    N = G.number_of_nodes()
    
    # Get Laplacian - unpack the tuple!
    L, nodelist, node_to_idx = get_laplacian(G, normalize=normalize)
    
    # Generate or use features
    if use_features and features is not None:
        X = features
    else:
        X = np.random.randn(N, embedding_dim)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)  # Normalize rows
    
    # Apply filter
    if filter_type == 'heat':
        logger.info(f"Generating baseline heat kernel embedding with t={t:.4f}")
        Z = apply_heat_filter(L, X, t)
    elif filter_type == 'poly':
        logger.info(f"Generating baseline polynomial embedding with K={K}")
        # Use simple coefficients: [1, -1, 0.5, -0.25, ...] (alternating signs, decreasing magnitude)
        coeffs = np.array([(-1)**k / (2**k) for k in range(K + 1)])
        Z = apply_polynomial_filter(L, X, coeffs)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    return Z


# =====================================================================
# Walk Feature Matrix + Named Embedding Functions
# =====================================================================

def _compute_walk_feature_matrix(
    G: nx.Graph,
    walk_type: str,
    embedding_dim: int = 128,
    random_state: int = 42,
    time: float = 1.0,
    steps: int = 8,
    restart_prob: float = 0.15,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute a walk-diffused feature matrix X of shape (N, embedding_dim).

    Starts from random unit-norm features and applies walk dynamics as a
    linear operator, giving each method a distinct inductive bias:
      - rwr : RWR diffusion via sparse solve  α*(I-(1-α)A_norm)^{-1}
      - ctqw: heat kernel on adjacency matrix  exp(-A*time)
      - dtqw: polynomial filter on Laplacian with 'steps' terms
    """
    rng = np.random.default_rng(random_state)
    N = G.number_of_nodes()
    X = rng.normal(size=(N, embedding_dim))
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

    nodelist = list(G.nodes())

    if walk_type == 'rwr':
        A = nx.to_scipy_sparse_array(G, nodelist=nodelist, format='csr', dtype=float)
        d = np.array(A.sum(axis=1)).flatten()
        d_inv = np.where(d > 0, 1.0 / d, 0.0)
        A_norm = sp.diags(d_inv) @ A
        M = sp.eye(N, format='csr') - (1.0 - restart_prob) * A_norm
        X = restart_prob * spla.spsolve(M, X)

    elif walk_type == 'ctqw':
        # Real-valued approximation of CTQW using adjacency-based heat kernel
        A = nx.to_scipy_sparse_array(G, nodelist=nodelist, format='csr', dtype=float)
        X = spla.expm_multiply(-time * A, X)

    elif walk_type == 'dtqw':
        # Polynomial approximation of DTQW using Laplacian with alternating coefficients.
        # Use numpy vectorised power to avoid Python float overflow when steps is large
        # (Python 2.0**k raises OverflowError for k >= 1024; numpy returns 0.0 safely).
        L, _, _ = get_laplacian(G, normalize=normalize)
        k_vals = np.arange(steps + 1, dtype=np.float64)
        coeffs = np.power(-0.5, k_vals)  # = (-1)**k / 2**k, but overflow-safe
        X = apply_polynomial_filter(L, X, coeffs)

    else:
        raise ValueError(f"Unknown walk_type: {walk_type!r}. Choose 'rwr', 'ctqw', or 'dtqw'.")

    return np.asarray(X)


def generate_baseline_heat_embedding(
    G: nx.Graph,
    embedding_dim: int = 128,
    scale: float = 1.0,
    normalize: bool = True,
    random_state: int = 42,
) -> np.ndarray:
    """Baseline heat-kernel embedding on random features (no quantum calibration)."""
    return generate_baseline_filter_embedding(
        G, filter_type='heat', t=scale,
        embedding_dim=embedding_dim, normalize=normalize, random_state=random_state,
    )


def generate_baseline_poly_embedding(
    G: nx.Graph,
    embedding_dim: int = 128,
    order: int = 4,
    normalize: bool = True,
    random_state: int = 42,
) -> np.ndarray:
    """Baseline polynomial-filter embedding on random features (no quantum calibration)."""
    return generate_baseline_filter_embedding(
        G, filter_type='poly', K=order,
        embedding_dim=embedding_dim, normalize=normalize, random_state=random_state,
    )


# =====================================================================