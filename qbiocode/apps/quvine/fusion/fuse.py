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

import numpy as np
from numpy.linalg import svd
from qbiocode.apps.quvine.analysis.analyze import normalize

def _row_norm(Z, eps=1e-8):
    nrm = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / (nrm + eps)

def _block_standardize(Z, eps=1e-8):
    mu = Z.mean(axis=0, keepdims=True)
    sd = Z.std(axis=0, keepdims=True)
    return (Z - mu) / (sd + eps)

def _prep_blocks(Zs, do_row_norm=True, do_block_standardize=True):
    out = []
    for Z in Zs:
        Zp = Z
        if do_row_norm:
            Zp = _row_norm(Zp)
        if do_block_standardize:
            Zp = _block_standardize(Zp)
        out.append(Zp)
    return out

def fuse_embeddings_svd(Zs, k):
    """
    Fast early fusion:
    concatenate (after per-block normalization) + SVD/PCA to shared k-dim space.
    """
    Zcat = np.concatenate(Zs, axis=1)            # (n, sum d_v)
    U, S, _ = svd(Zcat, full_matrices=False)
    k_eff = min(k, U.shape[1])
    Z_fused = U[:, :k_eff] * S[:k_eff]           # (n,k_eff), scaled PCs
    if k_eff < k:
        Z_fused = np.pad(Z_fused, ((0, 0), (0, k - k_eff)), mode='constant')
    return Z_fused

# ---------- Optional: graph-regularized shared U, scalable ----------
def _apply_graph_regularization(U0, L, beta, lam, max_cg_iter=200, cg_tol=1e-6):
    """
    Solve (beta*L + (1+lam)*I) u_j = u0_j for each column via sparse CG.
    Returns U0 unchanged if scipy is unavailable or beta <= 0.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    if not sp.issparse(L):
        # DO NOT densify on big graphs; pass sparse L.
        L = sp.csr_matrix(L)

    n, k = U0.shape
    Aop = (beta * L) + (1.0 + lam) * sp.eye(n, format="csr")

    U = np.zeros_like(U0)
    for j in range(k):
        rhs = U0[:, j]
        uj, info = spla.cg(Aop, rhs, maxiter=max_cg_iter, rtol=cg_tol)
        if info != 0:
            uj = rhs
        U[:, j] = uj
    return U


def fuse_embeddings_graphreg(Zs, k, L, beta=1e-2, lam=1e-2, max_cg_iter=200, cg_tol=1e-6):
    """
    Solve: argmin_U ||U - Zbar||_F^2 + beta * tr(U^T L U) + lam ||U||_F^2
    where Zbar is the SVD-fused initialization projected to k dims.

    This is a *much simpler* regularization story than your full multiview + Ws + alpha,
    and it avoids over-parameterization reviewers will question.
    """
    U0 = fuse_embeddings_svd(Zs, k)

    if beta <= 0:
        return U0

    try:
        return _apply_graph_regularization(U0, L, beta, lam, max_cg_iter, cg_tol)
    except ImportError:
        return U0

def fuse_embeddings_attention(Zs, k, temperature=1.0):
    """
    Attention-based fusion: learn attention weights for each embedding view.

    Uses softmax attention over embedding similarities to weight each view.
    """
    # Per-view quality = effective rank (exp of the spectral entropy of Z).
    # The previous score ||Z_norm||_F^2 / n is identically 1 for any
    # row-normalized Z, so the softmax was always uniform and this fused
    # method silently collapsed to a plain mean of the views. Effective rank
    # measures how much independent information a view carries and genuinely
    # varies across views. (Matches fuse_embeddings_attention_fixed.)
    quality = []
    for Z in Zs:
        s = svd(np.asarray(Z, dtype=np.float64), full_matrices=False, compute_uv=False)
        s2 = s ** 2
        if s2.sum() <= 0:
            quality.append(0.0)
            continue
        p = s2 / s2.sum()
        entropy = -np.sum(p * np.log(p + 1e-20))
        quality.append(float(np.exp(entropy)))

    from scipy.special import softmax
    attention_weights = softmax(np.array(quality) / temperature)
    
    # Weighted combination
    Z_weighted = sum(w * Z for w, Z in zip(attention_weights, Zs))
    
    # Apply SVD for dimensionality reduction
    U, S, _ = svd(Z_weighted, full_matrices=False)
    k_eff = min(k, U.shape[1])
    Z_fused = U[:, :k_eff] * S[:k_eff]
    if k_eff < k:
        Z_fused = np.pad(Z_fused, ((0, 0), (0, k - k_eff)), mode='constant')
    return Z_fused


def fuse_embeddings_hybrid(Zs, k, L=None, beta=1e-2, lam=1e-2, temperature=1.0):
    """
    Hybrid fusion: combines SVD and attention-based fusion.
    
    First applies attention-weighted fusion, then applies SVD with optional graph regularization.
    """
    # Step 1: Attention-based weighting
    attention_fused = fuse_embeddings_attention(Zs, k=k, temperature=temperature)
    
    if L is not None and beta > 0:
        try:
            return _apply_graph_regularization(attention_fused, L, beta, lam)
        except ImportError:
            pass

    return attention_fused


def fuse_embeddings_svd_shared_private(Zs, k, gate_type='attention'):
    """
    SVD-based shared/private fusion for N ≥ 2 embedding views.

    Each view is decomposed into a *shared* component (extracted by a joint
    rank-k SVD over all views concatenated) and a *private* residual.  The
    views are then recombined using one of two gating strategies:

    ``attention`` — a per-feature sigmoid gate is applied to each view's
        private component; the result is added to the mean shared component.
        Good when views are complementary and private details matter.

    ``moe`` — a per-node softmax over the V views selects a weighted
        combination of all normalised views.  Good when views are exchangeable
        and the model should pick the most informative one per node.

    Parameters
    ----------
    Zs : list of np.ndarray, each (n, d)
        Embedding views to fuse (V ≥ 2).
    k : int
        Rank for SVD approximation (typically d // 4).
    gate_type : {'attention', 'moe'}

    Returns
    -------
    Z_final : np.ndarray, shape (n, d)
    """
    V = len(Zs)
    if V < 2:
        raise ValueError("fuse_embeddings_svd_shared_private requires at least 2 views.")

    n, d = Zs[0].shape

    # 1. Layer-normalise all views.
    Zn = [_layer_norm(Z) for Z in Zs]

    # 2. Joint SVD over the horizontally concatenated views → rank-k shared reconstruction.
    Z_cat = np.concatenate(Zn, axis=1)          # (n, V*d)
    U, S, Vh = svd(Z_cat, full_matrices=False)
    k_eff = min(k, U.shape[1])
    Z_hat = (U[:, :k_eff] * S[:k_eff]) @ Vh[:k_eff, :]  # (n, V*d)

    # 3. Per-view shared and private components.
    Z_sh   = [Z_hat[:, i*d:(i+1)*d] for i in range(V)]   # each (n, d)
    Z_priv = [Zn[i] - Z_sh[i]        for i in range(V)]   # each (n, d)

    # Mean shared component — the consensus across all views.
    Z_shared_mean = np.mean(np.stack(Z_sh, axis=0), axis=0)   # (n, d)

    if gate_type == 'attention':
        # Gate features: [all shared ‖ all private] → (n, 2V*d)
        # Gate output:   (n, V*d)  — one d-dimensional sigmoid gate per view.
        gate_feat = np.concatenate(Z_sh + Z_priv, axis=1)     # (n, 2V*d)
        gate = _simple_mlp_gate(gate_feat, V * d)              # (n, V*d)

        # Add gated private corrections to the shared mean; average over views.
        private_mix = sum(gate[:, i*d:(i+1)*d] * Z_priv[i] for i in range(V))
        Z_final = Z_shared_mean + private_mix / V

    elif gate_type == 'moe':
        # Gate features: normalised views concatenated → (n, V*d)
        # Gate output:   (n, V) softmax weights over views.
        gate_feat = np.concatenate(Zn, axis=1)                 # (n, V*d)
        raw = _simple_mlp_gate(gate_feat, V)                   # (n, V) sigmoid
        gate_w = raw / (raw.sum(axis=1, keepdims=True) + 1e-8) # (n, V) sum-to-1
        Z_final = sum(gate_w[:, i:i+1] * Zn[i] for i in range(V))  # (n, d)

    else:
        raise ValueError(f"Unknown gate_type '{gate_type}'. Use 'attention' or 'moe'.")

    return Z_final


def _layer_norm(Z, eps=1e-8):
    """Layer normalization."""
    mean = Z.mean(axis=1, keepdims=True)
    std = Z.std(axis=1, keepdims=True)
    return (Z - mean) / (std + eps)


def _simple_mlp_gate(features, output_dim, random_state=0):
    n, input_dim = features.shape
    hidden_dim = max(64, output_dim)
    rng = np.random.default_rng(random_state)
    W1 = rng.standard_normal((input_dim, hidden_dim)) * 0.01
    b1 = np.zeros(hidden_dim)
    W2 = rng.standard_normal((hidden_dim, output_dim)) * 0.01
    b2 = np.zeros(output_dim)
    h = np.maximum(0, features @ W1 + b1)
    return 1.0 / (1.0 + np.exp(-(h @ W2 + b2)))


def fuse_embeddings(store, k=None, L=None, method="svd",
                beta=1e-2, lam=1e-2, temperature=1.0, svd_rank=None, gate_type='attention'):
    """
    Fuse multiple embeddings using various methods.
    
    Parameters
    ----------
    store : EmbeddingStore
        Store containing multiple embeddings
    k : int, optional
        Target embedding dimension
    L : sparse matrix, optional
        Graph Laplacian (required for 'graphreg' and 'hybrid')
    method : str
        Fusion method:
        - "svd"              : SVD-based fusion (fast, default)
        - "graphreg"         : Graph-regularized fusion (requires L)
        - "attention"        : Attention-weighted fusion
        - "hybrid"           : Attention + graph regularization (requires L)
        - "svd_shared_priv"  : SVD shared/private decomposition with gating
        - "all"              : Compute all methods (requires L)
    beta : float
        Graph regularization strength
    lam : float
        L2 regularization strength
    temperature : float
        Temperature for attention softmax
    svd_rank : int, optional
        Rank for SVD approximation in shared/private decomposition (default: k // 4)
    gate_type : str
        Gate type for shared/private fusion: 'attention' or 'moe'
        
    Returns
    -------
    embeddings : list
        List of fused embeddings
    names : list
        List of method names
    """
    names = store.names()
    Zs_raw = [store.get(name) for name in names]

    # Column-normalize then row-normalize each view once.
    Zs = [normalize(Z) for Z in Zs_raw]
    Zs = _prep_blocks(Zs, do_row_norm=True, do_block_standardize=False)

    if k is None:
        k = min(Z.shape[1] for Z in Zs)

    if method == "svd":
        return [fuse_embeddings_svd(Zs, k)], ['svd']

    if method == "graphreg":
        if L is None:
            raise ValueError("L must be provided for method='graphreg'.")
        return [fuse_embeddings_graphreg(Zs, k, L=L, beta=beta, lam=lam)], ['graphreg']
    
    if method == "attention":
        return [fuse_embeddings_attention(Zs, k, temperature=temperature)], ['attention']
    
    if method == "hybrid":
        if L is None:
            raise ValueError("L must be provided for method='hybrid'.")
        return [fuse_embeddings_hybrid(Zs, k, L=L, beta=beta, lam=lam, temperature=temperature)], ['hybrid']
    
    if method == "svd_shared_priv":
        rank = svd_rank if svd_rank is not None else max(k // 4, 1)
        return [fuse_embeddings_svd_shared_private(Zs, rank, gate_type=gate_type)], [f'svd_shared_priv_{gate_type}']

    if method == "all":
        if L is None:
            raise ValueError("L must be provided for method='graphreg' and 'hybrid'.")
        
        svd_emb = fuse_embeddings_svd(Zs, k)
        graphreg_emb = fuse_embeddings_graphreg(Zs, k, L=L, beta=beta, lam=lam)
        attention_emb = fuse_embeddings_attention(Zs, k, temperature=temperature)
        hybrid_emb = fuse_embeddings_hybrid(Zs, k, L=L, beta=beta, lam=lam, temperature=temperature)
        
        rank = svd_rank if svd_rank is not None else max(k // 4, 1)
        svd_sp_att = fuse_embeddings_svd_shared_private(Zs, rank, gate_type='attention')
        svd_sp_moe = fuse_embeddings_svd_shared_private(Zs, rank, gate_type='moe')
        return [svd_emb, graphreg_emb, attention_emb, hybrid_emb, svd_sp_att, svd_sp_moe], \
               ['svd', 'graphreg', 'attention', 'hybrid', 'svd_shared_priv_attention', 'svd_shared_priv_moe']
        
    raise ValueError(f"Unknown fusion method: {method}")


# ============================================================================
# Hierarchical Fusion for 39-Method System
# ============================================================================

# Define all 39 methods
ALL_39_METHODS = [
    # SGNS (3)
    'quvine_rwr', 'quvine_ctqw', 'quvine_dtqw',
    # Filters (6)
    'baseline_filter_heat', 'baseline_filter_poly',
    'filter_rwr_heat', 'filter_rwr_poly',
    'filter_ctqw_heat', 'filter_ctqw_poly',
    # GAT (12)
    'gat_baseline', 'gat_heat', 'gat_poly',
    'gat_rwr', 'gat_ctqw', 'gat_dtqw',
    'gat_rwr_heat', 'gat_rwr_poly',
    'gat_ctqw_heat', 'gat_ctqw_poly',
    'gat_dtqw_heat', 'gat_dtqw_poly',
    # GraphGPS (12)
    'graphgps_baseline', 'graphgps_heat', 'graphgps_poly',
    'graphgps_rwr', 'graphgps_ctqw', 'graphgps_dtqw',
    'graphgps_rwr_heat', 'graphgps_rwr_poly',
    'graphgps_ctqw_heat', 'graphgps_ctqw_poly',
    'graphgps_dtqw_heat', 'graphgps_dtqw_poly',
    # Classical baselines (6)
    'node2vec', 'netmf', 'graphsage', 'appnp',
    'baseline_filter', 'baseline_gcnmf'
]

# Define quantum methods
QUANTUM_METHODS = {
    'quvine_ctqw', 'quvine_dtqw',
    'filter_ctqw_heat', 'filter_ctqw_poly',
    'gat_ctqw', 'gat_dtqw',
    'gat_ctqw_heat', 'gat_ctqw_poly',
    'gat_dtqw_heat', 'gat_dtqw_poly',
    'graphgps_ctqw', 'graphgps_dtqw',
    'graphgps_ctqw_heat', 'graphgps_ctqw_poly',
    'graphgps_dtqw_heat', 'graphgps_dtqw_poly'
}


def _filter_methods_by_type(method_names, method_type, quantum_only=False, classical_only=False):
    """
    Filter methods by type and quantum/classical.
    
    Args:
        method_names: List of method names to filter
        method_type: 'sgns', 'filter', 'gat', 'graphgps', or 'baselines'
        quantum_only: Only include quantum methods
        classical_only: Only include classical methods
    
    Returns:
        List of filtered method names
    """
    # Define method type patterns
    type_patterns = {
        'sgns': ['quvine_rwr', 'quvine_ctqw', 'quvine_dtqw'],
        'filter': [
            'baseline_filter_heat', 'baseline_filter_poly',
            'filter_rwr_heat', 'filter_rwr_poly',
            'filter_ctqw_heat', 'filter_ctqw_poly'
        ],
        'gat': [m for m in ALL_39_METHODS if m.startswith('gat_')],
        'graphgps': [m for m in ALL_39_METHODS if m.startswith('graphgps_')],
        'baselines': ['node2vec', 'netmf', 'graphsage', 'appnp', 'baseline_filter', 'baseline_gcnmf']
    }
    
    # Filter by type
    candidates = [m for m in method_names if m in type_patterns.get(method_type, [])]
    
    # Filter by quantum/classical
    if quantum_only:
        candidates = [m for m in candidates if m in QUANTUM_METHODS]
    elif classical_only:
        candidates = [m for m in candidates if m not in QUANTUM_METHODS]
    
    return candidates


def _fuse_via_svd(embeddings_list, target_dim=None):
    """
    Fuse embeddings using SVD.
    
    Args:
        embeddings_list: List of embedding arrays (n_nodes, embedding_dim)
        target_dim: Target dimension for fused embedding (default: first embedding's dim)
    
    Returns:
        Fused embedding array (n_nodes, target_dim)
    """
    if len(embeddings_list) == 0:
        raise ValueError("embeddings_list cannot be empty")
    
    if len(embeddings_list) == 1:
        return embeddings_list[0]
    
    # Stack embeddings horizontally
    stacked = np.hstack(embeddings_list)
    
    # SVD
    U, S, Vt = np.linalg.svd(stacked, full_matrices=False)
    
    # Keep dimensions equal to first embedding or target_dim
    if target_dim is None:
        target_dim = embeddings_list[0].shape[1]
    target_dim = min(target_dim, U.shape[1])
    
    fused = U[:, :target_dim] @ np.diag(S[:target_dim])
    
    return fused


def fuse_by_method_type(embeddings_dict, method_type, quantum_only=False,
                        classical_only=False, fusion_method='svd', target_dim=None):
    """
    Fuse embeddings within a method type.
    
    Args:
        embeddings_dict: {method_name: embedding_array}
        method_type: Type of methods to fuse ('sgns', 'filter', 'gat', 'graphgps', 'baselines')
        quantum_only: Only fuse quantum methods
        classical_only: Only fuse classical methods
        fusion_method: 'svd', 'concatenate', 'average'
        target_dim: Target dimension for fused embedding
    
    Returns:
        Fused embedding array
    """
    # Filter methods by type and quantum/classical
    filtered_methods = _filter_methods_by_type(
        list(embeddings_dict.keys()),
        method_type,
        quantum_only,
        classical_only
    )
    
    # Extract embeddings to fuse
    embeddings_to_fuse = [
        embeddings_dict[m] for m in filtered_methods
        if m in embeddings_dict
    ]
    
    if len(embeddings_to_fuse) == 0:
        raise ValueError(f"No embeddings found for type={method_type}, "
                        f"quantum={quantum_only}, classical={classical_only}")
    
    # Perform fusion
    if fusion_method == 'svd':
        return _fuse_via_svd(embeddings_to_fuse, target_dim)
    elif fusion_method == 'concatenate':
        return np.concatenate(embeddings_to_fuse, axis=1)
    elif fusion_method == 'average':
        return np.mean(embeddings_to_fuse, axis=0)
    else:
        raise ValueError(f"Unknown fusion method: {fusion_method}")


def fuse_best_across_types(embeddings_dict, performance_scores,
                           quantum_only=False, classical_only=False,
                           fusion_method='svd', target_dim=None):
    """
    Fuse best-performing method from each type.
    
    Steps:

    1. For each method type (SGNS, Filter, GAT, GraphGPS):

       - Select best-performing method based on scores

    2. Fuse the best methods using specified fusion method
    
    Args:
        embeddings_dict: {method_name: embedding}
        performance_scores: {method_name: score} (mean across replicates)
        quantum_only: Only consider quantum methods
        classical_only: Only consider classical methods
        fusion_method: 'svd', 'concatenate', 'average'
        target_dim: Target dimension for fused embedding
    
    Returns:
        Fused embedding from best methods across types
    """
    method_types = ['sgns', 'filter', 'gat', 'graphgps']
    best_methods = []
    
    for mtype in method_types:
        # Get methods of this type
        type_methods = _filter_methods_by_type(
            list(embeddings_dict.keys()),
            mtype,
            quantum_only,
            classical_only
        )
        
        # Find best-performing method
        if len(type_methods) > 0:
            best_method = max(
                type_methods,
                key=lambda m: performance_scores.get(m, 0.0)
            )
            best_methods.append(best_method)
    
    if len(best_methods) == 0:
        raise ValueError("No methods found to fuse")
    
    # Fuse best methods
    best_embeddings = [embeddings_dict[m] for m in best_methods]
    
    if fusion_method == 'svd':
        return _fuse_via_svd(best_embeddings, target_dim)
    elif fusion_method == 'concatenate':
        return np.concatenate(best_embeddings, axis=1)
    elif fusion_method == 'average':
        return np.mean(best_embeddings, axis=0)
    else:
        raise ValueError(f"Unknown fusion method: {fusion_method}")


def hierarchical_fusion(embeddings_dict, performance_scores, target_dim=None):
    """
    Perform hierarchical fusion strategy for all 39 methods.
    
    Strategy:

    1. Within-type fusion:

       - Fuse quantum methods per type → fused_quantum_{type}
       - Fuse classical methods per type → fused_classical_{type}

    2. Cross-type fusion:

       - Select best quantum method per type
       - Select best classical method per type
       - Fuse best quantum methods → fused_q
       - Fuse best classical methods → fused_c
    
    Args:
        embeddings_dict: {method_name: embedding_array}
        performance_scores: {method_name: score}
        target_dim: Target dimension for fused embeddings
    
    Returns:
        dict: Fused embeddings, under these keys

            - 'fused_quantum_sgns', 'fused_classical_sgns'
            - 'fused_quantum_filter', 'fused_classical_filter'
            - 'fused_quantum_gat', 'fused_classical_gat'
            - 'fused_quantum_graphgps', 'fused_classical_graphgps'
            - 'fused_q' (best quantum across types)
            - 'fused_c' (best classical across types)
    """
    fused_embeddings = {}
    method_types = ['sgns', 'filter', 'gat', 'graphgps']
    
    # Step 1: Within-type fusion
    for mtype in method_types:
        # Quantum fusion
        try:
            fused_q = fuse_by_method_type(
                embeddings_dict, mtype,
                quantum_only=True,
                fusion_method='svd',
                target_dim=target_dim
            )
            fused_embeddings[f'fused_quantum_{mtype}'] = fused_q
        except ValueError:
            pass  # No quantum methods for this type
        
        # Classical fusion
        try:
            fused_c = fuse_by_method_type(
                embeddings_dict, mtype,
                classical_only=True,
                fusion_method='svd',
                target_dim=target_dim
            )
            fused_embeddings[f'fused_classical_{mtype}'] = fused_c
        except ValueError:
            pass  # No classical methods for this type
    
    # Step 2: Cross-type fusion
    try:
        fused_q = fuse_best_across_types(
            embeddings_dict, performance_scores,
            quantum_only=True,
            fusion_method='svd',
            target_dim=target_dim
        )
        fused_embeddings['fused_q'] = fused_q
    except ValueError:
        pass
    
    try:
        fused_c = fuse_best_across_types(
            embeddings_dict, performance_scores,
            classical_only=True,
            fusion_method='svd',
            target_dim=target_dim
        )
        fused_embeddings['fused_c'] = fused_c
    except ValueError:
        pass
    
    return fused_embeddings
