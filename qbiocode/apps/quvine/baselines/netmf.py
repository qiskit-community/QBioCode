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
NetMF: Network Embedding as Matrix Factorization

Implementation based on:
Qiu, J., Dong, Y., Ma, H., Li, J., Wang, K., & Tang, J. (2018).
Network embedding as matrix factorization: Unifying deepwalk, line, pte, and node2vec.
In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining (pp. 459-467).

Reference: https://github.com/xptree/NetMF
"""

import numpy as np
import networkx as nx
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
from typing import List, Optional


def run_netmf(
    graph: nx.Graph,
    nodes: List,
    dimensions: int = 128,
    window_size: int = 10,
    negative: int = 1,
    rank: Optional[int] = None,
    use_svd: bool = True,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Run NetMF (Network Embedding as Matrix Factorization) on a graph.
    
    NetMF provides a closed-form solution for network embedding by computing
    the matrix factorization of a modified adjacency matrix that captures
    higher-order proximity.
    
    Parameters
    ----------
    graph : nx.Graph
        Input graph
    nodes : list
        List of nodes (for ordering)
    dimensions : int, default=128
        Embedding dimension
    window_size : int, default=10
        Context window size (similar to DeepWalk/Node2Vec)
    negative : int, default=1
        Number of negative samples (affects the matrix transformation)
    rank : int, optional
        Rank for SVD approximation. If None, uses dimensions.
    use_svd : bool, default=True
        Whether to use SVD (True) or eigendecomposition (False)
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Node embeddings matrix of shape (n_nodes, dimensions)
        
    Notes
    -----
    NetMF computes embeddings by:

    1. Computing the transition matrix P from the adjacency matrix
    2. Computing the volume (sum of degrees)
    3. Computing the DeepWalk matrix: log(vol(G) * (sum_{r=1}^T P^r) / T / b) - log(b)
       where b is the number of negative samples
    4. Performing SVD/eigendecomposition to get the embedding
    
    References
    ----------
    Qiu et al. (2018). Network embedding as matrix factorization: Unifying 
    deepwalk, line, pte, and node2vec. WSDM 2018.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create node to index mapping
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    n_nodes = len(nodes)
    
    # Build adjacency matrix
    A = nx.to_scipy_sparse_array(graph, nodelist=nodes, format='csr', dtype=np.float64)
    
    # Compute degree matrix
    degrees = np.array(A.sum(axis=1)).flatten()
    
    # Avoid division by zero
    degrees[degrees == 0] = 1
    
    # Compute transition matrix P = D^{-1} A
    D_inv = sparse.diags(1.0 / degrees, format='csr')
    P = D_inv @ A
    
    # Compute volume (sum of all degrees)
    vol = degrees.sum()
    
    # Compute the sum of powers of P: sum_{r=1}^{window_size} P^r
    # This captures multi-hop proximity
    M = sparse.csr_matrix((n_nodes, n_nodes), dtype=np.float64)
    P_power = P.copy()
    
    for r in range(1, window_size + 1):
        M = M + P_power
        if r < window_size:
            P_power = P_power @ P
    
    # Average over window size
    M = M / window_size
    
    # Compute the DeepWalk matrix
    # M_dw = log(vol * M / b) - log(b)
    # where b is the number of negative samples
    M = M.multiply(vol / negative)
    
    # Convert to dense for log operation (only for non-zero elements)
    M_dense = M.toarray()
    
    # Apply log transformation (avoid log(0))
    M_dense[M_dense > 0] = np.log(M_dense[M_dense > 0])
    M_dense = M_dense - np.log(negative)
    
    # Set negative values to 0 (as in the original NetMF paper)
    M_dense[M_dense < 0] = 0
    
    # Perform matrix factorization
    if rank is None:
        rank = min(dimensions, n_nodes - 1)
    
    if use_svd:
        # Use SVD for factorization
        try:
            # Use sparse SVD if matrix is large
            if n_nodes > 500:
                U, S, Vt = sparse_linalg.svds(sparse.csr_matrix(M_dense), k=rank)
                # svds returns singular values in ascending order, reverse them
                U = U[:, ::-1]
                S = S[::-1]
            else:
                U, S, Vt = np.linalg.svd(M_dense, full_matrices=False)
                U = U[:, :rank]
                S = S[:rank]
            
            # Embedding is U * sqrt(S)
            embeddings = U @ np.diag(np.sqrt(S))
            
        except (np.linalg.LinAlgError, sparse_linalg.ArpackError, ValueError) as exc:
            # Do NOT fall back to random noise. This function previously returned
            # np.random.randn(...) here, so a failed factorization was reported to
            # the caller as a successful embedding and scored as if it were one.
            # run_netmf is reached through baselines.registry, which catches this
            # and records the method as failed with the reason attached -- which is
            # the honest outcome for "NetMF could not factorize this graph".
            raise RuntimeError(
                f"NetMF truncated SVD failed on a {n_nodes}-node graph at rank "
                f"{rank}: {exc}. Try a smaller `dimensions`, or a graph with more "
                f"than {rank + 1} nodes."
            ) from exc
    else:
        # Use eigendecomposition (symmetric approximation)
        M_sym = (M_dense + M_dense.T) / 2
        try:
            if n_nodes > 500:
                eigenvalues, eigenvectors = sparse_linalg.eigsh(
                    sparse.csr_matrix(M_sym), k=rank, which='LA'
                )
                # eigsh returns eigenvalues in ascending order, reverse them
                eigenvalues = eigenvalues[::-1]
                eigenvectors = eigenvectors[:, ::-1]
            else:
                eigenvalues, eigenvectors = np.linalg.eigh(M_sym)
                # Take top k eigenvalues
                idx = eigenvalues.argsort()[::-1][:rank]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
            
            # Embedding is eigenvectors * sqrt(eigenvalues)
            eigenvalues[eigenvalues < 0] = 0  # Ensure non-negative
            embeddings = eigenvectors @ np.diag(np.sqrt(eigenvalues))
            
        except (np.linalg.LinAlgError, sparse_linalg.ArpackError, ValueError) as exc:
            # See the SVD branch above: a random-noise embedding is not a result.
            raise RuntimeError(
                f"NetMF eigendecomposition failed on a {n_nodes}-node graph at "
                f"rank {rank}: {exc}. Try a smaller `dimensions`, or a graph with "
                f"more than {rank + 1} nodes."
            ) from exc
    
    # Ensure we have the right number of dimensions
    if embeddings.shape[1] < dimensions:
        # Pad with zeros if needed
        padding = np.zeros((n_nodes, dimensions - embeddings.shape[1]))
        embeddings = np.hstack([embeddings, padding])
    elif embeddings.shape[1] > dimensions:
        # Truncate if needed
        embeddings = embeddings[:, :dimensions]
    
    return embeddings
