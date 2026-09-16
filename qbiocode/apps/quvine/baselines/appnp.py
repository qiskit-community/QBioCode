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
APPNP: Approximate Personalized Propagation of Neural Predictions

Implementation of APPNP (Predict then Propagate) for node embedding generation.
APPNP combines neural network predictions with personalized PageRank propagation.

Reference:

- Klicpera et al. (2019). Predict then Propagate: Graph Neural Networks meet Personalized PageRank
  https://arxiv.org/abs/1810.05997
- Original implementation: https://github.com/gasteigerjo/ppnp

Key Innovation:

- Decouples feature transformation from propagation
- Uses personalized PageRank for efficient propagation
- More robust to oversmoothing than standard GCNs
"""

import logging
import numpy as np
import scipy.sparse as sp
import networkx as nx
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Import PyTorch utilities for better error handling
try:
    from ..utils.torch_utils import (
        check_torch_available,
        require_torch,
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
    
    def require_torch(func):
        def wrapper(*args, **kwargs):
            if not check_torch_available():
                raise ImportError(
                    "PyTorch is required for APPNP. Install with: pip install torch\n"
                    "See: https://pytorch.org/get-started/locally/"
                )
            return func(*args, **kwargs)
        return wrapper
    
    class TorchNotAvailableError(ImportError):
        pass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    optim = None
    TORCH_AVAILABLE = False


def normalize_adjacency(adj: sp.spmatrix, add_self_loops: bool = True) -> sp.spmatrix:
    """
    Normalize adjacency matrix for APPNP propagation.
    
    Computes: D^(-1/2) @ (A + I) @ D^(-1/2) if add_self_loops=True
              D^(-1/2) @ A @ D^(-1/2) otherwise
    
    Args:
        adj: Adjacency matrix (sparse)
        add_self_loops: Whether to add self-loops
    
    Returns:
        Normalized adjacency matrix
    """
    if add_self_loops:
        adj = adj + sp.eye(adj.shape[0], format='csr')
    
    # Compute degree matrix
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    # Normalize: D^(-1/2) @ A @ D^(-1/2)
    adj_normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    
    return adj_normalized.tocsr()


def personalized_pagerank_propagation(
    features: np.ndarray,
    adj_normalized: Union[sp.spmatrix, np.ndarray],
    alpha: float = 0.1,
    K: int = 10,
    use_sparse: bool = True
) -> np.ndarray:
    """
    Apply personalized PageRank propagation to features.
    
    Computes: Z = (1-alpha) * sum_{k=0}^{K-1} alpha^k * A^k @ X
    
    This is the power iteration approximation of:
    Z = (1-alpha) * (I - alpha*A)^(-1) @ X
    
    Args:
        features: Input features [N, d]
        adj_normalized: Normalized adjacency matrix [N, N]
        alpha: Teleport probability (1-alpha is restart probability)
        K: Number of propagation steps
        use_sparse: Whether to use sparse matrix operations
    
    Returns:
        Propagated features [N, d]
    """
    N, d = features.shape
    
    # Initialize with input features
    Z = (1 - alpha) * features
    H = features.copy()
    
    # Power iteration
    for k in range(K):
        if use_sparse and sp.issparse(adj_normalized):
            H = adj_normalized @ H
        else:
            H = adj_normalized @ H
        Z = Z + (1 - alpha) * (alpha ** (k + 1)) * H
    
    return Z


class MLPPredictor(nn.Module):
    """
    Multi-layer perceptron for feature transformation (Predict step).
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_layers: int = 2, dropout: float = 0.5):
        """
        Initialize MLP predictor.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            n_layers: Number of layers
            dropout: Dropout rate
        """
        super(MLPPredictor, self).__init__()
        
        self.n_layers = n_layers
        self.dropout = dropout
        
        # Build layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(n_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output layer
        if n_layers > 1:
            self.layers.append(nn.Linear(hidden_dim, output_dim))
        else:
            # Single layer case
            self.layers[0] = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        """
        Forward pass through MLP.
        
        Args:
            x: Input features [N, input_dim]
        
        Returns:
            Transformed features [N, output_dim]
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer (no activation)
        x = self.layers[-1](x)
        
        return x


def generate_appnp_embedding(
    G: nx.Graph,
    embedding_dim: int = 128,
    hidden_dim: int = 64,
    n_layers: int = 2,
    alpha: float = 0.1,
    K: int = 10,
    dropout: float = 0.5,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    use_features: bool = False,
    features: Optional[np.ndarray] = None,
    random_state: int = 42,
    device: str = "cpu",
) -> np.ndarray:
    """
    Generate node embeddings using APPNP.
    
    Workflow:
    1. Initialize random features (or use provided features)
    2. Train MLP to transform features (Predict step)
    3. Apply personalized PageRank propagation (Propagate step)
    4. Return final embeddings
    
    Args:
        G: NetworkX graph
        embedding_dim: Output embedding dimension
        hidden_dim: Hidden layer dimension for MLP
        n_layers: Number of MLP layers
        alpha: Teleport probability for PageRank (typically 0.1-0.2)
        K: Number of propagation steps (typically 10)
        dropout: Dropout rate
        lr: Learning rate
        weight_decay: L2 regularization
        epochs: Number of training epochs
        use_features: Whether to use provided features
        features: Node features [N, d] (optional)
        random_state: Random seed
    
    Returns:
        Node embeddings [N, embedding_dim]
    
    Raises:
        ImportError: If PyTorch is not installed
    """
    # Convert parameters to correct types (in case they come from config as strings)
    lr = float(lr)
    weight_decay = float(weight_decay)
    dropout = float(dropout)
    alpha = float(alpha)
    
    if not TORCH_AVAILABLE:
        raise TorchNotAvailableError(
            "PyTorch is required for APPNP embedding generation.\n"
            "Install PyTorch with: pip install torch\n"
            "See installation guide: https://pytorch.org/get-started/locally/\n"
            "For CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )
    
    # Set random seeds
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_state)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if device == "auto" else torch.device(device)

    N = G.number_of_nodes()
    node_list = list(G.nodes())

    # Get adjacency matrix and normalize
    adj = nx.to_scipy_sparse_array(G, nodelist=node_list, format='csr')
    adj_normalized = normalize_adjacency(adj, add_self_loops=True)

    # Convert to torch sparse tensor
    adj_normalized_coo = adj_normalized.tocoo()
    indices = torch.LongTensor(np.vstack([adj_normalized_coo.row, adj_normalized_coo.col])).to(dev)
    values = torch.FloatTensor(adj_normalized_coo.data).to(dev)
    shape = adj_normalized_coo.shape
    adj_torch = torch.sparse_coo_tensor(indices, values, shape, device=dev)

    # Generate or use features
    if use_features and features is not None:
        X = features
        input_dim = X.shape[1]
    else:
        # Generate random features
        input_dim = hidden_dim
        X = np.random.randn(N, input_dim)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)  # Normalize

    X_torch = torch.FloatTensor(X).to(dev)

    # Build dense normalized adjacency once so propagation remains differentiable
    adj_dense = torch.FloatTensor(adj_normalized.toarray()).to(dev)

    # Initialize MLP predictor
    model = MLPPredictor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=embedding_dim,
        n_layers=n_layers,
        dropout=dropout
    ).to(dev)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Sample edges for link reconstruction loss
    edges = list(G.edges())
    num_edges = len(edges)
    num_neg_samples = min(num_edges, N * 2)  # Limit negative samples

    # Create edge index tensor
    edge_src = [node_list.index(u) for u, v in edges]
    edge_dst = [node_list.index(v) for u, v in edges]
    pos_edges = torch.LongTensor([edge_src, edge_dst]).to(dev)

    # Training loop (unsupervised: link reconstruction)
    logger.info(f"Training APPNP for {epochs} epochs...")
    model.train()

    best_loss = float('inf')
    patience_counter = 0
    patience = 20

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Predict step: transform features
        H = model(X_torch)

        # Propagate step: differentiable personalized PageRank power iteration
        Z = (1 - alpha) * H
        H_prop = H
        for k in range(K):
            H_prop = adj_dense @ H_prop
            Z = Z + (1 - alpha) * (alpha ** (k + 1)) * H_prop

        # Link reconstruction loss: predict edges using dot product
        pos_src_emb = Z[pos_edges[0]]
        pos_dst_emb = Z[pos_edges[1]]
        pos_scores = (pos_src_emb * pos_dst_emb).sum(dim=1)

        # Sample negative edges
        neg_src = torch.randint(0, N, (num_neg_samples,), device=dev)
        neg_dst = torch.randint(0, N, (num_neg_samples,), device=dev)
        neg_src_emb = Z[neg_src]
        neg_dst_emb = Z[neg_dst]
        neg_scores = (neg_src_emb * neg_dst_emb).sum(dim=1)

        # Binary cross-entropy loss
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
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
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 50 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Generate final embeddings
    model.eval()
    with torch.no_grad():
        H = model(X_torch)
        H_np = H.cpu().numpy()
        Z_final = personalized_pagerank_propagation(
            H_np, adj_normalized, alpha=alpha, K=K, use_sparse=True
        )
    
    logger.info(f"APPNP embedding generation complete. Shape: {Z_final.shape}")
    
    return Z_final


def run_appnp(
    graph: nx.Graph,
    nodes: list,
    dimensions: int = 64,
    hidden_dim: int = 64,
    n_layers: int = 2,
    alpha: float = 0.1,
    K: int = 10,
    dropout: float = 0.5,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    seed: Optional[int] = None,
    device: str = "cpu",
) -> np.ndarray:
    """
    Run APPNP and return embeddings aligned to `nodes`.
    
    This is a wrapper function compatible with the existing baseline interface.
    
    Parameters
    ----------
    graph : networkx.Graph
        Input graph
    nodes : List[node]
        Canonical node ordering
    dimensions : int
        Embedding dimension
    hidden_dim : int
        Hidden layer dimension
    n_layers : int
        Number of MLP layers
    alpha : float
        Teleport probability (0.1-0.2 recommended)
    K : int
        Number of propagation steps
    dropout : float
        Dropout rate
    lr : float
        Learning rate
    weight_decay : float
        L2 regularization
    epochs : int
        Training epochs
    seed : int, optional
        Random seed
    
    Returns
    -------
    np.ndarray
        Node embeddings [N, dimensions]
    """
    if seed is None:
        seed = 42
    
    embeddings = generate_appnp_embedding(
        G=graph,
        embedding_dim=dimensions,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        alpha=alpha,
        K=K,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        random_state=seed,
        device=device,
    )
    
    return embeddings

