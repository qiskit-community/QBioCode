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
GCN-MF: Graph Convolutional Network with Matrix Factorization

A baseline method for disease gene prioritization that combines:
1. Graph Convolutional Networks (GCN) for learning node representations
2. Matrix Factorization for capturing latent features

Reference:
- Kipf & Welling (2017). Semi-Supervised Classification with Graph Convolutional Networks
- Matrix factorization approaches for biological network analysis
"""

import logging
import numpy as np
import scipy.sparse as sp
import networkx as nx

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False


class GCNLayer(nn.Module):
    """Single Graph Convolutional Layer."""
    
    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Forward pass.
        
        Args:
            x: Node features [N, in_features]
            adj: Normalized adjacency matrix [N, N] (sparse or dense)
        
        Returns:
            Output features [N, out_features]
        """
        support = torch.mm(x, self.weight)

        if getattr(adj, "is_sparse", False):
            output = torch.sparse.mm(adj, support)
        else:
            output = torch.mm(adj, support)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class GCNMF(nn.Module):
    """
    GCN-MF: Graph Convolutional Network with Matrix Factorization.
    
    Architecture:
    1. GCN layers for learning graph-based representations
    2. Matrix factorization component for latent features
    3. Fusion of GCN and MF representations
    """
    
    def __init__(self, n_nodes, input_dim, hidden_dim, output_dim, 
                 mf_dim=64, n_layers=2, dropout=0.5):
        """
        Initialize GCN-MF model.
        
        Args:
            n_nodes: Number of nodes in the graph
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (number of classes)
            mf_dim: Matrix factorization embedding dimension
            n_layers: Number of GCN layers
            dropout: Dropout rate
        """
        super(GCNMF, self).__init__()
        
        self.n_nodes = n_nodes
        self.mf_dim = mf_dim
        self.dropout = dropout
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNLayer(input_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.gcn_layers.append(GCNLayer(hidden_dim, hidden_dim))
        
        # Matrix factorization embeddings
        self.node_embeddings = nn.Parameter(torch.FloatTensor(n_nodes, mf_dim))
        nn.init.xavier_uniform_(self.node_embeddings)
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_dim + mf_dim, hidden_dim)
        
        # Output layer
        self.output = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, adj, node_indices=None):
        """
        Forward pass.
        
        Args:
            x: Node features [N, input_dim]
            adj: Normalized adjacency matrix [N, N]
            node_indices: Optional node indices for batch processing
        
        Returns:
            Output predictions [N, output_dim]
        """
        # GCN forward pass (computes for all N nodes)
        h = x
        for i, gcn_layer in enumerate(self.gcn_layers):
            h = gcn_layer(h, adj)
            if i < len(self.gcn_layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Slice both h and MF embeddings to keep shapes consistent
        if node_indices is not None:
            h = h[node_indices]
            mf_emb = self.node_embeddings[node_indices]
        else:
            mf_emb = self.node_embeddings
        
        # Fuse GCN and MF representations
        combined = torch.cat([h, mf_emb], dim=1)
        fused = F.relu(self.fusion(combined))
        fused = F.dropout(fused, p=self.dropout, training=self.training)
        
        # Output
        out = self.output(fused)
        
        return out
    
    def get_embeddings(self, x, adj):
        """
        Get node embeddings (before classification layer).
        
        Args:
            x: Node features
            adj: Normalized adjacency matrix
        
        Returns:
            Node embeddings
        """
        # GCN forward pass
        h = x
        for i, gcn_layer in enumerate(self.gcn_layers):
            h = gcn_layer(h, adj)
            if i < len(self.gcn_layers) - 1:
                h = F.relu(h)
        
        # Fuse with MF embeddings
        combined = torch.cat([h, self.node_embeddings], dim=1)
        fused = F.relu(self.fusion(combined))
        
        return fused


def normalize_adjacency(adj):
    """
    Normalize adjacency matrix: D^(-1/2) * A * D^(-1/2)
    
    Args:
        adj: Adjacency matrix (scipy sparse or numpy array)
    
    Returns:
        Normalized adjacency matrix
    """
    if isinstance(adj, np.ndarray):
        adj = sp.csr_matrix(adj)
    
    # Add self-loops
    adj = adj + sp.eye(adj.shape[0])
    
    # Compute D^(-1/2)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    # Normalize: D^(-1/2) * A * D^(-1/2)
    adj_normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    
    return adj_normalized




def precompute_quantum_diffusion(X, L, diffusion_type='heat', t_star=None, poly_coeffs=None):
    """
    Precompute quantum-calibrated diffusion OFFLINE (before training).
    
    This function should be called ONCE before training to compute diffused features.
    The result is then passed to QuVINEGCNMF during training.
    
    Args:
        X: Original node features [N, input_dim] (numpy array or torch tensor)
        L: Laplacian matrix (scipy sparse or numpy array)
        diffusion_type: 'heat' or 'poly'
        t_star: Calibrated heat kernel time parameter (for heat diffusion)
        poly_coeffs: Calibrated polynomial coefficients (for polynomial diffusion)
    
    Returns:
        X_diffused: Diffused features [N, input_dim] (torch tensor)
    
    Example:
        >>> # Offline: Compute diffusion once
        >>> X_diffused_heat = precompute_quantum_diffusion(X, L, 'heat', t_star=0.5)
        >>> X_diffused_poly = precompute_quantum_diffusion(X, L, 'poly', poly_coeffs=coeffs)
        >>>
        >>> # Online: Use in training (no diffusion computation)
        >>> model = QuVINEGCNMF(n_nodes, input_dim, hidden_dim, output_dim)
        >>> output = model(X_diffused_heat, adj)  # Fast!
    """
    import scipy.sparse.linalg as spla

    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for precompute_quantum_diffusion. Install with: pip install torch")
    
    # Convert to numpy if needed
    if isinstance(X, torch.Tensor):
        X_np = X.detach().cpu().numpy()
    else:
        X_np = np.array(X)
    
    if diffusion_type == 'heat' and t_star is not None:
        logger.info(f"Precomputing heat kernel diffusion (t={t_star})...")
        Z_np = spla.expm_multiply((-t_star) * L, X_np)

    elif diffusion_type == 'poly' and poly_coeffs is not None:
        logger.info(f"Precomputing polynomial diffusion (degree={len(poly_coeffs)-1})...")
        from qbiocode.apps.quvine.embedding.quantum_filters import apply_polynomial_filter
        Z_np = apply_polynomial_filter(L, X_np, poly_coeffs)
    else:
        logger.info("No diffusion applied (identity)")
        Z_np = X_np

    logger.info("Diffusion precomputed successfully")
    return torch.from_numpy(Z_np).float()


class QuVINEGCNMF(GCNMF):
    """
    QuVINE GCN-MF: GCN-MF with precomputed quantum-calibrated features.
    
    DESIGN PHILOSOPHY:
    - Diffusion is computed ONCE offline (before training)
    - Diffused features are passed as input
    - No CPU conversion or diffusion computation during training
    - Scalable and efficient for large graphs
    
    This model expects precomputed diffused features as input, combining:
    1. Precomputed quantum-calibrated diffusion (offline)
    2. GCN layers for learning representations (online)
    3. Matrix factorization for latent features (online)
    """
    
    def __init__(self, n_nodes, input_dim, hidden_dim, output_dim,
                mf_dim=64, n_layers=2, dropout=0.5):
        """
        Initialize QuVINE GCN-MF model.
        
        Args:
            n_nodes: Number of nodes
            input_dim: Input feature dimension (should match diffused features)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            mf_dim: Matrix factorization dimension
            n_layers: Number of GCN layers
            dropout: Dropout rate
        
        Note:
            This model expects precomputed diffused features as input.
            Compute diffusion offline using::

                X_diffused = apply_quantum_diffusion(X, L, t_star, poly_coeffs)

            Then pass X_diffused to forward().
        """
        super().__init__(n_nodes, input_dim, hidden_dim, output_dim,
                        mf_dim, n_layers, dropout)
    
    def forward(self, x, adj, node_indices=None):
        """
        Forward pass with precomputed diffused features.
        
        Args:
            x: PRECOMPUTED diffused features [N, input_dim]
               (computed offline before training)
            adj: Normalized adjacency matrix [N, N]
            node_indices: Optional node indices
        
        Returns:
            Output predictions [N, output_dim]
        
        Note:
            x should be the result of offline diffusion:
            x = exp(-t*L) @ X_original  (heat kernel)
            or
            x = polynomial_filter(L, coeffs) @ X_original  (polynomial)
        """
        # Simply use the precomputed diffused features
        # No diffusion computation here - it's done offline!
        return super().forward(x, adj, node_indices)
    
    def get_embeddings(self, x, adj):
        """
        Get node embeddings from precomputed diffused features.
        
        Args:
            x: PRECOMPUTED diffused features [N, input_dim]
               (computed offline before training)
            adj: Normalized adjacency matrix
        
        Returns:
            Node embeddings
        
        Note:
            x should already be diffused features from offline computation.
            No diffusion is applied here - it's done offline!
        """
        # x is already precomputed diffused features
        return super().get_embeddings(x, adj)



def train_gcn_mf(model, x, y, adj, train_mask, val_mask, 
                epochs=200, lr=0.01, weight_decay=5e-4, patience=20):
    """
    Train GCN-MF model.
    
    Args:
        model: GCN-MF model
        x: Node features
        y: Labels
        adj: Normalized adjacency matrix
        train_mask: Training mask
        val_mask: Validation mask
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for regularization
        patience: Early stopping patience
    
    Returns:
        Dictionary with training history
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'best_val_acc': 0.0,
        'best_epoch': 0
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        out = model(x, adj)
        loss = criterion(out[train_mask], y[train_mask])
        
        loss.backward()
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            out = model(x, adj)
            pred = out.argmax(dim=1)
            
            train_acc = (pred[train_mask] == y[train_mask]).float().mean().item()
            val_acc = (pred[val_mask] == y[val_mask]).float().mean().item()
        
        history['train_loss'].append(loss.item())
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history['best_val_acc'] = best_val_acc
            history['best_epoch'] = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch {epoch+1}/{epochs}: Loss={loss.item():.4f}, '
                  f'Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}')
    
    return history


def generate_quvine_gcnmf_embedding(G, q_targets, embedding_dim=128,
                                    diffusion_type='heat', t_star=None,
                                    poly_coeffs=None, K=4, ridge=1e-6,
                                    hidden_dim=64, mf_dim=64, n_layers=2,
                                    epochs=200, lr=0.01, weight_decay=5e-4,
                                    normalize_laplacian=True, random_state=42):
    """
    Generate QuVINE GCN-MF embeddings (Heat or Poly variant).
    
    WORKFLOW:
    1. Calibrate diffusion parameters using quantum walks (if not provided)
    2. Precompute diffused features OFFLINE
    3. Train GCN-MF model with diffused features
    4. Extract final embeddings
    
    This is a HIGH-LEVEL wrapper that combines:
    - QuVINE quantum filter calibration (from quantum_filters.py)
    - Offline diffusion precomputation
    - GCN-MF training
    - Embedding extraction
    
    Args:
        G: NetworkX graph
        q_targets: List of quantum walk targets for calibration
                Format: [{'nodes': [...], 'center': int, 'pQ': array}, ...]
        embedding_dim: Final embedding dimension
        diffusion_type: 'heat' or 'poly'
        t_star: Pre-calibrated heat kernel time (optional, will calibrate if None)
        poly_coeffs: Pre-calibrated polynomial coefficients (optional)
        K: Polynomial degree (for poly diffusion)
        ridge: Ridge regularization for polynomial calibration
        hidden_dim: GCN hidden dimension
        mf_dim: Matrix factorization dimension
        n_layers: Number of GCN layers
        epochs: Training epochs
        lr: Learning rate
        weight_decay: Weight decay for regularization
        normalize_laplacian: Whether to normalize Laplacian
        random_state: Random seed
    
    Returns:
        embeddings: Node embeddings [N, embedding_dim]
        metadata: Dictionary with calibration info and training history
    
    Example:
        >>> from qbiocode.apps.quvine.baselines.gcn_mf import generate_quvine_gcnmf_embedding
        >>>
        >>> # Generate QuVINE GCN-MF (Heat) embeddings
        >>> embeddings_heat, meta_heat = generate_quvine_gcnmf_embedding(
        ...     G, q_targets, embedding_dim=128, diffusion_type='heat'
        ... )
        >>>
        >>> # Generate QuVINE GCN-MF (Poly) embeddings
        >>> embeddings_poly, meta_poly = generate_quvine_gcnmf_embedding(
        ...     G, q_targets, embedding_dim=128, diffusion_type='poly', K=4
        ... )
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for GCN-MF. Install with: pip install torch")
    
    # Import QuVINE quantum filters
    try:
        from qbiocode.apps.quvine.embedding.quantum_filters import (
            calibrate_heat_kernel, calibrate_polynomial_filter,
            apply_heat_filter, apply_polynomial_filter
        )
    except ImportError:
        raise ImportError("QuVINE quantum filters module not found. Ensure quantum_filters.py exists.")

    
    logger.info(f"Generating QuVINE GCN-MF embeddings ({diffusion_type} diffusion)...")
    
    # Set random seed
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    
    # Get graph properties
    N = G.number_of_nodes()
    node_list = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    
    # Get Laplacian
    L = nx.laplacian_matrix(G, nodelist=node_list).astype(float)
    if normalize_laplacian:
        from qbiocode.apps.quvine.embedding.quantum_filters import _normalize_laplacian
        L = _normalize_laplacian(L)
    
    # Generate random features for diffusion
    X = np.random.randn(N, embedding_dim).astype(np.float32)
    row_norms = np.linalg.norm(X, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    X = X / row_norms
    
    metadata = {
        'diffusion_type': diffusion_type,
        'n_nodes': N,
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'mf_dim': mf_dim,
        'n_layers': n_layers
    }
    
    # STEP 1: Calibrate diffusion parameters (if not provided)
    if diffusion_type == 'heat':
        if t_star is None:
            logger.info("Calibrating heat kernel time parameter...")
            t_grid = np.linspace(0.1, 5.0, 50)
            best_loss, t_star = calibrate_heat_kernel(
                L, q_targets, t_grid, node_to_idx, loss='l2'
            )
            logger.info(f"✓ Calibrated t_star = {t_star:.4f} (loss = {best_loss:.6f})")
        else:
            logger.info(f"Using provided t_star = {t_star:.4f}")
        
        metadata['t_star'] = t_star
        
        # STEP 2: Precompute diffused features OFFLINE
        logger.info("Precomputing heat kernel diffusion...")
        X_diffused = apply_heat_filter(L, X, t_star)
        
    elif diffusion_type == 'poly':
        if poly_coeffs is None:
            logger.info(f"Calibrating polynomial filter (degree K={K})...")
            poly_coeffs = calibrate_polynomial_filter(
                L, q_targets, node_to_idx, K=K, ridge=ridge
            )
            logger.info(f"✓ Calibrated polynomial coefficients: {poly_coeffs}")
        else:
            logger.info(f"Using provided polynomial coefficients (degree={len(poly_coeffs)-1})")
        
        metadata['poly_coeffs'] = poly_coeffs.tolist() if hasattr(poly_coeffs, 'tolist') else poly_coeffs
        metadata['K'] = len(poly_coeffs) - 1
        
        # STEP 2: Precompute diffused features OFFLINE
        logger.info("Precomputing polynomial diffusion...")
        X_diffused = apply_polynomial_filter(L, X, poly_coeffs)
    
    else:
        raise ValueError(f"Unknown diffusion_type: {diffusion_type}. Use 'heat' or 'poly'.")
    
    # Convert to torch tensors
    X_diffused_torch = torch.from_numpy(X_diffused).float()
    
    # Get adjacency matrix and normalize
    adj = nx.adjacency_matrix(G, nodelist=node_list)
    adj_normalized = normalize_adjacency(adj)
    
    # Convert to torch sparse tensor
    adj_coo = adj_normalized.tocoo()
    indices = torch.LongTensor(np.vstack([adj_coo.row, adj_coo.col]))
    values = torch.FloatTensor(adj_coo.data)
    shape = adj_coo.shape
    adj_torch = torch.sparse_coo_tensor(indices, values, shape)
    
    # STEP 3: Train Q-Caliber GCN-MF model
    logger.info("Training QuVINE GCN-MF model...")
    
    # Create model
    model = QuVINEGCNMF(
        n_nodes=N,
        input_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=embedding_dim,  # For unsupervised embedding
        mf_dim=mf_dim,
        n_layers=n_layers,
        dropout=0.5
    )
    
    # For unsupervised training, we use reconstruction loss
    # Train to reconstruct adjacency matrix from embeddings
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Pre-sample balanced edges and non-edges for training
    edges = list(G.edges())
    n_edges = len(edges)
    n_samples_per_class = min(500, n_edges)  # Sample 500 positive and 500 negative
    
    # Sample positive edges (actual edges in graph)
    if n_samples_per_class == 0:
        raise ValueError("GCN-MF training requires at least one edge in the graph.")
    pos_edge_indices = np.random.choice(n_edges, size=n_samples_per_class, replace=False)
    pos_edges = [edges[i] for i in pos_edge_indices]
    
    # Sample negative edges (non-edges), avoiding duplicates
    neg_edges = set()
    attempts = 0
    max_attempts = max(n_samples_per_class * 20, 100)
    while len(neg_edges) < n_samples_per_class and attempts < max_attempts:
        u = np.random.randint(0, N)
        v = np.random.randint(0, N)
        if u != v and not G.has_edge(node_list[u], node_list[v]):
            edge = (u, v) if u < v else (v, u)
            neg_edges.add(edge)
        attempts += 1
    neg_edges = list(neg_edges)
    
    # Combine positive and negative samples
    all_edges = [(node_to_idx[u], node_to_idx[v]) for u, v in pos_edges] + neg_edges
    all_labels = [1.0] * len(pos_edges) + [0.0] * len(neg_edges)
    
    # Convert to tensors
    edge_indices_train = torch.LongTensor(all_edges).t()  # Shape: (2, n_samples)
    true_adj_train = torch.FloatTensor(all_labels)
    
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    patience = 15
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Get embeddings
        embeddings = model.get_embeddings(X_diffused_torch, adj_torch)
        
        # Compute similarity scores for sampled edges
        scores = (embeddings[edge_indices_train[0]] * embeddings[edge_indices_train[1]]).sum(dim=1)
        
        # Binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(scores, true_adj_train)
        
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
            logger.info(f"Epoch {epoch+1}/{epochs}: Loss = {loss.item():.4f}")
    
    # STEP 4: Extract final embeddings
    model.eval()
    with torch.no_grad():
        final_embeddings = model.get_embeddings(X_diffused_torch, adj_torch)
        final_embeddings = final_embeddings.cpu().numpy()
    
    logger.info(f"✓ QuVINE GCN-MF embeddings generated: shape {final_embeddings.shape}")
    
    metadata['final_embedding_shape'] = final_embeddings.shape
    
    return final_embeddings, metadata


def generate_qcaliber_gcnmf_heat_embedding(G, q_targets, embedding_dim=128,
                                           t_star=None, **kwargs):
    """
    Generate QuVINE Heat+GCN-MF (HGCNMF) embeddings.
    
    Convenience wrapper for heat kernel variant.
    
    Args:
        G: NetworkX graph
        q_targets: Quantum walk targets for calibration
        embedding_dim: Embedding dimension
        t_star: Pre-calibrated time parameter (optional)
        **kwargs: Additional arguments for generate_qcaliber_gcnmf_embedding
    
    Returns:
        embeddings: Node embeddings [N, embedding_dim]
        metadata: Calibration and training info
    """
    return generate_quvine_gcnmf_embedding(
        G, q_targets, embedding_dim=embedding_dim,
        diffusion_type='heat', t_star=t_star, **kwargs
    )


def generate_qcaliber_gcnmf_poly_embedding(G, q_targets, embedding_dim=128,
                                           poly_coeffs=None, K=4, ridge=1e-6, **kwargs):
    """
    Generate QuVINE Polynomial+GCN-MF (PGCNMF) embeddings.
    
    Convenience wrapper for polynomial filter variant.
    
    Args:
        G: NetworkX graph
        q_targets: Quantum walk targets for calibration
        embedding_dim: Embedding dimension
        poly_coeffs: Pre-calibrated coefficients (optional)
        K: Polynomial degree
        ridge: Ridge regularization
        **kwargs: Additional arguments for generate_qcaliber_gcnmf_embedding
    
    Returns:
        embeddings: Node embeddings [N, embedding_dim]
        metadata: Calibration and training info
    """
    return generate_quvine_gcnmf_embedding(
        G, q_targets, embedding_dim=embedding_dim,
        diffusion_type='poly', poly_coeffs=poly_coeffs, K=K, ridge=ridge, **kwargs
    )


def generate_baseline_gcnmf_embedding(
    G: nx.Graph,
    embedding_dim: int = 128,
    hidden_dim: int = 64,
    mf_dim: int = 64,
    n_layers: int = 2,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    random_state: int = 42,
    device: str = "cpu",
) -> np.ndarray:
    """
    Generate baseline GCN-MF embeddings (WITHOUT quantum calibration).
    
    This serves as a classical baseline to compare against QuVINE methods.
    Uses standard GCN-MF without any quantum-calibrated diffusion.
    
    Args:
        G: NetworkX graph
        embedding_dim: Final embedding dimension
        hidden_dim: GCN hidden dimension
        mf_dim: Matrix factorization dimension
        n_layers: Number of GCN layers
        epochs: Training epochs
        lr: Learning rate
        weight_decay: L2 regularization
        random_state: Random seed
    
    Returns:
        Node embeddings [N, embedding_dim]
    
    Example:
        >>> import networkx as nx
        >>> from qbiocode.apps.quvine.baselines.gcn_mf import generate_baseline_gcnmf_embedding
        >>> 
        >>> G = nx.karate_club_graph()
        >>> embeddings = generate_baseline_gcnmf_embedding(G, embedding_dim=128)
        >>> print(embeddings.shape)  # (34, 128)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for GCN-MF. Install with: pip install torch")

    import torch
    import torch.nn.functional as F
    import torch.optim as optim

    np.random.seed(random_state)
    torch.manual_seed(random_state)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if device == "auto" else torch.device(device)

    N = G.number_of_nodes()
    node_list = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    # Get adjacency matrix and normalize
    A = nx.adjacency_matrix(G, nodelist=node_list)
    A_norm = normalize_adjacency(A)

    # Convert to PyTorch tensors
    A_norm_torch = torch.FloatTensor(A_norm.toarray()).to(dev)

    # Generate random input features (no diffusion)
    X = np.random.randn(N, embedding_dim)
    row_norms = np.linalg.norm(X, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    X = X / row_norms
    X_torch = torch.FloatTensor(X).to(dev)

    # Create baseline GCNMF model
    model = GCNMF(
        n_nodes=N,
        input_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=embedding_dim,
        mf_dim=mf_dim,
        n_layers=n_layers
    ).to(dev)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Pre-sample balanced edges and non-edges for training
    edges = list(G.edges())
    n_edges = len(edges)
    n_samples_per_class = min(500, n_edges)  # Sample 500 positive and 500 negative
    
    # Sample positive edges (actual edges in graph)
    if n_samples_per_class == 0:
        raise ValueError("Baseline GCN-MF training requires at least one edge in the graph.")
    pos_edge_indices = np.random.choice(n_edges, size=n_samples_per_class, replace=False)
    pos_edges = [edges[i] for i in pos_edge_indices]
    
    # Sample negative edges (non-edges), respecting arbitrary node labels and avoiding duplicates
    neg_edges = set()
    attempts = 0
    max_attempts = max(n_samples_per_class * 20, 100)
    while len(neg_edges) < n_samples_per_class and attempts < max_attempts:
        u_idx = np.random.randint(0, N)
        v_idx = np.random.randint(0, N)
        u = node_list[u_idx]
        v = node_list[v_idx]
        if u != v and not G.has_edge(u, v):
            edge = (u_idx, v_idx) if u_idx < v_idx else (v_idx, u_idx)
            neg_edges.add(edge)
        attempts += 1
    neg_edges = list(neg_edges)
    
    # Combine positive and negative samples
    all_edges = [(node_to_idx[u], node_to_idx[v]) for u, v in pos_edges] + neg_edges
    all_labels = [1.0] * len(pos_edges) + [0.0] * len(neg_edges)

    # Convert to tensors
    edge_indices_train = torch.LongTensor(all_edges).t().to(dev)  # Shape: (2, n_samples)
    true_adj_train = torch.FloatTensor(all_labels).to(dev)

    # Training loop with early stopping
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    patience = 15
    
    for epoch in range(epochs):
        optimizer.zero_grad()

        embeddings = model.get_embeddings(X_torch, A_norm_torch)
        scores = (embeddings[edge_indices_train[0]] * embeddings[edge_indices_train[1]]).sum(dim=1)
        loss = F.binary_cross_entropy_with_logits(scores, true_adj_train)
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
            logger.info(f"Baseline GCN-MF Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    # Extract final embeddings
    model.eval()
    with torch.no_grad():
        final_embeddings = model.get_embeddings(X_torch, A_norm_torch)
        embeddings_np = final_embeddings.cpu().numpy()
    
    return embeddings_np


def generate_baseline_filter_embedding_wrapper(
    G: nx.Graph,
    filter_type: str = 'heat',
    t: float = 1.0,
    K: int = 4,
    embedding_dim: int = 128,
    normalize: bool = True,
    random_state: int = 42
) -> np.ndarray:
    """
    Generate baseline graph filter embeddings (WITHOUT quantum calibration).
    
    Wrapper for the baseline filter function in quantum_filters.py.
    This serves as a classical baseline to compare against QuVINE methods.
    
    Args:
        G: NetworkX graph
        filter_type: Type of filter ('heat' or 'poly')
        t: Time parameter for heat kernel (if filter_type='heat')
        K: Polynomial degree (if filter_type='poly')
        embedding_dim: Embedding dimension
        normalize: Whether to normalize Laplacian
        random_state: Random seed
    
    Returns:
        Node embeddings [N, embedding_dim]
    
    Example:
        >>> import networkx as nx
        >>> from qbiocode.apps.quvine.baselines.gcn_mf import generate_baseline_filter_embedding_wrapper
        >>> 
        >>> G = nx.karate_club_graph()
        >>> 
        >>> # Heat kernel baseline
        >>> embeddings_heat = generate_baseline_filter_embedding_wrapper(
        ...     G, filter_type='heat', t=1.0, embedding_dim=128
        ... )
        >>> 
        >>> # Polynomial filter baseline
        >>> embeddings_poly = generate_baseline_filter_embedding_wrapper(
        ...     G, filter_type='poly', K=4, embedding_dim=128
        ... )
    """
    from qbiocode.apps.quvine.embedding.quantum_filters import generate_baseline_filter_embedding
    
    return generate_baseline_filter_embedding(
        G=G,
        filter_type=filter_type,
        t=t,
        K=K,
        embedding_dim=embedding_dim,
        use_features=False,
        features=None,
        normalize=normalize,
        random_state=random_state
    )
