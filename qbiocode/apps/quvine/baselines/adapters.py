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
Method Adapters for Baseline Methods

This module provides adapter functions that bridge between the registry's config
objects and the actual baseline method implementations. Each adapter:
1. Takes a config object and other required parameters
2. Converts config to method-specific parameters
3. Calls the actual baseline method
4. Returns the embedding in a consistent format
"""

import logging
from typing import List, Optional
import numpy as np

# Import baseline methods. These come from the modules directly, never from the
# package namespace: __init__.py binds them behind try/except ImportError, so
# `from . import run_node2vec` turns any one baseline's missing dependency into
# an ImportError that names node2vec and takes down the whole method registry
# with it -- including the methods that need nothing optional at all.
from .node2vec import run_node2vec
from .appnp import run_appnp
from .graphsage import run_graphsage
from .netmf import run_netmf
from .gat import generate_gat_embedding, GATConfig as GATModelConfig, TrainConfig as GATTrainConfig
from .graphgps import generate_graphgps_embedding, GraphGPSConfig as GraphGPSModelConfig, TrainConfig as GraphGPSTrainConfig
from .gcn_mf import generate_baseline_gcnmf_embedding, generate_quvine_gcnmf_embedding
from ..embedding.quantum_filters import (
    generate_baseline_filter_embedding,
    generate_quvine_heat_embedding,
    generate_quvine_poly_embedding
)

# Import config types
from .configs import (
    Node2VecConfig,
    APPNPConfig,
    BaselineFilterConfig,
    GCNMFConfig,
    GATMethodConfig,
    GraphGPSMethodConfig,
    GraphSAGEConfig,
    QuvineFilterConfig,
    QuvineGCNMFConfig,
    QuvineSGNSConfig,
    NetMFConfig
)

logger = logging.getLogger(__name__)


# ============================================================================
# Simple Baseline Adapters
# ============================================================================

def run_node2vec_adapter(graph_data, config: Node2VecConfig, **kwargs) -> np.ndarray:
    """
    Adapter for Node2Vec baseline.
    
    Args:
        graph_data: NetworkX graph
        config: Node2VecConfig object
        **kwargs: Additional arguments (ignored)
        
    Returns:
        Embedding matrix (n_nodes x dimensions)
    """
    return run_node2vec(
        graph=graph_data,
        nodes=list(graph_data.nodes),
        dimensions=config.dimensions,
        walk_length=config.walk_length,
        num_walks=config.num_walks,
        p=config.p,
        q=config.q,
        window=config.window,
        min_count=config.min_count,
        workers=config.workers,
        seed=config.seed
    )


def run_appnp_adapter(graph_data, config: APPNPConfig, **kwargs) -> np.ndarray:
    """
    Adapter for APPNP baseline.
    
    Args:
        graph_data: NetworkX graph
        config: APPNPConfig object
        **kwargs: Additional arguments (ignored)
        
    Returns:
        Embedding matrix (n_nodes x dimensions)
    """
    return run_appnp(
        graph=graph_data,
        nodes=list(graph_data.nodes),
        dimensions=config.dimensions,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        alpha=config.alpha,
        K=config.K,
        dropout=config.dropout,
        lr=config.lr,
        weight_decay=config.weight_decay,
        epochs=config.epochs,
        seed=config.seed
    )


def run_graphsage_adapter(graph_data, config: GraphSAGEConfig, **kwargs) -> np.ndarray:
    """
    Adapter for GraphSAGE baseline.
    
    Args:
        graph_data: NetworkX graph
        config: GraphSAGEConfig object
        **kwargs: Additional arguments (ignored)
        
    Returns:
        Embedding matrix (n_nodes x dimensions)
    """
    return run_graphsage(
        graph=graph_data,
        nodes=list(graph_data.nodes),
        dimensions=config.dimensions,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        epochs=config.epochs,
        lr=config.lr,
        neg_samples=config.neg_samples,
        seed=config.seed or 42
    )


def run_netmf_adapter(graph_data, config: NetMFConfig, **kwargs) -> np.ndarray:
    """
    Adapter for NetMF baseline.
    
    Args:
        graph_data: NetworkX graph
        config: NetMFConfig object
        **kwargs: Additional arguments (ignored)
        
    Returns:
        Embedding matrix (n_nodes x dimensions)
    """
    return run_netmf(
        graph=graph_data,
        nodes=list(graph_data.nodes),
        dimensions=config.dimensions,
        window_size=config.window_size,
        rank=config.rank,
        negative=config.negative,
        seed=config.seed or 42
    )


# ============================================================================
# Filter-based Method Adapters
# ============================================================================

def run_baseline_filter_adapter(graph_data, config: BaselineFilterConfig, **kwargs) -> np.ndarray:
    """
    Adapter for baseline filter methods (heat/poly without quantum calibration).
    
    Args:
        graph_data: NetworkX graph
        config: BaselineFilterConfig object
        **kwargs: Additional arguments (ignored)
        
    Returns:
        Embedding matrix (n_nodes x embedding_dim)
    """
    return generate_baseline_filter_embedding(
        G=graph_data,
        filter_type=config.filter_type,
        t=config.t,
        K=config.K,
        embedding_dim=config.embedding_dim,
        use_features=config.use_features,
        features=config.features,
        normalize=config.normalize,
        random_state=config.random_state or 42
    )


def run_quvine_heat_adapter(graph_data, config: QuvineFilterConfig, q_targets: List, **kwargs) -> np.ndarray:
    """
    Adapter for quantum-calibrated heat kernel filter.
    
    Args:
        graph_data: NetworkX graph
        config: QuvineFilterConfig object
        q_targets: Quantum targets for calibration
        **kwargs: Additional arguments (ignored)
        
    Returns:
        Embedding matrix (n_nodes x embedding_dim)
    """
    return generate_quvine_heat_embedding(
        G=graph_data,
        q_targets=q_targets,
        embedding_dim=config.embedding_dim,
        use_features=config.use_features,
        features=config.features,
        normalize=config.normalize,
        random_state=config.random_state or 42
    )


def run_quvine_poly_adapter(graph_data, config: QuvineFilterConfig, q_targets: List, **kwargs) -> np.ndarray:
    """
    Adapter for quantum-calibrated polynomial filter.
    
    Args:
        graph_data: NetworkX graph
        config: QuvineFilterConfig object
        q_targets: Quantum targets for calibration
        **kwargs: Additional arguments (ignored)
        
    Returns:
        Embedding matrix (n_nodes x embedding_dim)
    """
    return generate_quvine_poly_embedding(
        G=graph_data,
        q_targets=q_targets,
        K=config.K,
        ridge=config.ridge,
        embedding_dim=config.embedding_dim,
        use_features=config.use_features,
        features=config.features,
        normalize=config.normalize,
        random_state=config.random_state or 42
    )


# ============================================================================
# GCN-MF Adapters
# ============================================================================

def run_baseline_gcnmf_adapter(graph_data, config: GCNMFConfig, **kwargs) -> np.ndarray:
    """
    Adapter for baseline GCN-MF.
    
    Args:
        graph_data: NetworkX graph
        config: GCNMFConfig object
        **kwargs: Additional arguments (ignored)
        
    Returns:
        Embedding matrix (n_nodes x embedding_dim)
    """
    return generate_baseline_gcnmf_embedding(
        G=graph_data,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        mf_dim=config.mf_dim,
        n_layers=config.n_layers,
        epochs=config.epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        random_state=config.random_state or 42
    )


def run_quvine_gcnmf_adapter(graph_data, config: QuvineGCNMFConfig, q_targets: List, **kwargs) -> np.ndarray:
    """
    Adapter for quantum-calibrated GCN-MF.
    
    Args:
        graph_data: NetworkX graph
        config: QuvineGCNMFConfig object
        q_targets: Quantum targets for calibration
        **kwargs: Additional arguments (ignored)
        
    Returns:
        Embedding matrix (n_nodes x embedding_dim)
    """
    embedding, _ = generate_quvine_gcnmf_embedding(
        G=graph_data,
        q_targets=q_targets,
        embedding_dim=config.embedding_dim,
        diffusion_type=config.diffusion_type,
        hidden_dim=config.hidden_dim,
        mf_dim=config.mf_dim,
        n_layers=config.n_layers,
        epochs=config.epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        K=config.K,
        ridge=config.ridge,
        normalize_laplacian=config.normalize_laplacian,
        random_state=config.random_state or 42
    )
    return embedding


def run_quvine_sgns_adapter(graph_data, config: QuvineSGNSConfig, **kwargs) -> np.ndarray:
    """
    Adapter for the QuVINE SGNS walk embeddings (quvine_rwr/ctqw/dtqw).

    Runs the shared SGNS core (views -> walks -> corpus -> word2vec) for the
    single ``config.walk_kind`` and returns that walk kind's embedding, with
    rows in ``list(graph_data.nodes)`` order.

    Args:
        graph_data: NetworkX graph
        config: QuvineSGNSConfig carrying the walk kind and full OmegaConf cfg
        **kwargs: Additional arguments (ignored; SGNS needs no quantum targets)

    Returns:
        Embedding matrix (n_nodes x embedding_dim)
    """
    from ..api.sgns import run_sgns

    embeddings = run_sgns(
        config.cfg,
        graph_data,
        it=0,
        kinds=[config.walk_kind],
        n_jobs=config.n_jobs,
        chunk_size=config.chunk_size,
    )
    return embeddings[config.walk_kind]


# ============================================================================
# GAT Adapters
# ============================================================================

def run_gat_adapter(graph_data, config: GATMethodConfig, q_targets: Optional[List] = None, **kwargs) -> np.ndarray:
    """
    Adapter for GAT-based methods.
    
    Handles all GAT variants:
    - raw: Standard GAT
    - heat_qcal_ctqw/dtqw/rwr: Quantum-calibrated with heat kernel
    - poly_qcal_ctqw/dtqw/rwr: Quantum-calibrated with polynomial filter
    
    Args:
        graph_data: NetworkX graph
        config: GATMethodConfig object
        q_targets: Quantum targets (optional, required for quantum variants)
        **kwargs: Additional arguments (ignored)
        
    Returns:
        Embedding matrix (n_nodes x embedding_dim)
    """
    # Convert our config to GAT's expected config format
    gat_config = GATModelConfig(
        hidden_dim=config.model.hidden_dim,
        output_dim=config.model.output_dim,
        num_layers=config.model.num_layers,
        heads=config.model.heads,
        dropout=config.model.dropout,
        attention_dropout=config.model.attention_dropout,
        negative_slope=config.model.negative_slope,
        residual=config.model.residual
    )
    
    train_config = GATTrainConfig(
        epochs=config.train.epochs,
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
        patience=config.train.patience,
        edge_batch_size=config.train.edge_batch_size,
        val_edge_fraction=config.train.val_edge_fraction,
        device=config.train.device,
        random_state=config.train.random_state or 42,
        verbose=config.train.verbose
    )
    
    # Prepare kwargs for generate_gat_embedding
    gat_kwargs = {
        'G': graph_data,
        'variant': config.variant,
        'nodelist': list(graph_data.nodes),
        'gat_config': gat_config,
        'train_config': train_config,
        # Pass filter params so classical single-axis variants work correctly.
        'heat_t': config.heat_t,
        'poly_K': config.poly_K,
        'poly_ridge': config.poly_ridge,
        'rwr_alpha': config.rwr_alpha,
        'rwr_steps': config.rwr_steps,
    }
    
    # Add quantum targets based on variant
    if 'ctqw' in config.variant:
        gat_kwargs['ctqw_targets'] = q_targets
    elif 'dtqw' in config.variant:
        gat_kwargs['dtqw_targets'] = q_targets
    elif 'rwr' in config.variant:
        gat_kwargs['rwr_targets'] = q_targets
    
    embedding, _ = generate_gat_embedding(**gat_kwargs)
    return embedding


# ============================================================================
# GraphGPS Adapters
# ============================================================================

def run_graphgps_adapter(graph_data, config: GraphGPSMethodConfig, q_targets: Optional[List] = None, **kwargs) -> np.ndarray:
    """
    Adapter for GraphGPS-based methods.
    
    Handles all GraphGPS variants:
    - raw: Standard GraphGPS
    - heat_qcal_ctqw/dtqw/rwr: Quantum-calibrated with heat kernel
    - poly_qcal_ctqw/dtqw/rwr: Quantum-calibrated with polynomial filter
    
    Args:
        graph_data: NetworkX graph
        config: GraphGPSMethodConfig object
        q_targets: Quantum targets (optional, required for quantum variants)
        **kwargs: Additional arguments (ignored)
        
    Returns:
        Embedding matrix (n_nodes x embedding_dim)
    """
    # Convert our config to GraphGPS's expected config format
    gps_config = GraphGPSModelConfig(
        hidden_dim=config.model.hidden_dim,
        output_dim=config.model.output_dim,
        num_layers=config.model.num_layers,
        heads=config.model.heads,
        dropout=config.model.dropout,
        attn_dropout=config.model.attn_dropout,
        local_gnn=config.model.local_gnn,
        attn_type=config.model.attn_type,
        use_layer_norm=config.model.use_layer_norm,
        activation=config.model.activation,
        lap_pe_dim=config.model.lap_pe_dim,
        standardize_features=config.model.standardize_features
    )
    
    train_config = GraphGPSTrainConfig(
        task=config.train.task,
        epochs=config.train.epochs,
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
        patience=config.train.patience,
        edge_batch_size=config.train.edge_batch_size,
        val_edge_fraction=config.train.val_edge_fraction,
        device=config.train.device,
        random_state=config.train.random_state or 42,
        verbose=config.train.verbose
    )
    
    # Prepare kwargs for generate_graphgps_embedding
    gps_kwargs = {
        'G': graph_data,
        'variant': config.variant,
        'nodelist': list(graph_data.nodes),
        'embedding_dim': config.embedding_dim,
        'gps_config': gps_config,
        'train_config': train_config,
        # Pass filter params so classical single-axis variants work correctly.
        'heat_t': config.heat_t,
        'poly_K': config.poly_K,
        'poly_ridge': config.poly_ridge,
        'rwr_alpha': config.rwr_alpha,
        'rwr_steps': config.rwr_steps,
    }
    
    # Add quantum targets based on variant
    if 'ctqw' in config.variant:
        gps_kwargs['ctqw_targets'] = q_targets
    elif 'dtqw' in config.variant:
        gps_kwargs['dtqw_targets'] = q_targets
    elif 'rwr' in config.variant:
        gps_kwargs['rwr_targets'] = q_targets
    
    embedding, _ = generate_graphgps_embedding(**gps_kwargs)
    return embedding


# ============================================================================
# Adapter Registry
# ============================================================================

# Map method names to their adapters
ADAPTER_MAP = {
    'node2vec': run_node2vec_adapter,
    'appnp': run_appnp_adapter,
    'graphsage': run_graphsage_adapter,
    'netmf': run_netmf_adapter,
    'baseline_filter_heat': run_baseline_filter_adapter,
    'baseline_filter_poly': run_baseline_filter_adapter,
    'baseline_gcnmf': run_baseline_gcnmf_adapter,
    'quvine_heat': run_quvine_heat_adapter,
    'quvine_poly': run_quvine_poly_adapter,
    'quvine_hgcnmf': run_quvine_gcnmf_adapter,
    'quvine_pgcnmf': run_quvine_gcnmf_adapter,
    # Filter variants
    'filter_rwr_heat': run_quvine_heat_adapter,
    'filter_rwr_poly': run_quvine_poly_adapter,
    'filter_ctqw_heat': run_quvine_heat_adapter,
    'filter_ctqw_poly': run_quvine_poly_adapter,
    'filter_dtqw_heat': run_quvine_heat_adapter,
    'filter_dtqw_poly': run_quvine_poly_adapter,
    # GAT variants — canonical names (gat_baseline replaces baseline_gat)
    'gat_baseline': run_gat_adapter,
    'gat_heat': run_gat_adapter,
    'gat_poly': run_gat_adapter,
    'gat_rwr': run_gat_adapter,
    'gat_ctqw_heat': run_gat_adapter,
    'gat_ctqw_poly': run_gat_adapter,
    'gat_dtqw_heat': run_gat_adapter,
    'gat_dtqw_poly': run_gat_adapter,
    'gat_rwr_heat': run_gat_adapter,
    'gat_rwr_poly': run_gat_adapter,
    # GraphGPS variants — canonical names (graphgps_baseline replaces baseline_graphgps)
    'graphgps_baseline': run_graphgps_adapter,
    'graphgps_heat': run_graphgps_adapter,
    'graphgps_poly': run_graphgps_adapter,
    'graphgps_rwr': run_graphgps_adapter,
    'graphgps_ctqw_heat': run_graphgps_adapter,
    'graphgps_ctqw_poly': run_graphgps_adapter,
    'graphgps_dtqw_heat': run_graphgps_adapter,
    'graphgps_dtqw_poly': run_graphgps_adapter,
    'graphgps_rwr_heat': run_graphgps_adapter,
    'graphgps_rwr_poly': run_graphgps_adapter,
    # QuVINE SGNS walk embeddings (views -> walks -> corpus -> word2vec)
    'quvine_ctqw': run_quvine_sgns_adapter,
    'quvine_dtqw': run_quvine_sgns_adapter,
    'quvine_rwr': run_quvine_sgns_adapter,
}


def get_adapter(method_name: str):
    """
    Get the adapter function for a method.
    
    Args:
        method_name: Name of the method
        
    Returns:
        Adapter function
        
    Raises:
        KeyError: If method not found
    """
    if method_name not in ADAPTER_MAP:
        raise KeyError(f"No adapter found for method: {method_name}")
    return ADAPTER_MAP[method_name]

