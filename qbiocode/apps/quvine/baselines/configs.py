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
Configuration dataclasses for all baseline methods.

This module provides type-safe configuration classes for each baseline method,
eliminating the need for getattr calls and providing clear defaults.
"""

from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# Import hyperparameter loader (will be available after module loads)
try:
    from .hyperparameter_loader import load_and_override_config
except ImportError:
    # Fallback if not available
    def load_and_override_config(config, method_name, task):
        return config

from typing import Optional, List, Any


@dataclass
class Node2VecConfig:
    """Configuration for Node2Vec baseline."""
    enabled: bool = False
    dimensions: int = 128
    walk_length: int = 80
    num_walks: int = 10
    p: float = 1.0
    q: float = 1.0
    window: int = 10
    min_count: int = 1
    workers: int = 4
    seed: Optional[int] = None


@dataclass
class APPNPConfig:
    """Configuration for APPNP baseline."""
    enabled: bool = False
    dimensions: int = 128
    hidden_dim: int = 64
    n_layers: int = 2
    alpha: float = 0.1
    K: int = 10
    dropout: float = 0.5
    lr: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 200
    seed: Optional[int] = None


@dataclass
class BaselineFilterConfig:
    """Configuration for baseline filter methods (heat/poly without quantum calibration)."""
    enabled: bool = False
    filter_type: str = "heat"  # "heat" or "poly"
    embedding_dim: int = 128
    t: float = 1.0  # for heat kernel
    K: int = 4  # for polynomial filter
    normalize: bool = True
    use_features: bool = False
    features: Optional[Any] = None
    random_state: Optional[int] = None


@dataclass
class GCNMFConfig:
    """Configuration for GCN-MF baseline."""
    enabled: bool = False
    embedding_dim: int = 128
    hidden_dim: int = 64
    mf_dim: int = 64
    n_layers: int = 2
    epochs: int = 200
    lr: float = 0.01
    weight_decay: float = 5e-4
    random_state: Optional[int] = None


@dataclass
class GATTrainConfig:
    """Training configuration for GAT models."""
    epochs: int = 200
    lr: float = 5e-3
    weight_decay: float = 5e-4
    patience: int = 25
    edge_batch_size: int = 4096
    val_edge_fraction: float = 0.1
    device: str = "cpu"
    random_state: Optional[int] = None
    verbose: bool = False


@dataclass
class GATModelConfig:
    """Model architecture configuration for GAT."""
    hidden_dim: int = 64
    output_dim: int = 128
    num_layers: int = 2
    heads: int = 4
    dropout: float = 0.2
    attention_dropout: float = 0.2
    negative_slope: float = 0.2
    residual: bool = True


@dataclass
class GATMethodConfig:
    """Complete configuration for GAT-based methods."""
    enabled: bool = False
    variant: str = "raw"  # raw, heat_fixed, poly_fixed, rwr, heat_qcal_ctqw, etc.
    embedding_dim: int = 128
    # Feature-construction params (used by the classical heat/poly/rwr variants;
    # the qcal variants fit their own params, so these are ignored there).
    heat_t: float = 1.0
    poly_K: int = 4
    poly_ridge: float = 1e-5
    rwr_alpha: float = 0.15
    rwr_steps: int = 50
    model: GATModelConfig = field(default_factory=GATModelConfig)
    train: GATTrainConfig = field(default_factory=GATTrainConfig)


@dataclass
class GraphGPSTrainConfig:
    """Training configuration for GraphGPS models."""
    task: str = "link_reconstruction"
    epochs: int = 200
    lr: float = 5e-3
    weight_decay: float = 5e-4
    patience: int = 30
    edge_batch_size: int = 8192
    val_edge_fraction: float = 0.1
    device: str = "cpu"
    random_state: Optional[int] = None
    verbose: bool = False


@dataclass
class GraphGPSModelConfig:
    """Model architecture configuration for GraphGPS."""
    hidden_dim: int = 64
    output_dim: int = 128
    num_layers: int = 2
    heads: int = 4
    dropout: float = 0.2
    attn_dropout: float = 0.2
    local_gnn: str = "gcn"
    attn_type: str = "multihead"
    use_layer_norm: bool = True
    activation: str = "relu"
    lap_pe_dim: int = 0
    standardize_features: bool = True


@dataclass
class GraphGPSMethodConfig:
    """Complete configuration for GraphGPS-based methods."""
    enabled: bool = False
    variant: str = "raw"  # raw, heat_fixed, poly_fixed, rwr, heat_qcal_ctqw, etc.
    embedding_dim: int = 128
    # Feature-construction params (classical heat/poly/rwr variants).
    heat_t: float = 1.0
    poly_K: int = 4
    poly_ridge: float = 1e-5
    rwr_alpha: float = 0.15
    rwr_steps: int = 50
    model: GraphGPSModelConfig = field(default_factory=GraphGPSModelConfig)
    train: GraphGPSTrainConfig = field(default_factory=GraphGPSTrainConfig)


@dataclass
class GraphSAGEConfig:
    """Configuration for GraphSAGE baseline."""
    enabled: bool = False
    dimensions: int = 128
    hidden_dim: int = 256
    n_layers: int = 2
    epochs: int = 50
    lr: float = 0.01
    neg_samples: int = 5
    seed: Optional[int] = None


@dataclass
class QuvineFilterConfig:
    """Configuration for quantum-calibrated filter methods."""
    enabled: bool = False
    filter_type: str = "heat"  # "heat" or "poly"
    embedding_dim: int = 128
    t: Optional[float] = None  # for heat kernel (calibrated if None)
    K: int = 4  # for polynomial filter
    ridge: float = 1e-6  # for polynomial filter
    normalize: bool = True
    use_features: bool = False
    features: Optional[Any] = None
    random_state: Optional[int] = None


@dataclass
class QuvineGCNMFConfig:
    """Configuration for quantum-calibrated GCN-MF methods."""
    enabled: bool = False
    diffusion_type: str = "heat"  # "heat" or "poly"
    embedding_dim: int = 128
    hidden_dim: int = 64
    mf_dim: int = 64
    n_layers: int = 2
    epochs: int = 200
    lr: float = 0.01
    weight_decay: float = 5e-4
    K: int = 4  # for poly
    ridge: float = 1e-6  # for poly
    normalize_laplacian: bool = True
    random_state: Optional[int] = None


@dataclass
class QuvineSGNSConfig:
    """
    Configuration for the QuVINE SGNS walk embeddings (quvine_rwr/ctqw/dtqw).

    These methods are the core QuVINE embedding: per-root views -> walks
    (rwr/ctqw/dtqw) -> corpus -> word2vec (SGNS). They do NOT use quantum
    calibration targets; the walk itself is the quantum component. The full
    OmegaConf ``cfg`` is carried through because the SGNS core reads
    ``walks.*``, ``views.*``, ``train.*``, ``min_count`` and
    ``experiment.base_seed`` from it.
    """
    enabled: bool = False
    walk_kind: str = "rwr"  # "rwr", "ctqw", or "dtqw"
    cfg: Any = None         # full OmegaConf config, consumed by run_sgns
    n_jobs: int = 1
    chunk_size: int = 30


@dataclass
class NetMFConfig:
    """Configuration for NetMF baseline."""
    enabled: bool = False
    dimensions: int = 128
    window_size: int = 10
    rank: int = 256
    negative: int = 1
    seed: Optional[int] = None


# Helper function to get default embedding dimension from global config
def get_embedding_dim(cfg, method_cfg, default: int = 128) -> int:
    """
    Get embedding dimension with fallback logic.
    
    Priority:
    1. Method-specific embedding_dim
    2. Global train.embedding_dim
    3. Default value
    """
    if hasattr(method_cfg, 'embedding_dim') and method_cfg.embedding_dim is not None:
        return method_cfg.embedding_dim
    if hasattr(cfg, 'train') and hasattr(cfg.train, 'embedding_dim'):
        return cfg.train.embedding_dim
    return default


"""
Configuration dataclasses for all baseline methods.

This module provides type-safe configuration classes for each baseline method,
eliminating the need for getattr calls and providing clear defaults.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Any
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Config Builder Functions
# ============================================================================

def build_node2vec_config(cfg, base_seed: int) -> Node2VecConfig:
    """Build Node2Vec config from OmegaConf."""
    if not hasattr(cfg.baselines, 'node2vec'):
        return Node2VecConfig(enabled=False)
    
    n2v = cfg.baselines.node2vec
    return Node2VecConfig(
        enabled=getattr(n2v, 'enabled', False),
        dimensions=getattr(n2v, 'dimensions', 128),
        walk_length=getattr(n2v, 'walk_length', 80),
        num_walks=getattr(n2v, 'num_walks', 10),
        p=getattr(n2v, 'p', 1.0),
        q=getattr(n2v, 'q', 1.0),
        window=getattr(n2v, 'window', 10),
        min_count=getattr(n2v, 'min_count', 1),
        workers=getattr(n2v, 'workers', 4),
        seed=getattr(n2v, 'seed', base_seed)
    )


def build_appnp_config(cfg, base_seed: int) -> APPNPConfig:
    """Build APPNP config from OmegaConf."""
    if not hasattr(cfg.baselines, 'appnp'):
        return APPNPConfig(enabled=False)
    
    appnp = cfg.baselines.appnp
    return APPNPConfig(
        enabled=getattr(appnp, 'enabled', False),
        dimensions=getattr(appnp, 'dimensions', 128),
        hidden_dim=getattr(appnp, 'hidden_dim', 64),
        n_layers=getattr(appnp, 'n_layers', 2),
        alpha=getattr(appnp, 'alpha', 0.1),
        K=getattr(appnp, 'K', 10),
        dropout=getattr(appnp, 'dropout', 0.5),
        lr=getattr(appnp, 'lr', 0.01),
        weight_decay=getattr(appnp, 'weight_decay', 5e-4),
        epochs=getattr(appnp, 'epochs', 200),
        seed=getattr(appnp, 'seed', base_seed)
    )


def build_baseline_filter_config(cfg, base_seed: int, filter_type: str = "heat") -> BaselineFilterConfig:
    """Build baseline filter config from OmegaConf."""
    if not hasattr(cfg.baselines, 'baseline_filter'):
        return BaselineFilterConfig(enabled=False)
    
    bf = cfg.baselines.baseline_filter
    embedding_dim = getattr(bf, 'embedding_dim', None)
    if embedding_dim is None and hasattr(cfg, 'train'):
        embedding_dim = getattr(cfg.train, 'embedding_dim', 128)
    
    return BaselineFilterConfig(
        enabled=getattr(bf, 'enabled', False),
        filter_type=getattr(bf, 'filter_type', filter_type),
        embedding_dim=embedding_dim or 128,
        t=getattr(bf, 't', 1.0),
        K=getattr(bf, 'K', 4),
        normalize=getattr(bf, 'normalize', True),
        use_features=getattr(bf, 'use_features', False),
        features=getattr(bf, 'features', None),
        random_state=getattr(bf, 'random_state', base_seed)
    )


def build_gcnmf_config(cfg, base_seed: int, config_name: str = "baseline_gcnmf") -> GCNMFConfig:
    """Build GCN-MF config from OmegaConf."""
    if not hasattr(cfg.baselines, config_name):
        return GCNMFConfig(enabled=False)
    
    gcnmf = getattr(cfg.baselines, config_name)
    embedding_dim = getattr(gcnmf, 'embedding_dim', None)
    if embedding_dim is None and hasattr(cfg, 'train'):
        embedding_dim = getattr(cfg.train, 'embedding_dim', 128)
    
    return GCNMFConfig(
        enabled=getattr(gcnmf, 'enabled', False),
        embedding_dim=embedding_dim or 128,
        hidden_dim=getattr(gcnmf, 'hidden_dim', 64),
        mf_dim=getattr(gcnmf, 'mf_dim', 64),
        n_layers=getattr(gcnmf, 'n_layers', 2),
        epochs=getattr(gcnmf, 'epochs', 200),
        lr=getattr(gcnmf, 'lr', 0.01),
        weight_decay=getattr(gcnmf, 'weight_decay', 5e-4),
        random_state=getattr(gcnmf, 'random_state', base_seed)
    )


def build_gat_config(cfg, base_seed: int, config_name: str) -> GATMethodConfig:
    """Build GAT method config from OmegaConf."""
    if not hasattr(cfg.baselines, config_name):
        return GATMethodConfig(enabled=False)
    
    gat_cfg = getattr(cfg.baselines, config_name)
    embedding_dim = getattr(gat_cfg, 'embedding_dim', None)
    if embedding_dim is None and hasattr(cfg, 'train'):
        embedding_dim = getattr(cfg.train, 'embedding_dim', 128)
    
    model_config = GATModelConfig(
        hidden_dim=getattr(gat_cfg, 'hidden_dim', 64),
        output_dim=embedding_dim or 128,
        num_layers=getattr(gat_cfg, 'num_layers', 2),
        heads=getattr(gat_cfg, 'heads', 4),
        dropout=getattr(gat_cfg, 'dropout', 0.2),
        attention_dropout=getattr(gat_cfg, 'attention_dropout', 0.2),
        negative_slope=getattr(gat_cfg, 'negative_slope', 0.2),
        residual=getattr(gat_cfg, 'residual', True)
    )
    
    train_config = GATTrainConfig(
        epochs=getattr(gat_cfg, 'epochs', 200),
        lr=getattr(gat_cfg, 'lr', 5e-3),
        weight_decay=getattr(gat_cfg, 'weight_decay', 5e-4),
        patience=getattr(gat_cfg, 'patience', 25),
        edge_batch_size=getattr(gat_cfg, 'edge_batch_size', 4096),
        val_edge_fraction=getattr(gat_cfg, 'val_edge_fraction', 0.1),
        device=getattr(gat_cfg, 'device', 'cpu'),
        random_state=getattr(gat_cfg, 'random_state', base_seed),
        verbose=getattr(gat_cfg, 'verbose', False)
    )
    
    return GATMethodConfig(
        enabled=getattr(gat_cfg, 'enabled', False),
        variant=getattr(gat_cfg, 'variant', 'raw'),
        embedding_dim=embedding_dim or 128,
        heat_t=getattr(gat_cfg, 'heat_t', 1.0),
        poly_K=getattr(gat_cfg, 'poly_K', 4),
        poly_ridge=getattr(gat_cfg, 'poly_ridge', 1e-5),
        rwr_alpha=getattr(gat_cfg, 'rwr_alpha', 0.15),
        rwr_steps=getattr(gat_cfg, 'rwr_steps', 50),
        model=model_config,
        train=train_config
    )


def build_graphgps_config(cfg, base_seed: int, config_name: str) -> GraphGPSMethodConfig:
    """Build GraphGPS method config from OmegaConf."""
    if not hasattr(cfg.baselines, config_name):
        return GraphGPSMethodConfig(enabled=False)
    
    gps_cfg = getattr(cfg.baselines, config_name)
    embedding_dim = getattr(gps_cfg, 'embedding_dim', None)
    if embedding_dim is None and hasattr(cfg, 'train'):
        embedding_dim = getattr(cfg.train, 'embedding_dim', 128)
    
    model_config = GraphGPSModelConfig(
        hidden_dim=getattr(gps_cfg, 'hidden_dim', 64),
        output_dim=embedding_dim or 128,
        num_layers=getattr(gps_cfg, 'num_layers', 2),
        heads=getattr(gps_cfg, 'heads', 4),
        dropout=getattr(gps_cfg, 'dropout', 0.2),
        attn_dropout=getattr(gps_cfg, 'attn_dropout', 0.2),
        local_gnn=getattr(gps_cfg, 'local_gnn', 'gcn'),
        attn_type=getattr(gps_cfg, 'attn_type', 'multihead'),
        use_layer_norm=getattr(gps_cfg, 'use_layer_norm', True),
        activation=getattr(gps_cfg, 'activation', 'relu'),
        lap_pe_dim=getattr(gps_cfg, 'lap_pe_dim', 0),
        standardize_features=getattr(gps_cfg, 'standardize_features', True)
    )
    
    train_config = GraphGPSTrainConfig(
        task=getattr(gps_cfg, 'task', 'link_reconstruction'),
        epochs=getattr(gps_cfg, 'epochs', 200),
        lr=getattr(gps_cfg, 'lr', 5e-3),
        weight_decay=getattr(gps_cfg, 'weight_decay', 5e-4),
        patience=getattr(gps_cfg, 'patience', 30),
        edge_batch_size=getattr(gps_cfg, 'edge_batch_size', 8192),
        val_edge_fraction=getattr(gps_cfg, 'val_edge_fraction', 0.1),
        device=getattr(gps_cfg, 'device', 'cpu'),
        random_state=getattr(gps_cfg, 'random_state', base_seed),
        verbose=getattr(gps_cfg, 'verbose', False)
    )
    
    return GraphGPSMethodConfig(
        enabled=getattr(gps_cfg, 'enabled', False),
        variant=getattr(gps_cfg, 'variant', 'raw'),
        embedding_dim=embedding_dim or 128,
        heat_t=getattr(gps_cfg, 'heat_t', 1.0),
        poly_K=getattr(gps_cfg, 'poly_K', 4),
        poly_ridge=getattr(gps_cfg, 'poly_ridge', 1e-5),
        rwr_alpha=getattr(gps_cfg, 'rwr_alpha', 0.15),
        rwr_steps=getattr(gps_cfg, 'rwr_steps', 50),
        model=model_config,
        train=train_config
    )


def build_graphsage_config(cfg, base_seed: int) -> GraphSAGEConfig:
    """Build GraphSAGE config from OmegaConf."""
    if not hasattr(cfg.baselines, 'graphsage'):
        return GraphSAGEConfig(enabled=False)
    
    gs = cfg.baselines.graphsage
    dimensions = getattr(gs, 'dimensions', None)
    if dimensions is None and hasattr(cfg, 'train'):
        dimensions = getattr(cfg.train, 'embedding_dim', 128)
    
    return GraphSAGEConfig(
        enabled=getattr(gs, 'enabled', False),
        dimensions=dimensions or 128,
        hidden_dim=getattr(gs, 'hidden_dim', min(256, (dimensions or 128) * 2)),
        n_layers=getattr(gs, 'n_layers', 2),
        epochs=getattr(gs, 'epochs', 50),
        lr=getattr(gs, 'lr', 0.01),
        neg_samples=getattr(gs, 'neg_samples', 5),
        seed=getattr(gs, 'seed', base_seed)
    )


def build_quvine_filter_config(cfg, base_seed: int, config_name: str, filter_type: str) -> QuvineFilterConfig:
    """Build quantum-calibrated filter config from OmegaConf."""
    if not hasattr(cfg.baselines, config_name):
        return QuvineFilterConfig(enabled=False)
    
    qf = getattr(cfg.baselines, config_name)
    embedding_dim = getattr(qf, 'embedding_dim', None)
    if embedding_dim is None and hasattr(cfg, 'train'):
        embedding_dim = getattr(cfg.train, 'embedding_dim', 128)
    
    return QuvineFilterConfig(
        enabled=getattr(qf, 'enabled', False),
        filter_type=filter_type,
        embedding_dim=embedding_dim or 128,
        t=getattr(qf, 't', None),
        K=getattr(qf, 'K', 4),
        ridge=getattr(qf, 'ridge', 1e-6),
        normalize=getattr(qf, 'normalize', True),
        use_features=getattr(qf, 'use_features', False),
        features=getattr(qf, 'features', None),
        random_state=getattr(qf, 'random_state', base_seed)
    )


def build_quvine_gcnmf_config(cfg, base_seed: int, config_name: str, diffusion_type: str) -> QuvineGCNMFConfig:
    """Build quantum-calibrated GCN-MF config from OmegaConf."""
    if not hasattr(cfg.baselines, config_name):
        return QuvineGCNMFConfig(enabled=False)
    
    qg = getattr(cfg.baselines, config_name)
    embedding_dim = getattr(qg, 'embedding_dim', None)
    if embedding_dim is None and hasattr(cfg, 'train'):
        embedding_dim = getattr(cfg.train, 'embedding_dim', 128)
    
    return QuvineGCNMFConfig(
        enabled=getattr(qg, 'enabled', False),
        diffusion_type=diffusion_type,
        embedding_dim=embedding_dim or 128,
        hidden_dim=getattr(qg, 'hidden_dim', 64),
        mf_dim=getattr(qg, 'mf_dim', 64),
        n_layers=getattr(qg, 'n_layers', 2),
        epochs=getattr(qg, 'epochs', 200),
        lr=getattr(qg, 'lr', 0.01),
        weight_decay=getattr(qg, 'weight_decay', 5e-4),
        K=getattr(qg, 'K', 4),
        ridge=getattr(qg, 'ridge', 1e-6),
        normalize_laplacian=getattr(qg, 'normalize_laplacian', True),
        random_state=getattr(qg, 'random_state', base_seed)
    )


def build_quvine_sgns_config(cfg, base_seed: int, config_name: str, walk_kind: str) -> QuvineSGNSConfig:
    """
    Build config for a QuVINE SGNS walk embedding (quvine_rwr/ctqw/dtqw).

    The executor runs the shared SGNS core (views -> walks -> corpus ->
    word2vec) for the single ``walk_kind``, so the whole ``cfg`` is carried
    through. Enabled follows the same ``cfg.baselines.<name>.enabled`` pattern
    as the other methods.
    """
    enabled = False
    if hasattr(cfg, 'baselines') and hasattr(cfg.baselines, config_name):
        enabled = getattr(getattr(cfg.baselines, config_name), 'enabled', False)

    n_jobs, chunk_size = 1, 30
    if hasattr(cfg, 'runtime'):
        n_jobs = getattr(cfg.runtime, 'n_jobs', 1)
        chunk_size = getattr(cfg.runtime, 'chunk_size', 30)

    return QuvineSGNSConfig(
        enabled=enabled,
        walk_kind=walk_kind,
        cfg=cfg,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
    )


def build_netmf_config(cfg, base_seed: int) -> NetMFConfig:
    """Build NetMF config from OmegaConf."""
    if not hasattr(cfg.baselines, 'netmf'):
        return NetMFConfig(enabled=False)
    
    nmf = cfg.baselines.netmf
    dimensions = getattr(nmf, 'dimensions', None)
    if dimensions is None and hasattr(cfg, 'train'):
        dimensions = getattr(cfg.train, 'embedding_dim', 128)
    
    return NetMFConfig(
        enabled=getattr(nmf, 'enabled', False),
        dimensions=dimensions or 128,
        window_size=getattr(nmf, 'window_size', 10),
        rank=getattr(nmf, 'rank', 256),
        negative=getattr(nmf, 'negative', 1),
        seed=getattr(nmf, 'seed', base_seed)
    )


# ============================================================================
# Dataclass Definitions
# ============================================================================
