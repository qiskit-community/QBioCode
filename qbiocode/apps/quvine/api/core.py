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
Public embedding-extraction API for QuVINE.

    from qbiocode.apps.quvine import embed
    result = embed(graph, "quvine_rwr")              # SGNS walk embedding
    result = embed(graph, "gat_ctqw_heat", seeds=s)  # one registry method
    Z = result.embedding                             # (n_nodes, dim) ndarray

``embed`` returns an :class:`EmbedResult` carrying the matrix, the node order
its rows align to, and metadata about what was actually run.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import networkx as nx
import numpy as np
from omegaconf import DictConfig, OmegaConf

from qbiocode.apps.quvine.api.aliases import FUSED_ALIASES, config_group_for, list_methods, resolve_method, variant_for
from qbiocode.apps.quvine.api.config import load_config
from qbiocode.apps.quvine.api.sgns import run_sgns
from qbiocode.apps.quvine.api.targets import build_walk_targets, select_calibration_seeds

logger = logging.getLogger(__name__)

# Names that mean "run all walk kinds and fuse them into one embedding". Sourced
# from aliases so the CLI's --list-methods and resolve_method agree with what
# embed() actually accepts.
_FUSED_NAMES = set(FUSED_ALIASES)


class QuvineMethodError(RuntimeError):
    """Raised when a requested method cannot be run or fails during execution."""


@dataclass
class EmbedResult:
    """Result of :func:`embed`."""

    embedding: np.ndarray            # (n_nodes, dim), rows aligned to node_order
    node_order: list                 # == list(graph.nodes)
    method: str                      # canonical name actually executed
    requested_method: str            # the string the caller passed
    kind: str                        # "sgns" | "registry" | "sgns_fused"
    dim: int
    execution_time: float = 0.0
    used_quantum_targets: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[Any, np.ndarray]:
        """Return a ``{node: vector}`` mapping."""
        return {n: self.embedding[i] for i, n in enumerate(self.node_order)}


def embed(
    graph: nx.Graph,
    method: str,
    config: Optional[Union[DictConfig, dict, str]] = None,
    *,
    seeds: Optional[Sequence] = None,
    base_seed: Optional[int] = None,
    fuse: bool = False,
    fuse_method: Optional[str] = None,
    overrides: Optional[Union[dict, list, DictConfig]] = None,
    n_jobs: int = 1,
    chunk_size: int = 30,
    verbose: bool = False,
) -> EmbedResult:
    """
    Extract a single QuVINE embedding from ``graph`` using ``method``.

    Args:
        graph: NetworkX graph to embed.
        method: Method name. SGNS walks (``"quvine_rwr"``, ``"ctqw"``, ...),
            a registry method (``"node2vec"``, ``"gat_ctqw_heat"``,
            ``"quvine_graphgps_dtqw_poly"``, ...), or a fused name
            (``"quvine_fused"``). See :func:`quvine.api.list_methods`.
        config: OmegaConf/dict/path or None for the packaged default schema.
        seeds: Seed nodes used to build quantum targets for quantum-calibrated
            registry methods (gat_*/graphgps_*/filter_*/quvine_* GCN-MF).
        base_seed: Override ``cfg.experiment.base_seed``.
        fuse: If True, run all ``cfg.walks.kinds`` SGNS embeddings and fuse them
            (equivalent to passing a fused method name).
        fuse_method: Fusion strategy; defaults to ``cfg.fusion.method``.
        overrides: Deep-merged onto ``config`` (dict / DictConfig / dotlist).
        n_jobs, chunk_size: joblib parallelism for the SGNS walk loop.
        verbose: Verbose logging for registry execution.

    Returns:
        An :class:`EmbedResult`.

    Raises:
        QuvineMethodError: unknown method, missing quantum targets, or a failed
            registry executor.
    """
    cfg = load_config(config, overrides)
    # Resolve the base seed and thread it back into cfg so every path (SGNS,
    # fused, and registry) uses the same seed -- otherwise base_seed would be
    # silently ignored by the walk-based embeddings.
    if base_seed is None:
        base_seed = int(cfg.experiment.base_seed)
    else:
        cfg = OmegaConf.merge(cfg, {"experiment": {"base_seed": int(base_seed)}})
    node_order = list(graph.nodes)

    # Fused requests (explicit name or fuse=True) run all walk kinds + fusion.
    if fuse or method.strip().lower() in _FUSED_NAMES:
        return _embed_fused(
            cfg, graph, node_order, method, base_seed,
            fuse_method=fuse_method, n_jobs=n_jobs, chunk_size=chunk_size,
        )

    try:
        kind, key = resolve_method(method)
    except KeyError as exc:
        raise QuvineMethodError(str(exc)) from exc

    if kind == "fused":
        # Reachable only if a fused alias is added to the tables without the
        # short-circuit above matching it; dispatch correctly rather than
        # falling through to the registry executor.
        return _embed_fused(
            cfg, graph, node_order, method, base_seed,
            fuse_method=fuse_method, n_jobs=n_jobs, chunk_size=chunk_size,
        )
    if kind == "sgns":
        return _embed_sgns(
            cfg, graph, node_order, method, key, base_seed,
            n_jobs=n_jobs, chunk_size=chunk_size,
        )
    return _embed_registry(
        cfg, graph, node_order, method, key, base_seed,
        seeds=seeds, verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------

def _embed_sgns(cfg, graph, node_order, requested, walk_kind, base_seed,
                *, n_jobs, chunk_size) -> EmbedResult:
    start = time.time()
    embeddings = run_sgns(
        cfg, graph, it=0, kinds=[walk_kind], n_jobs=n_jobs, chunk_size=chunk_size
    )
    Z = embeddings[walk_kind]
    return EmbedResult(
        embedding=Z,
        node_order=node_order,
        method=f"quvine_{walk_kind}",
        requested_method=requested,
        kind="sgns",
        dim=int(Z.shape[1]),
        execution_time=time.time() - start,
        used_quantum_targets=False,
        extra={"walk_kind": walk_kind},
    )


def _embed_fused(cfg, graph, node_order, requested, base_seed,
                 *, fuse_method, n_jobs, chunk_size) -> EmbedResult:
    from qbiocode.apps.quvine.embedding.registry import EmbeddingStore
    from qbiocode.apps.quvine.fusion.fuse import fuse_embeddings

    start = time.time()
    kinds = list(cfg.walks.kinds)
    embeddings = run_sgns(
        cfg, graph, it=0, kinds=kinds, n_jobs=n_jobs, chunk_size=chunk_size
    )
    store = EmbeddingStore()
    for name, Z in embeddings.items():
        store.add(name, Z)

    method_name = fuse_method or cfg.fusion.method
    # Mirror the pipeline: provide a dense normalized Laplacian for fusion
    # strategies that need it.
    L = nx.normalized_laplacian_matrix(
        G=graph, nodelist=graph.nodes
    ).toarray().astype(np.float32)

    fused_list, fuse_names = fuse_embeddings(
        store, method=method_name, k=cfg.fusion.k, L=L
    )
    if not fused_list:
        raise QuvineMethodError(
            f"Fusion with method {method_name!r} returned no embeddings."
        )
    Z = fused_list[0]
    return EmbedResult(
        embedding=Z,
        node_order=node_order,
        method=f"fused:{fuse_names[0]}",
        requested_method=requested,
        kind="sgns_fused",
        dim=int(Z.shape[1]),
        execution_time=time.time() - start,
        used_quantum_targets=False,
        extra={
            "walk_kinds": kinds,
            "fusion_method": method_name,
            "fused": {name: emb for name, emb in zip(fuse_names, fused_list)},
        },
    )


def _embed_registry(cfg, graph, node_order, requested, key, base_seed,
                    *, seeds, verbose) -> EmbedResult:
    from qbiocode.apps.quvine.baselines.registration import register_all_methods
    from qbiocode.apps.quvine.baselines.registry import MethodRegistry

    # Auto-enable the explicitly requested method (it may be off in YAML) by
    # merging an override into a fresh copy of the config.
    group = config_group_for(key)
    group_override = {"enabled": True}
    # GAT/GraphGPS: set the `variant` implied by the method name (e.g.
    # gat_rwr_heat -> heat_qcal_rwr) unless the caller already specified one,
    # otherwise the config builder defaults to a plain 'raw' (non-quantum) run.
    variant = variant_for(key)
    if variant is not None:
        existing = OmegaConf.select(cfg, f"baselines.{group}.variant", default=None)
        if existing is None:
            group_override["variant"] = variant
    cfg_run = OmegaConf.merge(cfg, {"baselines": {group: group_override}})

    registry = MethodRegistry(cfg_run, base_seed=base_seed, verbose=verbose)
    register_all_methods(registry)

    meta = registry.get_metadata(key)
    if meta is None:
        raise QuvineMethodError(f"Method {key!r} is not registered.")

    # Quantum-calibrated methods need walk targets. Derive the walk type from the
    # variant string (e.g. "heat_qcal_rwr" -> "rwr", "heat_qcal_ctqw" -> "ctqw").
    # For filter_ methods the key itself encodes the walk (filter_rwr_heat -> "rwr").
    # If the caller passed no seeds, infer dispersed landmarks (farthest-point/k-center).
    q_targets = None
    inferred_seeds = None
    walk_type = None
    if meta.requires_q_targets:
        walk_type = _walk_type_from_variant_or_key(variant, key)
        cal_seeds = list(seeds) if seeds else None
        if not cal_seeds:
            inferred_seeds = select_calibration_seeds(graph)
            cal_seeds = inferred_seeds
            logger.info(
                "Method %r needs calibration seeds; none provided. Inferred %d "
                "data-driven landmarks (farthest-point): %s",
                requested, len(cal_seeds), cal_seeds,
            )
        q_targets = build_walk_targets(graph, cal_seeds, walk_type=walk_type)
        if q_targets is None:
            raise QuvineMethodError(
                f"Method {requested!r} (registry key {key!r}) requires calibration "
                f"targets but none could be built from seeds {cal_seeds!r}."
            )

    result = registry.run_method(key, graph, q_targets)
    if not result.success or result.embedding is None:
        raise QuvineMethodError(
            f"Method {requested!r} (registry key {key!r}) failed: {result.error}"
        ) from result.exception

    Z = result.embedding
    extra: dict = {"category": meta.category}
    if inferred_seeds is not None:
        extra["inferred_seeds"] = inferred_seeds
        extra["seed_selection"] = "kcenter"
    if walk_type is not None and q_targets is not None:
        extra["walk_type"] = walk_type

    return EmbedResult(
        embedding=Z,
        node_order=node_order,
        method=key,
        requested_method=requested,
        kind="registry",
        dim=int(Z.shape[1]),
        execution_time=result.execution_time,
        used_quantum_targets=q_targets is not None and meta.requires_q_targets,
        extra=extra,
    )


def _walk_type_from_variant_or_key(variant: Optional[str], key: str) -> str:
    """
    Derive the walk type for ``build_walk_targets`` from the variant string or key.

    Priority: variant (set by ``variant_for(key)`` before this runs) > key suffix.

    Examples:
        variant="heat_qcal_rwr"  -> "rwr"
        variant="heat_qcal_ctqw" -> "ctqw"
        key="filter_rwr_heat"    -> "rwr"   (variant=None for filter methods)
    """
    if variant is not None:
        # variant like "heat_qcal_ctqw" or "poly_qcal_rwr"
        for walk in ("ctqw", "dtqw", "rwr"):
            if variant.endswith(f"_{walk}"):
                return walk
    # Fall back: second-to-last token in key (filter_rwr_heat, quvine_ctqw, …)
    parts = key.split("_")
    for walk in ("ctqw", "dtqw", "rwr"):
        if walk in parts:
            return walk
    # Last resort: use key's trailing token
    return parts[-1] if parts else "rwr"


__all__ = ["embed", "EmbedResult", "QuvineMethodError", "list_methods", "load_config"]
