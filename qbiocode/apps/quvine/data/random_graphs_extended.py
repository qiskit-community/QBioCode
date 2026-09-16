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
Extended Random Graph Generator for QuVINE
==========================================

Combines original QuVINE random graph generators with five new synthetic graph
families especially useful for studying quantum-vs-classical random-walk embedding:

Original generators:

- Erdős-Rényi, Barabási-Albert, Watts-Strogatz, Powerlaw Cluster
- Stochastic Block Model, Random Geometric, Modular, Hierarchical
- Core-Periphery, Bipartite Random

New extended generators:

1. Random regular / expander-like graphs
2. Heterophilic / disassortative stochastic block models
3. Degree-corrected stochastic block models
4. Grid / torus lattices
5. Configuration-model graphs with power-law or log-normal degrees

Design notes
------------
- New generators favor target average degree parameterizations for fair comparison
- Graphs can optionally be made connected by adding minimal bridge edges
- Metadata is returned with every generated graph for downstream analysis
- All generators maintain backward compatibility with existing QuVINE code

Dependencies: networkx, numpy
"""

from __future__ import annotations

# Import everything from the original random_graphs module
from .random_graphs import *

# Additional imports for extended generators
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import math
import warnings

import networkx as nx
import numpy as np


# ===========================================================================
# UTILITIES (for new extended generators)
# ===========================================================================

GraphWithMeta = Tuple[nx.Graph, Dict[str, Any]]


def _rng(seed: Optional[int] = None) -> np.random.Generator:
    """Create numpy random generator."""
    return np.random.default_rng(seed)


def _stable_seed(base_seed: int, *items: Any) -> int:
    """Create a reproducible positive int seed from a base seed and parameters."""
    h = int(base_seed) & 0xFFFFFFFF
    for item in items:
        for ch in str(item):
            h = (1664525 * h + ord(ch) + 1013904223) & 0xFFFFFFFF
    return int(h)


def _add_metadata(G: nx.Graph, metadata: Dict[str, Any]) -> nx.Graph:
    """Attach graph-level metadata to G.graph and return G."""
    G.graph.update(metadata)
    return G


def _connect_components_by_bridges(G: nx.Graph, seed: Optional[int] = None) -> nx.Graph:
    """
    Connect disconnected components by adding one random bridge between adjacent
    components in a random component ordering. Preserves all nodes.
    """
    if G.number_of_nodes() == 0 or nx.is_connected(G):
        return G

    rng = _rng(seed)
    G = G.copy()
    comps = [list(c) for c in nx.connected_components(G)]
    rng.shuffle(comps)

    for c1, c2 in zip(comps[:-1], comps[1:]):
        u = rng.choice(c1)
        v = rng.choice(c2)
        G.add_edge(int(u), int(v))
    return G


def _postprocess_graph(
    G: nx.Graph,
    seed: Optional[int] = None,
    make_connected: bool = True,
    remove_selfloops: bool = True,
) -> nx.Graph:
    """Common cleanup for generated graphs."""
    if not isinstance(G, nx.Graph) or isinstance(G, nx.DiGraph):
        G = nx.Graph(G)
    else:
        G = G.copy()

    if remove_selfloops:
        G.remove_edges_from(nx.selfloop_edges(G))

    # Ensure integer labels 0..n-1
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")

    if make_connected and G.number_of_nodes() > 0 and not nx.is_connected(G):
        G = _connect_components_by_bridges(G, seed=seed)

    return G


def _graph_basic_metadata(G: nx.Graph) -> Dict[str, Any]:
    """Compute basic graph statistics for metadata."""
    degrees = np.array([d for _, d in G.degree()], dtype=float)
    return {
        "n_nodes_actual": int(G.number_of_nodes()),
        "n_edges_actual": int(G.number_of_edges()),
        "avg_degree_actual": float(degrees.mean()) if degrees.size else 0.0,
        "density_actual": float(nx.density(G)) if G.number_of_nodes() > 1 else 0.0,
        "is_connected": bool(nx.is_connected(G)) if G.number_of_nodes() > 0 else False,
    }


def _even_degree_sequence(deg: np.ndarray, n: int) -> np.ndarray:
    """Clip, round, and force a graphical-ish even degree sequence."""
    deg = np.asarray(deg, dtype=float)
    deg = np.nan_to_num(deg, nan=1.0, posinf=n - 1, neginf=1.0)
    deg = np.clip(np.rint(deg), 1, n - 1).astype(int)
    if deg.sum() % 2 == 1:
        idx = int(np.argmax(deg < n - 1))
        deg[idx] += 1
    return deg


def _rescale_degrees_to_target(raw: np.ndarray, target_avg_degree: float, n: int) -> np.ndarray:
    """Rescale degree sequence to target average degree."""
    raw = np.asarray(raw, dtype=float)
    raw = np.maximum(raw, 1e-12)
    raw = raw / raw.mean() * float(target_avg_degree)
    return _even_degree_sequence(raw, n=n)


def _balanced_block_sizes(n: int, n_blocks: int) -> List[int]:
    """Create balanced block sizes for SBM."""
    sizes = [n // n_blocks] * n_blocks
    sizes[-1] += n - sum(sizes)
    return sizes


def _sbm_prob_matrix_from_out_in_ratio(
    sizes: Sequence[int],
    target_avg_degree: float,
    out_in_ratio: float,
    p_in_floor: float = 1e-8,
    max_prob: float = 0.95,
) -> np.ndarray:
    """
    Build an SBM probability matrix with p_out / p_in = out_in_ratio and
    approximately target_avg_degree expected average degree.
    """
    sizes = np.asarray(sizes, dtype=float)
    q = float(out_in_ratio)
    if q < 0:
        raise ValueError("out_in_ratio must be non-negative")

    n = int(sizes.sum())
    desired_edges = n * float(target_avg_degree) / 2.0

    n_blocks = len(sizes)
    R = np.full((n_blocks, n_blocks), q, dtype=float)
    np.fill_diagonal(R, 1.0)

    raw_expected = 0.0
    for a in range(n_blocks):
        raw_expected += R[a, a] * sizes[a] * (sizes[a] - 1) / 2.0
        for b in range(a + 1, n_blocks):
            raw_expected += R[a, b] * sizes[a] * sizes[b]

    if raw_expected <= 0:
        raise ValueError("Invalid SBM expected edge count")

    scale = desired_edges / raw_expected
    P = np.clip(scale * R, p_in_floor, max_prob)
    return P


# Export all new functions
__all__ = [
    # Re-export from original module
    'generate_erdos_renyi',
    'generate_barabasi_albert',
    'generate_watts_strogatz',
    'generate_powerlaw_cluster',
    'generate_stochastic_block_model',
    'generate_random_geometric',
    'generate_modular_network',
    'generate_hierarchical_network',
    'generate_core_periphery',
    'generate_bipartite_random',
    'add_hub_nodes',
    'generate_graph_with_seeds_and_targets',
    'get_graph_statistics',
    # New extended generators (to be added in next part)
]
