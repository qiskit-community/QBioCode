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

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Set, Tuple, List, Dict

import networkx as nx
import numpy as np
from .sparsify import materialize_undirected_simple_graph, sparsify_edges_biological
from .subgraph import subsample_nodes


def keep_largest_connected_component(
    G: nx.Graph,
    *,
    required_nodes: Iterable | None = None,
) -> nx.Graph:
    """
    Return a connected component while preserving required nodes when possible.

    If required_nodes are supplied and they all lie in the same connected
    component, that component is preferred over the global largest component.
    Otherwise, fall back to the largest connected component.
    """
    if G.number_of_nodes() == 0:
        return G

    if nx.is_connected(G):
        return G

    components = [set(c) for c in nx.connected_components(G)]
    required = set(required_nodes or []) & set(G.nodes())
    if required:
        for comp in components:
            if required.issubset(comp):
                return G.subgraph(comp).copy()

    lcc_nodes = max(components, key=len)
    return G.subgraph(lcc_nodes).copy()


# ---------------------------
# prepare_graph(): end-to-end, optional subsample, then sparsify edges
# ---------------------------

@dataclass
class PrepareGraphConfig:
    # Debug subsampling (optional)
    subsample_nodes: bool = False
    max_nodes: int = 3000
    radius: int = 2

    # Sparsification (optional but usually enabled)
    sparsify_edges: bool = True
    retain_ratio: float = 0.7
    max_degree: int = 40
    scoring: Literal["common_neighbors", "prefer_low_degree"] = "common_neighbors"

    # Verbose stats
    verbose: bool = False


def prepare_graph(
    cfg: PrepareGraphConfig,
    graph: nx.Graph,
    seed: int,
    *,
    seeds: Iterable | None = None,
    targets: Iterable | None = None
    ) -> nx.Graph:
    """
    Step-by-step graph preparation:

    1) Materialize a clean undirected simple graph (no views).
    2) If cfg.subsample_nodes: subsample nodes while protecting seeds+targets,
        expand neighborhood (radius hops), fill remaining budget.
    3) If cfg.sparsify_edges: degree-capped, biology-aware edge sparsification
        (triangle support or low-degree preference).
    4) Return a clean nx.Graph.

    Designed to avoid the KeyError / _succ / neighbor traversal issues you saw:
    - no subgraph views
    - no nx.is_connected / connected_components
    - no G.neighbors() calls
    - only adjacency dict access (G.adj[u])
    """
    if seed:
        rng = np.random.default_rng(seed)
    else:
        raise ValueError('Seed need to be passed')

    # Step 1: canonical materialization (always)
    G = materialize_undirected_simple_graph(graph)

    # Step 2: optional subsample (debug/notebook)
    if cfg.subsample_nodes:
        G = subsample_nodes(
            G,
            seeds=seeds,
            targets=targets,
            max_nodes=cfg.max_nodes,
            radius=cfg.radius,
            rng=rng,
        )

        G = keep_largest_connected_component(G, required_nodes=set(seeds or []) | set(targets or []))
        
    # Step 3: sparsify edges (biology-aware, degree-capped)
    if cfg.sparsify_edges:
        G = sparsify_edges_biological(
            G,
            retain_ratio=cfg.retain_ratio,
            max_degree=cfg.max_degree,
            rng=rng,
            scoring=cfg.scoring,
        )
    G = keep_largest_connected_component(G, required_nodes=set(seeds or []) | set(targets or []))
    # Step 4: lightweight stats (optional)
    if cfg.verbose and G.number_of_nodes() > 0:
        avg_deg = (2.0 * G.number_of_edges() / G.number_of_nodes()) if G.number_of_nodes() else 0.0
        max_deg = max(dict(G.degree()).values()) if G.number_of_nodes() else 0
        print("-" * 50)
        print("After prepare_graph()")
        print(f"Nodes: {G.number_of_nodes()}")
        print(f"Edges: {G.number_of_edges()}")
        print(f"Avg degree: {avg_deg:.2f}")
        print(f"Max degree: {max_deg}")
        print("-" * 50)

    return G
