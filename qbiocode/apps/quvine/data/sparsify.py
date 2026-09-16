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


def materialize_undirected_simple_graph(G: nx.Graph) -> nx.Graph:
    """
    Return a fully materialized undirected simple nx.Graph (no views, no DiGraph internals).
    This avoids the weird KeyErrors you were seeing with adjacency traversal.
    Preserves graph-level metadata and node attributes.
    """
    H = nx.Graph()
    # Copy graph-level metadata
    H.graph.update(G.graph)
    # nodes with attributes
    H.add_nodes_from(G.nodes(data=True))
    # edges() - if input is directed/multi, nx.Graph() will collapse direction/multiedges
    H.add_edges_from(list(G.edges()))
    return H

def edge_triangle_support_scores(G: nx.Graph, candidate_edges: List[Tuple]) -> List[Tuple[Tuple, int]]:
    """
    Score edges by triangle support (# common neighbors). Uses adjacency dict access only.
    """
    # Ensure fully materialized to avoid any internal weirdness
    G = materialize_undirected_simple_graph(G)
    node_set = set(G.nodes())

    # Precompute neighbor sets via adjacency (safe)
    neigh: Dict = {u: set(G.adj[u].keys()) for u in node_set}

    scored: List[Tuple[Tuple, int]] = []
    for u, v in candidate_edges:
        if u not in node_set or v not in node_set:
            continue
        scored.append(((u, v), len(neigh[u].intersection(neigh[v]))))
    return scored



def sparsify_edges_biological(
    G: nx.Graph,
    retain_ratio: float,
    max_degree: int,
    rng: np.random.Generator,
    scoring: Literal["common_neighbors", "prefer_low_degree"] = "common_neighbors",
) -> nx.Graph:
    """
    Edge-only sparsification: degree-capped selection prioritizing biologically coherent edges.
    No BFS, no connectivity checks, no nx.is_connected/connected_components.
    """
    if not (0 < retain_ratio <= 1):
        raise ValueError("retain_ratio must be in (0, 1].")
    if max_degree <= 0:
        raise ValueError("max_degree must be positive.")

    G = materialize_undirected_simple_graph(G)
    if G.number_of_nodes() == 0:
        return G

    m0 = G.number_of_edges()
    target_edges = int(round(retain_ratio * m0))
    # Guarantee we don't request a negative/zero edge budget
    target_edges = max(0, target_edges)

    edges = list(G.edges())

    if scoring == "common_neighbors":
        # Score all edges
        scored = edge_triangle_support_scores(G, edges)
        rng.shuffle(scored)  # tie-breaking
        scored.sort(key=lambda x: x[1], reverse=True)
        sorted_edges = [e for (e, _) in scored]
    elif scoring == "prefer_low_degree":
        rng.shuffle(edges)
        edges.sort(key=lambda e: (G.degree[e[0]] + G.degree[e[1]]))
        sorted_edges = edges
    else:
        raise ValueError(f"Unknown scoring method: {scoring}")

    H = nx.Graph()
    H.add_nodes_from(G.nodes())

    # Degree-capped edge addition
    for u, v in sorted_edges:
        if H.number_of_edges() >= target_edges:
            break
        # nodes must exist (should, but guard anyway)
        if u not in H or v not in H:
            continue
        if H.degree[u] >= max_degree or H.degree[v] >= max_degree:
            continue
        if H.has_edge(u, v):
            continue
        H.add_edge(u, v)

    return H
