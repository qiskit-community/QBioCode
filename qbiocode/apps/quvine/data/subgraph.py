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

from typing import Iterable, Set, Dict, Hashable, Any

import networkx as nx
import numpy as np


def materialize_undirected_simple_graph(G: nx.Graph) -> nx.Graph:
    """
    Return a fully materialized undirected simple nx.Graph (no views, no DiGraph internals).
    This avoids traversal issues from graph views and preserves node/edge attributes.
    """
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    H.add_edges_from(G.edges(data=True))
    return H


def induce_subgraph_by_nodes(G: nx.Graph, nodes: Set[Hashable]) -> nx.Graph:
    """
    Materialize an induced subgraph without using G.subgraph(...) to avoid view contamination.
    """
    nodes = set(nodes)
    H = nx.Graph()
    H.add_nodes_from((n, G.nodes[n]) for n in nodes)
    H.add_edges_from(
        (u, v, d) for (u, v, d) in G.edges(data=True) if u in nodes and v in nodes
    )
    return H


def expand_neighborhood(G: nx.Graph, roots: Set[Hashable], radius: int) -> Set[Hashable]:
    """
    Expand neighborhood around roots up to given hop radius using adjacency dict access.
    No G.neighbors() calls; 
    """
    # Filter absent roots *before* the radius check: callers sample roots from
    # G.nodes() and may prune the graph afterwards, and returning a node that is
    # not in G would make radius=0 disagree with every larger radius.
    node_set = set(G.nodes())
    roots = set(r for r in roots if r in node_set)

    if radius <= 0:
        return roots

    visited = set(roots)
    frontier = set(roots)

    for _ in range(radius):
        if not frontier:
            break
        next_frontier = set()
        for u in frontier:
            if u not in node_set:
                continue
            next_frontier.update(G.adj[u].keys())
        next_frontier -= visited
        visited |= next_frontier
        frontier = next_frontier

    return visited


def _degree_map(G: nx.Graph) -> Dict[Hashable, int]:
    return {node: int(G.degree[node]) for node in G.nodes()}


def _shuffle_nodes(nodes: Iterable[Hashable], rng: np.random.Generator) -> list[Hashable]:
    nodes = list(nodes)
    if nodes:
        rng.shuffle(nodes)
    return nodes


def _sample_degree_matched_nodes(
    G: nx.Graph,
    candidate_nodes: Iterable[Hashable],
    required_count: int,
    reference_degrees: Iterable[int],
    rng: np.random.Generator,
) -> Set[Hashable]:
    """
    Sample nodes whose full-graph degree distribution approximately matches
    the supplied reference degrees.

    The matching is exact at the degree level when enough candidates exist for
    those degrees; otherwise the sampler backs off to nearest available degrees.
    """
    if required_count <= 0:
        return set()

    degree_map = _degree_map(G)
    candidates = set(candidate_nodes)
    if len(candidates) <= required_count:
        return set(candidates)

    degree_to_candidates: Dict[int, list[Hashable]] = {}
    for node in candidates:
        deg = degree_map[node]
        degree_to_candidates.setdefault(deg, []).append(node)

    for nodes in degree_to_candidates.values():
        rng.shuffle(nodes)

    ref_degrees = list(reference_degrees)
    if not ref_degrees:
        ref_degrees = [degree_map[n] for n in candidates]

    rng.shuffle(ref_degrees)

    selected: Set[Hashable] = set()
    selected_degrees: Dict[int, int] = {}
    available_degrees = sorted(degree_to_candidates.keys())

    def take_from_degree(target_deg: int) -> bool:
        bucket = degree_to_candidates.get(target_deg, [])
        while bucket and bucket[-1] in selected:
            bucket.pop()
        if bucket:
            node = bucket.pop()
            selected.add(node)
            selected_degrees[target_deg] = selected_degrees.get(target_deg, 0) + 1
            return True
        return False

    for deg in ref_degrees:
        if len(selected) >= required_count:
            break
        if take_from_degree(deg):
            continue

        remaining_degrees = [
            d for d in available_degrees
            if any(node not in selected for node in degree_to_candidates.get(d, []))
        ]
        if not remaining_degrees:
            break

        nearest_deg = min(remaining_degrees, key=lambda d: (abs(d - deg), d))
        take_from_degree(nearest_deg)

    if len(selected) < required_count:
        leftovers = _shuffle_nodes(candidates - selected, rng)
        selected.update(leftovers[: required_count - len(selected)])

    return selected


def _trim_degree_matched(
    G: nx.Graph,
    nodes: Iterable[Hashable],
    max_nodes: int,
    protected: Set[Hashable],
    reference_degrees: Iterable[int],
    rng: np.random.Generator,
) -> Set[Hashable]:
    nodes = set(nodes)
    if len(nodes) <= max_nodes:
        return nodes

    protected = set(protected) & nodes
    if len(protected) > max_nodes:
        raise ValueError(
            f"Too many protected nodes ({len(protected)}) for max_nodes={max_nodes}"
        )

    remaining_budget = max_nodes - len(protected)
    optional_nodes = nodes - protected
    matched_optional = _sample_degree_matched_nodes(
        G=G,
        candidate_nodes=optional_nodes,
        required_count=remaining_budget,
        reference_degrees=reference_degrees,
        rng=rng,
    )
    return protected | matched_optional


def add_neighbors_until_budget(G, roots, keep, max_nodes, rng):
    frontier = list(roots)
    rng.shuffle(frontier)

    while frontier and len(keep) < max_nodes:
        u = frontier.pop()
        if u not in G:
            continue

        nbrs = list(G.adj[u].keys())
        rng.shuffle(nbrs)

        for v in nbrs:
            if len(keep) >= max_nodes:
                break
            if v not in keep:
                keep.add(v)
                frontier.append(v)

    return keep


def fill_random_until_budget(G, keep, max_nodes, rng):
    """Fill remaining budget preferring nodes adjacent to keep; fall back to random."""
    if len(keep) >= max_nodes:
        return keep

    # First pass: only nodes that have at least one edge into keep (won't be isolated)
    connectable = [n for n in set(G.nodes()) - keep
                   if any(v in keep for v in G.adj[n])]
    rng.shuffle(connectable)
    needed = max_nodes - len(keep)
    keep |= set(connectable[:needed])

    # Second pass: pure random fill for any remaining budget (last resort)
    if len(keep) < max_nodes:
        remaining = list(set(G.nodes()) - keep)
        rng.shuffle(remaining)
        needed = max_nodes - len(keep)
        keep |= set(remaining[:needed])

    return keep


def fill_degree_matched_until_budget(
    G: nx.Graph,
    keep: Set[Hashable],
    max_nodes: int,
    rng: np.random.Generator,
    *,
    reference_nodes: Iterable[Hashable] | None = None,
) -> Set[Hashable]:
    if len(keep) >= max_nodes:
        return keep

    degree_map = _degree_map(G)
    protected_reference = list(reference_nodes) if reference_nodes is not None else list(keep)
    reference_degrees = [degree_map[n] for n in protected_reference if n in degree_map]

    remaining = set(G.nodes()) - set(keep)
    needed = max_nodes - len(keep)
    matched = _sample_degree_matched_nodes(
        G=G,
        candidate_nodes=remaining,
        required_count=needed,
        reference_degrees=reference_degrees,
        rng=rng,
    )
    return set(keep) | matched


def subsample_nodes_with_protected(
    G: nx.Graph,
    protected: Iterable | None,
    max_nodes: int,
    rng: np.random.Generator,
    *,
    expand_radius: int | None = None,
    require_full_budget: bool = True,
    degree_matched_fill: bool = False,
    degree_matched_trim: bool = False,
    degree_reference_nodes: Iterable[Hashable] | None = None,
) -> nx.Graph:
    """
    Subsample nodes while preserving a protected set.

    Workflow:

    - keep all protected nodes
    - optionally expand by hop radius around them
    - if expansion exceeds budget, optionally trim in a degree-matched way
    - fill remaining budget either with neighborhood/random expansion or with
      degree-matched sampling from the remaining graph
    """
    G = materialize_undirected_simple_graph(G)
    node_set = set(G.nodes())
    if len(node_set) <= max_nodes:
        return G

    protected = set(protected or []) & node_set
    if len(protected) > max_nodes:
        raise ValueError(
            f"Too many protected nodes ({len(protected)}) for max_nodes={max_nodes}"
        )

    keep = set(protected)
    degree_reference_nodes = (
        list(degree_reference_nodes)
        if degree_reference_nodes is not None
        else list(node_set)
    )
    degree_map = _degree_map(G)
    reference_degrees = [degree_map[n] for n in degree_reference_nodes if n in degree_map]

    if expand_radius is not None and expand_radius > 0 and len(keep) > 0:
        keep = expand_neighborhood(G, keep, radius=expand_radius)
        if len(keep) > max_nodes:
            if degree_matched_trim:
                keep = _trim_degree_matched(
                    G=G,
                    nodes=keep,
                    max_nodes=max_nodes,
                    protected=protected,
                    reference_degrees=reference_degrees,
                    rng=rng,
                )
            else:
                keep = set(_shuffle_nodes(keep, rng)[:max_nodes])

    if len(keep) < max_nodes:
        if degree_matched_fill:
            keep = fill_degree_matched_until_budget(
                G,
                keep,
                max_nodes,
                rng,
                reference_nodes=degree_reference_nodes,
            )
        else:
            keep = add_neighbors_until_budget(
                G,
                roots=keep,
                keep=keep,
                max_nodes=max_nodes,
                rng=rng,
            )
            keep = fill_random_until_budget(G, keep, max_nodes, rng)

    H = induce_subgraph_by_nodes(G, keep)

    # Remove any isolated nodes introduced by last-resort random fill.
    # These have no edge to any other kept node and would produce degenerate
    # walk views. The fill_random_until_budget second pass can produce them
    # when the graph itself has low connectivity near the budget boundary.
    isolated = {n for n, d in H.degree() if d == 0}
    if isolated:
        H.remove_nodes_from(isolated)

    if require_full_budget and not isolated:
        assert H.number_of_nodes() >= min(max_nodes, len(node_set)), (
            H.number_of_nodes(), max_nodes, len(node_set)
        )

    return H


def subsample_nodes(
    G: nx.Graph,
    seeds: Iterable | None,
    targets: Iterable | None,
    max_nodes: int,
    radius: int,
    rng: np.random.Generator,
    *,
    degree_matched_fill: bool = False,
    degree_matched_trim: bool = False,
    degree_reference_nodes: Iterable[Hashable] | None = None,
) -> nx.Graph:
    protected = set(seeds or []) | set(targets or [])
    return subsample_nodes_with_protected(
        G,
        protected=protected,
        max_nodes=max_nodes,
        rng=rng,
        expand_radius=radius,
        degree_matched_fill=degree_matched_fill,
        degree_matched_trim=degree_matched_trim,
        degree_reference_nodes=degree_reference_nodes,
    )
