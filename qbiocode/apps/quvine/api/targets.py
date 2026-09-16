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
Quantum-target construction for quantum-calibrated registry methods.

``build_walk_targets`` is the real, walk-based target generator used by
:func:`quvine.embed`: for each calibration seed it samples a local subgraph and
runs the requested walk (CTQW/DTQW via hiperwalk, or classical RWR) to produce
the probability distribution ``pQ`` that the heat/poly filter is calibrated to.
This is the same construction the HPC engine
(``comprehensive_embedding_analysis._generate_quantum_targets``) uses, so the
``ctqw``/``dtqw``/``rwr`` variants produce genuinely different calibrations.

``select_calibration_seeds`` infers dispersed, traversal-useful calibration
centers (farthest-point / k-center landmarks) when the caller passes none.

``build_quantum_targets`` is the legacy index-distance stub, kept only for
backward compatibility; ``embed()`` no longer uses it.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


def select_calibration_seeds(
    G: nx.Graph,
    k: Optional[int] = None,
    seed: int = 0,
) -> List:
    """
    Select dispersed, traversal-useful calibration seeds via farthest-point
    (k-center) landmark selection.

    A naive "top-k degree" choice tends to pick mutually-adjacent hub nodes in
    one dense core, giving overlapping calibration subgraphs. Instead: anchor on
    the highest-degree node, then greedily add the node that is *farthest* (in
    shortest-path hops) from the already-selected set, maintaining a running
    nearest-landmark-distance array. This guarantees the landmarks are pairwise
    non-adjacent (>=2 hops) and spread across the graph.

    Cost is K BFS sweeps total (``O(K*(V+E))``) using scipy CSR BFS; runs
    per-connected-component so disconnected graphs still get coverage.

    Args:
        G: graph.
        k: number of landmarks; defaults to ``max(3, min(10, n // 20))``.
        seed: reserved for reproducibility (selection is deterministic).

    Returns:
        List of node ids (the graph's own labels).
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import breadth_first_order

    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return []
    if k is None:
        k = max(3, min(10, n // 20))
    k = min(k, n)

    idx_of = {node: i for i, node in enumerate(nodes)}
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr", dtype=float)

    degrees = np.asarray(A.sum(axis=1)).ravel()

    def bfs_hops(src_i: int) -> np.ndarray:
        """Unweighted shortest-path hop counts from src_i (inf if unreachable)."""
        order, preds = breadth_first_order(A, src_i, directed=False, return_predecessors=True)
        dist = np.full(n, np.inf)
        dist[src_i] = 0.0
        # Reconstruct hop distance by walking predecessors in BFS order.
        for v in order:
            p = preds[v]
            if p >= 0:
                dist[v] = dist[p] + 1.0
        return dist

    # Anchor: highest degree, deterministic id tie-break.
    first = max(range(n), key=lambda i: (degrees[i], -i))
    chosen = [first]
    min_dist = bfs_hops(first)

    while len(chosen) < k:
        # Farthest node from the chosen set; ignore unreachable (inf) unless we
        # must (then we still pick the max finite, else any unchosen node).
        candidate = int(np.argmax(np.where(np.isin(np.arange(n), chosen), -1.0, min_dist)))
        if candidate in chosen or not np.isfinite(min_dist[candidate]) or min_dist[candidate] <= 0:
            # No reachable node left with positive distance (e.g. another
            # component): fall back to the highest-degree unchosen node.
            remaining = [i for i in range(n) if i not in chosen]
            if not remaining:
                break
            candidate = max(remaining, key=lambda i: (degrees[i], -i))
        chosen.append(candidate)
        min_dist = np.minimum(min_dist, bfs_hops(candidate))

    return [nodes[i] for i in chosen]


def build_walk_targets(
    G: nx.Graph,
    seeds: Sequence,
    walk_type: str = "ctqw",
    num_subgraphs: int = 5,
    subgraph_size: int = 20,
    steps: int = 20,
    seed: int = 42,
) -> Optional[List[Dict]]:
    """
    Build real walk-based calibration targets.

    For up to ``num_subgraphs`` seeds, expand a <=2-hop subgraph (capped at
    ``subgraph_size``, restricted to the component containing the center), run
    the requested walk from the center, and record the node-probability
    distribution as ``pQ``. The returned ``{"nodes","center","pQ"}`` dicts
    satisfy the contract validated by ``gat.calibrate_heat_kernel`` /
    ``calibrate_polynomial_filter``.

    Args:
        G: graph.
        seeds: calibration center node ids (only those present are used).
        walk_type: ``"ctqw"`` | ``"dtqw"`` (need hiperwalk) | ``"rwr"`` (classical).
        num_subgraphs, subgraph_size, steps: sampling controls.
        seed: RNG seed for subgraph sampling (deterministic).

    Returns:
        List of target dicts, or ``None`` if no valid seed is present.

    Raises:
        QuvineMethodError: if ``walk_type`` is ctqw/dtqw but hiperwalk is missing.
        ValueError: on unknown ``walk_type``.
    """
    from qbiocode.apps.quvine.data.subgraph import expand_neighborhood

    node_set = set(G.nodes())
    valid_seeds = [s for s in seeds if s in node_set]
    if not valid_seeds:
        return None

    if walk_type == "ctqw":
        scorer = _ctqw_scorer(steps)
    elif walk_type == "dtqw":
        scorer = _dtqw_scorer(steps)
    elif walk_type == "rwr":
        from qbiocode.apps.quvine.walks.rwr import get_RWR_pagerank_scores
        scorer = lambda H, center: get_RWR_pagerank_scores(H, root=center)
    else:
        raise ValueError(f"Unknown walk_type {walk_type!r}; choose 'ctqw', 'dtqw', or 'rwr'.")

    rng = np.random.default_rng(seed)
    sampled = rng.choice(
        np.array(valid_seeds, dtype=object),
        size=min(num_subgraphs, len(valid_seeds)),
        replace=False,
    )

    targets: List[Dict] = []
    for center in sampled:
        try:
            subgraph_nodes = expand_neighborhood(G, {center}, radius=2)
            if len(subgraph_nodes) > subgraph_size:
                picked = rng.choice(list(subgraph_nodes), size=subgraph_size, replace=False)
                subgraph_nodes = set(picked.tolist())
                subgraph_nodes.add(center)
            H = G.subgraph(subgraph_nodes).copy()
            if not nx.is_connected(H):
                H = H.subgraph(nx.node_connected_component(H, center)).copy()
            if H.number_of_nodes() < 3:
                continue

            scores = scorer(H, center)
            nodes_list = list(H.nodes())
            pQ = np.array([scores.get(nd, 0.0) for nd in nodes_list], dtype=np.float64)
            total = pQ.sum()
            if total <= 0:
                continue
            targets.append({"nodes": nodes_list, "center": center, "pQ": pQ / total})
        except Exception as exc:  # noqa: BLE001 -- per-seed best effort
            logger.warning("walk-target generation failed for seed %r: %s", center, exc)
            continue

    return targets or None


def _ctqw_scorer(steps: int):
    try:
        from qbiocode.apps.quvine.walks.ctqw import generate_ctqw_hiperwalk_scores
    except ImportError as exc:  # hiperwalk missing
        from qbiocode.apps.quvine.api.core import QuvineMethodError
        raise QuvineMethodError(
            "ctqw calibration targets require the 'hiperwalk' package. Install it, "
            "or use an '_rwr_' method (classical, no hiperwalk needed)."
        ) from exc
    return lambda H, center: generate_ctqw_hiperwalk_scores(H, root=center, steps=steps)


def _dtqw_scorer(steps: int):
    try:
        from qbiocode.apps.quvine.walks.dtqw import get_coined_hiperwalk_scores
    except ImportError as exc:
        from qbiocode.apps.quvine.api.core import QuvineMethodError
        raise QuvineMethodError(
            "dtqw calibration targets require the 'hiperwalk' package. Install it, "
            "or use an '_rwr_' method (classical, no hiperwalk needed)."
        ) from exc
    return lambda H, center: get_coined_hiperwalk_scores(H, root=center, steps=steps)


def build_quantum_targets(
    graph: nx.Graph,
    seeds: Sequence,
    max_support: int = 64,
) -> Optional[List[dict]]:
    """
    DEPRECATED — use :func:`build_walk_targets`. This builds a walk-AGNOSTIC
    placeholder (``pQ`` from node-index distance), so it cannot distinguish
    ctqw/dtqw/rwr. Kept only for backward compatibility; ``embed()`` no longer
    calls it.

    Build per-seed quantum targets used to calibrate quantum filters.

    For each valid seed, collect its <=2-hop neighborhood (capped at
    ``max_support`` nodes) and assign a normalized inverse-distance target
    distribution ``pQ`` over that support.

    Args:
        graph: NetworkX graph.
        seeds: Seed node ids (only those present in the graph are used).
        max_support: Maximum number of support nodes per seed.

    Returns:
        A list of ``{"nodes", "center", "pQ"}`` dicts, or ``None`` if no seed
        is present in the graph.
    """
    node_order = list(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_order)}
    valid_seeds = [node for node in seeds if node in node_to_idx]
    if not valid_seeds:
        return None

    targets = []
    for center in valid_seeds:
        lengths = nx.single_source_shortest_path_length(graph, center, cutoff=2)
        support_nodes = [node for node, dist in lengths.items() if dist <= 2]
        if center not in support_nodes:
            support_nodes.append(center)
        if len(support_nodes) > max_support:
            support_nodes = support_nodes[:max_support]

        center_idx = node_to_idx[center]
        support_idx = np.array([node_to_idx[node] for node in support_nodes], dtype=np.int64)
        dist = np.abs(support_idx - center_idx).astype(np.float64) + 1.0
        p = 1.0 / dist
        p = p / p.sum()
        targets.append({"nodes": support_nodes, "center": center, "pQ": p})

    return targets
