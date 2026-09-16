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

import numpy as np
import random
import networkx as nx
from collections import defaultdict
import pandas as pd


class SeedTargetEvaluator:
    """
    Evaluates seed→target prioritization on a subgraph.
    """

    def __init__(self, subgraph, seeds, targets, nodes=None):
        self.subgraph = nx.Graph(subgraph)
        self.nodes = list(nodes) if nodes is not None else list(subgraph.nodes())

        self.node2i = {v: i for i, v in enumerate(self.nodes)}
        self.i2node = {i: v for v, i in self.node2i.items()}

        self.seeds = set(seeds) & set(self.nodes)
        self.targets = set(targets) & set(self.nodes)

        if not self.seeds:
            raise ValueError("SeedTargetEvaluator requires at least one seed node in the subgraph.")

        self.seed_indices = {self.node2i[v] for v in self.seeds}
        self.target_indices = {self.node2i[v] for v in self.targets}

        self.candidate_indices = [
            self.node2i[v] for v in self.nodes if v not in self.seeds
        ]
        self.candidate_nodes = [self.i2node[i] for i in self.candidate_indices]

        self.deg = dict(self.subgraph.degree())
        self.dist = self._compute_min_distance_to_seeds()

    # ---------------------------
    # Internals
    # ---------------------------

    def _compute_min_distance_to_seeds(self):
        # Single O(V+E) multi-source BFS instead of O(|seeds|*(V+E)) individual calls.
        lengths = nx.multi_source_dijkstra_path_length(self.subgraph, self.seeds)
        return {v: lengths.get(v, float("inf")) for v in self.nodes}

    def _rank_indices(self, scores):
        return sorted(
            self.candidate_indices,
            key=lambda i: scores[i],
            reverse=True
        )

    # ---------------------------
    # Metrics
    # ---------------------------

    def recall_at_k(self, scores, target_indices, k):
        ranked = self._rank_indices(scores)
        top_k = ranked[:k]
        hits = sum(i in target_indices for i in top_k)
        return hits / max(len(target_indices), 1)

    def precision_at_k(self, scores, target_indices, k):
        ranked = self._rank_indices(scores)
        top_k = ranked[:k]
        hits = sum(i in target_indices for i in top_k)
        return hits / k

    # ---------------------------
    # Controls
    # ---------------------------

    def degree_matched_targets(self, tolerance=0.1, random_state=None):
        if random_state is not None:
            random.seed(random_state)

        matched = set()
        candidates = set(self.candidate_nodes) - self.targets

        for t in self.targets:
            dt = self.deg[t]
            low = int((1 - tolerance) * dt)
            high = int((1 + tolerance) * dt)

            pool = [
                v for v in candidates
                if low <= self.deg[v] <= high
            ]
            if pool:
                matched.add(random.choice(pool))

        return matched

    def distance_matched_targets(self, random_state=None):
        if random_state is not None:
            random.seed(random_state)

        matched = set()
        candidates = set(self.candidate_nodes) - self.targets

        bins = defaultdict(list)
        for v in candidates:
            bins[self.dist[v]].append(v)

        for t in self.targets:
            d = self.dist[t]
            if bins[d]:
                matched.add(random.choice(bins[d]))

        return matched

    # ---------------------------
    # Main evaluation
    # ---------------------------

    def evaluate(self, scores, k_values=[20, 50, 100], n_repeats=20, deg_tol=0.1):
        results = {}

        # True targets
        results["true"] = {
            "recall": {
                k: self.recall_at_k(scores, self.target_indices, k)
                for k in k_values
            },
            "precision": {
                k: self.precision_at_k(scores, self.target_indices, k)
                for k in k_values
            }
        }

        # Degree-matched
        deg_recalls = {k: [] for k in k_values}
        deg_precisions = {k: [] for k in k_values}

        for r in range(n_repeats):
            dm = self.degree_matched_targets(
                tolerance=deg_tol,
                random_state=r
            )
            dm_idx = {self.node2i[v] for v in dm}

            for k in k_values:
                deg_recalls[k].append(self.recall_at_k(scores, dm_idx, k))
                deg_precisions[k].append(self.precision_at_k(scores, dm_idx, k))

        results["degree_matched"] = {
            "recall": {
                k: (np.mean(deg_recalls[k]), np.std(deg_recalls[k]))
                for k in k_values
            },
            "precision": {
                k: (np.mean(deg_precisions[k]), np.std(deg_precisions[k]))
                for k in k_values
            }
        }

        # Distance-matched
        dist_recalls = {k: [] for k in k_values}
        dist_precisions = {k: [] for k in k_values}

        for r in range(n_repeats):
            dm = self.distance_matched_targets(random_state=r)
            dm_idx = {self.node2i[v] for v in dm}

            for k in k_values:
                dist_recalls[k].append(self.recall_at_k(scores, dm_idx, k))
                dist_precisions[k].append(self.precision_at_k(scores, dm_idx, k))

        results["distance_matched"] = {
            "recall": {
                k: (np.mean(dist_recalls[k]), np.std(dist_recalls[k]))
                for k in k_values
            },
            "precision": {
                k: (np.mean(dist_precisions[k]), np.std(dist_precisions[k]))
                for k in k_values
            }
        }

        return results

def evaluate_embeddings_ranking(
    scores_by_method,
    subgraph,
    seeds,
    targets,
    nodes,
    k_values=[30, 60, 90, 120],
    n_repeats=30,
    deg_tol=0.1,
    iteration=None,
):
    """Score every method's node ranking against the target set.

    Args:
        scores_by_method (Dict[str, np.ndarray]): Mapping from method name to
            node-level score vector.

    Returns:
        pd.DataFrame: One tidy row per method, k value and repeat.
    """
    subgraph = nx.Graph(subgraph)
    valid_seeds = set(seeds) & set(subgraph.nodes())
    valid_targets = set(targets) & set(subgraph.nodes())

    if len(valid_seeds) == 0:
        raise ValueError("No seed nodes present in evaluation graph")

    if len(valid_targets) == 0:
        raise ValueError("No target nodes present in evaluation graph")

    st_eval = SeedTargetEvaluator(
        subgraph=subgraph,
        seeds=valid_seeds,
        targets=valid_targets,
        nodes=nodes,
    )

    rows = []

    for method, scores in scores_by_method.items():
        res = st_eval.evaluate(
            scores=scores,
            k_values=k_values,
            n_repeats=n_repeats,
            deg_tol=deg_tol,
        )

        for control in ["true", "degree_matched", "distance_matched"]:
            for metric in ["recall", "precision"]:
                for k in k_values:
                    if control == "true":
                        mean = res[control][metric][k]
                        std = 0.0
                    else:
                        mean, std = res[control][metric][k]

                    rows.append({
                        "iteration": iteration,
                        "method": method,
                        "control": control,
                        "metric": metric,
                        "k": k,
                        "mean": mean,
                        "std": std,
                    })

    return pd.DataFrame(rows)


def l2_normalize(X, axis=1, eps=1e-12):
    norm = np.linalg.norm(X, axis=axis, keepdims=True)
    return X / (norm + eps)

def seed_centroid_scores(Z, seed_indices):
    """
    Generates centroid scores based on seed node embeddings. 
    
    Z: (n,e) embedding matrix aligned with subgraph nodes 
    seed_indices: list of indices corresponding to seed nodes in Z
    
    Returns
    -------
    scores: cosine similarity to seed centroid 
    
    """
    if len(seed_indices) == 0:
        return np.zeros(Z.shape[0], dtype=float)

    Z_norm = l2_normalize(Z, axis=1)
    Z_seed = Z_norm[seed_indices]
    centroid = Z_seed.mean(axis=0)
    centroid /= max(np.linalg.norm(centroid), 1e-12)
    scores = Z_norm @ centroid
    return scores

def max_seed_cosine_scores(Z, seed_indices, block=4096):
    """
    Returns scores[v] = max_{s in seeds} cosine(z_v, z_s)
    Works for any embedding Z.

    block: compute in chunks to reduce memory if needed.
    """
    Z = l2_normalize(Z)
    seed_idx = list(seed_indices)

    if len(seed_idx) == 0:
        return np.zeros(Z.shape[0], dtype=float)

    S = Z[seed_idx]  # (m, e)

    scores = np.full(Z.shape[0], -np.inf, dtype=float)

    # chunked matmul: (block × e) @ (e × m) -> (block × m)
    for start in range(0, Z.shape[0], block):
        end = min(start + block, Z.shape[0])
        sims = Z[start:end] @ S.T
        scores[start:end] = sims.max(axis=1)

    return scores


