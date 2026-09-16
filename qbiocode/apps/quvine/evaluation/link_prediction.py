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
Link Prediction Evaluation Module

This module provides functions for evaluating node embeddings on link prediction tasks.
Includes edge sampling strategies, edge feature computation, and comprehensive evaluation metrics.

Edge Sampling Strategies:

1. Random negative sampling
2. Hard negatives: 2-hop node pairs (common neighbors but no edge)
3. Same-community non-edges
4. Stratified sampling by node degree

Edge Feature Computation:

1. Hadamard product: ``u ⊙ v``
2. Average: ``(u + v) / 2``
3. L1 distance: ``|u - v|``
4. L2 distance: ``||u - v||₂``
5. Concatenation: ``[u; v]``
6. Inner product (dot product): ``u · v``
7. Cosine similarity: ``(u · v) / (||u|| ||v||)``

Evaluation Metrics:

- AUC-ROC, AUC-PR
- Precision@K, Recall@K
- Hit@K, MRR (Mean Reciprocal Rank)
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings


def _make_bidi(edges) -> Set[Tuple]:
    """Return a bidirectional (both-direction) copy of an edge collection."""
    s = set(edges)
    s.update((v, u) for u, v in edges)
    return s


def sample_negative_edges(
    G: nx.Graph,
    n_samples: int,
    existing_edges: Optional[Set[Tuple[int, int]]] = None,
    strategy: str = 'random',
    seed: int = 42
) -> List[Tuple[int, int]]:
    """
    Sample negative edges (non-existent edges) from the graph.
    
    Args:
        G: NetworkX graph
        n_samples: Number of negative edges to sample
        existing_edges: Set of existing edges to avoid
        strategy: Sampling strategy ('random', 'hard_2hop', 'same_community')
        seed: Random seed
    
    Returns:
        List of negative edge tuples
    """
    # A local generator, not np.random.seed(): reseeding the global RNG makes every
    # subsequent random draw in the calling process depend on this function having been
    # called, which silently couples unrelated results together.
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    
    if existing_edges is None:
        existing_edges = set(G.edges())

    existing_edges_bidirectional = _make_bidi(existing_edges)
    
    negative_edges = []
    
    if strategy == 'random':
        # Random negative sampling
        max_attempts = n_samples * 10
        attempts = 0
        
        while len(negative_edges) < n_samples and attempts < max_attempts:
            u = nodes[rng.integers(n_nodes)]
            v = nodes[rng.integers(n_nodes)]
            
            if u != v and (u, v) not in existing_edges_bidirectional:
                negative_edges.append((u, v))
                existing_edges_bidirectional.add((u, v))
                existing_edges_bidirectional.add((v, u))
            
            attempts += 1
    
    elif strategy == 'hard_2hop':
        # Hard negatives: 2-hop node pairs (common neighbors but no edge)
        for u in nodes:
            u_neighbors = set(G.neighbors(u))
            # Find 2-hop neighbors
            two_hop_neighbors = set()
            for neighbor in u_neighbors:
                two_hop_neighbors.update(G.neighbors(neighbor))
            
            # Remove 1-hop neighbors and self
            two_hop_neighbors -= u_neighbors
            two_hop_neighbors.discard(u)
            
            # Sample from 2-hop neighbors
            candidates = [(u, v) for v in two_hop_neighbors 
                         if (u, v) not in existing_edges_bidirectional]
            
            if candidates:
                sample_size = min(len(candidates), max(1, n_samples // n_nodes))
                sampled = rng.choice(len(candidates), size=sample_size, replace=False)
                for idx in sampled:
                    if len(negative_edges) < n_samples:
                        edge = candidates[idx]
                        negative_edges.append(edge)
                        existing_edges_bidirectional.add(edge)
                        existing_edges_bidirectional.add((edge[1], edge[0]))
    
    elif strategy == 'same_community':
        # Same-community non-edges (requires community detection)
        try:
            import community as community_louvain
            communities = community_louvain.best_partition(G)
        except ImportError:
            warnings.warn("python-louvain not installed, falling back to random sampling")
            return sample_negative_edges(G, n_samples, existing_edges, 'random', seed)
        
        # Group nodes by community
        community_nodes = {}
        for node, comm in communities.items():
            if comm not in community_nodes:
                community_nodes[comm] = []
            community_nodes[comm].append(node)
        
        # Sample within communities
        for comm, comm_nodes in community_nodes.items():
            if len(comm_nodes) < 2:
                continue
            
            # Sample pairs within community
            sample_size = max(1, n_samples // len(community_nodes))
            attempts = 0
            max_attempts = sample_size * 10
            
            while len(negative_edges) < n_samples and attempts < max_attempts:
                u, v = rng.choice(comm_nodes, size=2, replace=False)
                if (u, v) not in existing_edges_bidirectional:
                    negative_edges.append((u, v))
                    existing_edges_bidirectional.add((u, v))
                    existing_edges_bidirectional.add((v, u))
                attempts += 1
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    if len(negative_edges) < n_samples:
        warnings.warn(f"Could only sample {len(negative_edges)} negative edges (requested {n_samples})")
    
    return negative_edges[:n_samples]


def split_edges(
    G: nx.Graph,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    negative_sampling_strategy: str = 'random',
    seed: int = 42
) -> Tuple[nx.Graph, List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Split graph edges into train/validation/test sets.
    
    Args:
        G: NetworkX graph
        test_ratio: Fraction of edges for testing
        val_ratio: Fraction of edges for validation
        negative_sampling_strategy: Strategy for sampling negative edges
        seed: Random seed
    
    Returns:
        Tuple of (train_graph, val_edges, test_edges, negative_edges)
    """
    # Local generator rather than np.random.seed(seed) -- see sample_negative_edges.
    rng = np.random.default_rng(seed)

    edges = list(G.edges())
    n_edges = len(edges)

    # Shuffle edges
    rng.shuffle(edges)
    
    # Split edges
    n_test = int(n_edges * test_ratio)
    n_val = int(n_edges * val_ratio)
    n_train = n_edges - n_test - n_val
    
    train_edges = edges[:n_train]
    val_edges = edges[n_train:n_train + n_val]
    test_edges = edges[n_train + n_val:]
    
    # Create training graph
    train_graph = nx.Graph()
    train_graph.add_nodes_from(G.nodes())
    train_graph.add_edges_from(train_edges)
    
    # Ensure training graph is connected (if original was connected)
    if nx.is_connected(G) and not nx.is_connected(train_graph):
        warnings.warn("Training graph is disconnected after edge split. Consider reducing test_ratio.")
    
    # Sample negative edges (same number as test edges)
    negative_edges = sample_negative_edges(
        G, n_test, set(edges), strategy=negative_sampling_strategy, seed=seed
    )
    
    return train_graph, val_edges, test_edges, negative_edges


def compute_edge_features(
    embeddings: np.ndarray,
    node_list: List[int],
    edges: List[Tuple[int, int]],
    method: str = 'hadamard'
) -> np.ndarray:
    """
    Compute edge features from node embeddings.
    
    Args:
        embeddings: Node embedding matrix (n_nodes x embedding_dim)
        node_list: List of node IDs corresponding to embedding rows
        edges: List of edge tuples
        method: Feature computation method
    
    Returns:
        Edge feature matrix (n_edges x feature_dim)
    """
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    edge_features = []
    
    for u, v in edges:
        if u not in node_to_idx or v not in node_to_idx:
            warnings.warn(f"Edge ({u}, {v}) contains unknown node(s). Skipping.")
            continue
        
        u_emb = embeddings[node_to_idx[u]]
        v_emb = embeddings[node_to_idx[v]]
        
        if method == 'hadamard':
            feature = u_emb * v_emb
        elif method == 'average':
            feature = (u_emb + v_emb) / 2
        elif method == 'l1':
            feature = np.abs(u_emb - v_emb)
        elif method == 'l2':
            feature = (u_emb - v_emb) ** 2
        elif method == 'concat':
            feature = np.concatenate([u_emb, v_emb])
        elif method == 'inner_product' or method == 'dot':
            # Inner product (dot product) - returns scalar
            feature = np.array([np.dot(u_emb, v_emb)])
        elif method == 'cosine':
            # Cosine similarity - returns scalar
            norm_u = np.linalg.norm(u_emb)
            norm_v = np.linalg.norm(v_emb)
            if norm_u > 0 and norm_v > 0:
                feature = np.array([np.dot(u_emb, v_emb) / (norm_u * norm_v)])
            else:
                feature = np.array([0.0])
        else:
            raise ValueError(f"Unknown method: {method}")
        
        edge_features.append(feature)
    
    return np.array(edge_features)


def _extract_edge_endpoints(
    embeddings: np.ndarray,
    node_list: List,
    edges: List[Tuple],
    node_to_idx: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (U, V) embedding matrices for valid edges."""
    U, V = [], []
    for u, v in edges:
        if u not in node_to_idx or v not in node_to_idx:
            warnings.warn(f"Edge ({u}, {v}) contains unknown node(s). Skipping.")
            continue
        U.append(embeddings[node_to_idx[u]])
        V.append(embeddings[node_to_idx[v]])
    if not U:
        return np.empty((0, embeddings.shape[1])), np.empty((0, embeddings.shape[1]))
    return np.array(U), np.array(V)


def _apply_edge_method(U: np.ndarray, V: np.ndarray, method: str) -> np.ndarray:
    """Vectorised edge feature from pre-extracted endpoint embeddings."""
    if method == 'hadamard':
        return U * V
    elif method == 'average':
        return (U + V) / 2
    elif method == 'l1':
        return np.abs(U - V)
    elif method == 'l2':
        return (U - V) ** 2
    elif method in ('inner_product', 'dot'):
        return (U * V).sum(axis=1, keepdims=True)
    elif method == 'cosine':
        norms = np.linalg.norm(U, axis=1, keepdims=True) * np.linalg.norm(V, axis=1, keepdims=True)
        return np.where(norms > 0, (U * V).sum(axis=1, keepdims=True) / norms, 0.0)
    elif method == 'concat':
        return np.concatenate([U, V], axis=1)
    else:
        raise ValueError(f"Unknown method: {method}")


def _classify_and_score(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_pos: int,
    n_neg: int,
    k_values: List[int],
    random_state: int,
    classifier: str = 'logistic',
) -> Dict[str, float]:
    """Scale, train classifier, and return link prediction metrics."""
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    if classifier == 'logistic':
        clf = LogisticRegression(max_iter=1000, random_state=random_state)
    elif classifier == 'random_forest':
        clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    clf.fit(X_train_sc, y_train)
    y_pred = clf.predict(X_test_sc)
    y_scores = (clf.predict_proba(X_test_sc)[:, 1]
                if hasattr(clf, 'predict_proba')
                else clf.decision_function(X_test_sc))

    out = {
        'auc_roc': roc_auc_score(y_test, y_scores),
        'auc_pr': average_precision_score(y_test, y_scores),
        'f1': f1_score(y_test, y_pred, average='binary', zero_division=0),
        'n_positive': n_pos,
        'n_negative': n_neg,
        'n_train': len(y_train),
        'n_test': len(y_test),
    }
    sorted_idx = np.argsort(y_scores)[::-1]
    y_sorted = y_test[sorted_idx]
    for k in k_values:
        if k <= len(y_sorted):
            top_k = y_sorted[:k]
            out[f'precision@{k}'] = np.sum(top_k) / k
            out[f'recall@{k}'] = np.sum(top_k) / np.sum(y_test) if np.sum(y_test) > 0 else 0.0
            out[f'hit@{k}'] = 1.0 if np.sum(top_k) > 0 else 0.0
    pos_idx = np.where(y_sorted == 1)[0]
    out['mrr'] = float(np.mean(1.0 / (pos_idx + 1))) if len(pos_idx) > 0 else 0.0
    return out


def evaluate_link_prediction(
    embeddings: np.ndarray,
    node_list: List[int],
    positive_edges: List[Tuple[int, int]],
    negative_edges: List[Tuple[int, int]],
    edge_feature_method: str = 'hadamard',
    classifier: str = 'logistic',
    k_values: List[int] = [10, 50, 100],
    test_size: float = 0.3,
    random_state: int = 42,
    train_positive_edges: Optional[List[Tuple[int, int]]] = None,
    train_negative_edges: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, float]:
    """
    Evaluate link prediction performance with no train-test leakage.

    When *train_positive_edges* and *train_negative_edges* are provided the
    classifier is trained on those edges and evaluated on *positive_edges* /
    *negative_edges* (the held-out test set).  The StandardScaler is always
    fit exclusively on the training features so that test statistics are
    never seen during normalisation.

    When train edges are omitted the function falls back to an internal
    stratified split controlled by *test_size*.

    Args:
        embeddings: Node embedding matrix
        node_list: List of node IDs
        positive_edges: Test positive (existing) edges
        negative_edges: Test negative (non-existing) edges
        edge_feature_method: Method for computing edge features
        classifier: Classifier type ('logistic' or 'random_forest')
        k_values: K values for Precision@K and Recall@K
        test_size: Fraction for internal test split (only used when train edges not provided)
        random_state: Random seed
        train_positive_edges: Training positive edges (prevents leakage)
        train_negative_edges: Training negative edges (prevents leakage)

    Returns:
        Dictionary of evaluation metrics
    """
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    U_pos, V_pos = _extract_edge_endpoints(embeddings, node_list, positive_edges, node_to_idx)
    U_neg, V_neg = _extract_edge_endpoints(embeddings, node_list, negative_edges, node_to_idx)

    if len(U_pos) == 0 or len(U_neg) == 0:
        warnings.warn("No valid edge features computed. Skipping evaluation.")
        return {'error': 'no_valid_features', 'auc_roc': 0.0, 'auc_pr': 0.0,
                'f1': 0.0, 'n_positive': 0, 'n_negative': 0}

    pos_features = _apply_edge_method(U_pos, V_pos, edge_feature_method)
    neg_features = _apply_edge_method(U_neg, V_neg, edge_feature_method)
    X_test_all = np.vstack([pos_features, neg_features])
    y_test_all = np.repeat([1, 0], [len(pos_features), len(neg_features)])

    # Pre-initialize to avoid unbound-variable hazard when have_train is False.
    U_tp = V_tp = U_tn = V_tn = None

    if train_positive_edges is not None and train_negative_edges is not None:
        U_tp, V_tp = _extract_edge_endpoints(embeddings, node_list, train_positive_edges, node_to_idx)
        U_tn, V_tn = _extract_edge_endpoints(embeddings, node_list, train_negative_edges, node_to_idx)

    if U_tp is not None and len(U_tp) > 0 and U_tn is not None and len(U_tn) > 0:
        X_train = np.vstack([
            _apply_edge_method(U_tp, V_tp, edge_feature_method),
            _apply_edge_method(U_tn, V_tn, edge_feature_method),
        ])
        y_train = np.repeat([1, 0], [len(U_tp), len(U_tn)])
        X_test, y_test = X_test_all, y_test_all
    else:
        if train_positive_edges is not None or train_negative_edges is not None:
            warnings.warn("Train edge features empty; falling back to internal split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X_test_all, y_test_all, test_size=test_size,
            random_state=random_state, stratify=y_test_all,
        )

    return _classify_and_score(
        X_train, y_train, X_test, y_test,
        n_pos=len(pos_features), n_neg=len(neg_features),
        k_values=k_values, random_state=random_state, classifier=classifier,
    )


def evaluate_link_prediction_cv(
    G: nx.Graph,
    embeddings: np.ndarray,
    node_list: List[int],
    test_ratio: float = 0.2,
    edge_feature_method: str = 'hadamard',
    negative_sampling_strategy: str = 'random',
    k_values: List[int] = [10, 50, 100],
    random_state: int = 42
) -> Dict[str, float]:
    """
    Evaluate link prediction with train-test split.
    
    Args:
        G: NetworkX graph
        embeddings: Node embedding matrix
        node_list: List of node IDs
        test_ratio: Fraction of edges for testing
        edge_feature_method: Method for computing edge features
        negative_sampling_strategy: Strategy for negative sampling
        k_values: K values for metrics
        random_state: Random seed
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Split edges
    train_graph, _, test_edges, negative_edges = split_edges(
        G, test_ratio=test_ratio, val_ratio=0.0, 
        negative_sampling_strategy=negative_sampling_strategy,
        seed=random_state
    )
    
    # Evaluate
    results = evaluate_link_prediction(
        embeddings=embeddings,
        node_list=node_list,
        positive_edges=test_edges,
        negative_edges=negative_edges,
        edge_feature_method=edge_feature_method,
        k_values=k_values,
        test_size=0.3,  # Further split for train/test within evaluation
        random_state=random_state
    )
    
    return results


def evaluate_all_edge_feature_methods(
    embeddings: np.ndarray,
    node_list: List[int],
    positive_edges: List[Tuple[int, int]],
    negative_edges: List[Tuple[int, int]],
    k_values: List[int] = [10, 50, 100],
    random_state: int = 42,
    train_positive_edges: Optional[List[Tuple[int, int]]] = None,
    train_negative_edges: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate link prediction using all edge feature methods.

    Args:
        embeddings: Node embedding matrix
        node_list: List of node IDs
        positive_edges: Test positive edges
        negative_edges: Test negative edges
        k_values: K values for metrics
        random_state: Random seed
        train_positive_edges: Training positive edges (no leakage when provided)
        train_negative_edges: Training negative edges (no leakage when provided)

    Returns:
        Dictionary mapping method names to evaluation results
    """
    # Pre-extract endpoint embeddings once for each edge set (avoids rebuilding
    # node_to_idx and re-indexing embeddings for every one of the 6 methods).
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    U_pos, V_pos = _extract_edge_endpoints(embeddings, node_list, positive_edges, node_to_idx)
    U_neg, V_neg = _extract_edge_endpoints(embeddings, node_list, negative_edges, node_to_idx)

    _empty_result = {'error': 'no_valid_features', 'auc_roc': 0.0, 'auc_pr': 0.0,
                     'f1': 0.0, 'n_positive': 0, 'n_negative': 0}
    if len(U_pos) == 0 or len(U_neg) == 0:
        return {m: _empty_result for m in ['hadamard', 'average', 'l1', 'l2', 'inner_product', 'cosine']}

    have_train = train_positive_edges is not None and train_negative_edges is not None
    if have_train:
        U_tp, V_tp = _extract_edge_endpoints(embeddings, node_list, train_positive_edges, node_to_idx)
        U_tn, V_tn = _extract_edge_endpoints(embeddings, node_list, train_negative_edges, node_to_idx)
        have_train = len(U_tp) > 0 and len(U_tn) > 0
        if not have_train:
            warnings.warn("Train edge features empty; falling back to internal split.")

    y_test_all = np.repeat([1, 0], [len(U_pos), len(U_neg)])
    y_train_fixed = np.repeat([1, 0], [len(U_tp), len(U_tn)]) if have_train else None

    results = {}
    for method in ['hadamard', 'average', 'l1', 'l2', 'inner_product', 'cosine']:
        try:
            X_test_all = np.vstack([
                _apply_edge_method(U_pos, V_pos, method),
                _apply_edge_method(U_neg, V_neg, method),
            ])
            if have_train:
                X_train = np.vstack([
                    _apply_edge_method(U_tp, V_tp, method),
                    _apply_edge_method(U_tn, V_tn, method),
                ])
                X_tr, y_tr, X_te, y_te = X_train, y_train_fixed, X_test_all, y_test_all
            else:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_test_all, y_test_all, test_size=0.3,
                    random_state=random_state, stratify=y_test_all,
                )
            results[method] = _classify_and_score(
                X_tr, y_tr, X_te, y_te,
                n_pos=len(U_pos), n_neg=len(U_neg),
                k_values=k_values, random_state=random_state,
            )
        except Exception as e:
            warnings.warn(f"Link prediction with {method} failed: {e}")
            results[method] = {'error': str(e)}

    return results


def summarize_link_prediction_results(
    results: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """
    Summarize link prediction results across all edge feature methods.
    
    Args:
        results: Dictionary of results from evaluate_all_edge_feature_methods
    
    Returns:
        Dictionary of summary statistics
    """
    # Extract AUC scores
    auc_roc_scores = []
    auc_pr_scores = []
    mrr_scores = []
    
    for method, metrics in results.items():
        if 'error' not in metrics:
            if 'auc_roc' in metrics:
                auc_roc_scores.append(metrics['auc_roc'])
            if 'auc_pr' in metrics:
                auc_pr_scores.append(metrics['auc_pr'])
            if 'mrr' in metrics:
                mrr_scores.append(metrics['mrr'])
    
    if not auc_roc_scores:
        # nan, not 0.0: no method produced a score, which is not the same claim
        # as every method scoring zero.
        return {
            'mean_auc_roc': float('nan'),
            'mean_auc_pr': float('nan'),
            'mean_mrr': float('nan'),
            'n_defined_auc_roc': 0,
            'n_defined_auc_pr': 0,
            'n_defined_mrr': 0,
            'n_successful_methods': 0,
        }
    
    # nan-aware: a method whose ranking metric was undefined (single-class test
    # split) reports nan, and plain np.mean would propagate that one nan into
    # every summary field, erasing the methods that did succeed. n_defined_* says
    # how many values each statistic was actually computed over.
    def _stats(prefix, values):
        arr = np.asarray(values, dtype=float)
        n_defined = int(np.count_nonzero(~np.isnan(arr)))
        if n_defined == 0:
            return {f'mean_{prefix}': float('nan'), f'std_{prefix}': float('nan'),
                    f'max_{prefix}': float('nan'), f'min_{prefix}': float('nan'),
                    f'n_defined_{prefix}': 0}
        return {f'mean_{prefix}': float(np.nanmean(arr)),
                f'std_{prefix}': float(np.nanstd(arr)),
                f'max_{prefix}': float(np.nanmax(arr)),
                f'min_{prefix}': float(np.nanmin(arr)),
                f'n_defined_{prefix}': n_defined}

    summary = {
        **_stats('auc_roc', auc_roc_scores),
        **_stats('auc_pr', auc_pr_scores),
        **{k: v for k, v in _stats('mrr', mrr_scores).items()
           if not (k.startswith('max_') or k.startswith('min_'))},
        'n_successful_methods': len(auc_roc_scores),
    }

    return summary


def compute_structural_link_features(
    G: nx.Graph,
    edges: List[Tuple[int, int]]
) -> np.ndarray:
    """
    Compute structural features for edges (for baseline comparison).
    
    Features:
    - Common neighbors
    - Jaccard coefficient
    - Adamic-Adar index
    - Preferential attachment
    
    Args:
        G: NetworkX graph
        edges: List of edge tuples
    
    Returns:
        Structural feature matrix (n_edges x 4)
    """
    features = []
    
    for u, v in edges:
        if u not in G or v not in G:
            features.append([0, 0, 0, 0])
            continue
        
        # Common neighbors
        common_neighbors = len(list(nx.common_neighbors(G, u, v)))
        
        # Jaccard coefficient
        u_neighbors = set(G.neighbors(u))
        v_neighbors = set(G.neighbors(v))
        union = len(u_neighbors | v_neighbors)
        jaccard = len(u_neighbors & v_neighbors) / union if union > 0 else 0
        
        # Adamic-Adar index
        adamic_adar = 0
        for w in nx.common_neighbors(G, u, v):
            deg_w = G.degree(w)
            if deg_w > 1:
                adamic_adar += 1 / np.log(deg_w)
        
        # Preferential attachment
        pref_attach = G.degree(u) * G.degree(v)
        
        features.append([common_neighbors, jaccard, adamic_adar, pref_attach])
    
    return np.array(features)


