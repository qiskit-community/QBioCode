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
Method Adapters for Reproducible Benchmarking Pipeline

This module provides adapter functions that wrap all 42 QuVINE methods
to work with the reproducible pipeline interface.

Each adapter:
1. Accepts pre-generated graph and split
2. Uses the provided canonical seed
3. Returns standardized metrics
4. Does NOT modify input data or create its own splits
"""

import numpy as np
import networkx as nx
from typing import Dict, Any, Optional, List
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    average_precision_score, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Utility Functions
# ============================================================================

def train_classifier(
    embeddings: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    labels: np.ndarray,
    seed: int
) -> Dict[str, float]:
    """
    Train a logistic regression classifier on embeddings.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Node embeddings [N, d]
    train_idx : np.ndarray
        Training indices
    val_idx : np.ndarray
        Validation indices
    test_idx : np.ndarray
        Test indices
    labels : np.ndarray
        Node labels
    seed : int
        Random seed
    
    Returns
    -------
    dict
        Classification metrics
    """
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(embeddings[train_idx])
    X_val = scaler.transform(embeddings[val_idx])
    X_test = scaler.transform(embeddings[test_idx])
    
    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]
    
    # Train classifier
    clf = LogisticRegression(
        max_iter=1000,
        random_state=seed,
        solver='lbfgs'
    )
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred_test = clf.predict(X_test)
    
    # Metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred_test)),
        "f1_macro": float(f1_score(y_test, y_pred_test, average='macro')),
        "f1_micro": float(f1_score(y_test, y_pred_test, average='micro')),
        "f1_weighted": float(f1_score(y_test, y_pred_test, average='weighted')),
        "num_train": len(train_idx),
        "num_val": len(val_idx),
        "num_test": len(test_idx),
        "num_classes": int(labels.max() + 1)
    }
    
    return metrics


def evaluate_link_prediction(
    embeddings: np.ndarray,
    test_edges: List,
    neg_test_edges: List
) -> Dict[str, float]:
    """
    Evaluate link prediction using dot product similarity.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Node embeddings [N, d]
    test_edges : list
        Positive test edges
    neg_test_edges : list
        Negative test edges
    
    Returns
    -------
    dict
        Link prediction metrics: ``auc_roc``, ``auc_pr``, ``f1``.

    Notes
    -----
    Ranking metrics are undefined unless both classes are present. When either
    ``test_edges`` or ``neg_test_edges`` is empty the three metrics are returned
    as ``nan`` -- not as 0.5 -- because 0.5 is an achievable score and reporting
    it would make an undefined evaluation indistinguishable from a genuinely
    chance-level one, and would drag any average over methods toward chance.
    Aggregate these with ``np.nanmean``.

    Raises
    ------
    ValueError
        If both edge lists are empty, or if an edge references a node index
        outside ``embeddings``.
    """
    n_nodes = len(embeddings)
    for name, edges in (("test_edges", test_edges), ("neg_test_edges", neg_test_edges)):
        bad = [(u, v) for u, v in edges
               if not (0 <= u < n_nodes) or not (0 <= v < n_nodes)]
        if bad:
            raise ValueError(
                f"{name} references node indices outside embeddings "
                f"(which has {n_nodes} rows): {bad[:5]}"
                f"{' ...' if len(bad) > 5 else ''}"
            )
    if not test_edges and not neg_test_edges:
        raise ValueError(
            "evaluate_link_prediction requires at least one edge; both "
            "test_edges and neg_test_edges are empty."
        )
    if not test_edges or not neg_test_edges:
        logger.warning(
            "Link-prediction metrics are undefined with a single class "
            "(%d positive, %d negative edges); returning nan.",
            len(test_edges), len(neg_test_edges),
        )
        return {"auc_roc": float("nan"), "auc_pr": float("nan"), "f1": float("nan")}

    # Compute scores for positive edges
    pos_scores = []
    for u, v in test_edges:
        score = np.dot(embeddings[u], embeddings[v])
        pos_scores.append(score)
    
    # Compute scores for negative edges
    neg_scores = []
    for u, v in neg_test_edges:
        score = np.dot(embeddings[u], embeddings[v])
        neg_scores.append(score)
    
    # Combine scores and labels
    scores = np.array(pos_scores + neg_scores)
    labels = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
    
    # Both classes are guaranteed present by the check above, so these are
    # defined. ValueError is still caught -- non-finite scores from a diverged
    # embedding reach sklearn as a genuine input error -- and the sentinel is
    # nan, matching the single-class path, so undefined never masquerades as
    # chance-level. Narrow to ValueError: anything else here is a real bug.
    try:
        auc_roc = float(roc_auc_score(labels, scores))
    except ValueError as exc:
        logger.warning("roc_auc_score failed (%s); reporting nan.", exc)
        auc_roc = float("nan")

    try:
        auc_pr = float(average_precision_score(labels, scores))
    except ValueError as exc:
        logger.warning("average_precision_score failed (%s); reporting nan.", exc)
        auc_pr = float("nan")

    # F1 score at optimal threshold
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_f1 = float(np.max(f1_scores))
    
    metrics = {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "f1": best_f1,
        "num_test_edges": len(test_edges),
        "num_neg_test_edges": len(neg_test_edges)
    }
    
    return metrics


def evaluate_node_ranking(
    embeddings: np.ndarray,
    seed_nodes: List[int],
    target_nodes: List[int],
    k_values: List[int] = [10, 20, 50]
) -> Dict[str, float]:
    """
    Evaluate node ranking task.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Node embeddings [N, d]
    seed_nodes : list
        Seed node indices
    target_nodes : list
        Target node indices
    k_values : list
        K values for precision@k
    
    Returns
    -------
    dict
        Ranking metrics
    """
    # Compute centroid of seed nodes
    seed_centroid = embeddings[seed_nodes].mean(axis=0)
    
    # Compute similarities to all nodes
    similarities = embeddings @ seed_centroid
    
    # Rank nodes by similarity
    ranked_indices = np.argsort(-similarities)
    
    # Compute metrics
    target_set = set(target_nodes)
    metrics = {}
    
    for k in k_values:
        top_k = set(ranked_indices[:k].tolist())
        hits = len(top_k & target_set)
        precision_at_k = hits / k if k > 0 else 0.0
        metrics[f"precision@{k}"] = float(precision_at_k)
    
    # Mean reciprocal rank
    mrr = 0.0
    for target in target_nodes:
        rank = np.where(ranked_indices == target)[0]
        if len(rank) > 0:
            mrr += 1.0 / (rank[0] + 1)
    mrr /= len(target_nodes) if target_nodes else 1.0
    
    metrics["mrr"] = float(mrr)
    metrics["num_seed_nodes"] = len(seed_nodes)
    metrics["num_target_nodes"] = len(target_nodes)
    
    return metrics


# ============================================================================
# Classical Baseline Adapters
# ============================================================================

def run_node2vec_adapter(
    G: nx.Graph,
    split: Dict[str, Any],
    task: str,
    config: Optional[Dict] = None,
    seed: int = 42
) -> Dict[str, float]:
    """Adapter for Node2Vec method."""
    from qbiocode.apps.quvine.baselines import run_node2vec
    
    config = config or {}
    nodes = list(G.nodes())
    
    # Generate embeddings
    embeddings = run_node2vec(
        graph=G,
        nodes=nodes,
        dimensions=config.get('dimensions', 128),
        walk_length=config.get('walk_length', 80),
        num_walks=config.get('num_walks', 10),
        p=config.get('p', 1.0),
        q=config.get('q', 1.0),
        window=config.get('window', 10),
        min_count=config.get('min_count', 1),
        workers=config.get('workers', 4),
        seed=seed
    )
    
    # Evaluate based on task
    if task == "node_classification":
        return train_classifier(
            embeddings,
            np.array(split["train_idx"]),
            np.array(split["val_idx"]),
            np.array(split["test_idx"]),
            np.array(split["labels"]),
            seed
        )
    elif task == "link_prediction":
        return evaluate_link_prediction(
            embeddings,
            split["test_edges"],
            split["neg_test_edges"]
        )
    elif task == "node_ranking":
        return evaluate_node_ranking(
            embeddings,
            split["seed_nodes"],
            split["target_nodes"]
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def run_netmf_adapter(
    G: nx.Graph,
    split: Dict[str, Any],
    task: str,
    config: Optional[Dict] = None,
    seed: int = 42
) -> Dict[str, float]:
    """Adapter for NetMF method."""
    from qbiocode.apps.quvine.baselines import run_netmf
    
    config = config or {}
    nodes = list(G.nodes())
    
    # Generate embeddings
    embeddings = run_netmf(
        graph=G,
        nodes=nodes,
        dimensions=config.get('dimensions', 128),
        window_size=config.get('window_size', 10),
        negative=config.get('negative', 1),
        rank=config.get('rank', None),
        use_svd=config.get('use_svd', True),
        seed=seed
    )
    
    # Evaluate based on task
    if task == "node_classification":
        return train_classifier(
            embeddings,
            np.array(split["train_idx"]),
            np.array(split["val_idx"]),
            np.array(split["test_idx"]),
            np.array(split["labels"]),
            seed
        )
    elif task == "link_prediction":
        return evaluate_link_prediction(
            embeddings,
            split["test_edges"],
            split["neg_test_edges"]
        )
    elif task == "node_ranking":
        return evaluate_node_ranking(
            embeddings,
            split["seed_nodes"],
            split["target_nodes"]
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def run_graphsage_adapter(
    G: nx.Graph,
    split: Dict[str, Any],
    task: str,
    config: Optional[Dict] = None,
    seed: int = 42
) -> Dict[str, float]:
    """Adapter for GraphSAGE method."""
    from qbiocode.apps.quvine.baselines import run_graphsage
    
    config = config or {}
    nodes = list(G.nodes())
    
    # Generate embeddings
    embeddings = run_graphsage(
        graph=G,
        nodes=nodes,
        dimensions=config.get('dimensions', 128),
        hidden_dim=config.get('hidden_dim', 256),
        n_layers=config.get('n_layers', 2),
        epochs=config.get('epochs', 50),
        lr=config.get('lr', 0.01),
        neg_samples=config.get('neg_samples', 5),
        seed=seed,
        device=config.get('device', 'cpu')
    )
    
    # Evaluate based on task
    if task == "node_classification":
        return train_classifier(
            embeddings,
            np.array(split["train_idx"]),
            np.array(split["val_idx"]),
            np.array(split["test_idx"]),
            np.array(split["labels"]),
            seed
        )
    elif task == "link_prediction":
        return evaluate_link_prediction(
            embeddings,
            split["test_edges"],
            split["neg_test_edges"]
        )
    elif task == "node_ranking":
        return evaluate_node_ranking(
            embeddings,
            split["seed_nodes"],
            split["target_nodes"]
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def run_appnp_adapter(
    G: nx.Graph,
    split: Dict[str, Any],
    task: str,
    config: Optional[Dict] = None,
    seed: int = 42
) -> Dict[str, float]:
    """Adapter for APPNP method."""
    from qbiocode.apps.quvine.baselines import run_appnp
    
    config = config or {}
    nodes = list(G.nodes())
    
    # Generate embeddings
    embeddings = run_appnp(
        graph=G,
        nodes=nodes,
        dimensions=config.get('dimensions', 128),
        hidden_dim=config.get('hidden_dim', 64),
        n_layers=config.get('n_layers', 2),
        alpha=config.get('alpha', 0.1),
        K=config.get('K', 10),
        dropout=config.get('dropout', 0.5),
        lr=config.get('lr', 0.01),
        weight_decay=config.get('weight_decay', 5e-4),
        epochs=config.get('epochs', 200),
        seed=seed,
        device=config.get('device', 'cpu')
    )
    
    # Evaluate based on task
    if task == "node_classification":
        return train_classifier(
            embeddings,
            np.array(split["train_idx"]),
            np.array(split["val_idx"]),
            np.array(split["test_idx"]),
            np.array(split["labels"]),
            seed
        )
    elif task == "link_prediction":
        return evaluate_link_prediction(
            embeddings,
            split["test_edges"],
            split["neg_test_edges"]
        )
    elif task == "node_ranking":
        return evaluate_node_ranking(
            embeddings,
            split["seed_nodes"],
            split["target_nodes"]
        )
    else:
        raise ValueError(f"Unknown task: {task}")


# ============================================================================
# QuVINE SGNS Methods (Quantum Walk + Skip-Gram)
# ============================================================================

def run_quvine_sgns(
    G: nx.Graph,
    walk_type: str,
    config: Optional[Dict] = None,
    seed: int = 42
) -> np.ndarray:
    """
    Run QuVINE SGNS embedding with specified walk type.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    walk_type : str
        Type of walk: 'rwr', 'ctqw', or 'dtqw'
    config : dict, optional
        Configuration parameters
    seed : int
        Random seed
    
    Returns
    -------
    np.ndarray
        Node embeddings (n_nodes x embedding_dim)
    """
    from qbiocode.apps.quvine.walks.rwr import generate_RWR_pagerank_walks
    from qbiocode.apps.quvine.walks.ctqw import generate_CTQW_walks
    from qbiocode.apps.quvine.walks.dtqw import generate_DTQW_walks
    from qbiocode.apps.quvine.corpus.builder import CorpusBuilder
    from qbiocode.apps.quvine.embedding.word2vec import corpus_to_embedding
    
    config = config or {}
    nodes = list(G.nodes())
    
    # Convert nodes to strings for Word2Vec
    node_to_str = {n: str(n) for n in nodes}
    str_to_node = {str(n): n for n in nodes}
    G_str = nx.relabel_nodes(G, node_to_str, copy=True)
    nodes_str = [node_to_str[n] for n in nodes]
    
    # Set random seed
    rng = np.random.default_rng(seed)
    
    # Walk parameters
    num_walks = config.get('num_walks', 10)
    walk_length = config.get('walk_length', 80)
    
    # Generate walks for all nodes
    corpus_builder = CorpusBuilder()
    
    for node_str in nodes_str:
        try:
            if walk_type == 'rwr':
                walks = generate_RWR_pagerank_walks(
                    G=G_str,
                    root=node_str,
                    view_nodes=None,
                    num_walks=num_walks,
                    walk_length=walk_length,
                    restart_prob=config.get('restart_prob', 0.15),
                    max_iter=config.get('max_iter', 100),
                    rng=rng
                )
            elif walk_type == 'ctqw':
                walks = generate_CTQW_walks(
                    G=G_str,
                    root=node_str,
                    view_nodes=None,
                    num_walks=num_walks,
                    walk_length=walk_length,
                    time=config.get('time', 1.0),
                    steps=config.get('steps', 20),
                    rng=rng
                )
            elif walk_type == 'dtqw':
                walks = generate_DTQW_walks(
                    G=G_str,
                    root=node_str,
                    view_nodes=None,
                    num_walks=num_walks,
                    walk_length=walk_length,
                    steps=config.get('steps', 25),
                    coin=config.get('coin', 'grover'),
                    rng=rng
                )
            else:
                raise ValueError(f"Unknown walk type: {walk_type}")
            
            # Add walks to corpus
            corpus_builder.add(node_str, walks)
            
        except Exception as e:
            logger.warning(f"Walk generation failed for node {node_str}: {e}")
            continue
    
    # Build corpus
    corpus = corpus_builder.build()
    
    # Train SGNS embeddings
    embeddings = corpus_to_embedding(
        corpus=corpus,
        nodes=nodes_str,
        vector_size=config.get('dimensions', 128),
        window=config.get('window', 10),
        sg=1,  # Skip-gram
        negative=config.get('negative', 5),
        min_count=config.get('min_count', 0),
        workers=config.get('workers', 4),
        epochs=config.get('epochs', 5)
    )
    
    return embeddings


def run_quvine_rwr_adapter(
    G: nx.Graph,
    split: Dict[str, Any],
    task: str,
    config: Optional[Dict] = None,
    seed: int = 42
) -> Dict[str, float]:
    """Adapter for QuVINE RWR (Random Walk with Restart + SGNS)."""
    config = config or {}
    
    # Generate embeddings using RWR walks
    embeddings = run_quvine_sgns(G, 'rwr', config, seed)
    
    # Evaluate based on task
    if task == "node_classification":
        return train_classifier(
            embeddings,
            np.array(split["train_idx"]),
            np.array(split["val_idx"]),
            np.array(split["test_idx"]),
            np.array(split["labels"]),
            seed
        )
    elif task == "link_prediction":
        return evaluate_link_prediction(
            embeddings,
            split["test_edges"],
            split["neg_test_edges"]
        )
    elif task == "node_ranking":
        return evaluate_node_ranking(
            embeddings,
            split["seed_nodes"],
            split["target_nodes"]
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def run_quvine_ctqw_adapter(
    G: nx.Graph,
    split: Dict[str, Any],
    task: str,
    config: Optional[Dict] = None,
    seed: int = 42
) -> Dict[str, float]:
    """Adapter for QuVINE CTQW (Continuous-Time Quantum Walk + SGNS)."""
    config = config or {}
    
    # Generate embeddings using CTQW walks
    embeddings = run_quvine_sgns(G, 'ctqw', config, seed)
    
    # Evaluate based on task
    if task == "node_classification":
        return train_classifier(
            embeddings,
            np.array(split["train_idx"]),
            np.array(split["val_idx"]),
            np.array(split["test_idx"]),
            np.array(split["labels"]),
            seed
        )
    elif task == "link_prediction":
        return evaluate_link_prediction(
            embeddings,
            split["test_edges"],
            split["neg_test_edges"]
        )
    elif task == "node_ranking":
        return evaluate_node_ranking(
            embeddings,
            split["seed_nodes"],
            split["target_nodes"]
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def run_quvine_dtqw_adapter(
    G: nx.Graph,
    split: Dict[str, Any],
    task: str,
    config: Optional[Dict] = None,
    seed: int = 42
) -> Dict[str, float]:
    """Adapter for QuVINE DTQW (Discrete-Time Quantum Walk + SGNS)."""
    config = config or {}
    
    # Generate embeddings using DTQW walks
    embeddings = run_quvine_sgns(G, 'dtqw', config, seed)
    
    # Evaluate based on task
    if task == "node_classification":
        return train_classifier(
            embeddings,
            np.array(split["train_idx"]),
            np.array(split["val_idx"]),
            np.array(split["test_idx"]),
            np.array(split["labels"]),
            seed
        )
    elif task == "link_prediction":
        return evaluate_link_prediction(
            embeddings,
            split["test_edges"],
            split["neg_test_edges"]
        )
    elif task == "node_ranking":
        return evaluate_node_ranking(
            embeddings,
            split["seed_nodes"],
            split["target_nodes"]
        )
    else:
        raise ValueError(f"Unknown task: {task}")


# ============================================================================
# Filter Methods (Baseline and QuVINE-calibrated)
# ============================================================================

def run_baseline_filter_heat_adapter(
    G: nx.Graph,
    split: Dict[str, Any],
    task: str,
    config: Optional[Dict] = None,
    seed: int = 42
) -> Dict[str, float]:
    """Adapter for baseline heat kernel filter (no quantum calibration)."""
    from qbiocode.apps.quvine.embedding.quantum_filters import generate_baseline_heat_embedding
    
    config = config or {}
    
    # Generate embeddings
    embeddings = generate_baseline_heat_embedding(
        G=G,
        embedding_dim=config.get('embedding_dim', 128),
        scale=config.get('scale', 1.0),
        use_features=False,
        normalize=True,
        random_state=seed
    )
    
    # Evaluate based on task
    if task == "node_classification":
        return train_classifier(
            embeddings,
            np.array(split["train_idx"]),
            np.array(split["val_idx"]),
            np.array(split["test_idx"]),
            np.array(split["labels"]),
            seed
        )
    elif task == "link_prediction":
        return evaluate_link_prediction(
            embeddings,
            split["test_edges"],
            split["neg_test_edges"]
        )
    elif task == "node_ranking":
        return evaluate_node_ranking(
            embeddings,
            split["seed_nodes"],
            split["target_nodes"]
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def run_baseline_filter_poly_adapter(
    G: nx.Graph,
    split: Dict[str, Any],
    task: str,
    config: Optional[Dict] = None,
    seed: int = 42
) -> Dict[str, float]:
    """Adapter for baseline polynomial filter (no quantum calibration)."""
    from qbiocode.apps.quvine.embedding.quantum_filters import generate_baseline_poly_embedding
    
    config = config or {}
    
    # Generate embeddings
    embeddings = generate_baseline_poly_embedding(
        G=G,
        embedding_dim=config.get('embedding_dim', 128),
        order=config.get('order', 4),
        use_features=False,
        normalize=True,
        random_state=seed
    )
    
    # Evaluate based on task
    if task == "node_classification":
        return train_classifier(
            embeddings,
            np.array(split["train_idx"]),
            np.array(split["val_idx"]),
            np.array(split["test_idx"]),
            np.array(split["labels"]),
            seed
        )
    elif task == "link_prediction":
        return evaluate_link_prediction(
            embeddings,
            split["test_edges"],
            split["neg_test_edges"]
        )
    elif task == "node_ranking":
        return evaluate_node_ranking(
            embeddings,
            split["seed_nodes"],
            split["target_nodes"]
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def generate_quantum_targets_from_walks(
    G: nx.Graph,
    walk_type: str,
    config: Dict,
    seed: int,
    n_samples: int = 10
) -> List[Dict]:
    """
    Generate quantum walk targets for filter calibration.
    
    Samples subnetworks and computes quantum walk distributions.
    Returns targets with integer node IDs (not strings).
    """
    from qbiocode.apps.quvine.walks.rwr import get_RWR_pagerank_scores
    from qbiocode.apps.quvine.walks.ctqw import generate_ctqw_hiperwalk_scores
    from qbiocode.apps.quvine.walks.dtqw import get_coined_hiperwalk_scores
    
    nodes = list(G.nodes())
    rng = np.random.default_rng(seed)
    
    q_targets = []
    
    # Sample random centers (use original integer node IDs)
    centers = rng.choice(nodes, size=min(n_samples, len(nodes)), replace=False)
    
    for center in centers:
        # Sample local neighborhood
        neighbors = list(nx.single_source_shortest_path_length(G, center, cutoff=2).keys())
        if len(neighbors) < 3:
            continue
            
        try:
            if walk_type == 'rwr':
                scores = get_RWR_pagerank_scores(
                    G,
                    center,
                    restart_prob=config.get('restart_prob', 0.15),
                    view_nodes=set(neighbors),
                    max_iter=config.get('max_iter', 100)
                )
            elif walk_type == 'ctqw':
                scores = generate_ctqw_hiperwalk_scores(
                    G,
                    center,
                    view_nodes=set(neighbors),
                    steps=config.get('steps', 20),
                    time=config.get('time', 1.0)
                )
            elif walk_type == 'dtqw':
                scores = get_coined_hiperwalk_scores(
                    G,
                    center,
                    view_nodes=set(neighbors),
                    steps=config.get('steps', 25),
                    coin=config.get('coin', 'grover')
                )
            else:
                raise ValueError(f"Unknown walk type: {walk_type}")
            
            # Normalize to probability distribution
            nodes_in_view = list(scores.keys())
            probs = np.array([scores[n] for n in nodes_in_view])
            probs = probs / probs.sum() if probs.sum() > 0 else probs
            
            q_targets.append({
                'nodes': nodes_in_view,
                'center': center,
                'pQ': probs
            })
        except Exception as e:
            logger.warning(f"Failed to generate quantum target for center {center}: {e}")
            continue
    
    return q_targets


def run_filter_rwr_heat_adapter(
    G: nx.Graph,
    split: Dict[str, Any],
    task: str,
    config: Optional[Dict] = None,
    seed: int = 42
) -> Dict[str, float]:
    """Adapter for RWR + heat kernel filter."""
    from qbiocode.apps.quvine.embedding.quantum_filters import (
        get_laplacian, calibrate_heat_kernel, apply_heat_filter
    )
    
    config = config or {}
    
    # Generate quantum targets from RWR walks
    q_targets = generate_quantum_targets_from_walks(G, 'rwr', config, seed)
    
    if not q_targets:
        logger.warning("No quantum targets generated, falling back to baseline")
        from qbiocode.apps.quvine.embedding.quantum_filters import generate_baseline_heat_embedding
        embeddings = generate_baseline_heat_embedding(
            G, embedding_dim=config.get('embedding_dim', 128), random_state=seed
        )
    else:
        # Manual implementation to work around tuple unpacking bug
        np.random.seed(seed)
        N = G.number_of_nodes()
        
        # Get Laplacian (unpack tuple properly)
        L, nodelist, node_to_idx = get_laplacian(G, normalize=True)
        
        # Calibrate heat kernel
        t_grid = np.logspace(-2, 2, 40)
        _, t_star = calibrate_heat_kernel(L, q_targets, t_grid, node_to_idx, loss='l2')
        
        # Generate random features
        embedding_dim = config.get('embedding_dim', 128)
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(N, embedding_dim))
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norm, 1e-12)
        
        # Apply heat kernel filter
        embeddings = apply_heat_filter(L, X, t_star)
    
    # Evaluate based on task
    if task == "node_classification":
        return train_classifier(
            embeddings,
            np.array(split["train_idx"]),
            np.array(split["val_idx"]),
            np.array(split["test_idx"]),
            np.array(split["labels"]),
            seed
        )
    elif task == "link_prediction":
        return evaluate_link_prediction(
            embeddings,
            split["test_edges"],
            split["neg_test_edges"]
        )
    elif task == "node_ranking":
        return evaluate_node_ranking(
            embeddings,
            split["seed_nodes"],
            split["target_nodes"]
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def run_filter_rwr_poly_adapter(
    G: nx.Graph,
    split: Dict[str, Any],
    task: str,
    config: Optional[Dict] = None,
    seed: int = 42
) -> Dict[str, float]:
    """Adapter for RWR + polynomial filter."""
    from qbiocode.apps.quvine.embedding.quantum_filters import generate_quvine_poly_embedding
    
    config = config or {}
    
    # Generate quantum targets from RWR walks
    q_targets = generate_quantum_targets_from_walks(G, 'rwr', config, seed)
    
    if not q_targets:
        logger.warning("No quantum targets generated, falling back to baseline")
        from qbiocode.apps.quvine.embedding.quantum_filters import generate_baseline_poly_embedding
        embeddings = generate_baseline_poly_embedding(G, embedding_dim=config.get('embedding_dim', 128), random_state=seed)
    else:
        # Generate embeddings with quantum calibration
        embeddings = generate_quvine_poly_embedding(
            G=G,
            q_targets=q_targets,
            K=config.get('K', 4),
            ridge=config.get('ridge', 1e-6),
            embedding_dim=config.get('embedding_dim', 128),
            use_features=False,
            normalize=True,
            random_state=seed
        )
    
    # Evaluate based on task
    if task == "node_classification":
        return train_classifier(
            embeddings,
            np.array(split["train_idx"]),
            np.array(split["val_idx"]),
            np.array(split["test_idx"]),
            np.array(split["labels"]),
            seed
        )
    elif task == "link_prediction":
        return evaluate_link_prediction(
            embeddings,
            split["test_edges"],
            split["neg_test_edges"]
        )
    elif task == "node_ranking":
        return evaluate_node_ranking(
            embeddings,
            split["seed_nodes"],
            split["target_nodes"]
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def run_filter_ctqw_heat_adapter(
    G: nx.Graph,
    split: Dict[str, Any],
    task: str,
    config: Optional[Dict] = None,
    seed: int = 42
) -> Dict[str, float]:
    """Adapter for CTQW + heat kernel filter."""
    from qbiocode.apps.quvine.embedding.quantum_filters import (
        get_laplacian, calibrate_heat_kernel, apply_heat_filter
    )
    
    config = config or {}
    
    # Generate quantum targets from CTQW walks
    q_targets = generate_quantum_targets_from_walks(G, 'ctqw', config, seed)
    
    if not q_targets:
        logger.warning("No quantum targets generated, falling back to baseline")
        from qbiocode.apps.quvine.embedding.quantum_filters import generate_baseline_heat_embedding
        embeddings = generate_baseline_heat_embedding(
            G, embedding_dim=config.get('embedding_dim', 128), random_state=seed
        )
    else:
        # Manual implementation to work around tuple unpacking bug
        np.random.seed(seed)
        N = G.number_of_nodes()
        
        # Get Laplacian (unpack tuple properly)
        L, nodelist, node_to_idx = get_laplacian(G, normalize=True)
        
        # Calibrate heat kernel
        t_grid = np.logspace(-2, 2, 40)
        _, t_star = calibrate_heat_kernel(L, q_targets, t_grid, node_to_idx, loss='l2')
        
        # Generate random features
        embedding_dim = config.get('embedding_dim', 128)
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(N, embedding_dim))
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norm, 1e-12)
        
        # Apply heat kernel filter
        embeddings = apply_heat_filter(L, X, t_star)
    
    # Evaluate based on task
    if task == "node_classification":
        return train_classifier(
            embeddings,
            np.array(split["train_idx"]),
            np.array(split["val_idx"]),
            np.array(split["test_idx"]),
            np.array(split["labels"]),
            seed
        )
    elif task == "link_prediction":
        return evaluate_link_prediction(
            embeddings,
            split["test_edges"],
            split["neg_test_edges"]
        )
    elif task == "node_ranking":
        return evaluate_node_ranking(
            embeddings,
            split["seed_nodes"],
            split["target_nodes"]
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def run_filter_ctqw_poly_adapter(
    G: nx.Graph,
    split: Dict[str, Any],
    task: str,
    config: Optional[Dict] = None,
    seed: int = 42
) -> Dict[str, float]:
    """Adapter for CTQW + polynomial filter."""
    from qbiocode.apps.quvine.embedding.quantum_filters import generate_quvine_poly_embedding
    
    config = config or {}
    
    # Generate quantum targets from CTQW walks
    q_targets = generate_quantum_targets_from_walks(G, 'ctqw', config, seed)
    
    if not q_targets:
        logger.warning("No quantum targets generated, falling back to baseline")
        from qbiocode.apps.quvine.embedding.quantum_filters import generate_baseline_poly_embedding
        embeddings = generate_baseline_poly_embedding(G, embedding_dim=config.get('embedding_dim', 128), random_state=seed)
    else:
        # Generate embeddings with quantum calibration
        embeddings = generate_quvine_poly_embedding(
            G=G,
            q_targets=q_targets,
            K=config.get('K', 4),
            ridge=config.get('ridge', 1e-6),
            embedding_dim=config.get('embedding_dim', 128),
            use_features=False,
            normalize=True,
            random_state=seed
        )
    
    # Evaluate based on task
    if task == "node_classification":
        return train_classifier(
            embeddings,
            np.array(split["train_idx"]),
            np.array(split["val_idx"]),
            np.array(split["test_idx"]),
            np.array(split["labels"]),
            seed
        )
    elif task == "link_prediction":
        return evaluate_link_prediction(
            embeddings,
            split["test_edges"],
            split["neg_test_edges"]
        )
    elif task == "node_ranking":
        return evaluate_node_ranking(
            embeddings,
            split["seed_nodes"],
            split["target_nodes"]
        )
    else:
        raise ValueError(f"Unknown task: {task}")


# ============================================================================
# GAT Methods (Graph Attention Networks with various input features)
# ============================================================================

def run_gat_adapter(
    G: nx.Graph,
    split: Dict[str, Any],
    task: str,
    method_name: str,
    config: Optional[Dict] = None,
    seed: int = 42
) -> Dict[str, float]:
    """
    Generic GAT adapter that handles all 12 GAT variants.
    
    GAT variants differ only in their input features:
    - gat_baseline: raw structural features
    - gat_heat/poly: fixed filter parameters
    - gat_rwr/ctqw/dtqw: walk-based features
    - gat_*_heat/poly: quantum-calibrated filters
    """
    from qbiocode.apps.quvine.baselines.gat import generate_gat_embedding_by_method_name, GATConfig, TrainConfig
    
    config = config or {}
    
    # Prepare quantum targets if needed for calibrated variants
    ctqw_targets = None
    dtqw_targets = None
    rwr_targets = None
    
    if 'ctqw' in method_name and ('heat' in method_name or 'poly' in method_name):
        # Need CTQW targets for calibration
        ctqw_targets = generate_quantum_targets_from_walks(G, 'ctqw', config, seed)
    elif 'dtqw' in method_name and ('heat' in method_name or 'poly' in method_name):
        # Need DTQW targets for calibration
        dtqw_targets = generate_quantum_targets_from_walks(G, 'dtqw', config, seed)
    elif 'rwr' in method_name and ('heat' in method_name or 'poly' in method_name):
        # Need RWR targets for calibration
        rwr_targets = generate_quantum_targets_from_walks(G, 'rwr', config, seed)
    
    # Set up GAT configuration
    gat_config = GATConfig(
        hidden_dim=config.get('hidden_dim', 64),
        output_dim=config.get('embedding_dim', 128),
        num_layers=config.get('num_layers', 2),
        heads=config.get('heads', 4),
        dropout=config.get('dropout', 0.5),
        attention_dropout=config.get('attention_dropout', 0.2),
        negative_slope=config.get('negative_slope', 0.2),
        residual=config.get('residual', True)
    )
    
    # Set up training configuration
    train_config = TrainConfig(
        epochs=config.get('epochs', 200),
        lr=config.get('lr', 5e-3),
        weight_decay=config.get('weight_decay', 5e-4),
        patience=config.get('patience', 25),
        edge_batch_size=config.get('edge_batch_size', 4096),
        val_edge_fraction=config.get('val_edge_fraction', 0.1),
        device=config.get('device', 'cpu'),
        random_state=seed,
        verbose=config.get('verbose', False)
    )
    
    # Generate embeddings
    embeddings = generate_gat_embedding_by_method_name(
        G=G,
        method_name=method_name,
        embedding_dim=config.get('embedding_dim', 128),
        nodelist=list(G.nodes()),
        ctqw_targets=ctqw_targets,
        dtqw_targets=dtqw_targets,
        rwr_targets=rwr_targets,
        heat_t=config.get('heat_t', 1.0),
        poly_K=config.get('poly_K', 4),
        rwr_alpha=config.get('rwr_alpha', 0.15),
        gat_config=gat_config,
        train_config=train_config
    )
    
    # Evaluate based on task
    if task == "node_classification":
        return train_classifier(
            embeddings,
            np.array(split["train_idx"]),
            np.array(split["val_idx"]),
            np.array(split["test_idx"]),
            np.array(split["labels"]),
            seed
        )
    elif task == "link_prediction":
        return evaluate_link_prediction(
            embeddings,
            split["test_edges"],
            split["neg_test_edges"]
        )
    elif task == "node_ranking":
        return evaluate_node_ranking(
            embeddings,
            split["seed_nodes"],
            split["target_nodes"]
        )
    else:
        raise ValueError(f"Unknown task: {task}")


# Create individual adapters for each GAT variant
def run_gat_baseline_adapter(G, split, task, config=None, seed=42):
    """GAT with raw structural features."""
    return run_gat_adapter(G, split, task, 'gat_baseline', config, seed)

def run_gat_heat_adapter(G, split, task, config=None, seed=42):
    """GAT with fixed heat kernel features."""
    return run_gat_adapter(G, split, task, 'gat_heat', config, seed)

def run_gat_poly_adapter(G, split, task, config=None, seed=42):
    """GAT with fixed polynomial features."""
    return run_gat_adapter(G, split, task, 'gat_poly', config, seed)

def run_gat_rwr_adapter(G, split, task, config=None, seed=42):
    """GAT with RWR walk features."""
    return run_gat_adapter(G, split, task, 'gat_rwr', config, seed)

def run_gat_ctqw_adapter(G, split, task, config=None, seed=42):
    """GAT with direct CTQW features."""
    return run_gat_adapter(G, split, task, 'gat_ctqw', config, seed)

def run_gat_dtqw_adapter(G, split, task, config=None, seed=42):
    """GAT with direct DTQW features."""
    return run_gat_adapter(G, split, task, 'gat_dtqw', config, seed)

def run_gat_rwr_heat_adapter(G, split, task, config=None, seed=42):
    """GAT with RWR-calibrated heat kernel features."""
    return run_gat_adapter(G, split, task, 'gat_rwr_heat', config, seed)

def run_gat_rwr_poly_adapter(G, split, task, config=None, seed=42):
    """GAT with RWR-calibrated polynomial features."""
    return run_gat_adapter(G, split, task, 'gat_rwr_poly', config, seed)

def run_gat_ctqw_heat_adapter(G, split, task, config=None, seed=42):
    """GAT with CTQW-calibrated heat kernel features."""
    return run_gat_adapter(G, split, task, 'gat_ctqw_heat', config, seed)

def run_gat_ctqw_poly_adapter(G, split, task, config=None, seed=42):
    """GAT with CTQW-calibrated polynomial features."""
    return run_gat_adapter(G, split, task, 'gat_ctqw_poly', config, seed)

def run_gat_dtqw_heat_adapter(G, split, task, config=None, seed=42):
    """GAT with DTQW-calibrated heat kernel features."""
    return run_gat_adapter(G, split, task, 'gat_dtqw_heat', config, seed)

def run_gat_dtqw_poly_adapter(G, split, task, config=None, seed=42):
    """GAT with DTQW-calibrated polynomial features."""
    return run_gat_adapter(G, split, task, 'gat_dtqw_poly', config, seed)


# ============================================================================
# GraphGPS Methods (Graph GPS Transformer with various input features)
# ============================================================================

def run_graphgps_adapter(
    G: nx.Graph,
    split: Dict[str, Any],
    task: str,
    method_name: str,
    config: Optional[Dict] = None,
    seed: int = 42
) -> Dict[str, float]:
    """
    Generic GraphGPS adapter that handles all 12 GraphGPS variants.
    
    GraphGPS variants differ only in their input features (same as GAT):
    - graphgps_baseline: raw structural features
    - graphgps_heat/poly: fixed filter parameters
    - graphgps_rwr/ctqw/dtqw: walk-based features
    - graphgps_*_heat/poly: quantum-calibrated filters
    """
    from qbiocode.apps.quvine.baselines.graphgps import generate_graphgps_embedding_by_method_name, GraphGPSConfig, TrainConfig
    
    config = config or {}
    
    # Prepare quantum targets if needed for calibrated variants
    ctqw_targets = None
    dtqw_targets = None
    rwr_targets = None
    
    if 'ctqw' in method_name and ('heat' in method_name or 'poly' in method_name):
        ctqw_targets = generate_quantum_targets_from_walks(G, 'ctqw', config, seed)
    elif 'dtqw' in method_name and ('heat' in method_name or 'poly' in method_name):
        dtqw_targets = generate_quantum_targets_from_walks(G, 'dtqw', config, seed)
    elif 'rwr' in method_name and ('heat' in method_name or 'poly' in method_name):
        rwr_targets = generate_quantum_targets_from_walks(G, 'rwr', config, seed)
    
    # Set up GraphGPS configuration
    gps_config = GraphGPSConfig(
        hidden_dim=config.get('hidden_dim', 64),
        output_dim=config.get('embedding_dim', 128),
        num_layers=config.get('num_layers', 2),
        heads=config.get('heads', 4),
        dropout=config.get('dropout', 0.2),
        attn_dropout=config.get('attn_dropout', 0.2),
        local_gnn=config.get('local_gnn', 'gcn'),
        attn_type=config.get('attn_type', 'multihead'),
        use_layer_norm=config.get('use_layer_norm', True),
        activation=config.get('activation', 'relu'),
        lap_pe_dim=config.get('lap_pe_dim', 0),
        standardize_features=config.get('standardize_features', True)
    )
    
    # Set up training configuration
    train_config = TrainConfig(
        task='link_reconstruction',
        epochs=config.get('epochs', 200),
        lr=config.get('lr', 5e-3),
        weight_decay=config.get('weight_decay', 5e-4),
        patience=config.get('patience', 30),
        edge_batch_size=config.get('edge_batch_size', 8192),
        val_edge_fraction=config.get('val_edge_fraction', 0.1),
        device=config.get('device', 'cpu'),
        random_state=seed,
        verbose=config.get('verbose', False)
    )
    
    # Generate embeddings
    embeddings = generate_graphgps_embedding_by_method_name(
        G=G,
        method_name=method_name,
        embedding_dim=config.get('embedding_dim', 128),
        nodelist=list(G.nodes()),
        ctqw_targets=ctqw_targets,
        dtqw_targets=dtqw_targets,
        rwr_targets=rwr_targets,
        heat_t=config.get('heat_t', 1.0),
        poly_K=config.get('poly_K', 4),
        rwr_alpha=config.get('rwr_alpha', 0.15),
        gps_config=gps_config,
        train_config=train_config
    )
    
    # Evaluate based on task
    if task == "node_classification":
        return train_classifier(
            embeddings,
            np.array(split["train_idx"]),
            np.array(split["val_idx"]),
            np.array(split["test_idx"]),
            np.array(split["labels"]),
            seed
        )
    elif task == "link_prediction":
        return evaluate_link_prediction(
            embeddings,
            split["test_edges"],
            split["neg_test_edges"]
        )
    elif task == "node_ranking":
        return evaluate_node_ranking(
            embeddings,
            split["seed_nodes"],
            split["target_nodes"]
        )
    else:
        raise ValueError(f"Unknown task: {task}")


# Create individual adapters for each GraphGPS variant
def run_graphgps_baseline_adapter(G, split, task, config=None, seed=42):
    """GraphGPS with raw structural features."""
    return run_graphgps_adapter(G, split, task, 'graphgps_baseline', config, seed)

def run_graphgps_heat_adapter(G, split, task, config=None, seed=42):
    """GraphGPS with fixed heat kernel features."""
    return run_graphgps_adapter(G, split, task, 'graphgps_heat', config, seed)

def run_graphgps_poly_adapter(G, split, task, config=None, seed=42):
    """GraphGPS with fixed polynomial features."""
    return run_graphgps_adapter(G, split, task, 'graphgps_poly', config, seed)

def run_graphgps_rwr_adapter(G, split, task, config=None, seed=42):
    """GraphGPS with RWR walk features."""
    return run_graphgps_adapter(G, split, task, 'graphgps_rwr', config, seed)

def run_graphgps_ctqw_adapter(G, split, task, config=None, seed=42):
    """GraphGPS with direct CTQW features."""
    return run_graphgps_adapter(G, split, task, 'graphgps_ctqw', config, seed)

def run_graphgps_dtqw_adapter(G, split, task, config=None, seed=42):
    """GraphGPS with direct DTQW features."""
    return run_graphgps_adapter(G, split, task, 'graphgps_dtqw', config, seed)

def run_graphgps_rwr_heat_adapter(G, split, task, config=None, seed=42):
    """GraphGPS with RWR-calibrated heat kernel features."""
    return run_graphgps_adapter(G, split, task, 'graphgps_rwr_heat', config, seed)

def run_graphgps_rwr_poly_adapter(G, split, task, config=None, seed=42):
    """GraphGPS with RWR-calibrated polynomial features."""
    return run_graphgps_adapter(G, split, task, 'graphgps_rwr_poly', config, seed)

def run_graphgps_ctqw_heat_adapter(G, split, task, config=None, seed=42):
    """GraphGPS with CTQW-calibrated heat kernel features."""
    return run_graphgps_adapter(G, split, task, 'graphgps_ctqw_heat', config, seed)

def run_graphgps_ctqw_poly_adapter(G, split, task, config=None, seed=42):
    """GraphGPS with CTQW-calibrated polynomial features."""
    return run_graphgps_adapter(G, split, task, 'graphgps_ctqw_poly', config, seed)

def run_graphgps_dtqw_heat_adapter(G, split, task, config=None, seed=42):
    """GraphGPS with DTQW-calibrated heat kernel features."""
    return run_graphgps_adapter(G, split, task, 'graphgps_dtqw_heat', config, seed)

def run_graphgps_dtqw_poly_adapter(G, split, task, config=None, seed=42):
    """GraphGPS with DTQW-calibrated polynomial features."""
    return run_graphgps_adapter(G, split, task, 'graphgps_dtqw_poly', config, seed)


# ============================================================================
# Additional Classical Baselines
# ============================================================================

def run_baseline_gcnmf_adapter(
    G: nx.Graph,
    split: Dict[str, Any],
    task: str,
    config: Optional[Dict] = None,
    seed: int = 42
) -> Dict[str, float]:
    """Adapter for baseline GCN-MF (no quantum calibration)."""
    from qbiocode.apps.quvine.baselines.gcn_mf import generate_baseline_gcnmf_embedding
    
    config = config or {}
    
    # Generate embeddings
    embeddings = generate_baseline_gcnmf_embedding(
        G=G,
        embedding_dim=config.get('embedding_dim', 128),
        hidden_dim=config.get('hidden_dim', 64),
        mf_dim=config.get('mf_dim', 64),
        n_layers=config.get('n_layers', 2),
        epochs=config.get('epochs', 200),
        lr=config.get('lr', 0.01),
        weight_decay=config.get('weight_decay', 5e-4),
        random_state=seed,
        device=config.get('device', 'cpu')
    )
    
    # Evaluate based on task
    if task == "node_classification":
        return train_classifier(
            embeddings,
            np.array(split["train_idx"]),
            np.array(split["val_idx"]),
            np.array(split["test_idx"]),
            np.array(split["labels"]),
            seed
        )
    elif task == "link_prediction":
        return evaluate_link_prediction(
            embeddings,
            split["test_edges"],
            split["neg_test_edges"]
        )
    elif task == "node_ranking":
        return evaluate_node_ranking(
            embeddings,
            split["seed_nodes"],
            split["target_nodes"]
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def run_baseline_filter_adapter(
    G: nx.Graph,
    split: Dict[str, Any],
    task: str,
    config: Optional[Dict] = None,
    seed: int = 42
) -> Dict[str, float]:
    """
    Adapter for generic baseline filter (defaults to heat kernel).
    
    This is the generic 'baseline_filter' method that can be configured
    to use either heat or polynomial filters via config['filter_type'].
    Defaults to heat kernel for backward compatibility.
    """
    from qbiocode.apps.quvine.embedding.quantum_filters import generate_baseline_filter_embedding
    
    config = config or {}
    filter_type = config.get('filter_type', 'heat')
    
    # Generate embeddings using the generic baseline filter function
    embeddings = generate_baseline_filter_embedding(
        G=G,
        filter_type=filter_type,
        t=config.get('t', 1.0),
        K=config.get('K', 4),
        embedding_dim=config.get('embedding_dim', 128),
        use_features=False,
        normalize=config.get('normalize', True),
        random_state=seed
    )
    
    # Evaluate based on task
    if task == "node_classification":
        return train_classifier(
            embeddings,
            np.array(split["train_idx"]),
            np.array(split["val_idx"]),
            np.array(split["test_idx"]),
            np.array(split["labels"]),
            seed
        )
    elif task == "link_prediction":
        return evaluate_link_prediction(
            embeddings,
            split["test_edges"],
            split["neg_test_edges"]
        )
    elif task == "node_ranking":
        return evaluate_node_ranking(
            embeddings,
            split["seed_nodes"],
            split["target_nodes"]
        )
    else:
        raise ValueError(f"Unknown task: {task}")


# ============================================================================
# Fusion Methods
# ============================================================================

def run_rwr_fusion_adapter(
    G: nx.Graph,
    split: Dict[str, Any],
    task: str,
    config: Optional[Dict] = None,
    seed: int = 42
) -> Dict[str, float]:
    """
    Fusion method combining RWR-based embeddings.
    
    Combines:
    - quvine_rwr (SGNS with RWR)
    - filter_rwr_heat (RWR-calibrated heat kernel)
    - filter_rwr_poly (RWR-calibrated polynomial)
    
    Uses SVD-based fusion to combine the three views.
    """
    from qbiocode.apps.quvine.fusion.fuse import fuse_embeddings_svd, _prep_blocks
    from qbiocode.apps.quvine.embedding.quantum_filters import (
        get_laplacian, calibrate_heat_kernel, apply_heat_filter,
        calibrate_polynomial_filter, apply_polynomial_filter,
        generate_baseline_heat_embedding, generate_baseline_poly_embedding
    )
    
    config = config or {}
    embedding_dim = config.get('embedding_dim', 128)
    
    # Generate three RWR-based embeddings
    emb_sgns = run_quvine_sgns(G, 'rwr', config, seed)
    
    # Generate quantum targets
    q_targets = generate_quantum_targets_from_walks(G, 'rwr', config, seed)
    
    if q_targets:
        np.random.seed(seed)
        N = G.number_of_nodes()
        L, nodelist, node_to_idx = get_laplacian(G, normalize=True)
        
        # Generate random features
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(N, embedding_dim))
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norm, 1e-12)
        
        # Calibrate and apply heat filter
        t_grid = np.logspace(-2, 2, 40)
        _, t_star = calibrate_heat_kernel(L, q_targets, t_grid, node_to_idx, loss='l2')
        emb_heat = apply_heat_filter(L, X, t_star)
        
        # Calibrate and apply poly filter
        result = calibrate_polynomial_filter(L, q_targets, K=config.get('K', 4), node_to_idx=node_to_idx, ridge=1e-6)
        poly_coeffs = result[0] if isinstance(result, tuple) else result
        emb_poly = apply_polynomial_filter(L, X, poly_coeffs)
    else:
        emb_heat = generate_baseline_heat_embedding(G, embedding_dim=embedding_dim, random_state=seed)
        emb_poly = generate_baseline_poly_embedding(G, embedding_dim=embedding_dim, order=config.get('K', 4), random_state=seed)
    
    # Fuse the three views
    embeddings_list = _prep_blocks([emb_sgns, emb_heat, emb_poly])
    fused_embeddings = fuse_embeddings_svd(embeddings_list, k=embedding_dim)
    
    # Evaluate based on task
    if task == "node_classification":
        return train_classifier(
            fused_embeddings,
            np.array(split["train_idx"]),
            np.array(split["val_idx"]),
            np.array(split["test_idx"]),
            np.array(split["labels"]),
            seed
        )
    elif task == "link_prediction":
        return evaluate_link_prediction(
            fused_embeddings,
            split["test_edges"],
            split["neg_test_edges"]
        )
    elif task == "node_ranking":
        return evaluate_node_ranking(
            fused_embeddings,
            split["seed_nodes"],
            split["target_nodes"]
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def run_ctqw_fusion_adapter(
    G: nx.Graph,
    split: Dict[str, Any],
    task: str,
    config: Optional[Dict] = None,
    seed: int = 42
) -> Dict[str, float]:
    """
    Fusion method combining CTQW-based embeddings.
    
    Combines:
    - quvine_ctqw (SGNS with CTQW)
    - filter_ctqw_heat (CTQW-calibrated heat kernel)
    - filter_ctqw_poly (CTQW-calibrated polynomial)
    
    Uses SVD-based fusion to combine the three views.
    """
    from qbiocode.apps.quvine.fusion.fuse import fuse_embeddings_svd, _prep_blocks
    from qbiocode.apps.quvine.embedding.quantum_filters import (
        get_laplacian, calibrate_heat_kernel, apply_heat_filter,
        calibrate_polynomial_filter, apply_polynomial_filter,
        generate_baseline_heat_embedding, generate_baseline_poly_embedding
    )
    
    config = config or {}
    embedding_dim = config.get('embedding_dim', 128)
    
    # Generate three CTQW-based embeddings
    emb_sgns = run_quvine_sgns(G, 'ctqw', config, seed)
    
    # Generate quantum targets
    q_targets = generate_quantum_targets_from_walks(G, 'ctqw', config, seed)
    
    if q_targets:
        np.random.seed(seed)
        N = G.number_of_nodes()
        L, nodelist, node_to_idx = get_laplacian(G, normalize=True)
        
        # Generate random features
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(N, embedding_dim))
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norm, 1e-12)
        
        # Calibrate and apply heat filter
        t_grid = np.logspace(-2, 2, 40)
        _, t_star = calibrate_heat_kernel(L, q_targets, t_grid, node_to_idx, loss='l2')
        emb_heat = apply_heat_filter(L, X, t_star)
        
        # Calibrate and apply poly filter
        result = calibrate_polynomial_filter(L, q_targets, K=config.get('K', 4), node_to_idx=node_to_idx, ridge=1e-6)
        poly_coeffs = result[0] if isinstance(result, tuple) else result
        emb_poly = apply_polynomial_filter(L, X, poly_coeffs)
    else:
        emb_heat = generate_baseline_heat_embedding(G, embedding_dim=embedding_dim, random_state=seed)
        emb_poly = generate_baseline_poly_embedding(G, embedding_dim=embedding_dim, order=config.get('K', 4), random_state=seed)
    
    # Fuse the three views
    embeddings_list = _prep_blocks([emb_sgns, emb_heat, emb_poly])
    fused_embeddings = fuse_embeddings_svd(embeddings_list, k=embedding_dim)
    
    # Evaluate based on task
    if task == "node_classification":
        return train_classifier(
            fused_embeddings,
            np.array(split["train_idx"]),
            np.array(split["val_idx"]),
            np.array(split["test_idx"]),
            np.array(split["labels"]),
            seed
        )
    elif task == "link_prediction":
        return evaluate_link_prediction(
            fused_embeddings,
            split["test_edges"],
            split["neg_test_edges"]
        )
    elif task == "node_ranking":
        return evaluate_node_ranking(
            fused_embeddings,
            split["seed_nodes"],
            split["target_nodes"]
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def run_dtqw_fusion_adapter(
    G: nx.Graph,
    split: Dict[str, Any],
    task: str,
    config: Optional[Dict] = None,
    seed: int = 42
) -> Dict[str, float]:
    """
    Fusion method combining DTQW-based embeddings.
    
    Combines:
    - quvine_dtqw (SGNS with DTQW)
    - Baseline heat kernel (as DTQW doesn't have filter variants)
    - Baseline polynomial (as DTQW doesn't have filter variants)
    
    Uses SVD-based fusion to combine the three views.
    Note: DTQW doesn't have dedicated filter variants, so we use baseline filters.
    """
    from qbiocode.apps.quvine.embedding.quantum_filters import (
        generate_baseline_heat_embedding,
        generate_baseline_poly_embedding
    )
    from qbiocode.apps.quvine.fusion.fuse import fuse_embeddings_svd, _prep_blocks
    
    config = config or {}
    embedding_dim = config.get('embedding_dim', 128)
    
    # Generate DTQW-based SGNS embedding using existing helper
    emb_sgns = run_quvine_sgns(G, 'dtqw', config, seed)
    
    # Generate baseline filter embeddings (DTQW doesn't have calibrated filters)
    emb_heat = generate_baseline_heat_embedding(
        G=G,
        embedding_dim=embedding_dim,
        scale=config.get('scale', 1.0),
        use_features=False,
        normalize=True,
        random_state=seed
    )
    
    emb_poly = generate_baseline_poly_embedding(
        G=G,
        embedding_dim=embedding_dim,
        order=config.get('order', 4),
        use_features=False,
        normalize=True,
        random_state=seed
    )
    
    # Fuse the three views
    embeddings_list = _prep_blocks([emb_sgns, emb_heat, emb_poly])
    fused_embeddings = fuse_embeddings_svd(embeddings_list, k=embedding_dim)
    
    # Evaluate based on task
    if task == "node_classification":
        return train_classifier(
            fused_embeddings,
            np.array(split["train_idx"]),
            np.array(split["val_idx"]),
            np.array(split["test_idx"]),
            np.array(split["labels"]),
            seed
        )
    elif task == "link_prediction":
        return evaluate_link_prediction(
            fused_embeddings,
            split["test_edges"],
            split["neg_test_edges"]
        )
    elif task == "node_ranking":
        return evaluate_node_ranking(
            fused_embeddings,
            split["seed_nodes"],
            split["target_nodes"]
        )
    else:
        raise ValueError(f"Unknown task: {task}")


# ============================================================================
# Method Registry
# ============================================================================

METHOD_ADAPTERS = {
    # Classical baselines
    "node2vec": run_node2vec_adapter,
    "netmf": run_netmf_adapter,
    "graphsage": run_graphsage_adapter,
    "appnp": run_appnp_adapter,
    
    # QuVINE SGNS methods
    "quvine_rwr": run_quvine_rwr_adapter,
    "quvine_ctqw": run_quvine_ctqw_adapter,
    "quvine_dtqw": run_quvine_dtqw_adapter,
    
    # Baseline filter methods (no quantum calibration)
    "baseline_filter_heat": run_baseline_filter_heat_adapter,
    "baseline_filter_poly": run_baseline_filter_poly_adapter,
    
    # QuVINE filter methods (quantum-calibrated)
    "filter_rwr_heat": run_filter_rwr_heat_adapter,
    "filter_rwr_poly": run_filter_rwr_poly_adapter,
    "filter_ctqw_heat": run_filter_ctqw_heat_adapter,
    "filter_ctqw_poly": run_filter_ctqw_poly_adapter,
    
    # GAT variants (12 methods)
    "gat_baseline": run_gat_baseline_adapter,
    "gat_heat": run_gat_heat_adapter,
    "gat_poly": run_gat_poly_adapter,
    "gat_rwr": run_gat_rwr_adapter,
    "gat_ctqw": run_gat_ctqw_adapter,
    "gat_dtqw": run_gat_dtqw_adapter,
    "gat_rwr_heat": run_gat_rwr_heat_adapter,
    "gat_rwr_poly": run_gat_rwr_poly_adapter,
    "gat_ctqw_heat": run_gat_ctqw_heat_adapter,
    "gat_ctqw_poly": run_gat_ctqw_poly_adapter,
    "gat_dtqw_heat": run_gat_dtqw_heat_adapter,
    "gat_dtqw_poly": run_gat_dtqw_poly_adapter,
    
    # GraphGPS variants (12 methods)
    "graphgps_baseline": run_graphgps_baseline_adapter,
    "graphgps_heat": run_graphgps_heat_adapter,
    "graphgps_poly": run_graphgps_poly_adapter,
    "graphgps_rwr": run_graphgps_rwr_adapter,
    "graphgps_ctqw": run_graphgps_ctqw_adapter,
    "graphgps_dtqw": run_graphgps_dtqw_adapter,
    "graphgps_rwr_heat": run_graphgps_rwr_heat_adapter,
    "graphgps_rwr_poly": run_graphgps_rwr_poly_adapter,
    "graphgps_ctqw_heat": run_graphgps_ctqw_heat_adapter,
    "graphgps_ctqw_poly": run_graphgps_ctqw_poly_adapter,
    "graphgps_dtqw_heat": run_graphgps_dtqw_heat_adapter,
    "graphgps_dtqw_poly": run_graphgps_dtqw_poly_adapter,
    
    # Additional classical baselines
    "baseline_gcnmf": run_baseline_gcnmf_adapter,
    "baseline_filter": run_baseline_filter_adapter,
    
    # Fusion methods (combine multiple embedding views)
    "rwr_fusion": run_rwr_fusion_adapter,
    "ctqw_fusion": run_ctqw_fusion_adapter,
    "dtqw_fusion": run_dtqw_fusion_adapter,
}


def get_method_adapter(method_name: str):
    """
    Get the adapter function for a method.
    
    Parameters
    ----------
    method_name : str
        Name of the method
    
    Returns
    -------
    callable
        Adapter function
    
    Raises
    ------
    NotImplementedError
        If method is not yet implemented
    """
    if method_name not in METHOD_ADAPTERS:
        raise NotImplementedError(
            f"Method '{method_name}' not yet implemented. "
            f"Available methods: {list(METHOD_ADAPTERS.keys())}"
        )
    
    return METHOD_ADAPTERS[method_name]

