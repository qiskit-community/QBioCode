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
Node Classification Evaluation Module

This module provides functions for evaluating node embeddings on node classification tasks.
Includes multiple label generation strategies and comprehensive evaluation metrics.

Label Generation Strategies:
1. Community-based: Louvain, Label Propagation, Spectral Clustering
2. Degree-based: Structural role binning
3. Centrality-based: Betweenness, Closeness, Eigenvector, PageRank
4. Core-periphery: K-core decomposition, Rich-club
5. Homophily-based: Graph structure-aware labels (from Q-Caliber)

Evaluation Metrics:
- Accuracy, Precision, Recall, F1-score (macro/micro/weighted)
- Confusion matrix
- Per-class metrics
"""

import numpy as np
import scipy.sparse as sp
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import logging
import warnings


logger = logging.getLogger(__name__)

def generate_community_labels(
    G: nx.Graph,
    method: str = 'louvain',
    min_community_size: int = 5,
    resolution: float = 1.0
) -> Dict[int, int]:
    """
    Generate node labels based on community detection.
    
    Args:
        G: NetworkX graph
        method: Community detection method ('louvain', 'label_propagation', 'spectral')
        min_community_size: Minimum nodes per community (smaller communities merged)
        resolution: Resolution parameter for Louvain (higher = more communities)
    
    Returns:
        Dictionary mapping node IDs to community labels
    """
    if method == 'louvain':
        try:
            import community as community_louvain
            communities = community_louvain.best_partition(G, resolution=resolution)
        except ImportError:
            warnings.warn("python-louvain not installed, falling back to label propagation")
            method = 'label_propagation'
    
    if method == 'label_propagation':
        communities_gen = nx.algorithms.community.label_propagation_communities(G)
        communities = {}
        for label, nodes in enumerate(communities_gen):
            for node in nodes:
                communities[node] = label
    
    elif method == 'spectral':
        # Use spectral clustering with k determined by modularity optimization
        from sklearn.cluster import SpectralClustering
        adj_matrix = nx.to_scipy_sparse_array(G)
        
        # Try different k values and pick best modularity
        best_k = 2
        best_modularity = -1
        for k in range(2, min(20, G.number_of_nodes() // 10)):
            clustering = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=42)
            labels = clustering.fit_predict(adj_matrix.toarray())
            temp_communities = {node: labels[i] for i, node in enumerate(G.nodes())}
            mod = nx.algorithms.community.modularity(
                G, 
                [{n for n, l in temp_communities.items() if l == label} for label in set(labels)]
            )
            if mod > best_modularity:
                best_modularity = mod
                best_k = k
        
        clustering = SpectralClustering(n_clusters=best_k, affinity='precomputed', random_state=42)
        labels = clustering.fit_predict(adj_matrix.toarray())
        communities = {node: labels[i] for i, node in enumerate(G.nodes())}
    
    # Merge small communities
    label_counts = {}
    for label in communities.values():
        label_counts[label] = label_counts.get(label, 0) + 1
    
    small_labels = {label for label, count in label_counts.items() if count < min_community_size}
    
    if small_labels:
        # Merge small communities into largest community
        largest_label = max(label_counts.items(), key=lambda x: x[1])[0]
        communities = {
            node: (largest_label if label in small_labels else label)
            for node, label in communities.items()
        }
        
        # Relabel to consecutive integers
        unique_labels = sorted(set(communities.values()))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        communities = {node: label_map[label] for node, label in communities.items()}
    
    return communities


def generate_degree_labels(
    G: nx.Graph,
    n_bins: int = 5,
    method: str = 'quantile'
) -> Dict[int, int]:
    """
    Generate node labels based on degree binning.
    
    Args:
        G: NetworkX graph
        n_bins: Number of degree bins
        method: Binning method ('quantile' or 'uniform')
    
    Returns:
        Dictionary mapping node IDs to degree-based labels
    """
    degrees = dict(G.degree())
    degree_values = np.array(list(degrees.values()))
    
    if method == 'quantile':
        # Equal-sized bins
        bins = np.percentile(degree_values, np.linspace(0, 100, n_bins + 1))
    else:  # uniform
        # Equal-width bins
        bins = np.linspace(degree_values.min(), degree_values.max() + 1, n_bins + 1)
    
    labels = {}
    for node, degree in degrees.items():
        label = np.digitize(degree, bins[1:])  # Returns 0 to n_bins-1
        labels[node] = min(label, n_bins - 1)  # Ensure max label is n_bins-1
    
    return labels


def generate_centrality_labels(
    G: nx.Graph,
    centrality_type: str = 'betweenness',
    n_bins: int = 5
) -> Dict[int, int]:
    """
    Generate node labels based on centrality measures.
    
    Args:
        G: NetworkX graph
        centrality_type: Type of centrality ('betweenness', 'closeness', 'eigenvector', 'pagerank')
        n_bins: Number of centrality bins
    
    Returns:
        Dictionary mapping node IDs to centrality-based labels
    """
    if centrality_type == 'betweenness':
        centrality = nx.betweenness_centrality(G)
    elif centrality_type == 'closeness':
        centrality = nx.closeness_centrality(G)
    elif centrality_type == 'eigenvector':
        try:
            centrality = nx.eigenvector_centrality(G, max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            warnings.warn("Eigenvector centrality failed to converge, using PageRank")
            centrality = nx.pagerank(G)
    elif centrality_type == 'pagerank':
        centrality = nx.pagerank(G)
    else:
        raise ValueError(f"Unknown centrality type: {centrality_type}")
    
    centrality_values = np.array(list(centrality.values()))
    bins = np.percentile(centrality_values, np.linspace(0, 100, n_bins + 1))
    
    labels = {}
    for node, cent_value in centrality.items():
        label = np.digitize(cent_value, bins[1:])
        labels[node] = min(label, n_bins - 1)
    
    return labels


def generate_core_periphery_labels(
    G: nx.Graph,
    method: str = 'k_core',
    n_bins: int = 3
) -> Dict[int, int]:
    """
    Generate node labels based on core-periphery structure.
    
    Args:
        G: NetworkX graph
        method: Method ('k_core' or 'rich_club')
        n_bins: Number of bins for rich-club method
    
    Returns:
        Dictionary mapping node IDs to core-periphery labels
    """
    if method == 'k_core':
        # Use k-core decomposition
        core_numbers = nx.core_number(G)
        max_core = max(core_numbers.values())
        
        # Create labels: 0=periphery, 1=intermediate, 2=core
        labels = {}
        for node, k in core_numbers.items():
            if k <= max_core * 0.33:
                labels[node] = 0  # Periphery
            elif k <= max_core * 0.67:
                labels[node] = 1  # Intermediate
            else:
                labels[node] = 2  # Core
    
    elif method == 'rich_club':
        # Use rich-club coefficient
        degrees = dict(G.degree())
        degree_values = np.array(list(degrees.values()))
        bins = np.percentile(degree_values, np.linspace(0, 100, n_bins + 1))
        
        labels = {}
        for node, degree in degrees.items():
            label = np.digitize(degree, bins[1:])
            labels[node] = min(label, n_bins - 1)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return labels


def generate_homophily_labels(
    G: nx.Graph,
    embeddings: Optional[np.ndarray] = None,
    feature_weight: float = 0.5,
    neighbor_weight: float = 0.5,
    noise_std: float = 0.15,
    n_bins: int = 2,
    seed_nodes: Optional[List[int]] = None,
    random_state: int = 42
) -> Dict[int, int]:
    """
    Generate node labels with homophily (graph structure influence).
    
    This strategy creates labels that respect graph structure without direct data leakage.
    It combines feature-based scores with neighbor influence to create realistic labels
    where similar/connected nodes tend to have similar labels.
    
    Strategy (from Q-Caliber notebook):
    1. Create feature-based scores from embeddings or random features
    2. Initialize labels based on feature scores
    3. Add homophily: neighbors of positive nodes more likely positive
    4. Combine feature scores (50%) with neighbor influence (50%)
    5. Add noise and create final labels
    
    Args:
        G: NetworkX graph
        embeddings: Node embeddings (optional, if None uses random features)
        feature_weight: Weight for feature-based scores (default 0.5)
        neighbor_weight: Weight for neighbor influence (default 0.5)
        noise_std: Standard deviation of noise to add (default 0.15)
        n_bins: Number of label classes (default 2 for binary)
        seed_nodes: Optional list of nodes to force as positive class
        random_state: Random seed
    
    Returns:
        Dictionary mapping node IDs to homophily-based labels
    """
    np.random.seed(random_state)
    
    N = G.number_of_nodes()
    node_list = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    
    # Get adjacency matrix
    A = nx.to_scipy_sparse_array(G, nodelist=node_list, format='csr')
    
    # Step 1: Create feature-based scores
    if embeddings is not None:
        # Use provided embeddings
        F = embeddings.shape[1]
        X = embeddings
    else:
        # Generate random features with graph structure influence
        F = 32
        X_base = np.random.randn(N, F)

        # Degree-normalised adjacency stays sparse: avoids O(N^2) dense broadcast.
        A_tilde = A + sp.eye(N, format='csr')
        deg = np.array(A_tilde.sum(axis=1)).flatten()
        D_inv = sp.diags(1.0 / (deg + 1e-8), format='csr')
        A_norm = D_inv @ A_tilde  # sparse CSR, O(E) storage

        # Two-hop smoothing as two sparse matrix multiplies over all F columns at once.
        X_smooth = A_norm @ (A_norm @ X_base)

        # Mix base and smooth features with noise
        X = 0.4 * X_base + 0.4 * X_smooth + 0.2 * np.random.randn(N, F)
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)  # Normalize
    
    # Create decision weights and compute feature scores
    decision_weights = np.random.randn(F)
    decision_weights = decision_weights / np.linalg.norm(decision_weights)
    feature_scores = X @ decision_weights
    feature_scores = (feature_scores - feature_scores.min()) / (feature_scores.max() - feature_scores.min() + 1e-10)
    
    # Step 2: Initialize labels based on features
    threshold = np.median(feature_scores)
    y = (feature_scores > threshold).astype(int)
    
    # Step 3: Force seed nodes to be positive (if provided)
    if seed_nodes is not None:
        for seed in seed_nodes:
            if seed in node_to_idx:
                y[node_to_idx[seed]] = 1
    
    # Step 4: Add homophily - neighbors of positive nodes more likely positive
    # This makes graph structure useful without direct leakage
    y_float = y.astype(float)
    neighbor_influence = A @ y_float  # Sum of neighbor labels
    neighbor_influence = neighbor_influence / (neighbor_influence.max() + 1e-10)
    
    # Step 5: Combine feature scores with neighbor influence
    # Balance allows graph diffusion methods to be useful
    combined_scores = feature_weight * feature_scores + neighbor_weight * neighbor_influence
    
    # Add noise
    noise = np.random.randn(N) * noise_std
    noisy_scores = combined_scores + noise
    
    # Step 6: Create final labels
    if n_bins == 2:
        # Binary classification
        threshold_final = np.median(noisy_scores)
        y = (noisy_scores > threshold_final).astype(int)
    else:
        # Multi-class classification
        bins = np.percentile(noisy_scores, np.linspace(0, 100, n_bins + 1))
        y = np.digitize(noisy_scores, bins[1:])
        y = np.clip(y, 0, n_bins - 1)
    
    # Re-force seed nodes to be positive (if provided)
    if seed_nodes is not None:
        for seed in seed_nodes:
            if seed in node_to_idx:
                y[node_to_idx[seed]] = min(n_bins - 1, 1)  # Positive class
    
    # Convert to dictionary
    labels = {node: int(y[idx]) for node, idx in node_to_idx.items()}
    
    return labels


def evaluate_node_classification(
    embeddings: np.ndarray,
    labels: Dict[int, int],
    node_list: List[int],
    test_size: float = 0.3,
    classifier: str = 'logistic',
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Evaluate node embeddings on classification task.
    
    Args:
        embeddings: Node embedding matrix (n_nodes x embedding_dim)
        labels: Dictionary mapping node IDs to class labels
        node_list: List of node IDs corresponding to embedding rows
        test_size: Fraction of data for testing
        classifier: Classifier type ('logistic' or 'random_forest')
        n_splits: Number of cross-validation splits
        random_state: Random seed
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Prepare data
    X = embeddings
    y = np.array([labels[node] for node in node_list])
    
    # Check if we have enough samples per class
    unique_labels, label_counts = np.unique(y, return_counts=True)
    min_samples = label_counts.min()
    
    if min_samples < 2:
        warnings.warn(f"Some classes have only {min_samples} sample(s). Skipping evaluation.")
        return {
            'accuracy': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'f1_macro': 0.0,
            'n_classes': len(unique_labels),
            'error': 'insufficient_samples'
        }
    
    # Initialize classifier
    if classifier == 'logistic':
        clf = LogisticRegression(max_iter=1000, random_state=random_state)
    elif classifier == 'random_forest':
        clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    
    # Single train-test split evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # FIX: Prevent data leakage - fit scaler only on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Use transform, not fit_transform
    
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    
    # Compute metrics
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'precision_micro': precision_score(y_test, y_pred, average='micro', zero_division=0),
        'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_micro': recall_score(y_test, y_pred, average='micro', zero_division=0),
        'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_micro': f1_score(y_test, y_pred, average='micro', zero_division=0),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'n_classes': len(unique_labels),
        'n_train': len(y_train),
        'n_test': len(y_test)
    }
    
    # Cross-validation if enough samples
    if min_samples >= n_splits:
        cv_scores = []
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        for train_idx, test_idx in skf.split(X, y):
            X_train_cv, X_test_cv = X[train_idx], X[test_idx]
            cv_scaler = StandardScaler()
            X_train_cv_scaled = cv_scaler.fit_transform(X_train_cv)
            X_test_cv_scaled = cv_scaler.transform(X_test_cv)
            y_train_cv, y_test_cv = y[train_idx], y[test_idx]
            
            clf_cv = LogisticRegression(max_iter=1000, random_state=random_state)
            clf_cv.fit(X_train_cv_scaled, y_train_cv)
            y_pred_cv = clf_cv.predict(X_test_cv_scaled)
            
            cv_scores.append(f1_score(y_test_cv, y_pred_cv, average='macro', zero_division=0))
        
        results['f1_macro_cv_mean'] = np.mean(cv_scores)
        results['f1_macro_cv_std'] = np.std(cv_scores)
    
    return results


def _generate_ensemble_labels(
    node_list: List[int],
    label_dicts: Dict[str, Dict[int, int]],
) -> Dict[int, int]:
    """
    Majority-vote ensemble across all successful label strategies.

    Each strategy's labels are median-binarized (above median → 1) before
    voting, so multi-class strategies contribute a clean binary signal.
    A node is assigned class 1 when strictly more than half the strategies
    voted 1 for it.
    """
    N = len(node_list)
    votes = np.zeros(N, dtype=np.int32)
    n_used = 0
    for labels in label_dicts.values():
        arr = np.array([labels.get(n, 0) for n in node_list])
        if len(np.unique(arr)) < 2:
            continue  # degenerate strategy — skip
        binary = (arr > np.median(arr)).astype(np.int32)
        votes += binary
        n_used += 1
    if n_used == 0:
        return {n: 0 for n in node_list}
    ensemble = (votes * 2 > n_used).astype(int)
    return {n: int(ensemble[i]) for i, n in enumerate(node_list)}


def evaluate_all_label_strategies(
    G: nx.Graph,
    embeddings: np.ndarray,
    node_list: List[int],
    test_size: float = 0.3,
    random_state: int = 42,
    pregenerated_split=None,  # accepted for API compat, not used in label generation
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate embeddings using all label generation strategies plus an ensemble.

    Strategies (8 total):
      1. community_louvain
      2. community_label_propagation
      3. degree_based
      4. centrality_betweenness
      5. centrality_pagerank
      6. core_periphery
      7. homophily_based
      8. ensemble  (majority-vote across all successful strategies above)

    Args:
        G: NetworkX graph
        embeddings: Node embedding matrix
        node_list: List of node IDs
        test_size: Test set fraction
        random_state: Random seed
        pregenerated_split: Accepted for API compatibility; unused here.

    Returns:
        Dictionary mapping strategy names to evaluation results
    """
    results = {}
    _label_dicts: Dict[str, Dict[int, int]] = {}  # track for ensemble

    # 1. Community-based labels
    for method in ['louvain', 'label_propagation']:
        key = f'community_{method}'
        try:
            labels = generate_community_labels(G, method=method)
            eval_results = evaluate_node_classification(
                embeddings, labels, node_list, test_size, random_state=random_state
            )
            results[key] = eval_results
            _label_dicts[key] = labels
        except Exception as e:
            warnings.warn(f"Community detection ({method}) failed: {e}")
            results[key] = {'error': str(e)}

    # 2. Degree-based labels
    try:
        labels = generate_degree_labels(G, n_bins=5)
        eval_results = evaluate_node_classification(
            embeddings, labels, node_list, test_size, random_state=random_state
        )
        results['degree_based'] = eval_results
        _label_dicts['degree_based'] = labels
    except Exception as e:
        warnings.warn(f"Degree-based labeling failed: {e}")
        results['degree_based'] = {'error': str(e)}

    # 3. Centrality-based labels
    for cent_type in ['betweenness', 'pagerank']:
        key = f'centrality_{cent_type}'
        try:
            labels = generate_centrality_labels(G, centrality_type=cent_type, n_bins=5)
            eval_results = evaluate_node_classification(
                embeddings, labels, node_list, test_size, random_state=random_state
            )
            results[key] = eval_results
            _label_dicts[key] = labels
        except Exception as e:
            warnings.warn(f"Centrality-based labeling ({cent_type}) failed: {e}")
            results[key] = {'error': str(e)}

    # 4. Core-periphery labels
    try:
        labels = generate_core_periphery_labels(G, method='k_core')
        eval_results = evaluate_node_classification(
            embeddings, labels, node_list, test_size, random_state=random_state
        )
        results['core_periphery'] = eval_results
        _label_dicts['core_periphery'] = labels
    except Exception as e:
        warnings.warn(f"Core-periphery labeling failed: {e}")
        results['core_periphery'] = {'error': str(e)}

    # 5. Homophily-based labels (topology-only: prevents circular dependency
    #    where a method's own embedding defines the labels it is then scored on).
    try:
        labels = generate_homophily_labels(
            G,
            embeddings=None,
            feature_weight=0.5,
            neighbor_weight=0.5,
            random_state=random_state,
        )
        eval_results = evaluate_node_classification(
            embeddings, labels, node_list, test_size, random_state=random_state
        )
        results['homophily_based'] = eval_results
        _label_dicts['homophily_based'] = labels
    except Exception as e:
        warnings.warn(f"Homophily-based labeling failed: {e}")
        results['homophily_based'] = {'error': str(e)}

    # 6. Ensemble: majority-vote across all strategies that succeeded
    if len(_label_dicts) >= 2:
        try:
            ensemble_labels = _generate_ensemble_labels(node_list, _label_dicts)
            eval_results = evaluate_node_classification(
                embeddings, ensemble_labels, node_list, test_size, random_state=random_state
            )
            results['ensemble'] = eval_results
        except Exception as e:
            warnings.warn(f"Ensemble labeling failed: {e}")
            results['ensemble'] = {'error': str(e)}
    else:
        results['ensemble'] = {'error': 'insufficient_strategies_for_ensemble'}

    return results


def evaluate_nc_stratified(
    G: nx.Graph,
    embeddings: np.ndarray,
    node_list: List[int],
    label_strategy: str = 'louvain',
    n_degree_bins: int = 5,
    dist_max_bin: int = 5,
    test_size: float = 0.3,
    random_state: int = 42,
) -> List[Dict]:
    """
    Evaluate node classification stratified by node degree and distance from hubs.

    Produces one row per (bin_type, bin_label) for the primary label strategy,
    reporting per-bin accuracy on the test nodes.  This mirrors the degree/
    distance-matched controls used in LP and ranking evaluations.

    Parameters
    ----------
    G : nx.Graph
    embeddings : np.ndarray  (n_nodes × dim)
    node_list : list
        Nodes in the same order as rows of *embeddings*.
    label_strategy : str
        Community detection method to generate labels ('louvain' or 'label_propagation').
    n_degree_bins : int
        Number of degree quantile bins (default 5 → Q1–Q5).
    dist_max_bin : int
        Distances ≥ this value are grouped into a single "{dist_max_bin}+" bin.
    test_size : float
        Fraction of nodes held out for evaluation.
    random_state : int

    Returns
    -------
    list of dicts
        Each dict has keys: bin_type, bin_label, bin_n_nodes, accuracy, f1_macro.
        *method* and *network_id* are added by the caller.
    """
    try:
        labels = generate_community_labels(G, method=label_strategy)
    except Exception as exc:
        # label_propagation needs no optional dependency, so it is the fallback
        # when the requested strategy is unavailable (python-louvain absent) or
        # undefined on this graph. The first failure was previously discarded,
        # which made a silently-substituted labelling indistinguishable from the
        # requested one.
        logger.warning(
            "Label strategy %r failed (%s); falling back to label_propagation.",
            label_strategy, exc,
        )
        try:
            labels = generate_community_labels(G, method='label_propagation')
        except Exception as e:
            warnings.warn(
                f"NC stratified: label generation failed for both {label_strategy!r} "
                f"({exc}) and the label_propagation fallback ({e}); skipping."
            )
            return []

    X = embeddings
    y = np.array([labels[n] for n in node_list])
    unique_labels, label_counts = np.unique(y, return_counts=True)

    if label_counts.min() < 2 or len(unique_labels) < 2:
        return []

    try:
        X_train, X_test, y_train, y_test, _, test_nodes = train_test_split(
            X, y, list(node_list), test_size=test_size,
            random_state=random_state, stratify=y,
        )
    except Exception as e:
        warnings.warn(f"NC stratified: train_test_split failed: {e}")
        return []

    # Normalise: fit on train only
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    try:
        clf.fit(X_train_s, y_train)
    except Exception as e:
        warnings.warn(f"NC stratified: classifier fit failed: {e}")
        return []

    y_pred = clf.predict(X_test_s)

    # Degree bins — percentile thresholds on test-node degrees
    node_degrees = dict(G.degree())
    test_degrees = np.array([node_degrees.get(n, 0) for n in test_nodes])
    deg_thresholds = np.percentile(test_degrees, np.linspace(0, 100, n_degree_bins + 1)[1:])
    deg_thresholds[-1] += 1  # make the last bin right-inclusive
    test_deg_bins = [
        f"Q{min(int(np.digitize(d, deg_thresholds)) + 1, n_degree_bins)}"
        for d in test_degrees
    ]

    # Distance bins — min distance from top-sqrt(N) hub nodes
    hub_count = max(1, int(np.sqrt(G.number_of_nodes())))
    hubs = sorted(node_degrees, key=lambda v: node_degrees[v], reverse=True)[:hub_count]

    node_min_dist: Dict[int, int] = {}
    for hub in hubs:
        try:
            dists = nx.single_source_shortest_path_length(G, hub, cutoff=dist_max_bin)
        except nx.NetworkXException as exc:
            # A hub that is not in G (only reachable if G was mutated between the
            # degree scan and here) contributes no distances. Other hubs still do,
            # so this is a skip, not a failure.
            logger.debug("No distances from hub %r: %s", hub, exc)
            dists = {}
        for v, d in dists.items():
            if v not in node_min_dist or d < node_min_dist[v]:
                node_min_dist[v] = d

    test_dist_bins = [
        f"{dist_max_bin}+" if min(node_min_dist.get(n, dist_max_bin + 1), dist_max_bin) >= dist_max_bin
        else str(min(node_min_dist.get(n, dist_max_bin + 1), dist_max_bin))
        for n in test_nodes
    ]

    rows = []
    bin_labels_deg = [f"Q{i+1}" for i in range(n_degree_bins)]
    bin_labels_dist = [
        str(d) if d < dist_max_bin else f"{dist_max_bin}+"
        for d in range(1, dist_max_bin + 1)
    ]

    for bin_type, bin_labels, node_bins in [
        ('degree',   bin_labels_deg,  test_deg_bins),
        ('distance', bin_labels_dist, test_dist_bins),
    ]:
        for bl in bin_labels:
            idxs = [i for i, b in enumerate(node_bins) if b == bl]
            if len(idxs) == 0:
                continue
            yt = y_test[idxs]
            yp = y_pred[idxs]
            n_cls = len(np.unique(yt))
            rows.append({
                'bin_type': bin_type,
                'bin_label': bl,
                'bin_n_nodes': len(idxs),
                'accuracy': float(accuracy_score(yt, yp)),
                'f1_macro': float(f1_score(yt, yp, average='macro', zero_division=0))
                            if n_cls >= 2 else float(np.nan),
                'label_strategy': label_strategy,
            })

    return rows


_CLASSIFICATION_METRIC_KEYS = [
    'f1_macro', 'f1_micro', 'f1_weighted',
    'accuracy',
    'precision_macro', 'precision_weighted',
    'recall_macro', 'recall_weighted',
    'n_classes', 'n_train', 'n_test',
    'f1_macro_cv_mean', 'f1_macro_cv_std',
]


def flatten_classification_results(
    results: Dict[str, Dict[str, float]],
    network_id: str,
    method: str,
) -> List[Dict]:
    """
    Flatten results from evaluate_all_label_strategies into one dict per strategy.

    Returns one row per label strategy with columns: network_id, method,
    label_strategy, f1_macro, accuracy, precision_macro, recall_macro,
    n_classes, n_train, n_test. Failed strategies produce NaN values so the
    schema is always consistent.
    """
    rows = []
    for strategy, metrics in results.items():
        row = {
            'network_id': network_id,
            'method': method,
            'label_strategy': strategy,
        }
        if 'error' in metrics:
            row.update({k: np.nan for k in _CLASSIFICATION_METRIC_KEYS})
            row['error'] = metrics['error']
        else:
            row.update({k: metrics.get(k, np.nan) for k in _CLASSIFICATION_METRIC_KEYS})
        rows.append(row)
    return rows


def summarize_classification_results(
    results: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """
    Summarize classification results across all label strategies.
    
    Args:
        results: Dictionary of results from evaluate_all_label_strategies
    
    Returns:
        Dictionary of summary statistics
    """
    # Extract F1 scores
    f1_scores = []
    accuracy_scores = []
    
    for strategy, metrics in results.items():
        if 'error' not in metrics and 'f1_macro' in metrics:
            f1_scores.append(metrics['f1_macro'])
            accuracy_scores.append(metrics['accuracy'])
    
    if not f1_scores:
        return {
            'mean_f1_macro': 0.0,
            'std_f1_macro': 0.0,
            'mean_accuracy': 0.0,
            'std_accuracy': 0.0,
            'n_successful_strategies': 0
        }
    
    return {
        'mean_f1_macro': np.mean(f1_scores),
        'std_f1_macro': np.std(f1_scores),
        'max_f1_macro': np.max(f1_scores),
        'min_f1_macro': np.min(f1_scores),
        'mean_accuracy': np.mean(accuracy_scores),
        'std_accuracy': np.std(accuracy_scores),
        'max_accuracy': np.max(accuracy_scores),
        'min_accuracy': np.min(accuracy_scores),
        'n_successful_strategies': len(f1_scores)
    }


