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

"""Graph complexity evaluation for QBioCode.

This module is the graph-network analogue of :mod:`qbiocode.evaluation.dataset_evaluation`.
Where ``dataset_evaluation.evaluate(df, y, file)`` summarizes a tabular (samples x
features) dataset, ``graph_evaluation.evaluate_graph(G, name)`` summarizes a
:class:`networkx.Graph` with spectral, topological, and structural complexity
metrics and returns a one-row transposed :class:`pandas.DataFrame`.

The metric implementations below are ported from the QuVINE complexity modules
(``graph.py`` + ``graph_enhanced.py``) and are self-contained here so that
QBioCode owns the graph-complexity math directly (no ``quvine.complexity``
dependency). The heavy embedding machinery lives separately under
``qbiocode.apps.quvine``.

Optional dependencies:
- ``ripser``            -> persistent Betti / persistence-entropy metrics
- ``python-louvain`` (``community``) -> modularity / community metrics
Both are guarded; missing them degrades gracefully to defaults.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from scipy import linalg
from scipy.stats import entropy
from scipy.sparse.linalg import ArpackError, eigsh, eigs, expm_multiply


# =============================================================================
# Ported from quvine/complexity/graph.py
# =============================================================================


logger = logging.getLogger(__name__)

def compute_laplacian_spectrum(G: nx.Graph, normalized: bool = True) -> np.ndarray:
    """
    Compute the eigenvalues of the graph Laplacian.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    normalized : bool, default=True
        If True, use normalized Laplacian; otherwise use unnormalized

    Returns
    -------
    eigenvalues : np.ndarray
        Sorted eigenvalues of the Laplacian (ascending order)
    """
    if G.number_of_nodes() == 0:
        return np.array([])

    if normalized:
        L = nx.normalized_laplacian_matrix(G).toarray()
    else:
        L = nx.laplacian_matrix(G).toarray()

    eigenvalues = linalg.eigvalsh(L)
    return np.sort(eigenvalues)


def compute_spectral_gap(G: nx.Graph, normalized: bool = True) -> float:
    """
    Compute the spectral gap (difference between first and second eigenvalues).

    The spectral gap is related to graph connectivity and mixing time.
    Larger gaps indicate better connectivity and faster mixing.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    normalized : bool, default=True
        If True, use normalized Laplacian

    Returns
    -------
    float
        Spectral gap (lambda_2 - lambda_1)
    """
    eigenvalues = compute_laplacian_spectrum(G, normalized=normalized)

    if len(eigenvalues) < 2:
        return 0.0

    # For Laplacian, smallest eigenvalue is ~0
    return float(eigenvalues[1] - eigenvalues[0])


def fiedler_eigenvalue_sparse(
    G: nx.Graph, normalized: bool = False
) -> Tuple[float, np.ndarray]:
    """
    Compute Fiedler eigenvalue and eigenvector using sparse matrix methods.

    This is more efficient for large graphs than computing the full spectrum.
    The Fiedler eigenvalue is the second smallest eigenvalue of the Laplacian,
    and its eigenvector (Fiedler vector) is useful for graph partitioning.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    normalized : bool, default=False
        If True, use normalized Laplacian; otherwise use unnormalized

    Returns
    -------
    lambda2 : float
        Fiedler eigenvalue (second smallest eigenvalue)
    fiedler_vec : np.ndarray
        Fiedler eigenvector
    """
    if G.number_of_nodes() < 2:
        return 0.0, np.array([])

    if normalized:
        L = nx.normalized_laplacian_matrix(G)
    else:
        L = nx.laplacian_matrix(G)

    try:
        # Compute 2 smallest eigenvalues
        eigenvalues, eigenvectors = eigsh(L, k=2, which='SM')

        # Sort them
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        lambda2 = float(eigenvalues[1])
        fiedler_vec = eigenvectors[:, 1]

        return lambda2, fiedler_vec
    except (ArpackError, np.linalg.LinAlgError, ValueError) as exc:
        # ARPACK needs k < n and does not always converge on a near-singular or
        # disconnected Laplacian. The dense path computes the same quantity
        # exactly, so this is an alternative route rather than a degraded result.
        logger.debug("eigsh(k=2) failed (%s); falling back to the dense spectrum", exc)
        eigenvalues = compute_laplacian_spectrum(G, normalized=normalized)
        if len(eigenvalues) < 2:
            return 0.0, np.array([])
        return float(eigenvalues[1]), np.array([])


def compute_algebraic_connectivity(G: nx.Graph) -> float:
    """
    Compute algebraic connectivity (Fiedler value).

    This is the second smallest eigenvalue of the unnormalized Laplacian matrix.
    Higher values indicate better connectivity and robustness to node removal.

    Parameters
    ----------
    G : nx.Graph
        Input graph

    Returns
    -------
    float
        Algebraic connectivity (lambda_2)
    """
    if not nx.is_connected(G):
        return 0.0

    lambda2, _ = fiedler_eigenvalue_sparse(G, normalized=False)
    return lambda2


def compute_spectral_entropy(G: nx.Graph, normalized: bool = True) -> float:
    """
    Compute spectral entropy based on Laplacian eigenvalues.

    Spectral entropy measures the complexity/randomness of the graph structure
    by treating the normalized positive eigenvalues as a probability distribution.
    Higher entropy indicates more complex or random structure.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    normalized : bool, default=True
        If True, use normalized Laplacian

    Returns
    -------
    float
        Spectral entropy H = -sum(p_i * log(p_i)) where p_i = lambda_i / sum(lambda)
    """
    eigenvalues = compute_laplacian_spectrum(G, normalized=normalized)

    if len(eigenvalues) == 0:
        return 0.0

    # Remove near-zero eigenvalues (trivial zero mode of Laplacian)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) == 0:
        return 0.0

    # Normalize to create probability distribution
    probs = eigenvalues / eigenvalues.sum()

    return float(entropy(probs))


def compute_von_neumann_entropy(G: nx.Graph) -> float:
    """
    Compute von Neumann entropy of the graph.

    Implements the Passerini-Severini (2008) definition: the graph is
    associated with a density matrix rho = L / Tr(L), where L is the
    combinatorial (unnormalized) Laplacian and Tr(L) = sum of (weighted)
    degrees. The von Neumann entropy is then:

        S = -Tr(rho log2 rho) = -sum_i (lambda_i / Tr(L)) * log2(lambda_i / Tr(L))

    where the sum is over non-zero eigenvalues of L.

    Parameters
    ----------
    G : nx.Graph
        Input graph

    Returns
    -------
    float
        Von Neumann entropy S in bits (log base 2)
    """
    if G.number_of_nodes() == 0:
        return 0.0

    # Use unnormalized Laplacian; Tr(L) = sum of weighted degrees
    L = nx.laplacian_matrix(G).toarray()
    trace_L = float(np.trace(L))

    if trace_L == 0:
        return 0.0

    eigenvalues = np.sort(linalg.eigvalsh(L))

    # Normalize eigenvalues to form density matrix spectrum: rho_i = lambda_i / Tr(L)
    rho_eigs = eigenvalues / trace_L

    # Remove near-zero entries (zero eigenvalue of Laplacian gives 0 * log(0) = 0)
    rho_eigs = rho_eigs[rho_eigs > 1e-12]

    if len(rho_eigs) == 0:
        return 0.0

    # Von Neumann entropy: -sum(rho_i * log2(rho_i))
    vn_entropy = -np.sum(rho_eigs * np.log2(rho_eigs))

    return float(vn_entropy)


def compute_estrada_index(G: nx.Graph) -> float:
    """
    Compute the Laplacian Estrada index.

    The Laplacian Estrada Index (LEE) is defined as:

        LEE = sum_i exp(lambda_i)

    where lambda_i are the eigenvalues of the unnormalized Laplacian.
    It is related to the number of closed walks in the graph and captures
    the overall "folding" or connectivity complexity.

    Note: For large dense graphs the exponentials can be very large. This
    implementation uses log-space accumulation when any eigenvalue exceeds
    500 to avoid float64 overflow.

    Parameters
    ----------
    G : nx.Graph
        Input graph

    Returns
    -------
    float
        Laplacian Estrada index LEE = sum exp(lambda_i)
    """
    eigenvalues = compute_laplacian_spectrum(G, normalized=False)

    if len(eigenvalues) == 0:
        return 0.0

    # Guard against float64 overflow (exp overflows above ~709)
    if eigenvalues.max() > 500:
        # Use log-sum-exp: log(LEE) = max + log(sum(exp(x - max)))
        max_val = eigenvalues.max()
        log_estrada = max_val + np.log(np.sum(np.exp(eigenvalues - max_val)))
        return float(np.exp(log_estrada))

    return float(np.sum(np.exp(eigenvalues)))


def compute_quantum_complexity(G: nx.Graph) -> float:
    """
    Compute quantum complexity metric inspired by QBioCode.

    This combines spectral properties to measure how "quantum" or complex
    the graph structure is. Higher values indicate more complex structures
    that may benefit from quantum walks.

    The metric is a weighted combination (weights: 0.3, 0.3, 0.4) of:
    - Spectral gap ratio (gap / spectral radius)
    - Spectral participation ratio (fraction of active modes)
    - Normalised von Neumann entropy

    Parameters
    ----------
    G : nx.Graph
        Input graph

    Returns
    -------
    float
        Quantum complexity score in [0, 1]
    """
    if G.number_of_nodes() == 0:
        return 0.0

    eigenvalues = compute_laplacian_spectrum(G, normalized=True)

    if len(eigenvalues) < 2:
        return 0.0

    # Compute various spectral measures
    spectral_gap = eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0
    spectral_radius = eigenvalues[-1]

    # Effective dimension (spectral participation ratio)
    eigenvalues_pos = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues_pos) > 0:
        participation_ratio = (eigenvalues_pos.sum() ** 2) / (eigenvalues_pos ** 2).sum()
    else:
        participation_ratio = 1.0

    # Von Neumann entropy
    vn_entropy = compute_von_neumann_entropy(G)

    # Combine metrics (normalized)
    n = G.number_of_nodes()
    complexity = (
        0.3 * (spectral_gap / spectral_radius if spectral_radius > 0 else 0) +
        0.3 * (participation_ratio / n) +
        0.4 * (vn_entropy / np.log2(n) if n > 1 else 0)
    )

    return float(complexity)


def compute_spectral_concentration(G: nx.Graph, normalized: bool = True) -> float:
    """
    Compute spectral concentration from the Laplacian eigenvalue distribution.

    Measures how concentrated the spectral energy is among the eigenvalues:

        SC = sum(lambda_i^4) / (sum(lambda_i^2))^2

    This is analogous to an inverse participation ratio applied to the
    eigenvalue spectrum (not eigenvectors). Values near 1/k (where k is the
    number of non-zero eigenvalues) indicate uniform spectral spread; values
    near 1 indicate extreme spectral concentration in a few modes.

    Note: this metric operates on eigenvalues and measures the shape of the
    spectrum. For eigenvector-based localization, see
    compute_inverse_participation_ratio().

    Parameters
    ----------
    G : nx.Graph
        Input graph
    normalized : bool, default=True
        If True, use normalized Laplacian eigenvalues

    Returns
    -------
    float
        Spectral concentration in [1/k, 1] where k = number of non-zero eigenvalues
    """
    if G.number_of_nodes() == 0:
        return 0.0

    eigenvalues = compute_laplacian_spectrum(G, normalized=normalized)

    # Remove near-zero eigenvalues
    eigenvalues_pos = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues_pos) == 0:
        return 0.0

    # SC = sum(lambda^4) / (sum(lambda^2))^2
    sum_lambda_squared = np.sum(eigenvalues_pos ** 2)
    sum_lambda_fourth = np.sum(eigenvalues_pos ** 4)

    if sum_lambda_squared == 0:
        return 0.0

    return float(sum_lambda_fourth / (sum_lambda_squared ** 2))


def compute_inverse_participation_ratio(G: nx.Graph, normalized: bool = True) -> float:
    """
    Compute the mean Inverse Participation Ratio (IPR) over all Laplacian eigenmodes.

    For each normalised eigenvector v of the Laplacian, the IPR is defined as:

        IPR(v) = sum_j v_j^4

    Because the eigenvectors are L2-normalised (sum v_j^2 = 1), IPR(v) lies in
    [1/n, 1].  A value of 1/n corresponds to a perfectly delocalised mode
    (uniform over all n nodes), while IPR = 1 means the mode is entirely
    concentrated on a single node (Anderson localisation limit).

    This function returns the mean IPR averaged over all n eigenmodes.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    normalized : bool, default=True
        If True, use the normalised Laplacian; otherwise use the combinatorial
        (unnormalised) Laplacian

    Returns
    -------
    float
        Mean IPR in [1/n, 1]
    """
    if G.number_of_nodes() == 0:
        return 0.0

    # Compute Laplacian matrix
    L = (
        nx.normalized_laplacian_matrix(G).toarray()
        if normalized
        else nx.laplacian_matrix(G).toarray()
    )

    # Eigenvectors as columns of V; eigh guarantees real, orthonormal columns
    _, V = np.linalg.eigh(L)

    # IPR per mode: sum over nodes of (v_j)^4
    ipr_per_mode = np.sum(V ** 4, axis=0)  # shape: (n_nodes,)

    return float(np.mean(ipr_per_mode))


def compute_participation_ratio(G: nx.Graph, normalized: bool = True) -> float:
    """
    Compute the mean Participation Ratio (PR) over all Laplacian eigenmodes.

    The Participation Ratio is the inverse of the IPR for each eigenmode:

        PR(v) = 1 / IPR(v) = 1 / sum_j v_j^4

    PR(v) estimates the effective number of nodes over which eigenmode v is
    spread. It ranges from 1 (fully localised on one node) to n (perfectly
    delocalised across all nodes). The mean over all modes is returned.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    normalized : bool, default=True
        If True, use the normalised Laplacian; otherwise use the combinatorial
        Laplacian

    Returns
    -------
    float
        Mean participation ratio in [1, n]
    """
    if G.number_of_nodes() == 0:
        return 0.0

    L = (
        nx.normalized_laplacian_matrix(G).toarray()
        if normalized
        else nx.laplacian_matrix(G).toarray()
    )

    _, V = np.linalg.eigh(L)

    # PR per mode: 1 / sum(v_j^4); guard against exact zeros (shouldn't occur)
    ipr_per_mode = np.sum(V ** 4, axis=0)
    pr_per_mode = np.where(ipr_per_mode > 0, 1.0 / ipr_per_mode, 0.0)

    return float(np.mean(pr_per_mode))


def compute_effective_resistance(G: nx.Graph, source: int, target: int) -> float:
    """
    Compute effective resistance between two nodes.

    Effective resistance is related to random walk commute time and
    provides a distance metric on the graph.

        R(i, j) = L^+_ii + L^+_jj - 2 L^+_ij

    where L^+ is the Moore-Penrose pseudoinverse of the Laplacian.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    source : int
        Source node
    target : int
        Target node

    Returns
    -------
    float
        Effective resistance (non-negative)
    """
    if source not in G.nodes() or target not in G.nodes():
        return float('inf')

    if source == target:
        return 0.0

    # Compute pseudoinverse of Laplacian
    L = nx.laplacian_matrix(G).toarray()

    try:
        L_pinv = linalg.pinv(L)
    except (np.linalg.LinAlgError, ValueError) as exc:
        # Infinite resistance is the correct reading of "no finite pseudoinverse
        # of this Laplacian", i.e. no conducting path between the two nodes.
        logger.debug("Laplacian pseudoinverse failed (%s); resistance is infinite", exc)
        return float('inf')

    # Get node indices
    nodes = list(G.nodes())
    i = nodes.index(source)
    j = nodes.index(target)

    # Effective resistance: R(i,j) = L+_ii + L+_jj - 2 L+_ij
    resistance = float(L_pinv[i, i] + L_pinv[j, j] - 2 * L_pinv[i, j])

    return float(max(0.0, resistance))


def compute_laplacian_centrality_complexity(
    G: nx.Graph, normalized: bool = True
) -> Dict[str, float]:
    """
    Compute centrality-based complexity metrics from the Laplacian.

    Uses the Fiedler vector (eigenvector of the second-smallest eigenvalue)
    as a node-centrality proxy and characterises its distribution via entropy,
    variance, Gini coefficient, and range.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    normalized : bool, default=True
        If True, use normalized Laplacian

    Returns
    -------
    dict
        Dictionary of centrality complexity metrics including:
        - centrality_entropy: Shannon entropy of the Fiedler-vector distribution
        - centrality_variance: Variance of absolute Fiedler-vector entries
        - centrality_gini: Gini coefficient of absolute Fiedler-vector entries
        - centrality_range: Range (max - min) of absolute entries
        - dominant_eigenvector_centrality: Max entry of the largest-eigenvalue eigenvector
    """
    if G.number_of_nodes() == 0:
        return {
            'centrality_entropy': 0.0,
            'centrality_variance': 0.0,
            'centrality_gini': 0.0,
            'centrality_range': 0.0,
            'dominant_eigenvector_centrality': 0.0,
        }

    if normalized:
        L = nx.normalized_laplacian_matrix(G).toarray()
    else:
        L = nx.laplacian_matrix(G).toarray()

    eigenvalues, eigenvectors = linalg.eigh(L)

    # Sort by eigenvalue (ascending)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Use the Fiedler vector (second smallest eigenvalue's eigenvector)
    if eigenvectors.shape[1] > 1:
        fiedler_vector = np.abs(eigenvectors[:, 1])
    else:
        fiedler_vector = np.abs(eigenvectors[:, 0])

    # Normalize to probability distribution for entropy
    if fiedler_vector.sum() > 0:
        centrality_dist = fiedler_vector / fiedler_vector.sum()
    else:
        centrality_dist = np.ones(len(fiedler_vector)) / len(fiedler_vector)

    centrality_entropy = float(entropy(centrality_dist))
    centrality_variance = float(np.var(fiedler_vector))

    # Gini coefficient
    sorted_centrality = np.sort(fiedler_vector)
    n = len(sorted_centrality)
    index = np.arange(1, n + 1)
    gini = float(
        (2 * np.sum(index * sorted_centrality)) / (n * np.sum(sorted_centrality)) - (n + 1) / n
    )

    centrality_range = float(np.max(fiedler_vector) - np.min(fiedler_vector))

    if eigenvectors.shape[1] > 0:
        dominant_vector = np.abs(eigenvectors[:, -1])
        dominant_centrality = float(np.max(dominant_vector))
    else:
        dominant_centrality = 0.0

    return {
        'centrality_entropy': centrality_entropy,
        'centrality_variance': centrality_variance,
        'centrality_gini': gini,
        'centrality_range': centrality_range,
        'dominant_eigenvector_centrality': dominant_centrality,
    }


def compute_graph_complexity_metrics(G: nx.Graph) -> Dict[str, float]:
    """
    Compute comprehensive complexity metrics for a graph.

    Parameters
    ----------
    G : nx.Graph
        Input graph

    Returns
    -------
    dict
        Dictionary of complexity metrics
    """
    if G.number_of_nodes() == 0:
        return {
            'spectral_gap': 0.0,
            'algebraic_connectivity': 0.0,
            'spectral_entropy': 0.0,
            'von_neumann_entropy': 0.0,
            'estrada_index': 0.0,
            'quantum_complexity': 0.0,
            'centrality_entropy': 0.0,
            'centrality_variance': 0.0,
            'centrality_gini': 0.0,
            'centrality_range': 0.0,
            'num_nodes': 0,
            'num_edges': 0,
            'density': 0.0,
            # Topological metrics
            'orc_gJC_mean': 0.0,
            'orc_kLB_mean': 0.0,
            'orc_negative_fraction': 0.0,
            'cyclomatic_number': 0,
            'kirchhoff_index': 0.0,
            'betti_0': 0,
            'betti_1': 0,
            'betti_2': 0,
        }

    metrics = {
        # Basic properties
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),

        # Spectral properties
        'spectral_gap': compute_spectral_gap(G, normalized=True),
        'algebraic_connectivity': compute_algebraic_connectivity(G),
        'spectral_entropy': compute_spectral_entropy(G, normalized=True),

        # Quantum-inspired metrics
        'von_neumann_entropy': compute_von_neumann_entropy(G),
        'estrada_index': compute_estrada_index(G),
        'quantum_complexity': compute_quantum_complexity(G),

        # Participation metrics
        'inverse_participation_ratio': compute_inverse_participation_ratio(G, normalized=True),
        'participation_ratio': compute_participation_ratio(G, normalized=True),
        'spectral_concentration': compute_spectral_concentration(G, normalized=True),
    }

    # Add centrality complexity metrics
    centrality_metrics = compute_laplacian_centrality_complexity(G, normalized=True)
    metrics.update(centrality_metrics)

    # Add eigenvalue statistics
    eigenvalues = compute_laplacian_spectrum(G, normalized=True)
    if len(eigenvalues) > 0:
        metrics['eigenvalue_mean'] = float(np.mean(eigenvalues))
        metrics['eigenvalue_std'] = float(np.std(eigenvalues))
        metrics['eigenvalue_max'] = float(np.max(eigenvalues))
        metrics['eigenvalue_min'] = float(np.min(eigenvalues))

    # Add quantum advantage metrics
    qa_metrics = compute_quantum_advantage_metrics(G)
    metrics.update(qa_metrics)
    
    # Add topological/geometric complexity metrics
    # Betti computation is thresholded by graph size to control runtime/memory
    try:
        maxdim = 1 if G.number_of_nodes() >= 500 else 2
        include_betti = G.number_of_nodes() <= 5000
        topo_metrics = compute_topological_metrics(
            G,
            include_betti=include_betti,
            include_persistence_entropy=include_betti,
            maxdim=maxdim,
            filtration_scale=1.0
        )
        metrics.update(topo_metrics)
    except Exception as e:
        # If topological metrics fail (e.g., ripser not installed), continue
        import warnings
        warnings.warn(f"Topological metrics computation failed: {e}")
        # Add placeholder values
        metrics.update({
            'orc_gJC_mean': 0.0, 'orc_kLB_mean': 0.0,
            'cyclomatic_number': 0,
            'kirchhoff_index': 0.0,
            'betti_0': 0, 'betti_1': 0, 'betti_2': 0,
        })

    return metrics


def compare_graph_complexities(
    graphs: Dict[str, nx.Graph]
) -> Dict[str, Dict[str, float]]:
    """
    Compare complexity metrics across multiple graphs.

    Parameters
    ----------
    graphs : dict
        Dictionary mapping graph names to NetworkX graphs

    Returns
    -------
    dict
        Dictionary mapping graph names to their complexity metrics
    """
    return {name: compute_graph_complexity_metrics(G) for name, G in graphs.items()}


def compute_quantum_advantage_metrics(G: nx.Graph) -> Dict[str, float]:
    """
    Compute metrics that predict quantum advantage in graph algorithms.

    These metrics help identify when quantum walks are likely to outperform
    classical random walks based on graph structure.

    Parameters
    ----------
    G : nx.Graph
        Input graph

    Returns
    -------
    dict
        Dictionary including:
        - spectral_dimension: Effective number of active eigenvalues
        - modularity: Community structure strength (Louvain greedy)
        - path_length_ratio: avg_path_length / diameter
        - clustering_mean/std: Local clustering statistics
        - degree_heterogeneity: Coefficient of variation of degree sequence
        - quantum_advantage_score: Weighted composite prediction score
    """
    if G.number_of_nodes() == 0:
        # nan, not 0.0. Every value here is a measurement of structure, and a
        # graph with no nodes has no structure to measure -- which is a different
        # statement from "measured, and found to be zero". A 0.0
        # quantum_advantage_score in particular is a rankable value, so an empty
        # graph used to sort alongside genuinely unpromising ones. Aggregate these
        # with np.nanmean; compute_degree_metrics already reports nan here.
        return {
            'spectral_dimension': float('nan'),
            'modularity': float('nan'),
            'path_length_ratio': float('nan'),
            'clustering_mean': float('nan'),
            'clustering_std': float('nan'),
            'degree_heterogeneity': float('nan'),
            'quantum_advantage_score': float('nan'),
            'quantum_advantage_arithmetic': float('nan'),
            'quantum_advantage_geometric': float('nan'),
            'quantum_advantage_harmonic': float('nan'),
        }

    metrics = {}

    # 1. Spectral dimension (effective number of active eigenvalues)
    #    PR_spectral = (sum lambda_i)^2 / sum(lambda_i^2)
    eigenvalues = compute_laplacian_spectrum(G, normalized=True)
    eigenvalues_pos = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues_pos) > 0:
        metrics['spectral_dimension'] = float(
            (eigenvalues_pos.sum() ** 2) / (eigenvalues_pos ** 2).sum()
        )
    else:
        metrics['spectral_dimension'] = 1.0

    # 2. Modularity (community structure)
    try:
        communities = nx.community.greedy_modularity_communities(G)
        metrics['modularity'] = float(nx.community.modularity(G, communities))
    except (nx.NetworkXException, ValueError, ZeroDivisionError) as exc:
        # nan, not 0.0. Modularity 0.0 is a real measurement -- "no community
        # structure beyond chance" -- so reporting it for an undetectable
        # partition fabricates a finding. compute_community_metrics below already
        # uses nan for exactly this case; these two now agree.
        logger.warning("Modularity could not be computed (%s); reporting nan.", exc)
        metrics['modularity'] = float('nan')

    # 3. Path length ratio (compactness)
    if nx.is_connected(G):
        try:
            avg_path = nx.average_shortest_path_length(G)
            diameter = nx.diameter(G)
            metrics['path_length_ratio'] = float(avg_path / diameter if diameter > 0 else 0.0)
        except (nx.NetworkXException, ZeroDivisionError) as exc:
            # The 0.0 in the `else` branch below is deliberate: a disconnected
            # graph has no finite ratio by definition. This branch is different --
            # the graph *is* connected and the computation failed -- so it reports
            # nan rather than borrowing the disconnected graph's answer.
            logger.warning("Path length ratio could not be computed (%s); reporting nan.", exc)
            metrics['path_length_ratio'] = float('nan')
    else:
        metrics['path_length_ratio'] = 0.0

    # 4. Clustering coefficient distribution
    clustering_values = list(nx.clustering(G).values())
    if clustering_values:
        metrics['clustering_mean'] = float(np.mean(clustering_values))
        metrics['clustering_std'] = float(np.std(clustering_values))
    else:
        metrics['clustering_mean'] = 0.0
        metrics['clustering_std'] = 0.0

    # 5. Degree heterogeneity (coefficient of variation)
    degrees = [d for _, d in G.degree()]
    mean_deg = float(np.mean(degrees)) if degrees else 0.0
    metrics['degree_heterogeneity'] = float(
        np.std(degrees) / mean_deg if mean_deg > 0 else 0.0
    )

    # 6. Quantum advantage scores (multiple formulations)
    qc = compute_quantum_complexity(G)
    sg = compute_spectral_gap(G, normalized=True)
    ipr = compute_inverse_participation_ratio(G, normalized=True)

    # Normalized components (all in [0, 1])
    modularity_norm = metrics['modularity']                    # in [0, 1]
    spectral_gap_norm = 1.0 - min(sg, 1.0)                    # low gap → high advantage
    ipr_norm = min(ipr, 1.0)                                   # more localised → more advantage
    clustering_norm = metrics['clustering_mean']               # in [0, 1]
    
    # Add small epsilon to avoid log(0) and division by zero
    eps = 1e-10
    components = np.array([
        modularity_norm + eps,
        spectral_gap_norm + eps,
        ipr_norm + eps,
        clustering_norm + eps
    ])
    weights = np.array([0.30, 0.25, 0.25, 0.20])
    
    # Arithmetic mean (current default - additive contributions)
    qa_arithmetic = float(np.sum(weights * components))
    
    # Geometric mean (synergistic interactions - all features must be present)
    qa_geometric = float(np.prod(components ** weights))
    
    # Harmonic mean (emphasizes minimum - bottleneck-sensitive)
    qa_harmonic = float(1.0 / np.sum(weights / components))
    
    metrics['quantum_advantage_score'] = qa_arithmetic  # Keep as default
    metrics['quantum_advantage_arithmetic'] = qa_arithmetic
    metrics['quantum_advantage_geometric'] = qa_geometric
    metrics['quantum_advantage_harmonic'] = qa_harmonic

    return metrics


def rank_graphs_by_complexity(
    graphs: Dict[str, nx.Graph],
    metric: str = 'quantum_complexity'
) -> list:
    """
    Rank graphs by a specific complexity metric.

    Parameters
    ----------
    graphs : dict
        Dictionary mapping graph names to NetworkX graphs
    metric : str, default='quantum_complexity'
        Metric to use for ranking

    Returns
    -------
    list
        List of (name, score) tuples sorted by complexity (descending)
    """
    complexities = compare_graph_complexities(graphs)
    rankings = [
        (name, metrics.get(metric, 0.0))
        for name, metrics in complexities.items()
    ]
    return sorted(rankings, key=lambda x: x[1], reverse=True)



# ═════════════════════════════════════════════════════════════════════════════
# Topological & Geometric Complexity Metrics
# ═════════════════════════════════════════════════════════════════════════════

def _hop_distance_matrix(G: nx.Graph) -> np.ndarray:
    """
    Compute all-pairs shortest-path distance matrix using hop counts (unweighted).
    
    Disconnected pairs receive distance = max_finite + 1.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
        
    Returns
    -------
    np.ndarray
        Distance matrix with shape (n, n)
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path as _scipy_shortest_path
    
    n = G.number_of_nodes()
    if n == 0:
        return np.zeros((0, 0))
    
    # Build unweighted adjacency matrix
    A_bool = (nx.to_numpy_array(G, weight=None) > 0).astype(np.float64)
    D = _scipy_shortest_path(csr_matrix(A_bool), directed=False)
    
    # Replace inf with sentinel value
    finite_vals = D[np.isfinite(D)]
    sentinel = (finite_vals.max() + 1.0) if len(finite_vals) > 0 else 1.0
    D[~np.isfinite(D)] = sentinel
    
    return D


def _laplacian_nonzero_eigenvalues(G: nx.Graph, tol: float = 1e-10) -> np.ndarray:
    """
    Compute sorted positive eigenvalues of the combinatorial Laplacian.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    tol : float
        Threshold for filtering near-zero eigenvalues
        
    Returns
    -------
    np.ndarray
        Sorted positive eigenvalues
    """
    from scipy import linalg
    
    L = nx.laplacian_matrix(G).toarray()
    eigs = np.sort(linalg.eigvalsh(L))
    return eigs[eigs > tol]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ollivier-Ricci Curvature (ORC)
# ─────────────────────────────────────────────────────────────────────────────

def compute_orc_per_edge(G: nx.Graph) -> Dict[Tuple, Dict[str, float]]:
    """
    Compute Ollivier-Ricci curvature approximations for every edge.
    
    Two approximations are computed:
    
    1. **Generalized Jaccard (gJC)**: Fast O(d) proxy
       gJC(u, v) = ``|N(u) ∩ N(v)|`` / ``|N(u) ∪ N(v)|``
       
    2. **Jost-Liu lower bound (κ_LB)**: Tighter spectral bound
       κ_LB(u, v) = Δ / max(dᵤ, d_v) + 1/dᵤ + 1/d_v − 1
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
        
    Returns
    -------
    dict
        Mapping (u, v) → {'gJC': float, 'kappa_LB': float, 'triangles': int}
    """
    result = {}
    
    for u, v in G.edges():
        du = G.degree(u)
        dv = G.degree(v)
        
        # Common neighbours (excluding u and v)
        Nu = set(G.neighbors(u))
        Nv = set(G.neighbors(v))
        common = (Nu & Nv) - {u, v}
        Delta = len(common)
        
        # Generalized Jaccard
        union_size = du + dv - Delta
        gJC = Delta / union_size if union_size > 0 else 0.0
        
        # Jost-Liu lower bound
        if du == 0 or dv == 0:
            kappa_LB = -1.0
        else:
            kappa_LB = Delta / max(du, dv) + 1.0 / du + 1.0 / dv - 1.0
        
        result[(u, v)] = {
            "gJC": float(gJC),
            "kappa_LB": float(kappa_LB),
            "triangles": Delta,
        }
    
    return result


def compute_orc_stats(G: nx.Graph) -> Dict[str, float]:
    """
    Aggregate ORC statistics over all edges.
    
    Returns mean, min, max, std for both gJC and κ_LB, plus the fraction
    of edges with negative κ_LB (bottleneck indicator).
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
        
    Returns
    -------
    dict
        ORC statistics with keys:
        - orc_gJC_mean, orc_gJC_min, orc_gJC_max, orc_gJC_std
        - orc_kLB_mean, orc_kLB_min, orc_kLB_max, orc_kLB_std
        - orc_negative_fraction (fraction of edges with κ_LB < 0)
        - orc_num_edges
    """
    if G.number_of_edges() == 0:
        return {
            "orc_gJC_mean": 0.0, "orc_gJC_min": 0.0, "orc_gJC_max": 0.0, "orc_gJC_std": 0.0,
            "orc_kLB_mean": 0.0, "orc_kLB_min": 0.0, "orc_kLB_max": 0.0, "orc_kLB_std": 0.0,
            "orc_negative_fraction": 0.0,
            "orc_num_edges": 0.0,
        }
    
    per_edge = compute_orc_per_edge(G)
    gJC_vals = np.array([v["gJC"] for v in per_edge.values()])
    kLB_vals = np.array([v["kappa_LB"] for v in per_edge.values()])
    
    return {
        "orc_gJC_mean": float(gJC_vals.mean()),
        "orc_gJC_min": float(gJC_vals.min()),
        "orc_gJC_max": float(gJC_vals.max()),
        "orc_gJC_std": float(gJC_vals.std()),
        "orc_kLB_mean": float(kLB_vals.mean()),
        "orc_kLB_min": float(kLB_vals.min()),
        "orc_kLB_max": float(kLB_vals.max()),
        "orc_kLB_std": float(kLB_vals.std()),
        "orc_negative_fraction": float(np.mean(kLB_vals < 0)),
        "orc_num_edges": float(G.number_of_edges()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cyclomatic Number (Circuit Rank)
# ─────────────────────────────────────────────────────────────────────────────

def compute_cyclomatic_number(G: nx.Graph) -> int:
    """
    Compute cyclomatic number (circuit rank / first Betti number of 1-skeleton).
    
    μ(G) = m − n + c
    
    where m = ``|E|``, n = ``|V|``, c = number of connected components.
    
    Interpretation:
    - μ = 0 iff G is a forest (no cycles)
    - μ counts minimum edges to remove to make G acyclic
    - Dimension of cycle space H₁(G; ℤ₂)
    - For quantum walks: counts interference-generating loops
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
        
    Returns
    -------
    int
        Non-negative cyclomatic number
    """
    m = G.number_of_edges()
    n = G.number_of_nodes()
    c = nx.number_connected_components(G)
    return max(0, m - n + c)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Kirchhoff Index (Total Effective Resistance)
# ─────────────────────────────────────────────────────────────────────────────

def compute_kirchhoff_index(G: nx.Graph, tol: float = 1e-10) -> float:
    """
    Compute Kirchhoff index (total effective resistance).
    
    R_K = n · Σᵢ 1/λᵢ
    
    summing over all positive eigenvalues λᵢ of the Laplacian.
    
    Interpretation:
    - For disconnected graphs: R_K = ∞
    - Classical random-walk mixing time ∝ R_K
    - Large R_K indicates bottlenecked topology → potential quantum speedup
    - Complete graph K_n: R_K = n−1
    - Path graph P_n: R_K = n(n²−1)/6
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    tol : float
        Eigenvalue threshold for filtering zero mode
        
    Returns
    -------
    float
        Kirchhoff index, or np.inf for disconnected graphs
    """
    if G.number_of_nodes() == 0:
        return 0.0
    
    if not nx.is_connected(G):
        return float("inf")
    
    n = G.number_of_nodes()
    eigs_pos = _laplacian_nonzero_eigenvalues(G, tol=tol)
    
    if len(eigs_pos) == 0:
        return float("inf")
    
    return float(n * np.sum(1.0 / eigs_pos))


def compute_kirchhoff_stats(G: nx.Graph, tol: float = 1e-10) -> Dict[str, float]:
    """
    Compute Kirchhoff index and normalized variants.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    tol : float
        Eigenvalue threshold
        
    Returns
    -------
    dict
        - kirchhoff_index: Raw R_K
        - kirchhoff_per_pair: R_K / C(n, 2) (mean effective resistance)
        - kirchhoff_normalised: R_K / R_K(P_n) (fraction of path-graph value)
    """
    n = G.number_of_nodes()
    Rk = compute_kirchhoff_index(G, tol=tol)
    
    num_pairs = n * (n - 1) / 2 if n > 1 else 1.0
    Rk_path = n * (n ** 2 - 1) / 6.0 if n > 1 else 1.0
    
    return {
        "kirchhoff_index": Rk,
        "kirchhoff_per_pair": Rk / num_pairs if np.isfinite(Rk) else float("inf"),
        "kirchhoff_normalised": Rk / Rk_path if np.isfinite(Rk) else float("inf"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Persistent Betti Numbers via Ripser
# ─────────────────────────────────────────────────────────────────────────────

def compute_betti_numbers(
    G: nx.Graph,
    maxdim: int = 2,
    filtration_scale: float = 1.0,
) -> Dict[str, object]:
    """
    Compute persistent Betti numbers β₀, β₁, β₂ using Ripser.
    
    Uses hop-count shortest-path distance matrix for Vietoris-Rips filtration.
    
    At filtration scale ε = 1 (one hop = one edge):
    - β₀ = number of connected components
    - β₁ = number of independent cycles in clique complex
    - β₂ = voids (unfilled tetrahedra) in clique complex
    
    Relationship to cyclomatic number μ:
    - μ counts all cycles in 1-skeleton (graph edges only)
    - β₁ counts cycles not filled by triangles
    - β₁ ≤ μ (triangles reduce cycle count)
    - β₁ = μ iff G is triangle-free
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    maxdim : int, default=2
        Maximum homological dimension
    filtration_scale : float, default=1.0
        ε at which Betti numbers are evaluated
        
    Returns
    -------
    dict
        - betti_0, betti_1, betti_2: Betti numbers at filtration_scale
        - persistence_diagrams: list of numpy arrays
        - betti_sum: β₀ + β₁ + β₂
        - euler_characteristic: β₀ − β₁ + β₂
    """
    try:
        from ripser import ripser as _ripser
    except ImportError:
        import warnings
        warnings.warn("ripser not installed. Install with: pip install ripser")
        return {
            "betti_0": 0, "betti_1": 0, "betti_2": 0,
            "persistence_diagrams": [],
            "betti_sum": 0, "euler_characteristic": 0,
        }
    
    n = G.number_of_nodes()
    if n == 0:
        return {
            "betti_0": 0, "betti_1": 0, "betti_2": 0,
            "persistence_diagrams": [],
            "betti_sum": 0, "euler_characteristic": 0,
        }
    
    if n == 1:
        return {
            "betti_0": 1, "betti_1": 0, "betti_2": 0,
            "persistence_diagrams": [np.array([[0.0, np.inf]])],
            "betti_sum": 1, "euler_characteristic": 1,
        }
    
    # Build hop-count distance matrix
    D = _hop_distance_matrix(G)
    
    # Run Ripser
    result = _ripser(D, maxdim=maxdim, distance_matrix=True)
    dgms = result["dgms"]
    
    # Count features alive at filtration_scale ε
    eps = filtration_scale
    betti = []
    for dim, dgm in enumerate(dgms):
        if len(dgm) == 0:
            betti.append(0)
            continue
        births = dgm[:, 0]
        deaths = dgm[:, 1]
        alive = int(np.sum((births <= eps) & (deaths > eps)))
        betti.append(alive)
    
    # Pad to 3 dimensions
    while len(betti) < 3:
        betti.append(0)
    
    b0, b1, b2 = betti[0], betti[1], betti[2]
    
    return {
        "betti_0": b0,
        "betti_1": b1,
        "betti_2": b2,
        "persistence_diagrams": dgms,
        "betti_sum": b0 + b1 + b2,
        "euler_characteristic": b0 - b1 + b2,
    }


def compute_persistence_entropy(
    G: nx.Graph,
    maxdim: int = 2,
    filtration_scale: float = 1.0,
) -> Dict[str, float]:
    """
    Compute persistence entropy for each homological dimension.
    
    For persistence diagram D_k = {(bᵢ, dᵢ)}, persistence entropy is:
    
    H_k = −Σᵢ (lᵢ / L) · log(lᵢ / L)
    
    where lᵢ = dᵢ − bᵢ is persistence lifetime and L = Σᵢ lᵢ.
    
    Measures complexity of multi-scale topological structure:
    - High H_k: many cycles with diverse lifetimes
    - Low H_k: one dominant topological feature
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    maxdim : int, default=2
        Maximum dimension
    filtration_scale : float
        Unused (entropy computed over all features)
        
    Returns
    -------
    dict
        persistence_entropy_H0, persistence_entropy_H1, persistence_entropy_H2
    """
    betti_result = compute_betti_numbers(G, maxdim=maxdim, filtration_scale=filtration_scale)
    dgms = betti_result["persistence_diagrams"]
    
    entropies = {}
    for dim in range(3):
        key = f"persistence_entropy_H{dim}"
        if dim >= len(dgms) or len(dgms[dim]) == 0:
            entropies[key] = 0.0
            continue
        
        dgm = dgms[dim]
        births = dgm[:, 0]
        deaths = dgm[:, 1].copy()
        
        # Replace inf with max-finite + 1
        finite_mask = np.isfinite(deaths)
        if finite_mask.any():
            max_finite = deaths[finite_mask].max()
        else:
            max_finite = 0.0
        deaths[~finite_mask] = max_finite + 1.0
        
        lifetimes = deaths - births
        lifetimes = lifetimes[lifetimes > 0]
        
        if len(lifetimes) == 0:
            entropies[key] = 0.0
            continue
        
        L = lifetimes.sum()
        probs = lifetimes / L
        entropies[key] = float(-np.sum(probs * np.log(probs + 1e-300)))
    
    return entropies


# ─────────────────────────────────────────────────────────────────────────────
# 5. Combined Topological Metrics Interface
# ─────────────────────────────────────────────────────────────────────────────

def compute_topological_metrics(
    G: nx.Graph,
    include_betti: bool = True,
    include_persistence_entropy: bool = True,
    maxdim: int = 2,
    filtration_scale: float = 1.0,
) -> Dict[str, object]:
    """
    Compute all topological/geometric complexity metrics.
    
    Metrics computed:
    - ORC (Ollivier-Ricci curvature): gJC and κ_LB approximations
    - Cyclomatic number: Circuit rank μ
    - Kirchhoff index: Total effective resistance R_K
    - Betti numbers: β₀, β₁, β₂ (if include_betti=True)
    - Persistence entropy: H₀, H₁, H₂ (if include_persistence_entropy=True)
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    include_betti : bool, default=True
        Compute Betti numbers (expensive for large graphs)
    include_persistence_entropy : bool, default=True
        Compute persistence entropy (requires include_betti=True)
    maxdim : int, default=2
        Maximum homological dimension
    filtration_scale : float, default=1.0
        ε at which Betti numbers are evaluated
        
    Returns
    -------
    dict
        All topological metrics
    """
    metrics = {}
    
    # Basic properties
    metrics["num_nodes"] = G.number_of_nodes()
    metrics["num_edges"] = G.number_of_edges()
    
    # ORC stats
    metrics.update(compute_orc_stats(G))
    
    # Cyclomatic number
    metrics["cyclomatic_number"] = compute_cyclomatic_number(G)
    
    # Kirchhoff index
    metrics.update(compute_kirchhoff_stats(G))
    
    # Betti numbers + persistence entropy
    if include_betti and G.number_of_nodes() > 0:
        betti = compute_betti_numbers(G, maxdim=maxdim, filtration_scale=filtration_scale)
        # Don't store raw diagrams in flat dict
        betti.pop("persistence_diagrams", None)
        metrics.update(betti)
        
        if include_persistence_entropy:
            metrics.update(
                compute_persistence_entropy(G, maxdim=maxdim, filtration_scale=filtration_scale)
            )
    
    return metrics


# =============================================================================
# Ported from quvine/complexity/graph_enhanced.py
# =============================================================================


# -----------------------------------------------------------------------------
# Candidate metric lists
# -----------------------------------------------------------------------------

CANDIDATE_27_METRICS: List[str] = [
    # Size / density controls
    "log_num_nodes",
    "log_num_edges",
    "density",
    "avg_degree",

    # Connectivity / mixing
    "normalized_spectral_gap",
    "approx_avg_path_length",
    "approx_conductance",

    # Degree / centrality concentration
    "degree_gini",
    "max_degree_fraction",
    "pagerank_gini",
    "betweenness_gini_approx",

    # Community / cyclic structure
    "modularity",
    "transitivity",
    "cycle_density",
    "nonbacktracking_spectral_radius",

    # Curvature / bottleneck geometry
    "orc_kLB_mean",
    "orc_negative_fraction",

    # Spectral richness / localization
    "laplacian_effective_rank_partial",
    "ipr_low_mean",
    "ipr_high_mean",
    "spectral_degeneracy_fraction",

    # Symmetry / core-periphery
    "wl_compression_ratio",
    "core_number_gini",

    # Task signal
    "label_homophily",
    "feature_dirichlet_energy",

    # Additional controls/structure
    "degree_assortativity",
    "largest_cc_fraction",
]

CANDIDATE_NEW_METRICS: List[str] = [
    # Theory-grade additions tied to QW vs classical advantage literature
    "bipartite_proximity",
    "log_odd_girth",
    "algebraic_connectivity_ratio",
    "spectral_entropy_partial",
    "heat_kernel_trace_t1",
    "heat_kernel_trace_t10",
    "adjacency_ipr_low_mean",
    "adjacency_ipr_high_mean",
    "closeness_gini_approx",
]

CANDIDATE_ALL_METRICS: List[str] = CANDIDATE_27_METRICS + CANDIDATE_NEW_METRICS


@dataclass
class ComplexityConfig:
    """Runtime and approximation settings for scalable metrics."""

    spectral_k: int = 64
    eig_tol: float = 1e-5
    path_num_sources: int = 64
    betweenness_k: int = 256
    wl_iterations: int = 3
    nonbacktracking_max_directed_edges: int = 1_000_000  # raised from 200_000
    random_state: int = 0
    use_largest_cc_for_path: bool = True
    pagerank_alpha: float = 0.85
    pagerank_max_iter: int = 200
    pagerank_tol: float = 1e-6

    # Heat kernel trace (stochastic Hutchinson estimator)
    heat_kernel_t_values: Tuple[float, ...] = (1.0, 10.0)
    heat_kernel_n_probes: int = 20

    # Odd girth (BFS-based, sampled sources)
    odd_girth_max_sources: int = 32
    odd_girth_min_cycle_break: int = 5  # break early if found cycle <= this


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def sanitize_graph(G: nx.Graph, make_undirected: bool = True, remove_selfloops: bool = True) -> nx.Graph:
    """
    Return a simple NetworkX graph suitable for undirected complexity metrics.
    """
    if make_undirected and G.is_directed():
        H = nx.Graph(G)
    else:
        H = nx.Graph(G) if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)) else G.copy()

    if remove_selfloops:
        H.remove_edges_from(nx.selfloop_edges(H))
    return H


def safe_float(x: Any, default: float = np.nan) -> float:
    """Coerce ``x`` to a finite float, returning ``default`` if it is not one.

    Deliberately broad on the coercion: this is called on values from many metric
    backends, and ``float()`` raises ``TypeError`` for a None or a sequence,
    ``ValueError`` for an unparsable string, and ``OverflowError`` for an
    out-of-range value. Not logged -- it runs once per metric value, and the
    caller decides what a missing metric means.
    """
    try:
        y = float(x)
        if math.isfinite(y):
            return y
        return default
    except (TypeError, ValueError, OverflowError):
        return default


def gini_coefficient(values: Iterable[float]) -> float:
    """Compute Gini coefficient for a nonnegative vector."""
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    x = np.maximum(x, 0.0)
    total = x.sum()
    if total <= 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * x)) / (n * total) - (n + 1.0) / n)


def get_nodelist(G: nx.Graph) -> List[Hashable]:
    return list(G.nodes())


def get_sparse_laplacian(G: nx.Graph, normalized: bool = True) -> Tuple[sp.csr_matrix, List[Hashable]]:
    """Sparse Laplacian with explicit nodelist for reproducibility."""
    nodelist = get_nodelist(G)
    if normalized:
        L = nx.normalized_laplacian_matrix(G, nodelist=nodelist).astype(float).tocsr()
    else:
        L = nx.laplacian_matrix(G, nodelist=nodelist).astype(float).tocsr()
    return L, nodelist


def get_sparse_adjacency(G: nx.Graph) -> Tuple[sp.csr_matrix, List[Hashable]]:
    """Sparse adjacency with explicit nodelist for reproducibility."""
    nodelist = get_nodelist(G)
    A = nx.adjacency_matrix(G, nodelist=nodelist).astype(float).tocsr()
    return A, nodelist


def safe_eigsh(
    L: sp.spmatrix,
    k: int,
    which: str,
    tol: float = 1e-5,
    return_eigenvectors: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Robust wrapper around scipy.sparse.linalg.eigsh."""
    n = L.shape[0]
    if n < 3:
        return np.array([]), None if return_eigenvectors else None
    k_eff = min(max(1, k), n - 2)
    # ARPACK seeds its Lanczos start vector randomly by default, which makes the
    # spectra — and especially eigenvector-derived metrics (IPR) on near-degenerate
    # eigenvalues — non-reproducible run to run. Pin a deterministic start vector so
    # complexity metrics are reproducible.
    v0 = np.random.default_rng(0).standard_normal(n)
    try:
        vals, vecs = eigsh(L, k=k_eff, which=which, tol=tol, v0=v0, return_eigenvectors=True)
        vals = np.real(vals)
        vecs = np.real(vecs)
        idx = np.argsort(vals)
        return vals[idx], vecs[:, idx]
    except Exception as exc:
        warnings.warn(f"eigsh failed for which={which}, k={k_eff}: {exc}")
        if return_eigenvectors:
            return np.array([]), np.empty((n, 0))
        return np.array([]), None


# -----------------------------------------------------------------------------
# 1-4. Size and density controls
# -----------------------------------------------------------------------------

def compute_size_density_metrics(G: nx.Graph) -> Dict[str, float]:
    """Compute scale and density controls."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G) if n > 1 else 0.0
    avg_degree = (2.0 * m / n) if n > 0 else 0.0
    return {
        "log_num_nodes": float(np.log1p(n)),
        "log_num_edges": float(np.log1p(m)),
        "density": float(density),
        "avg_degree": float(avg_degree),
    }


# -----------------------------------------------------------------------------
# 5, 18-21 + new spectral metrics. Sparse Lanczos on the normalized Laplacian.
# -----------------------------------------------------------------------------

def compute_sparse_spectral_metrics(G: nx.Graph, config: ComplexityConfig = ComplexityConfig()) -> Dict[str, float]:
    """
    Compute scalable spectral descriptors using sparse Lanczos on the
    normalized Laplacian.

    Existing keys (unchanged semantics):
      * normalized_spectral_gap
      * laplacian_effective_rank_partial
      * ipr_low_mean
      * ipr_high_mean
      * spectral_degeneracy_fraction (BUG-FIXED: within-block gaps only)

    New keys:
      * bipartite_proximity            : max(0, 2 - lambda_n^(L_norm)).
                                          Equals 0 iff a bipartite component exists.
      * algebraic_connectivity_ratio   : lambda_2 / lambda_n^(L_norm).
      * spectral_entropy_partial       : Shannon entropy of normalized partial spectrum,
                                          normalized to [0,1].
    """
    n = G.number_of_nodes()
    if n < 2:
        return {
            "normalized_spectral_gap": 0.0,
            "laplacian_effective_rank_partial": 0.0,
            "ipr_low_mean": 0.0,
            "ipr_high_mean": 0.0,
            "spectral_degeneracy_fraction": 0.0,
            "bipartite_proximity": np.nan,
            "algebraic_connectivity_ratio": np.nan,
            "spectral_entropy_partial": np.nan,
        }

    L, _ = get_sparse_laplacian(G, normalized=True)
    k = min(config.spectral_k, max(2, n - 2))

    # FIX: which="SA" (smallest algebraic) is more stable than "SM" for PSD operators.
    vals_low, vecs_low = safe_eigsh(L, k=k, which="SA", tol=config.eig_tol)
    vals_high, vecs_high = safe_eigsh(L, k=k, which="LA", tol=config.eig_tol)

    out: Dict[str, float] = {}

    # ---- normalized_spectral_gap ----
    if vals_low.size >= 2:
        vals_low_clean = vals_low.copy()
        vals_low_clean[np.abs(vals_low_clean) < 1e-10] = 0.0
        out["normalized_spectral_gap"] = float(max(vals_low_clean[1] - vals_low_clean[0], 0.0))
    else:
        out["normalized_spectral_gap"] = np.nan

    # ---- laplacian_effective_rank_partial ----
    vals_obs = np.concatenate([vals_low, vals_high]) if vals_high.size else vals_low
    vals_pos = vals_obs[np.isfinite(vals_obs) & (vals_obs > 1e-10)]
    if vals_pos.size > 0:
        out["laplacian_effective_rank_partial"] = float((vals_pos.sum() ** 2) / np.sum(vals_pos ** 2))
    else:
        out["laplacian_effective_rank_partial"] = np.nan

    # ---- ipr_low_mean / ipr_high_mean (Laplacian eigenvectors) ----
    if vecs_low is not None and vecs_low.shape[1] > 0:
        out["ipr_low_mean"] = float(np.mean(np.sum(vecs_low ** 4, axis=0)))
    else:
        out["ipr_low_mean"] = np.nan
    if vecs_high is not None and vecs_high.shape[1] > 0:
        out["ipr_high_mean"] = float(np.mean(np.sum(vecs_high ** 4, axis=0)))
    else:
        out["ipr_high_mean"] = np.nan

    # ---- spectral_degeneracy_fraction (BUG FIX) ----
    # Count near-zero adjacent eigenvalue gaps within each contiguous block.
    deg_tol = max(1e-5, 10.0 * config.eig_tol)
    deg_counts, deg_total = 0, 0
    for block in (vals_low, vals_high):
        block_clean = np.sort(block[np.isfinite(block)])
        if block_clean.size >= 2:
            gaps = np.diff(block_clean)
            deg_counts += int(np.sum(np.abs(gaps) < deg_tol))
            deg_total += gaps.size
    out["spectral_degeneracy_fraction"] = float(deg_counts / deg_total) if deg_total > 0 else np.nan

    # ---- NEW: bipartite_proximity ----
    # For a normalized Laplacian, lambda_n in [0, 2], with lambda_n = 2 iff a
    # connected component is bipartite. We report 2 - lambda_n_max as a continuous
    # proximity-to-bipartite measure.
    if vals_high.size > 0:
        lam_max = float(vals_high.max())
        out["bipartite_proximity"] = float(max(0.0, 2.0 - lam_max))
    else:
        out["bipartite_proximity"] = np.nan

    # ---- NEW: algebraic_connectivity_ratio ----
    if vals_low.size >= 2 and vals_high.size > 0:
        lam2 = float(vals_low[1])
        lam_max = float(vals_high.max())
        out["algebraic_connectivity_ratio"] = float(lam2 / lam_max) if lam_max > 1e-10 else np.nan
    else:
        out["algebraic_connectivity_ratio"] = np.nan

    # ---- NEW: spectral_entropy_partial ----
    if vals_pos.size > 1:
        p = vals_pos / vals_pos.sum()
        ent = -float(np.sum(p * np.log(p + 1e-20)))
        # Normalize to [0,1] by dividing by log(k); 1 = uniform spectrum, 0 = single mode.
        out["spectral_entropy_partial"] = float(ent / np.log(vals_pos.size))
    else:
        out["spectral_entropy_partial"] = np.nan

    return out


# -----------------------------------------------------------------------------
# NEW. Adjacency-spectrum localization (band-edge IPR on A).
# -----------------------------------------------------------------------------

def compute_adjacency_spectral_metrics(
    G: nx.Graph,
    config: ComplexityConfig = ComplexityConfig(),
) -> Dict[str, float]:
    """
    Compute IPR of band-edge eigenvectors of the unsigned adjacency matrix A.

    Theoretically motivated for QW pathways that use H = A (rather than H = L),
    and complementary to Laplacian-IPR because adjacency eigenvectors are not
    degree-normalized; localization signals on hubs survive.
    """
    n = G.number_of_nodes()
    if n < 3 or G.number_of_edges() == 0:
        return {
            "adjacency_ipr_low_mean": np.nan,
            "adjacency_ipr_high_mean": np.nan,
        }

    A, _ = get_sparse_adjacency(G)
    k = min(config.spectral_k, max(2, n - 2))

    # SA = smallest algebraic (most negative for adjacency); LA = largest algebraic.
    vals_low, vecs_low = safe_eigsh(A, k=k, which="SA", tol=config.eig_tol)
    vals_high, vecs_high = safe_eigsh(A, k=k, which="LA", tol=config.eig_tol)

    out: Dict[str, float] = {}
    if vecs_low is not None and vecs_low.shape[1] > 0:
        out["adjacency_ipr_low_mean"] = float(np.mean(np.sum(vecs_low ** 4, axis=0)))
    else:
        out["adjacency_ipr_low_mean"] = np.nan
    if vecs_high is not None and vecs_high.shape[1] > 0:
        out["adjacency_ipr_high_mean"] = float(np.mean(np.sum(vecs_high ** 4, axis=0)))
    else:
        out["adjacency_ipr_high_mean"] = np.nan
    return out


# -----------------------------------------------------------------------------
# NEW. Heat-kernel trace via Hutchinson + scipy expm_multiply.
# -----------------------------------------------------------------------------

def compute_heat_kernel_traces(
    G: nx.Graph,
    config: ComplexityConfig = ComplexityConfig(),
) -> Dict[str, float]:
    """
    Compute normalized heat kernel traces tr(exp(-t L)) / n via the Hutchinson
    estimator with Rademacher probe vectors and scipy.sparse.linalg.expm_multiply.

    Theoretically motivated: tr(exp(-t L)) = sum_i exp(-t lambda_i) is the smooth
    spectral observable that integrates the diffusion behavior the relevant QW vs
    classical mixing bounds depend on. At small t it is dominated by the bulk
    spectrum; at large t it is dominated by the spectral gap.

    Each call is ~O(n_probes * matvec * scipy_internal_steps).
    For n=5000 with sparse L, this is at most a few seconds.
    """
    n = G.number_of_nodes()
    out: Dict[str, float] = {}
    for t in config.heat_kernel_t_values:
        out[f"heat_kernel_trace_t{int(round(t))}"] = np.nan
    if n < 2:
        return out

    L, _ = get_sparse_laplacian(G, normalized=True)
    rng = np.random.default_rng(config.random_state)
    Z = rng.choice(np.array([-1.0, 1.0]), size=(n, config.heat_kernel_n_probes)).astype(float)

    for t in config.heat_kernel_t_values:
        key = f"heat_kernel_trace_t{int(round(t))}"
        try:
            HZ = expm_multiply(-t * L, Z)
            # Hutchinson: E[z^T A z] = tr(A) for Rademacher z.
            trace_per_probe = np.sum(Z * HZ, axis=0)
            trace_est = float(np.mean(trace_per_probe))
            out[key] = float(trace_est / n)
        except Exception as exc:
            warnings.warn(f"heat kernel trace at t={t} failed: {exc}")
            out[key] = np.nan
    return out


# -----------------------------------------------------------------------------
# NEW. Odd girth (length of shortest odd cycle).
# -----------------------------------------------------------------------------

def compute_odd_girth_metric(
    G: nx.Graph,
    config: ComplexityConfig = ComplexityConfig(),
) -> Dict[str, float]:
    """
    Compute log(1 + shortest_odd_cycle_length).

    Procedure:
      1. If G is bipartite, return NaN (no odd cycle exists).
      2. Fast triangle existence check via shared-neighbor scan; if any triangle
         exists, return log(1 + 3).
      3. Otherwise, BFS from up to `odd_girth_max_sources` sampled sources;
         for each source s, scan all edges and identify same-level closures
         (level u == level v), giving an odd cycle of length 2 * level + 1
         passing through s. Track the minimum.

    Returns NaN if no odd cycle is found within the source budget.
    """
    n = G.number_of_nodes()
    if n < 3 or G.number_of_edges() == 0:
        return {"log_odd_girth": np.nan}

    if nx.is_bipartite(G):
        return {"log_odd_girth": np.nan}

    # Fast triangle existence check (early termination).
    for u, v in G.edges():
        nu = set(G.neighbors(u))
        nv = set(G.neighbors(v))
        if (nu & nv) - {u, v}:
            return {"log_odd_girth": float(np.log1p(3))}

    # No triangles: search via BFS from sampled sources.
    rng = np.random.default_rng(config.random_state)
    nodes = list(G.nodes())
    n_sources = min(config.odd_girth_max_sources, len(nodes))
    sources = rng.choice(nodes, size=n_sources, replace=False)

    best_odd: float = np.inf
    edges_list = list(G.edges())

    for src in sources:
        levels = nx.single_source_shortest_path_length(G, src)
        for u, v in edges_list:
            if u in levels and v in levels and levels[u] == levels[v]:
                cycle_len = 2 * levels[u] + 1
                if cycle_len < best_odd:
                    best_odd = cycle_len
        if best_odd <= config.odd_girth_min_cycle_break:
            break

    if not np.isfinite(best_odd):
        return {"log_odd_girth": np.nan}
    return {"log_odd_girth": float(np.log1p(best_odd))}


# -----------------------------------------------------------------------------
# 6 + new. Approximate path-length AND closeness-Gini (free from same BFS).
# -----------------------------------------------------------------------------

def compute_approx_path_length_metric(G: nx.Graph, config: ComplexityConfig = ComplexityConfig()) -> Dict[str, float]:
    """
    Approximate average shortest-path length using sampled BFS sources.

    Also returns:
      * largest_cc_fraction
      * closeness_gini_approx (NEW; free byproduct of same BFS calls).

    For disconnected graphs, the metric is computed on the largest connected
    component when use_largest_cc_for_path is True.
    """
    n = G.number_of_nodes()
    if n == 0:
        return {
            "approx_avg_path_length": np.nan,
            "largest_cc_fraction": 0.0,
            "closeness_gini_approx": np.nan,
        }

    if G.number_of_edges() == 0:
        return {
            "approx_avg_path_length": np.nan,
            "largest_cc_fraction": 1.0 / n,
            "closeness_gini_approx": np.nan,
        }

    if nx.is_connected(G):
        H = G
        lcc_frac = 1.0
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        lcc_frac = len(largest_cc) / n
        H = G.subgraph(largest_cc).copy() if config.use_largest_cc_for_path else G

    nodes = list(H.nodes())
    if len(nodes) < 2:
        return {
            "approx_avg_path_length": 0.0,
            "largest_cc_fraction": float(lcc_frac),
            "closeness_gini_approx": np.nan,
        }

    rng = np.random.default_rng(config.random_state)
    sources = rng.choice(nodes, size=min(config.path_num_sources, len(nodes)), replace=False)

    distances: List[int] = []
    closeness_values: List[float] = []
    for source in sources:
        d = nx.single_source_shortest_path_length(H, source)
        # Closeness centrality of source within H.
        total_dist = sum(v for v in d.values() if v > 0)
        n_reach = sum(1 for v in d.values() if v > 0)
        if total_dist > 0 and n_reach > 0:
            closeness_values.append(n_reach / total_dist)
        distances.extend(d.values())

    if not distances:
        avg_path = np.nan
    else:
        arr = np.asarray(distances, dtype=float)
        arr = arr[arr > 0]
        avg_path = float(arr.mean()) if arr.size > 0 else 0.0

    closeness_gini = (
        gini_coefficient(closeness_values) if len(closeness_values) >= 2 else np.nan
    )

    return {
        "approx_avg_path_length": avg_path,
        "largest_cc_fraction": float(lcc_frac),
        "closeness_gini_approx": float(closeness_gini) if np.isfinite(closeness_gini) else np.nan,
    }


# -----------------------------------------------------------------------------
# 7, 12. Community modularity and conductance
# -----------------------------------------------------------------------------

def _get_communities(G: nx.Graph, seed: int = 0) -> List[set]:
    """Compute communities using Louvain when available, else greedy modularity."""
    if G.number_of_nodes() == 0:
        return []
    try:
        return [set(c) for c in nx.community.louvain_communities(G, seed=seed)]
    except (nx.NetworkXException, ValueError, ZeroDivisionError) as exc:
        logger.debug("Louvain community detection failed (%s); trying greedy modularity", exc)
        try:
            return [set(c) for c in nx.community.greedy_modularity_communities(G)]
        except (nx.NetworkXException, ValueError, ZeroDivisionError) as exc2:
            # A single all-node community is not a detected partition: it forces
            # modularity to 0 and conductance to nan downstream. Warn, because a
            # caller reading those numbers needs to know they came from here.
            logger.warning(
                "No community structure could be detected (louvain: %s; greedy: %s); "
                "treating the graph as a single community, which forces modularity to 0.",
                exc, exc2,
            )
            return [set(G.nodes())]


def _conductance_for_set(G: nx.Graph, S: set) -> float:
    n = G.number_of_nodes()
    if not S or len(S) == n:
        return np.nan
    S = set(S)
    vol_S = sum(dict(G.degree(S)).values())
    T = set(G.nodes()) - S
    vol_T = sum(dict(G.degree(T)).values())
    if min(vol_S, vol_T) <= 0:
        return np.nan
    cut = nx.cut_size(G, S, T)
    return float(cut / min(vol_S, vol_T))


def compute_community_metrics(G: nx.Graph, config: ComplexityConfig = ComplexityConfig()) -> Dict[str, float]:
    """Compute modularity and approximate conductance from detected communities."""
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        # Modularity is Q = sum_c (e_c/m - (d_c/2m)^2), undefined when m == 0.
        # Returning 0.0 for it claimed "no community structure beyond chance",
        # which is an achievable measurement, and disagreed with the nan on the
        # very same line for approx_conductance.
        return {"modularity": np.nan, "approx_conductance": np.nan}

    communities = _get_communities(G, seed=config.random_state)
    try:
        mod = float(nx.community.modularity(G, communities)) if communities else 0.0
    except (nx.NetworkXException, ValueError, ZeroDivisionError) as exc:
        logger.debug("modularity failed for the detected partition (%s); reporting nan", exc)
        mod = np.nan

    conductances = [_conductance_for_set(G, c) for c in communities if 0 < len(c) < G.number_of_nodes()]
    conductances = [c for c in conductances if np.isfinite(c)]
    approx_cond = float(np.min(conductances)) if conductances else np.nan

    return {"modularity": mod, "approx_conductance": approx_cond}


# -----------------------------------------------------------------------------
# 8-11, 26. Degree and centrality concentration
# -----------------------------------------------------------------------------

def compute_degree_metrics(G: nx.Graph) -> Dict[str, float]:
    """Degree heterogeneity, hub dominance, and assortativity."""
    n = G.number_of_nodes()
    if n == 0:
        return {
            "degree_gini": np.nan,
            "max_degree_fraction": np.nan,
            "degree_assortativity": np.nan,
        }

    deg = np.asarray([d for _, d in G.degree()], dtype=float)
    max_possible = max(n - 1, 1)

    try:
        assort = nx.degree_assortativity_coefficient(G) if G.number_of_edges() > 0 else np.nan
    except (nx.NetworkXException, ValueError, ZeroDivisionError, FloatingPointError) as exc:
        # Undefined when every node has the same degree (zero variance).
        logger.debug("degree assortativity is undefined here (%s); reporting nan", exc)
        assort = np.nan

    return {
        "degree_gini": gini_coefficient(deg),
        "max_degree_fraction": float(deg.max() / max_possible) if deg.size else np.nan,
        "degree_assortativity": safe_float(assort, default=np.nan),
    }


def compute_centrality_concentration_metrics(
    G: nx.Graph,
    config: ComplexityConfig = ComplexityConfig(),
) -> Dict[str, float]:
    """Approximate betweenness Gini and PageRank Gini."""
    n = G.number_of_nodes()
    if n == 0:
        return {"pagerank_gini": np.nan, "betweenness_gini_approx": np.nan}

    try:
        pr = nx.pagerank(
            G,
            alpha=config.pagerank_alpha,
            max_iter=config.pagerank_max_iter,
            tol=config.pagerank_tol,
        )
        pagerank_gini = gini_coefficient(pr.values())
    except Exception as exc:
        warnings.warn(f"PageRank failed: {exc}")
        pagerank_gini = np.nan

    try:
        k = min(config.betweenness_k, n)
        btw = nx.betweenness_centrality(G, k=k, seed=config.random_state, normalized=True)
        betweenness_gini = gini_coefficient(btw.values())
    except Exception as exc:
        warnings.warn(f"Approximate betweenness failed: {exc}")
        betweenness_gini = np.nan

    return {
        "pagerank_gini": float(pagerank_gini) if np.isfinite(pagerank_gini) else np.nan,
        "betweenness_gini_approx": float(betweenness_gini) if np.isfinite(betweenness_gini) else np.nan,
    }


# -----------------------------------------------------------------------------
# 13-15. Cycles, transitivity, and non-backtracking structure
# -----------------------------------------------------------------------------

def compute_cycle_metrics(G: nx.Graph, config: ComplexityConfig = ComplexityConfig()) -> Dict[str, float]:
    """Compute transitivity, normalized cycle density, and nonbacktracking spectral radius."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n == 0:
        return {
            "transitivity": np.nan,
            "cycle_density": np.nan,
            "nonbacktracking_spectral_radius": np.nan,
        }

    try:
        trans = float(nx.transitivity(G)) if m > 0 else 0.0
    except (nx.NetworkXException, ValueError, ZeroDivisionError) as exc:
        logger.debug("transitivity is undefined here (%s); reporting nan", exc)
        trans = np.nan

    components = nx.number_connected_components(G) if n > 0 else 0
    cyclomatic = max(0, m - n + components)
    cycle_density = float(cyclomatic / max(m, 1))

    nbr = compute_nonbacktracking_spectral_radius(G, config=config)

    return {
        "transitivity": trans,
        "cycle_density": cycle_density,
        "nonbacktracking_spectral_radius": nbr,
    }


def compute_nonbacktracking_spectral_radius(
    G: nx.Graph,
    config: ComplexityConfig = ComplexityConfig(),
) -> float:
    """Approximate spectral radius of the Hashimoto/non-backtracking matrix."""
    m = G.number_of_edges()
    if m == 0:
        return 0.0

    directed_edges: List[Tuple[Hashable, Hashable]] = []
    for u, v in G.edges():
        directed_edges.append((u, v))
        directed_edges.append((v, u))

    q = len(directed_edges)
    if q > config.nonbacktracking_max_directed_edges:
        warnings.warn(
            f"Skipping nonbacktracking spectral radius: {q} directed edges exceed cap "
            f"{config.nonbacktracking_max_directed_edges}."
        )
        return np.nan

    edge_to_idx = {e: i for i, e in enumerate(directed_edges)}
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for i, (u, v) in enumerate(directed_edges):
        for w in G.neighbors(v):
            if w == u:
                continue
            j = edge_to_idx.get((v, w))
            if j is not None:
                rows.append(i)
                cols.append(j)
                data.append(1.0)

    if not data:
        return 0.0

    B = sp.csr_matrix((data, (rows, cols)), shape=(q, q), dtype=float)

    try:
        # Deterministic ARPACK start vector (see safe_eigsh) for reproducibility.
        v0 = np.random.default_rng(0).standard_normal(q)
        val = eigs(B, k=1, which="LM", v0=v0, return_eigenvectors=False, tol=config.eig_tol)[0]
        return float(abs(val))
    except Exception as exc:
        warnings.warn(f"Nonbacktracking eigs failed: {exc}")
        return np.nan


# -----------------------------------------------------------------------------
# 16-17. Ollivier-Ricci curvature proxies
# -----------------------------------------------------------------------------

def compute_orc_proxy_metrics(G: nx.Graph) -> Dict[str, float]:
    """
    Compute scalable ORC-inspired edge bottleneck proxies.

    Uses the Jost-Liu style lower-bound proxy:
        kappa_LB(u,v) = Delta/max(d_u,d_v) + 1/d_u + 1/d_v - 1
    """
    if G.number_of_edges() == 0:
        return {"orc_kLB_mean": np.nan, "orc_negative_fraction": np.nan}

    neighbor_sets = {u: set(G.neighbors(u)) for u in G.nodes()}
    kappa_vals: List[float] = []

    for u, v in G.edges():
        du = len(neighbor_sets[u])
        dv = len(neighbor_sets[v])
        if du == 0 or dv == 0:
            kappa = -1.0
        else:
            common = (neighbor_sets[u] & neighbor_sets[v]) - {u, v}
            Delta = len(common)
            kappa = Delta / max(du, dv) + 1.0 / du + 1.0 / dv - 1.0
        kappa_vals.append(kappa)

    arr = np.asarray(kappa_vals, dtype=float)
    return {
        "orc_kLB_mean": float(np.mean(arr)),
        "orc_negative_fraction": float(np.mean(arr < 0.0)),
    }


# -----------------------------------------------------------------------------
# 22. Weisfeiler-Lehman compression / symmetry proxy
# -----------------------------------------------------------------------------

def compute_wl_compression_ratio(G: nx.Graph, config: ComplexityConfig = ComplexityConfig()) -> Dict[str, float]:
    """WL color compression ratio after a few 1-WL refinement iterations."""
    n = G.number_of_nodes()
    if n == 0:
        return {"wl_compression_ratio": np.nan}

    colors: Dict[Hashable, int] = {u: int(G.degree(u)) for u in G.nodes()}

    for _ in range(config.wl_iterations):
        signatures = {}
        for u in G.nodes():
            neigh_colors = tuple(sorted(colors[v] for v in G.neighbors(u)))
            signatures[u] = (colors[u], neigh_colors)

        unique = {sig: i for i, sig in enumerate(sorted(set(signatures.values()), key=str))}
        colors = {u: unique[sig] for u, sig in signatures.items()}

    num_colors = len(set(colors.values()))
    return {"wl_compression_ratio": float(num_colors / n)}


# -----------------------------------------------------------------------------
# 23. Core-periphery proxy
# -----------------------------------------------------------------------------

def compute_core_metrics(G: nx.Graph) -> Dict[str, float]:
    """k-core concentration as a scalable core-periphery proxy."""
    n = G.number_of_nodes()
    if n == 0:
        return {"core_number_gini": np.nan}
    try:
        # core_number requires no self-loops, which we already strip in sanitize_graph.
        core = nx.core_number(G)
        vals = list(core.values())
        return {"core_number_gini": gini_coefficient(vals)}
    except Exception as exc:
        warnings.warn(f"core_number failed: {exc}")
        return {"core_number_gini": np.nan}


# -----------------------------------------------------------------------------
# 24. Label homophily
# -----------------------------------------------------------------------------

def _labels_to_dict(
    labels: Optional[Union[Mapping[Hashable, Any], Sequence[Any], np.ndarray]],
    nodelist: Sequence[Hashable],
) -> Optional[Dict[Hashable, Any]]:
    if labels is None:
        return None
    if isinstance(labels, Mapping):
        return dict(labels)
    arr = np.asarray(labels)
    if arr.shape[0] != len(nodelist):
        raise ValueError("labels length must match number of nodes when labels is an array/sequence.")
    return {node: arr[i] for i, node in enumerate(nodelist)}


def compute_label_homophily(
    G: nx.Graph,
    labels: Optional[Union[Mapping[Hashable, Any], Sequence[Any], np.ndarray]],
) -> Dict[str, float]:
    """Fraction of edges connecting nodes with identical labels."""
    nodelist = get_nodelist(G)
    label_dict = _labels_to_dict(labels, nodelist)
    if label_dict is None:
        return {"label_homophily": np.nan}

    same = 0
    total = 0
    for u, v in G.edges():
        if u in label_dict and v in label_dict:
            if label_dict[u] is None or label_dict[v] is None:
                continue
            same += int(label_dict[u] == label_dict[v])
            total += 1

    return {"label_homophily": float(same / total) if total > 0 else np.nan}


# -----------------------------------------------------------------------------
# 25. Feature Dirichlet energy
# -----------------------------------------------------------------------------

def _features_to_array(
    features: Optional[Union[np.ndarray, Mapping[Hashable, Sequence[float]]]],
    nodelist: Sequence[Hashable],
) -> Optional[np.ndarray]:
    if features is None:
        return None
    if isinstance(features, Mapping):
        X = []
        for node in nodelist:
            if node not in features:
                raise ValueError(f"Missing feature for node {node!r}.")
            X.append(features[node])
        return np.asarray(X, dtype=float)
    X = np.asarray(features, dtype=float)
    if X.shape[0] != len(nodelist):
        raise ValueError("features.shape[0] must match number of nodes.")
    if X.ndim == 1:
        X = X[:, None]
    return X


def compute_feature_dirichlet_energy(
    G: nx.Graph,
    features: Optional[Union[np.ndarray, Mapping[Hashable, Sequence[float]]]],
    normalized_laplacian: bool = True,
) -> Dict[str, float]:
    """Compute normalized feature Dirichlet energy Tr(X^T L X) / Tr(X^T X)."""
    L, nodelist = get_sparse_laplacian(G, normalized=normalized_laplacian)
    X = _features_to_array(features, nodelist)
    if X is None:
        return {"feature_dirichlet_energy": np.nan}

    denom = float(np.sum(X * X))
    if denom <= 0:
        return {"feature_dirichlet_energy": np.nan}

    LX = L @ X
    energy = float(np.sum(X * LX) / denom)
    return {"feature_dirichlet_energy": energy}


# -----------------------------------------------------------------------------
# Full metric interface (now returns 27 + 9 = 36 metrics).
# -----------------------------------------------------------------------------

def compute_enhanced_complexity_metrics(
    G: nx.Graph,
    labels: Optional[Union[Mapping[Hashable, Any], Sequence[Any], np.ndarray]] = None,
    features: Optional[Union[np.ndarray, Mapping[Hashable, Sequence[float]]]] = None,
    config: ComplexityConfig = ComplexityConfig(),
    sanitize: bool = True,
) -> Dict[str, float]:
    """
    Compute the enhanced QuVINE complexity metrics for a single graph.

    This function computes 36 comprehensive metrics (27 original + 9 new theory-grade
    metrics) that characterize graph structure and predict quantum advantage.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    labels : optional
        Node labels for computing label homophily
    features : optional
        Node features for computing feature Dirichlet energy
    config : ComplexityConfig
        Configuration for approximation parameters
    sanitize : bool, default=True
        If True, convert to simple undirected graph and remove self-loops

    Returns
    -------
    dict
        Dictionary containing all 36 complexity metrics
    """
    H = sanitize_graph(G) if sanitize else G.copy()

    metrics: Dict[str, float] = {
        "num_nodes_raw": float(H.number_of_nodes()),
        "num_edges_raw": float(H.number_of_edges()),
    }

    metric_functions = [
        lambda graph: compute_size_density_metrics(graph),
        lambda graph: compute_sparse_spectral_metrics(graph, config=config),
        lambda graph: compute_adjacency_spectral_metrics(graph, config=config),
        lambda graph: compute_heat_kernel_traces(graph, config=config),
        lambda graph: compute_odd_girth_metric(graph, config=config),
        lambda graph: compute_approx_path_length_metric(graph, config=config),
        lambda graph: compute_community_metrics(graph, config=config),
        lambda graph: compute_degree_metrics(graph),
        lambda graph: compute_centrality_concentration_metrics(graph, config=config),
        lambda graph: compute_cycle_metrics(graph, config=config),
        lambda graph: compute_orc_proxy_metrics(graph),
        lambda graph: compute_wl_compression_ratio(graph, config=config),
        lambda graph: compute_core_metrics(graph),
        lambda graph: compute_label_homophily(graph, labels=labels),
        lambda graph: compute_feature_dirichlet_energy(graph, features=features),
    ]

    for fn in metric_functions:
        try:
            metrics.update(fn(H))
        except Exception as exc:
            warnings.warn(f"Metric function {getattr(fn, '__name__', repr(fn))} failed: {exc}")

    for key in CANDIDATE_ALL_METRICS:
        metrics.setdefault(key, np.nan)

    return metrics


def compute_complexity_table(
    graphs: Mapping[str, nx.Graph],
    labels: Optional[Mapping[str, Union[Mapping[Hashable, Any], Sequence[Any], np.ndarray]]] = None,
    features: Optional[Mapping[str, Union[np.ndarray, Mapping[Hashable, Sequence[float]]]]] = None,
    config: ComplexityConfig = ComplexityConfig(),
) -> "Any":
    """
    Compute a pandas DataFrame of complexity metrics for many graphs.

    Parameters
    ----------
    graphs : dict
        Dictionary mapping graph names to NetworkX graphs
    labels : optional
        Dictionary mapping graph names to node labels
    features : optional
        Dictionary mapping graph names to node features
    config : ComplexityConfig
        Configuration for approximation parameters

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per graph and columns for each metric
    """
    import pandas as pd

    rows = []
    for name, G in graphs.items():
        lab = labels.get(name) if labels is not None else None
        feat = features.get(name) if features is not None else None
        row = compute_enhanced_complexity_metrics(G, labels=lab, features=feat, config=config)
        row["graph_name"] = name
        rows.append(row)

    df = pd.DataFrame(rows).set_index("graph_name")
    return df



# =============================================================================
# Orchestrator (parallels dataset_evaluation.evaluate)
# =============================================================================
def evaluate_graph(G: nx.Graph, name: str = "") -> pd.DataFrame:
    """Summarize a graph's complexity as a one-row DataFrame.

    Mirrors :func:`qbiocode.evaluation.dataset_evaluation.evaluate`: it runs the
    core spectral/topological metrics (:func:`compute_graph_complexity_metrics`)
    and the enhanced structural metrics (:func:`compute_enhanced_complexity_metrics`),
    merges them, and returns a transposed one-row summary keyed by ``name``.

    Args:
        G (networkx.Graph): Graph to evaluate. Any networkx graph class is
            accepted; metrics undefined for the given class are omitted rather
            than raised (see Notes).
        name (str): Identifier stored in the ``Graph`` column of the summary.

    Returns:
        pandas.DataFrame: One-row summary of graph complexity metrics.

    Raises:
        TypeError: if ``G`` is not a networkx graph, or ``name`` is not a string.

    Notes:
        An **empty** graph is a legitimate input and yields a size-only summary
        (``Graph``/``num_nodes``/``num_edges``) with a ``UserWarning``. ``None``
        is not: it is a caller error and raises, because the size-only summary
        previously returned for it reported "0 nodes" for a graph that was in
        fact missing.

        Individual metric groups degrade independently: if either the core or the
        enhanced block fails, its columns are absent from the returned frame and
        a ``UserWarning`` names the failure. The frame is never partially
        populated with fabricated values, so check for a column before reading
        it rather than assuming a fixed width.
    """
    if G is None:
        raise TypeError(
            "evaluate_graph() requires a networkx graph, got None. If you meant "
            "an empty graph, pass nx.Graph() -- that returns a size-only summary. "
            "None is treated as a caller error because silently summarizing it as "
            "'0 nodes' is indistinguishable from a real empty graph."
        )
    if not isinstance(G, nx.Graph):
        raise TypeError(
            f"evaluate_graph() requires a networkx graph (Graph, DiGraph, "
            f"MultiGraph or MultiDiGraph); got {type(G).__name__}. Convert first, "
            f"e.g. nx.from_numpy_array(adjacency) for a dense or sparse matrix, or "
            f"nx.from_pandas_edgelist(df) for an edge list."
        )
    if not isinstance(name, str):
        raise TypeError(
            f"name must be a string used to label the summary row; got "
            f"{type(name).__name__} ({name!r})."
        )

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if n_nodes == 0:
        warnings.warn("evaluate_graph received an empty graph; returning size-only summary.")
        return pd.DataFrame.from_dict(
            {"Graph": name, "num_nodes": 0, "num_edges": 0}, orient="index"
        ).T

    # Both blocks stay broad on purpose: this is the public entry point, and the
    # two metric families are independent, so one failing must not cost the caller
    # the other. What they must not do is fail opaquely -- the exception type is
    # named in the warning and the traceback is kept at DEBUG, because the columns
    # simply go missing from the returned frame and there is otherwise nothing to
    # trace it back to.
    try:
        base = compute_graph_complexity_metrics(G)
    except Exception as e:  # pragma: no cover - defensive
        warnings.warn(
            f"core graph complexity metrics failed ({type(e).__name__}: {e}); "
            f"those columns are omitted from the summary for {name!r}."
        )
        logger.debug("compute_graph_complexity_metrics failed", exc_info=True)
        base = {}

    try:
        enhanced = compute_enhanced_complexity_metrics(G)
    except Exception as e:  # pragma: no cover - defensive
        warnings.warn(
            f"enhanced graph complexity metrics failed ({type(e).__name__}: {e}); "
            f"those columns are omitted from the summary for {name!r}."
        )
        logger.debug("compute_enhanced_complexity_metrics failed", exc_info=True)
        enhanced = {}

    row: Dict[str, Any] = {"Graph": name, "num_nodes": n_nodes, "num_edges": n_edges}
    # Enhanced first, then core -- core's canonical values win on any key overlap.
    row.update(enhanced)
    row.update(base)

    summary_df = pd.DataFrame.from_dict(row, orient="index")
    return summary_df.T
