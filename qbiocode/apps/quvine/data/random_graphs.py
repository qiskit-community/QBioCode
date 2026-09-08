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
Random Graph Generator for QuVINE

This module provides functions to generate various types of random graphs with known
structures or specific key elements that are suitable for testing embedding algorithms.
Includes both classical graph models and biologically-inspired structures.
"""

import networkx as nx
import numpy as np
from typing import Optional, Dict, List, Tuple, Set, Sequence, Any, Iterable
import warnings


def generate_erdos_renyi(
    n: int,
    p: Optional[float] = None,
    m: Optional[int] = None,
    seed: Optional[int] = None,
    directed: bool = False
) -> nx.Graph:
    """
    Generate an Erdős-Rényi random graph.
    
    Parameters
    ----------
    n : int
        Number of nodes
    p : float, optional
        Probability of edge creation (G(n,p) model)
    m : int, optional
        Number of edges (G(n,m) model)
    seed : int, optional
        Random seed for reproducibility
    directed : bool, default=False
        If True, generate a directed graph
        
    Returns
    -------
    nx.Graph or nx.DiGraph
        Random graph
    """
    if p is not None and m is not None:
        raise ValueError("Specify either p or m, not both")
    
    if p is not None:
        if directed:
            return nx.erdos_renyi_graph(n, p, seed=seed, directed=True)
        return nx.erdos_renyi_graph(n, p, seed=seed)
    elif m is not None:
        if directed:
            return nx.gnm_random_graph(n, m, seed=seed, directed=True)
        return nx.gnm_random_graph(n, m, seed=seed)
    else:
        raise ValueError("Must specify either p or m")


def generate_barabasi_albert(
    n: int,
    m: int,
    seed: Optional[int] = None,
    initial_graph: Optional[nx.Graph] = None
) -> nx.Graph:
    """
    Generate a Barabási-Albert scale-free network using preferential attachment.
    
    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of edges to attach from a new node to existing nodes
    seed : int, optional
        Random seed for reproducibility
    initial_graph : nx.Graph, optional
        Initial connected graph with at least m nodes
        
    Returns
    -------
    nx.Graph
        Scale-free network
    """
    return nx.barabasi_albert_graph(n, m, seed=seed, initial_graph=initial_graph)


def generate_watts_strogatz(
    n: int,
    k: int,
    p: float,
    seed: Optional[int] = None
) -> nx.Graph:
    """
    Generate a Watts-Strogatz small-world network.
    
    Parameters
    ----------
    n : int
        Number of nodes
    k : int
        Each node is connected to k nearest neighbors in ring topology
    p : float
        Probability of rewiring each edge
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    nx.Graph
        Small-world network
    """
    return nx.watts_strogatz_graph(n, k, p, seed=seed)


def generate_powerlaw_cluster(
    n: int,
    m: int,
    p: float,
    seed: Optional[int] = None
) -> nx.Graph:
    """
    Generate a random graph with powerlaw degree distribution and clustering.
    
    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of random edges to add for each new node
    p : float
        Probability of adding a triangle after adding a random edge
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    nx.Graph
        Powerlaw cluster graph
    """
    return nx.powerlaw_cluster_graph(n, m, p, seed=seed)


def generate_stochastic_block_model(
    sizes: List[int],
    p_matrix: List[List[float]],
    seed: Optional[int] = None,
    directed: bool = False,
    selfloops: bool = False
) -> nx.Graph:
    """
    Generate a stochastic block model graph with community structure.
    
    Parameters
    ----------
    sizes : list of int
        Sizes of blocks (communities)
    p_matrix : list of list of float
        Matrix of edge probabilities between and within blocks
    seed : int, optional
        Random seed for reproducibility
    directed : bool, default=False
        If True, generate a directed graph
    selfloops : bool, default=False
        If True, allow self-loops
        
    Returns
    -------
    nx.Graph or nx.DiGraph
        Stochastic block model graph with community structure
    """
    return nx.stochastic_block_model(
        sizes, p_matrix, seed=seed, directed=directed, selfloops=selfloops
    )


def generate_random_geometric(
    n: int,
    radius: float,
    dim: int = 2,
    seed: Optional[int] = None,
    pos: Optional[Dict] = None
) -> nx.Graph:
    """
    Generate a random geometric graph in the unit cube.
    
    Parameters
    ----------
    n : int
        Number of nodes
    radius : float
        Distance threshold for edge creation
    dim : int, default=2
        Dimension of the space
    seed : int, optional
        Random seed for reproducibility
    pos : dict, optional
        Dictionary of node positions
        
    Returns
    -------
    nx.Graph
        Random geometric graph with 'pos' node attribute
    """
    return nx.random_geometric_graph(n, radius, dim=dim, seed=seed, pos=pos)


def generate_modular_network(
    num_communities: int,
    nodes_per_community: int,
    p_intra: float,
    p_inter: float,
    seed: Optional[int] = None
) -> Tuple[nx.Graph, Dict[int, int]]:
    """
    Generate a modular network with clear community structure.
    
    Parameters
    ----------
    num_communities : int
        Number of communities
    nodes_per_community : int
        Number of nodes in each community
    p_intra : float
        Probability of edges within communities
    p_inter : float
        Probability of edges between communities
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    G : nx.Graph
        Modular network
    communities : dict
        Mapping from node to community ID
    """
    sizes = [nodes_per_community] * num_communities
    p_matrix = [[p_intra if i == j else p_inter 
                 for j in range(num_communities)] 
                for i in range(num_communities)]
    
    G = generate_stochastic_block_model(sizes, p_matrix, seed=seed)
    
    # Create community mapping
    communities = {}
    node_idx = 0
    for comm_id, size in enumerate(sizes):
        for _ in range(size):
            communities[node_idx] = comm_id
            node_idx += 1
    
    # Add community as node attribute
    nx.set_node_attributes(G, communities, 'community')
    
    return G, communities


def generate_hierarchical_network(
    levels: int,
    branching_factor: int,
    p_level: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[nx.Graph, Dict[int, int]]:
    """
    Generate a hierarchical network with tree-like structure plus random edges.
    
    Parameters
    ----------
    levels : int
        Number of hierarchy levels
    branching_factor : int
        Number of children per parent node
    p_level : float, default=0.1
        Probability of random edges within same level
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    G : nx.Graph
        Hierarchical network
    node_levels : dict
        Mapping from node to hierarchy level
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    # Create tree structure
    G = nx.balanced_tree(branching_factor, levels - 1)
    
    # Track node levels
    node_levels = {}
    for node in G.nodes():
        # Calculate level based on tree structure
        level = 0
        temp_node = node
        while temp_node > 0:
            temp_node = (temp_node - 1) // branching_factor
            level += 1
        node_levels[node] = level
    
    # Add random edges within levels
    nodes_by_level = {}
    for node, level in node_levels.items():
        if level not in nodes_by_level:
            nodes_by_level[level] = []
        nodes_by_level[level].append(node)
    
    for level, nodes in nodes_by_level.items():
        if len(nodes) > 1:
            for i, u in enumerate(nodes):
                for v in nodes[i+1:]:
                    if rng.random() < p_level:
                        G.add_edge(u, v)
    
    nx.set_node_attributes(G, node_levels, 'level')
    
    return G, node_levels


def generate_core_periphery(
    n_core: int,
    n_periphery: int,
    p_core: float,
    p_core_periphery: float,
    p_periphery: float = 0.01,
    seed: Optional[int] = None
) -> Tuple[nx.Graph, Set[int], Set[int]]:
    """
    Generate a core-periphery network structure.
    
    Parameters
    ----------
    n_core : int
        Number of core nodes
    n_periphery : int
        Number of periphery nodes
    p_core : float
        Edge probability within core
    p_core_periphery : float
        Edge probability between core and periphery
    p_periphery : float, default=0.01
        Edge probability within periphery
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    G : nx.Graph
        Core-periphery network
    core_nodes : set
        Set of core node IDs
    periphery_nodes : set
        Set of periphery node IDs
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    n_total = n_core + n_periphery
    G = nx.Graph()
    G.add_nodes_from(range(n_total))
    
    core_nodes = set(range(n_core))
    periphery_nodes = set(range(n_core, n_total))
    
    # Core edges
    for i in core_nodes:
        for j in core_nodes:
            if i < j and rng.random() < p_core:
                G.add_edge(i, j)
    
    # Core-periphery edges
    for i in core_nodes:
        for j in periphery_nodes:
            if rng.random() < p_core_periphery:
                G.add_edge(i, j)
    
    # Periphery edges
    for i in periphery_nodes:
        for j in periphery_nodes:
            if i < j and rng.random() < p_periphery:
                G.add_edge(i, j)
    
    # Add node attributes
    node_types = {node: 'core' if node in core_nodes else 'periphery' 
                  for node in G.nodes()}
    nx.set_node_attributes(G, node_types, 'type')
    
    return G, core_nodes, periphery_nodes


def generate_bipartite_random(
    n1: int,
    n2: int,
    p: Optional[float] = None,
    m: Optional[int] = None,
    seed: Optional[int] = None
) -> Tuple[nx.Graph, Set[int], Set[int]]:
    """
    Generate a random bipartite graph.
    
    Parameters
    ----------
    n1 : int
        Number of nodes in first partition
    n2 : int
        Number of nodes in second partition
    p : float, optional
        Probability of edge creation
    m : int, optional
        Number of edges
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    G : nx.Graph
        Bipartite graph
    set1 : set
        First partition node IDs
    set2 : set
        Second partition node IDs
    """
    if p is not None and m is not None:
        raise ValueError("Specify either p or m, not both")
    
    if p is not None:
        G = nx.bipartite.random_graph(n1, n2, p, seed=seed)
    elif m is not None:
        G = nx.bipartite.gnmk_random_graph(n1, n2, m, seed=seed)
    else:
        raise ValueError("Must specify either p or m")
    
    set1 = {n for n, d in G.nodes(data=True) if d['bipartite'] == 0}
    set2 = set(G.nodes()) - set1
    
    return G, set1, set2


def add_hub_nodes(
    G: nx.Graph,
    num_hubs: int,
    hub_degree: int,
    seed: Optional[int] = None
) -> Tuple[nx.Graph, List[int]]:
    """
    Add hub nodes to an existing graph.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    num_hubs : int
        Number of hub nodes to add
    hub_degree : int
        Degree of each hub node
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    G : nx.Graph
        Graph with added hubs
    hub_nodes : list
        List of hub node IDs
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    G = G.copy()
    existing_nodes = list(G.nodes())
    n_existing = len(existing_nodes)
    
    hub_nodes = []
    for i in range(num_hubs):
        hub_id = n_existing + i
        G.add_node(hub_id)
        hub_nodes.append(hub_id)
        
        # Connect to random existing nodes
        targets = rng.choice(existing_nodes, size=min(hub_degree, n_existing), replace=False)
        for target in targets:
            G.add_edge(hub_id, target)
    
    return G, hub_nodes


def generate_graph_with_seeds_and_targets(
    n: int,
    num_seeds: int,
    num_targets: int,
    graph_type: str = 'barabasi_albert',
    seed: Optional[int] = None,
    **kwargs
) -> Tuple[nx.Graph, List[int], List[int]]:
    """
    Generate a random graph with designated seed and target nodes for embedding evaluation.
    
    Parameters
    ----------
    n : int
        Total number of nodes
    num_seeds : int
        Number of seed nodes
    num_targets : int
        Number of target nodes
    graph_type : str, default='barabasi_albert'
        Type of graph: 'erdos_renyi', 'barabasi_albert', 'watts_strogatz', 
        'powerlaw_cluster', 'modular'
    seed : int, optional
        Random seed for reproducibility
    **kwargs
        Additional parameters for specific graph types
        
    Returns
    -------
    G : nx.Graph
        Generated graph
    seeds : list
        List of seed node IDs
    targets : list
        List of target node IDs
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    # Generate base graph
    if graph_type == 'erdos_renyi':
        p = kwargs.get('p', 0.1)
        G = generate_erdos_renyi(n, p=p, seed=seed)
    elif graph_type == 'barabasi_albert':
        m = kwargs.get('m', 3)
        G = generate_barabasi_albert(n, m, seed=seed)
    elif graph_type == 'watts_strogatz':
        k = kwargs.get('k', 4)
        p = kwargs.get('p', 0.3)
        G = generate_watts_strogatz(n, k, p, seed=seed)
    elif graph_type == 'powerlaw_cluster':
        m = kwargs.get('m', 3)
        p = kwargs.get('p', 0.3)
        G = generate_powerlaw_cluster(n, m, p, seed=seed)
    elif graph_type == 'modular':
        num_communities = kwargs.get('num_communities', 5)
        nodes_per_community = n // num_communities
        p_intra = kwargs.get('p_intra', 0.3)
        p_inter = kwargs.get('p_inter', 0.01)
        G, _ = generate_modular_network(num_communities, nodes_per_community, 
                                       p_intra, p_inter, seed=seed)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")
    
    # Select seeds and targets
    nodes = list(G.nodes())
    selected = rng.choice(nodes, size=num_seeds + num_targets, replace=False)
    seeds = selected[:num_seeds].tolist()
    targets = selected[num_seeds:].tolist()
    
    # Add node attributes
    node_roles = {}
    for node in G.nodes():
        if node in seeds:
            node_roles[node] = 'seed'
        elif node in targets:
            node_roles[node] = 'target'
        else:
            node_roles[node] = 'regular'
    
    nx.set_node_attributes(G, node_roles, 'role')
    
    return G, seeds, targets


def get_graph_statistics(G: nx.Graph) -> Dict:
    """
    Compute comprehensive statistics for a graph.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
        
    Returns
    -------
    dict
        Dictionary of graph statistics
    """
    stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'is_connected': nx.is_connected(G),
    }
    
    if G.number_of_nodes() > 0:
        degrees = [d for n, d in G.degree()]
        stats['avg_degree'] = float(np.mean(degrees))
        stats['max_degree'] = np.max(degrees)
        stats['min_degree'] = np.min(degrees)
        
        if nx.is_connected(G):
            stats['diameter'] = nx.diameter(G)
            stats['avg_shortest_path'] = nx.average_shortest_path_length(G)
        else:
            stats['num_components'] = nx.number_connected_components(G)
            largest_cc = max(nx.connected_components(G), key=len)
            stats['largest_cc_size'] = len(largest_cc)
        
        stats['avg_clustering'] = nx.average_clustering(G)
        stats['transitivity'] = nx.transitivity(G)
    
    return stats



def generate_comprehensive_dataset(
    n_instances: int = 30,
    base_seed: int = 42,
    n_nodes: int = 200,
    save_dir: Optional[str] = None
) -> Dict[str, List[Tuple[nx.Graph, Dict]]]:
    """
    Generate a comprehensive dataset of random graphs for embedding analysis.
    
    This function generates multiple instances of each graph type with varying
    random seeds to capture natural parameter variations.
    
    Graph types included:
    - Erdős-Rényi (random)
    - Barabási-Albert (scale-free)
    - Watts-Strogatz (small-world)
    - Powerlaw Cluster (scale-free with clustering)
    - Stochastic Block Model (modular/community structure)
    - Random Geometric (spatial networks)
    - Hierarchical (tree-like structure)
    - Core-Periphery (hub-spoke structure)
    - Bipartite Random (two-mode networks)
    
    Parameters
    ----------
    n_instances : int, default=30
        Number of instances to generate for each graph type
    base_seed : int, default=42
        Base random seed (each instance uses base_seed + instance_id)
    n_nodes : int, default=200
        Target number of nodes for each graph
    save_dir : str, optional
        If provided, save graphs to this directory
        
    Returns
    -------
    dict
        Dictionary mapping graph type names to lists of (graph, metadata) tuples
        
    Examples
    --------
    >>> dataset = generate_comprehensive_dataset(n_instances=30, n_nodes=200)
    >>> print(f"Generated {sum(len(v) for v in dataset.values())} graphs")
    >>> print(f"Graph types: {list(dataset.keys())}")
    """
    import os
    from pathlib import Path
    
    dataset = {}
    
    # 1. Erdős-Rényi (Random)
    print(f"Generating {n_instances} Erdős-Rényi graphs...")
    dataset['erdos_renyi'] = []
    for i in range(n_instances):
        seed = base_seed + i
        p = 0.05 + (i / n_instances) * 0.10  # Vary density from 0.05 to 0.15
        G = generate_erdos_renyi(n_nodes, p=p, seed=seed)
        metadata = {
            'type': 'erdos_renyi',
            'instance': i,
            'seed': seed,
            'n_nodes': n_nodes,
            'p': p,
            'params': {'n': n_nodes, 'p': p}
        }
        dataset['erdos_renyi'].append((G, metadata))
    
    # 2. Barabási-Albert (Scale-Free)
    print(f"Generating {n_instances} Barabási-Albert graphs...")
    dataset['barabasi_albert'] = []
    for i in range(n_instances):
        seed = base_seed + i
        m = 2 + (i % 5)  # Vary m from 2 to 6
        G = generate_barabasi_albert(n_nodes, m=m, seed=seed)
        metadata = {
            'type': 'barabasi_albert',
            'instance': i,
            'seed': seed,
            'n_nodes': n_nodes,
            'm': m,
            'params': {'n': n_nodes, 'm': m}
        }
        dataset['barabasi_albert'].append((G, metadata))
    
    # 3. Watts-Strogatz (Small-World)
    print(f"Generating {n_instances} Watts-Strogatz graphs...")
    dataset['watts_strogatz'] = []
    for i in range(n_instances):
        seed = base_seed + i
        k = 4 + (i % 6) * 2  # Vary k from 4 to 14
        p = 0.1 + (i / n_instances) * 0.4  # Vary rewiring prob from 0.1 to 0.5
        G = generate_watts_strogatz(n_nodes, k=k, p=p, seed=seed)
        metadata = {
            'type': 'watts_strogatz',
            'instance': i,
            'seed': seed,
            'n_nodes': n_nodes,
            'k': k,
            'p': p,
            'params': {'n': n_nodes, 'k': k, 'p': p}
        }
        dataset['watts_strogatz'].append((G, metadata))
    
    # 4. Powerlaw Cluster (Scale-Free with Clustering)
    print(f"Generating {n_instances} Powerlaw Cluster graphs...")
    dataset['powerlaw_cluster'] = []
    for i in range(n_instances):
        seed = base_seed + i
        m = 2 + (i % 4)  # Vary m from 2 to 5
        p = 0.1 + (i / n_instances) * 0.3  # Vary triangle prob from 0.1 to 0.4
        G = generate_powerlaw_cluster(n_nodes, m=m, p=p, seed=seed)
        metadata = {
            'type': 'powerlaw_cluster',
            'instance': i,
            'seed': seed,
            'n_nodes': n_nodes,
            'm': m,
            'p': p,
            'params': {'n': n_nodes, 'm': m, 'p': p}
        }
        dataset['powerlaw_cluster'].append((G, metadata))
    
    # 5. Stochastic Block Model (Modular)
    print(f"Generating {n_instances} Stochastic Block Model graphs...")
    dataset['stochastic_block_model'] = []
    for i in range(n_instances):
        seed = base_seed + i
        n_communities = 3 + (i % 5)  # Vary communities from 3 to 7
        p_in = 0.3 + (i / n_instances) * 0.3  # Vary intra-community from 0.3 to 0.6
        p_out = 0.01 + (i / n_instances) * 0.04  # Vary inter-community from 0.01 to 0.05
        
        # Create block sizes
        block_sizes = [n_nodes // n_communities] * n_communities
        # Adjust last block to account for rounding
        block_sizes[-1] += n_nodes - sum(block_sizes)
        
        # Create probability matrix
        p_matrix = [[p_out] * n_communities for _ in range(n_communities)]
        for j in range(n_communities):
            p_matrix[j][j] = p_in
        
        G = generate_stochastic_block_model(block_sizes, p_matrix, seed=seed)
        metadata = {
            'type': 'stochastic_block_model',
            'instance': i,
            'seed': seed,
            'n_nodes': n_nodes,
            'n_communities': n_communities,
            'p_in': p_in,
            'p_out': p_out,
            'params': {'block_sizes': block_sizes, 'p_in': p_in, 'p_out': p_out}
        }
        dataset['stochastic_block_model'].append((G, metadata))
    
    # 6. Random Geometric (Spatial)
    print(f"Generating {n_instances} Random Geometric graphs...")
    dataset['random_geometric'] = []
    for i in range(n_instances):
        seed = base_seed + i
        radius = 0.1 + (i / n_instances) * 0.15  # Vary radius from 0.1 to 0.25
        dim = 2  # 2D space
        G = generate_random_geometric(n_nodes, radius=radius, dim=dim, seed=seed)
        metadata = {
            'type': 'random_geometric',
            'instance': i,
            'seed': seed,
            'n_nodes': n_nodes,
            'radius': radius,
            'dim': dim,
            'params': {'n': n_nodes, 'radius': radius, 'dim': dim}
        }
        dataset['random_geometric'].append((G, metadata))
    
    # 7. Hierarchical
    print(f"Generating {n_instances} Hierarchical graphs...")
    dataset['hierarchical'] = []
    for i in range(n_instances):
        seed = base_seed + i
        branching_factor = 2 + (i % 3)  # Vary branching from 2 to 4
        levels = 4 + (i % 3)  # Vary levels from 4 to 6
        p_level = 0.01 + (i / n_instances) * 0.09  # Vary cross-level from 0.01 to 0.1
        G, node_levels = generate_hierarchical_network(
            levels=levels,
            branching_factor=branching_factor,
            p_level=p_level,
            seed=seed
        )
        metadata = {
            'type': 'hierarchical',
            'instance': i,
            'seed': seed,
            'branching_factor': branching_factor,
            'levels': levels,
            'p_level': p_level,
            'params': {'branching_factor': branching_factor, 'levels': levels, 'p_level': p_level}
        }
        dataset['hierarchical'].append((G, metadata))
    
    # 8. Core-Periphery
    print(f"Generating {n_instances} Core-Periphery graphs...")
    dataset['core_periphery'] = []
    for i in range(n_instances):
        seed = base_seed + i
        core_size = int(n_nodes * (0.1 + (i / n_instances) * 0.2))  # Core 10-30% of nodes
        p_core = 0.5 + (i / n_instances) * 0.3  # Vary core density from 0.5 to 0.8
        p_periphery = 0.01 + (i / n_instances) * 0.04  # Vary periphery from 0.01 to 0.05
        p_core_periphery = 0.1 + (i / n_instances) * 0.2  # Vary connection from 0.1 to 0.3
        
        G = generate_core_periphery(
            n_core=core_size,
            n_periphery=n_nodes - core_size,
            p_core=p_core,
            p_periphery=p_periphery,
            p_core_periphery=p_core_periphery,
            seed=seed
        )
        metadata = {
            'type': 'core_periphery',
            'instance': i,
            'seed': seed,
            'n_nodes': n_nodes,
            'core_size': core_size,
            'p_core': p_core,
            'p_periphery': p_periphery,
            'p_core_periphery': p_core_periphery,
            'params': {
                'n_core': core_size,
                'n_periphery': n_nodes - core_size,
                'p_core': p_core,
                'p_periphery': p_periphery,
                'p_core_periphery': p_core_periphery
            }
        }
        dataset['core_periphery'].append((G, metadata))
    
    # 9. Bipartite Random
    print(f"Generating {n_instances} Bipartite Random graphs...")
    dataset['bipartite_random'] = []
    for i in range(n_instances):
        seed = base_seed + i
        n1 = n_nodes // 2 + (i % 20) - 10  # Vary split around 50/50
        n2 = n_nodes - n1
        p = 0.05 + (i / n_instances) * 0.15  # Vary edge probability from 0.05 to 0.2
        
        G = generate_bipartite_random(n1=n1, n2=n2, p=p, seed=seed)
        metadata = {
            'type': 'bipartite_random',
            'instance': i,
            'seed': seed,
            'n_nodes': n_nodes,
            'n1': n1,
            'n2': n2,
            'p': p,
            'params': {'n1': n1, 'n2': n2, 'p': p}
        }
        dataset['bipartite_random'].append((G, metadata))
    
    # 10. Random Regular / Expander-like
    print(f"Generating {n_instances} Random Regular graphs...")
    dataset['random_regular'] = []
    for i in range(n_instances):
        seed = base_seed + i
        d = 3 + (i % 8)  # Vary degree from 3 to 10
        # Ensure n*d is even
        if (n_nodes * d) % 2 != 0:
            d += 1
        G = generate_random_regular_expander_like(n_nodes, d=d, seed=seed)
        metadata = dict(G.graph)
        metadata['instance'] = i
        dataset['random_regular'].append((G, metadata))
    
    # 11. Heterophilic SBM
    print(f"Generating {n_instances} Heterophilic SBM graphs...")
    dataset['heterophilic_sbm'] = []
    for i in range(n_instances):
        seed = base_seed + i
        n_blocks = 2 + (i % 4)  # Vary blocks from 2 to 5
        target_avg_degree = 4.0 + (i / n_instances) * 8.0  # Vary from 4 to 12
        out_in_ratio = 1.0 + (i / n_instances) * 7.0  # Vary from 1.0 to 8.0
        G, labels = generate_heterophilic_sbm(
            n=n_nodes,
            n_blocks=n_blocks,
            target_avg_degree=target_avg_degree,
            out_in_ratio=out_in_ratio,
            seed=seed
        )
        metadata = dict(G.graph)
        metadata['instance'] = i
        dataset['heterophilic_sbm'].append((G, metadata))
    
    # 12. Degree-Corrected SBM
    print(f"Generating {n_instances} Degree-Corrected SBM graphs...")
    dataset['degree_corrected_sbm'] = []
    for i in range(n_instances):
        seed = base_seed + i
        n_blocks = 3 + (i % 3)  # Vary blocks from 3 to 5
        target_avg_degree = 4.0 + (i / n_instances) * 8.0  # Vary from 4 to 12
        out_in_ratio = 0.1 + (i / n_instances) * 3.9  # Vary from 0.1 to 4.0
        dist = 'powerlaw' if i % 2 == 0 else 'lognormal'
        G, labels = generate_degree_corrected_sbm(
            n=n_nodes,
            n_blocks=n_blocks,
            target_avg_degree=target_avg_degree,
            out_in_ratio=out_in_ratio,
            degree_distribution=dist,
            seed=seed
        )
        metadata = dict(G.graph)
        metadata['instance'] = i
        dataset['degree_corrected_sbm'].append((G, metadata))
    
    # 13. Grid/Torus Lattice
    print(f"Generating {n_instances} Grid/Torus Lattice graphs...")
    dataset['grid_torus'] = []
    for i in range(n_instances):
        seed = base_seed + i
        periodic = (i % 2 == 0)  # Alternate between grid and torus
        add_diagonals = (i % 3 == 0)  # Add diagonals every 3rd graph
        G = generate_grid_torus_lattice(
            n=n_nodes,
            periodic=periodic,
            add_diagonals=add_diagonals,
            seed=seed
        )
        metadata = dict(G.graph)
        metadata['instance'] = i
        dataset['grid_torus'].append((G, metadata))
    
    # 14. Configuration Model
    print(f"Generating {n_instances} Configuration Model graphs...")
    dataset['configuration_model'] = []
    for i in range(n_instances):
        seed = base_seed + i
        target_avg_degree = 4.0 + (i / n_instances) * 8.0  # Vary from 4 to 12
        if i % 3 == 0:
            dist = 'powerlaw'
            gamma = 2.2 + (i / n_instances) * 0.8  # Vary gamma from 2.2 to 3.0
        elif i % 3 == 1:
            dist = 'lognormal'
            gamma = 2.5  # Not used for lognormal
        else:
            dist = 'poisson'
            gamma = 2.5  # Not used for poisson
        
        G = generate_configuration_model_graph(
            n=n_nodes,
            distribution=dist,
            target_avg_degree=target_avg_degree,
            gamma=gamma,
            seed=seed
        )
        metadata = dict(G.graph)
        metadata['instance'] = i
        dataset['configuration_model'].append((G, metadata))
    
    # Save graphs if directory provided
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving graphs to {save_dir}...")
        for graph_type, graphs in dataset.items():
            type_dir = save_dir / graph_type
            type_dir.mkdir(exist_ok=True)
            
            for i, (G, metadata) in enumerate(graphs):
                # Save graph
                graph_file = type_dir / f"{graph_type}_{i:03d}.graphml"
                nx.write_graphml(G, graph_file)
                
                # Save metadata
                metadata_file = type_dir / f"{graph_type}_{i:03d}_metadata.json"
                import json
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("Dataset Generation Summary")
    print("="*60)
    total_graphs = 0
    for graph_type, graphs in dataset.items():
        print(f"{graph_type:25s}: {len(graphs):3d} graphs")
        total_graphs += len(graphs)
    print("-"*60)
    print(f"{'Total':25s}: {total_graphs:3d} graphs")
    print("="*60)
    
    return dataset


def load_comprehensive_dataset(load_dir: str) -> Dict[str, List[Tuple[nx.Graph, Dict]]]:
    """
    Load a previously saved comprehensive dataset.
    
    Parameters
    ----------
    load_dir : str
        Directory containing saved graphs
        
    Returns
    -------
    dict
        Dictionary mapping graph type names to lists of (graph, metadata) tuples
    """
    import json
    from pathlib import Path
    
    load_dir = Path(load_dir)
    dataset = {}
    
    # Find all graph type directories
    for type_dir in sorted(load_dir.iterdir()):
        if not type_dir.is_dir():
            continue
        
        graph_type = type_dir.name
        dataset[graph_type] = []
        
        # Load all graphs in this directory
        graph_files = sorted(type_dir.glob("*.graphml"))
        for graph_file in graph_files:
            # Load graph
            G = nx.read_graphml(graph_file)
            
            # Load metadata
            metadata_file = graph_file.with_suffix('').with_suffix('.json').with_name(
                graph_file.stem + '_metadata.json'
            )
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {'type': graph_type}
            
            dataset[graph_type].append((G, metadata))
    
    print(f"Loaded {sum(len(v) for v in dataset.values())} graphs from {load_dir}")
    return dataset

# ===========================================================================
# 1. Random Regular / Expander-like Graphs
# ===========================================================================

def generate_random_regular_expander_like(
    n: int,
    d: int,
    seed: Optional[int] = None,
    make_connected: bool = True,
    max_tries: int = 25,
) -> nx.Graph:
    """
    Generate a random d-regular graph. Random regular graphs are a practical
    expander-like family for QuVINE sweeps.

    Parameters
    ----------
    n : int
        Number of nodes.
    d : int
        Regular degree. Must satisfy 0 <= d < n and n*d even.
    seed : int, optional
        Random seed.
    make_connected : bool
        If True, retry until connected; if retries fail, connect components.
    max_tries : int
        Number of random draws before bridge-connecting components.
    """
    if d < 0 or d >= n:
        raise ValueError("d must satisfy 0 <= d < n")
    if (n * d) % 2 != 0:
        raise ValueError("n * d must be even for a d-regular graph")

    last_G = None
    for attempt in range(max_tries):
        s = None if seed is None else _stable_seed(seed, "rr", attempt)
        G = nx.random_regular_graph(d=d, n=n, seed=s)
        last_G = G
        if not make_connected or nx.is_connected(G):
            break
    else:
        G = _connect_components_by_bridges(last_G, seed=seed)

    G = _postprocess_graph(G, seed=seed, make_connected=False)
    metadata = {
        "type": "random_regular_expander_like",
        "n_nodes": int(n),
        "d": int(d),
        "target_avg_degree": float(d),
        "seed": seed,
        "params": {"n": int(n), "d": int(d)},
    }
    metadata.update(_graph_basic_metadata(G))
    return _add_metadata(G, metadata)


def sweep_random_regular_expander_like(
    n_values: Sequence[int] = (1000, 2000, 5000),
    d_values: Sequence[int] = (3, 6, 10, 20),
    seeds: Sequence[int] = (0, 1, 2),
    make_connected: bool = True,
) -> List[Tuple[nx.Graph, Dict[str, Any]]]:
    """Generate sweep of random regular graphs."""
    graphs = []
    for n, d, seed in product(n_values, d_values, seeds):
        if d >= n or (n * d) % 2 != 0:
            continue
        G = generate_random_regular_expander_like(n, d, seed=seed, make_connected=make_connected)
        graphs.append((G, dict(G.graph)))
    return graphs


# ===========================================================================
# 2. Heterophilic / Disassortative SBM
# ===========================================================================

def generate_heterophilic_sbm(
    n: int,
    n_blocks: int,
    target_avg_degree: float,
    out_in_ratio: float,
    seed: Optional[int] = None,
    make_connected: bool = True,
    selfloops: bool = False,
) -> Tuple[nx.Graph, Dict[int, int]]:
    """
    Generate an SBM where p_out / p_in is controlled. For out_in_ratio > 1,
    between-block edges are more likely than within-block edges, producing a
    heterophilic/disassortative block structure.
    """
    sizes = _balanced_block_sizes(n, n_blocks)
    P = _sbm_prob_matrix_from_out_in_ratio(
        sizes=sizes,
        target_avg_degree=target_avg_degree,
        out_in_ratio=out_in_ratio,
    )
    G = nx.stochastic_block_model(sizes, P.tolist(), seed=seed, selfloops=selfloops)
    G = _postprocess_graph(G, seed=seed, make_connected=make_connected)

    labels: Dict[int, int] = {}
    start = 0
    for b, size in enumerate(sizes):
        for node in range(start, start + size):
            if node in G:
                labels[node] = b
        start += size
    nx.set_node_attributes(G, labels, "block")

    metadata = {
        "type": "heterophilic_sbm",
        "n_nodes": int(n),
        "n_blocks": int(n_blocks),
        "target_avg_degree": float(target_avg_degree),
        "out_in_ratio": float(out_in_ratio),
        "p_in": float(np.diag(P).mean()),
        "p_out_mean": float((P.sum() - np.trace(P)) / max(P.size - len(P), 1)),
        "block_sizes": sizes,
        "seed": seed,
        "params": {
            "n": int(n),
            "n_blocks": int(n_blocks),
            "target_avg_degree": float(target_avg_degree),
            "out_in_ratio": float(out_in_ratio),
        },
    }
    metadata.update(_graph_basic_metadata(G))
    _add_metadata(G, metadata)
    return G, labels


def sweep_heterophilic_sbm(
    n_values: Sequence[int] = (1000, 2000, 5000),
    n_blocks_values: Sequence[int] = (2, 4, 8),
    avg_degree_values: Sequence[float] = (4, 8, 16),
    out_in_ratios: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
    seeds: Sequence[int] = (0, 1, 2),
    make_connected: bool = True,
) -> List[Tuple[nx.Graph, Dict[str, Any]]]:
    """Generate sweep of heterophilic SBM graphs."""
    graphs = []
    for n, b, avg_d, ratio, seed in product(
        n_values, n_blocks_values, avg_degree_values, out_in_ratios, seeds
    ):
        if b > n:
            continue
        G, _ = generate_heterophilic_sbm(
            n=n,
            n_blocks=b,
            target_avg_degree=avg_d,
            out_in_ratio=ratio,
            seed=seed,
            make_connected=make_connected,
        )
        graphs.append((G, dict(G.graph)))
    return graphs


# ===========================================================================
# 3. Degree-Corrected SBM
# ===========================================================================

def _sample_degree_weights(
    n: int,
    distribution: str,
    seed: Optional[int] = None,
    gamma: float = 2.5,
    lognormal_sigma: float = 1.0,
    max_weight_quantile: float = 0.995,
) -> np.ndarray:
    """Sample degree weights from specified distribution."""
    rng = _rng(seed)
    if distribution == "powerlaw":
        a = max(float(gamma) - 1.0, 0.2)
        w = rng.pareto(a=a, size=n) + 1.0
    elif distribution == "lognormal":
        w = rng.lognormal(mean=0.0, sigma=float(lognormal_sigma), size=n)
    elif distribution == "uniform":
        w = np.ones(n, dtype=float)
    else:
        raise ValueError("distribution must be 'powerlaw', 'lognormal', or 'uniform'")

    if n > 10:
        cap = np.quantile(w, max_weight_quantile)
        w = np.minimum(w, cap)
    w = np.maximum(w, 1e-12)
    return w


def generate_degree_corrected_sbm(
    n: int,
    n_blocks: int,
    target_avg_degree: float,
    out_in_ratio: float = 0.1,
    degree_distribution: str = "powerlaw",
    gamma: float = 2.5,
    lognormal_sigma: float = 1.0,
    seed: Optional[int] = None,
    max_prob: float = 0.95,
    make_connected: bool = True,
) -> Tuple[nx.Graph, Dict[int, int]]:
    """
    Generate a degree-corrected SBM.

    Edge probabilities are::

        P_ij = scale * R_{b_i,b_j} * theta_i * theta_j

    where theta values are normalized to have mean 1 within each block.
    """
    rng = _rng(seed)
    sizes = _balanced_block_sizes(n, n_blocks)
    blocks = np.empty(n, dtype=int)
    start = 0
    for b, size in enumerate(sizes):
        blocks[start:start + size] = b
        start += size

    theta = np.empty(n, dtype=float)
    start = 0
    for b, size in enumerate(sizes):
        s = None if seed is None else _stable_seed(seed, "theta", b)
        w = _sample_degree_weights(
            size,
            distribution=degree_distribution,
            seed=s,
            gamma=gamma,
            lognormal_sigma=lognormal_sigma,
        )
        theta[start:start + size] = w / w.mean()
        start += size

    R = np.full((n_blocks, n_blocks), float(out_in_ratio), dtype=float)
    np.fill_diagonal(R, 1.0)

    raw_expected = 0.0
    for a in range(n_blocks):
        idx_a = np.where(blocks == a)[0]
        ta = theta[idx_a]
        raw_expected += R[a, a] * (ta.sum() ** 2 - np.sum(ta ** 2)) / 2.0
        for b in range(a + 1, n_blocks):
            idx_b = np.where(blocks == b)[0]
            raw_expected += R[a, b] * ta.sum() * theta[idx_b].sum()

    desired_edges = n * float(target_avg_degree) / 2.0
    scale = desired_edges / max(raw_expected, 1e-12)

    G = nx.Graph()
    G.add_nodes_from(range(n))

    for a in range(n_blocks):
        idx_a = np.where(blocks == a)[0]
        ta = theta[idx_a]

        Paa = scale * R[a, a] * np.outer(ta, ta)
        Paa = np.clip(Paa, 0.0, max_prob)
        iu, ju = np.triu_indices(len(idx_a), k=1)
        mask = rng.random(len(iu)) < Paa[iu, ju]
        edges = [(int(idx_a[i]), int(idx_a[j])) for i, j in zip(iu[mask], ju[mask])]
        G.add_edges_from(edges)

        for b in range(a + 1, n_blocks):
            idx_b = np.where(blocks == b)[0]
            tb = theta[idx_b]
            Pab = scale * R[a, b] * np.outer(ta, tb)
            Pab = np.clip(Pab, 0.0, max_prob)
            coords = np.where(rng.random(Pab.shape) < Pab)
            edges = [(int(idx_a[i]), int(idx_b[j])) for i, j in zip(coords[0], coords[1])]
            G.add_edges_from(edges)

    G = _postprocess_graph(G, seed=seed, make_connected=make_connected)
    block_dict = {i: int(blocks[i]) for i in range(n)}
    theta_dict = {i: float(theta[i]) for i in range(n)}
    nx.set_node_attributes(G, block_dict, "block")
    nx.set_node_attributes(G, theta_dict, "degree_weight")

    metadata = {
        "type": "degree_corrected_sbm",
        "n_nodes": int(n),
        "n_blocks": int(n_blocks),
        "target_avg_degree": float(target_avg_degree),
        "out_in_ratio": float(out_in_ratio),
        "degree_distribution": degree_distribution,
        "gamma": float(gamma),
        "lognormal_sigma": float(lognormal_sigma),
        "scale": float(scale),
        "max_prob": float(max_prob),
        "block_sizes": sizes,
        "seed": seed,
        "params": {
            "n": int(n),
            "n_blocks": int(n_blocks),
            "target_avg_degree": float(target_avg_degree),
            "out_in_ratio": float(out_in_ratio),
            "degree_distribution": degree_distribution,
        },
    }
    metadata.update(_graph_basic_metadata(G))
    _add_metadata(G, metadata)
    return G, block_dict


def sweep_degree_corrected_sbm(
    n_values: Sequence[int] = (1000, 2000, 5000),
    n_blocks_values: Sequence[int] = (4, 8),
    avg_degree_values: Sequence[float] = (4, 8, 16),
    out_in_ratios: Sequence[float] = (0.1, 0.5, 1.0, 2.0, 4.0),
    degree_distributions: Sequence[str] = ("powerlaw", "lognormal"),
    seeds: Sequence[int] = (0, 1, 2),
    make_connected: bool = True,
) -> List[Tuple[nx.Graph, Dict[str, Any]]]:
    """Generate sweep of degree-corrected SBM graphs."""
    graphs = []
    for n, b, avg_d, ratio, dist, seed in product(
        n_values,
        n_blocks_values,
        avg_degree_values,
        out_in_ratios,
        degree_distributions,
        seeds,
    ):
        if b > n:
            continue
        G, _ = generate_degree_corrected_sbm(
            n=n,
            n_blocks=b,
            target_avg_degree=avg_d,
            out_in_ratio=ratio,
            degree_distribution=dist,
            seed=seed,
            make_connected=make_connected,
        )
        graphs.append((G, dict(G.graph)))
    return graphs


# ===========================================================================
# 4. Grid / Torus Lattice
# ===========================================================================

def _side_lengths_for_n(n: int, dim: int) -> Tuple[int, ...]:
    """Calculate side lengths for grid with approximately n nodes."""
    side = int(round(n ** (1.0 / dim)))
    side = max(side, 2)
    return tuple([side] * dim)


def generate_grid_torus_lattice(
    n: Optional[int] = None,
    side_lengths: Optional[Sequence[int]] = None,
    dim: int = 2,
    periodic: bool = True,
    add_diagonals: bool = False,
    seed: Optional[int] = None,
) -> nx.Graph:
    """
    Generate a regular grid or torus lattice. If periodic=True, this is a torus.

    Parameters
    ----------
    n : int, optional
        Approximate target number of nodes. Ignored if side_lengths is provided.
    side_lengths : sequence of int, optional
        Grid shape, e.g. (50, 50) for 2500 nodes.
    dim : int
        Dimension used when side_lengths is not provided.
    periodic : bool
        If True, use periodic boundary conditions.
    add_diagonals : bool
        For 2D grids only, add diagonal lattice edges to increase local cycles.
    """
    if side_lengths is None:
        if n is None:
            raise ValueError("Specify either n or side_lengths")
        side_lengths = _side_lengths_for_n(n, dim=dim)
    side_lengths = tuple(int(x) for x in side_lengths)

    G = nx.grid_graph(dim=list(side_lengths), periodic=periodic)

    if add_diagonals and len(side_lengths) == 2:
        m, k = side_lengths
        for i in range(m):
            for j in range(k):
                u = (i, j)
                candidates = [
                    ((i + 1) % m, (j + 1) % k),
                    ((i + 1) % m, (j - 1) % k),
                ] if periodic else [
                    (i + 1, j + 1),
                    (i + 1, j - 1),
                ]
                for v in candidates:
                    if 0 <= v[0] < m and 0 <= v[1] < k:
                        G.add_edge(u, v)

    # Relabel to integers and preserve positions
    old_nodes = list(G.nodes())
    mapping = {node: idx for idx, node in enumerate(old_nodes)}
    pos_attrs = {mapping[node]: tuple(float(x) for x in node) for node in old_nodes}
    G = nx.relabel_nodes(G, mapping)
    nx.set_node_attributes(G, pos_attrs, "pos")

    G = _postprocess_graph(G, seed=seed, make_connected=False)
    metadata = {
        "type": "grid_torus_lattice" if periodic else "grid_lattice",
        "side_lengths": side_lengths,
        "dim": int(len(side_lengths)),
        "periodic": bool(periodic),
        "add_diagonals": bool(add_diagonals),
        "n_nodes_requested": n,
        "seed": seed,
        "params": {
            "side_lengths": side_lengths,
            "periodic": bool(periodic),
            "add_diagonals": bool(add_diagonals),
        },
    }
    metadata.update(_graph_basic_metadata(G))
    return _add_metadata(G, metadata)


def sweep_grid_torus_lattice(
    n_values: Sequence[int] = (1024, 2025, 4900),
    dims: Sequence[int] = (2,),
    periodic_values: Sequence[bool] = (False, True),
    diagonal_values: Sequence[bool] = (False, True),
    seeds: Sequence[int] = (0,),
) -> List[Tuple[nx.Graph, Dict[str, Any]]]:
    """Generate sweep of grid/torus lattice graphs."""
    graphs = []
    for n, dim, periodic, diag, seed in product(n_values, dims, periodic_values, diagonal_values, seeds):
        if dim != 2 and diag:
            continue
        G = generate_grid_torus_lattice(n=n, dim=dim, periodic=periodic, add_diagonals=diag, seed=seed)
        graphs.append((G, dict(G.graph)))
    return graphs


# ===========================================================================
# 5. Configuration Model
# ===========================================================================

def sample_degree_sequence(
    n: int,
    distribution: str,
    target_avg_degree: float,
    seed: Optional[int] = None,
    gamma: float = 2.5,
    lognormal_sigma: float = 1.0,
    max_degree_fraction: float = 0.1,
) -> np.ndarray:
    """Sample a degree sequence and rescale it to target average degree."""
    rng = _rng(seed)
    if distribution == "powerlaw":
        a = max(float(gamma) - 1.0, 0.2)
        raw = rng.pareto(a=a, size=n) + 1.0
    elif distribution == "lognormal":
        raw = rng.lognormal(mean=0.0, sigma=float(lognormal_sigma), size=n)
    elif distribution == "poisson":
        raw = rng.poisson(lam=float(target_avg_degree), size=n) + 1
    else:
        raise ValueError("distribution must be 'powerlaw', 'lognormal', or 'poisson'")

    deg = _rescale_degrees_to_target(raw, target_avg_degree=target_avg_degree, n=n)
    max_degree = max(1, int(max_degree_fraction * (n - 1)))
    deg = np.clip(deg, 1, max_degree)
    if deg.sum() % 2 == 1:
        idx = int(np.argmax(deg < n - 1))
        deg[idx] += 1
    return deg.astype(int)


def generate_configuration_model_graph(
    n: int,
    distribution: str = "powerlaw",
    target_avg_degree: float = 8,
    seed: Optional[int] = None,
    gamma: float = 2.5,
    lognormal_sigma: float = 1.0,
    max_degree_fraction: float = 0.1,
    make_connected: bool = True,
) -> nx.Graph:
    """
    Generate a simple graph from a configuration model with power-law,
    log-normal, or Poisson degree sequence.
    """
    deg_seq = sample_degree_sequence(
        n=n,
        distribution=distribution,
        target_avg_degree=target_avg_degree,
        seed=seed,
        gamma=gamma,
        lognormal_sigma=lognormal_sigma,
        max_degree_fraction=max_degree_fraction,
    )

    MG = nx.configuration_model(deg_seq, seed=seed)
    G = nx.Graph(MG)
    G.remove_edges_from(nx.selfloop_edges(G))
    G = _postprocess_graph(G, seed=seed, make_connected=make_connected)

    nx.set_node_attributes(G, {i: int(deg_seq[i]) for i in range(n)}, "target_degree")

    metadata = {
        "type": "configuration_model",
        "degree_distribution": distribution,
        "n_nodes": int(n),
        "target_avg_degree": float(target_avg_degree),
        "target_degree_mean": float(deg_seq.mean()),
        "target_degree_max": int(deg_seq.max()) if len(deg_seq) else 0,
        "gamma": float(gamma),
        "lognormal_sigma": float(lognormal_sigma),
        "max_degree_fraction": float(max_degree_fraction),
        "seed": seed,
        "params": {
            "n": int(n),
            "distribution": distribution,
            "target_avg_degree": float(target_avg_degree),
            "gamma": float(gamma),
            "lognormal_sigma": float(lognormal_sigma),
        },
    }
    metadata.update(_graph_basic_metadata(G))
    return _add_metadata(G, metadata)


def sweep_configuration_model_graphs(
    n_values: Sequence[int] = (1000, 2000, 5000),
    distributions: Sequence[str] = ("powerlaw", "lognormal"),
    avg_degree_values: Sequence[float] = (4, 8, 16),
    gamma_values: Sequence[float] = (2.2, 2.5, 3.0),
    lognormal_sigma_values: Sequence[float] = (0.75, 1.0, 1.5),
    seeds: Sequence[int] = (0, 1, 2),
    make_connected: bool = True,
) -> List[Tuple[nx.Graph, Dict[str, Any]]]:
    """Generate sweep of configuration model graphs."""
    graphs = []
    for n, dist, avg_d, seed in product(n_values, distributions, avg_degree_values, seeds):
        if dist == "powerlaw":
            for gamma in gamma_values:
                G = generate_configuration_model_graph(
                    n=n,
                    distribution=dist,
                    target_avg_degree=avg_d,
                    gamma=gamma,
                    seed=_stable_seed(seed, dist, avg_d, gamma),
                    make_connected=make_connected,
                )
                graphs.append((G, dict(G.graph)))
        elif dist == "lognormal":
            for sigma in lognormal_sigma_values:
                G = generate_configuration_model_graph(
                    n=n,
                    distribution=dist,
                    target_avg_degree=avg_d,
                    lognormal_sigma=sigma,
                    seed=_stable_seed(seed, dist, avg_d, sigma),
                    make_connected=make_connected,
                )
                graphs.append((G, dict(G.graph)))
        else:
            G = generate_configuration_model_graph(
                n=n,
                distribution=dist,
                target_avg_degree=avg_d,
                seed=seed,
                make_connected=make_connected,
            )
            graphs.append((G, dict(G.graph)))
    return graphs
    
# ===========================================================================
# EXTENDED GENERATORS FOR QUVINE
# ===========================================================================
# Added: Random regular, heterophilic SBM, degree-corrected SBM,
# grid/torus lattices, and configuration model graphs

from dataclasses import dataclass
from itertools import product
from typing import Any, Sequence

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


# Note: Due to length constraints, the full extended generators code
# has been provided in the task description. The complete implementation
# includes 5 new generator families with sweep functions.
# 
# To integrate: append the complete extended generator code from the task
# to this file after line 962.

