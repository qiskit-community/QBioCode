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
Synthetic Graph Generator for QuVINE

Pre-generates and saves synthetic graphs to ensure all methods use identical instances.
"""

import json
import logging
import pickle
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
import networkx as nx
import numpy as np
import pandas as pd
import torch
from typing import TYPE_CHECKING, Callable, cast

if TYPE_CHECKING:
    pyg_from_networkx_type = Any
else:
    pyg_from_networkx_type = None

logger = logging.getLogger(__name__)

# PyTorch Geometric export is optional: every graph is always written as a
# pickle, and the .pt companion file is an extra for GNN baselines. Deliberately
# broad -- torch_geometric raises OSError or RuntimeError from its native
# extensions when it is installed against a mismatched torch, not just
# ImportError. The reason is kept and logged so a missing graph.pt is traceable
# instead of silent.
try:
    import importlib
    _pyg_utils = importlib.import_module("torch_geometric.utils")
    pyg_from_networkx = getattr(_pyg_utils, "from_networkx")
    HAS_TORCH_GEOMETRIC = True
    _PYG_IMPORT_ERROR = None
except Exception as exc:
    pyg_from_networkx = None
    HAS_TORCH_GEOMETRIC = False
    _PYG_IMPORT_ERROR = str(exc)
    logger.debug(
        "torch_geometric unavailable (%s); graphs will be saved as .pkl only. "
        'Install it with: pip install "qbiocode[quvine]"',
        _PYG_IMPORT_ERROR,
    )

from .seed_manager import SeedManager
from .dataset_registry import DatasetRegistry, DatasetType, DatasetMetadata


def _json_safe(value: Any) -> Any:
    """Recursively convert values into JSON-serializable Python primitives."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return value


def _largest_component_subgraph(G: nx.Graph) -> nx.Graph:
    """Return largest connected component subgraph copy."""
    if G.number_of_nodes() == 0:
        return G.copy()
    component_nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(component_nodes).copy()


class SyntheticGraphGenerator:
    """
    Generates and saves synthetic graphs for reproducible benchmarking.
    
    All synthetic graphs are pre-generated once and saved to disk.
    Methods then load these pre-generated graphs instead of generating their own.
    """
    
    # Define all synthetic network families
    SYNTHETIC_FAMILIES = [
        "configuration_model",
        "core_periphery",
        "degree_corrected_sbm",
        "erdos_renyi",
        "grid_torus",
        "heterophilic_sbm",
        "modular_medium",
        "modular_strong",
        "powerlaw_cluster",
        "random_geometric",
        "random_regular",
        "scale_free",
        "stochastic_block_model",
        "watts_strogatz_high_p",
        "watts_strogatz_low_p",
    ]
    
    def __init__(
        self,
        output_dir: Path,
        seed_manager: SeedManager,
        registry: DatasetRegistry
    ):
        """
        Initialize graph generator.
        
        Parameters
        ----------
        output_dir : Path
            Root directory for saving generated graphs
        seed_manager : SeedManager
            Seed manager for reproducible generation
        registry : DatasetRegistry
            Dataset registry to register generated graphs
        """
        self.output_dir = Path(output_dir)
        self.seed_manager = seed_manager
        self.registry = registry
        
        # Import graph generation functions
        from qbiocode.apps.quvine.data.random_graphs import (
            generate_modular_network,
            generate_core_periphery,
            generate_watts_strogatz,
            generate_erdos_renyi,
            generate_powerlaw_cluster,
            generate_random_geometric,
        )
        from qbiocode.apps.quvine.data.random_graphs_extended import (
            generate_degree_corrected_sbm,
            generate_heterophilic_sbm,
        )
        
        self.generators = {
            "modular_strong": self._gen_modular_strong,
            "modular_medium": self._gen_modular_medium,
            "core_periphery": self._gen_core_periphery,
            "watts_strogatz_low_p": self._gen_ws_low_p,
            "watts_strogatz_high_p": self._gen_ws_high_p,
            "erdos_renyi": self._gen_erdos_renyi,
            "powerlaw_cluster": self._gen_powerlaw_cluster,
            "random_geometric": self._gen_random_geometric,
            "scale_free": self._gen_scale_free,
            "stochastic_block_model": self._gen_sbm,
            "configuration_model": self._gen_configuration_model,
            "random_regular": self._gen_random_regular,
            "grid_torus": self._gen_grid_torus,
            "degree_corrected_sbm": self._gen_degree_corrected_sbm,
            "heterophilic_sbm": self._gen_heterophilic_sbm,
        }
    
    def generate_all(
        self,
        n_nodes_list: List[int] = [500, 2000, 5000],
        n_replicates: int = 30
    ) -> None:
        """
        Generate all synthetic graphs for all families, sizes, and replicates.
        
        Parameters
        ----------
        n_nodes_list : List[int]
            List of node counts to generate
        n_replicates : int
            Number of replicates per family-size combination
        """
        print(f"Generating synthetic graphs...")
        print(f"  Families: {len(self.SYNTHETIC_FAMILIES)}")
        print(f"  Sizes: {n_nodes_list}")
        print(f"  Replicates: {n_replicates}")
        print(f"  Total graphs: {len(self.SYNTHETIC_FAMILIES) * len(n_nodes_list) * n_replicates}")
        print()
        
        for family in self.SYNTHETIC_FAMILIES:
            print(f"Family: {family}")
            for n_nodes in n_nodes_list:
                print(f"  Size: {n_nodes} nodes")
                for rep in range(n_replicates):
                    self.generate_single(family, n_nodes, rep)
                    if (rep + 1) % 10 == 0:
                        print(f"    Generated {rep + 1}/{n_replicates} replicates")
            print()
    
    def generate_single(
        self,
        family: str,
        n_nodes: int,
        repetition_id: int
    ) -> Tuple[nx.Graph, Path]:
        """
        Generate a single synthetic graph instance.
        
        Parameters
        ----------
        family : str
            Graph family name
        n_nodes : int
            Number of nodes
        repetition_id : int
            Repetition index
        
        Returns
        -------
        G : nx.Graph
            Generated graph
        graph_path : Path
            Path where graph was saved
        """
        # Get canonical seed
        seed = self.seed_manager.get_seed(family, repetition_id)
        
        # Generate graph
        if family not in self.generators:
            raise ValueError(f"Unknown graph family: {family}")
        
        G, gen_params = self.generators[family](n_nodes, seed)
        
        # Create output directory — include n_nodes so that different sizes don't overwrite each other.
        dataset_dir = self.output_dir / family / f"n{n_nodes}" / f"rep_{repetition_id:02d}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Save graph as pickle (handles all data types without issues)
        graph_path = dataset_dir / "graph.pkl"
        with open(graph_path, 'wb') as f:
            pickle.dump(G, f)
        
        # Also save as PyG format for GNN methods (if available)
        if HAS_TORCH_GEOMETRIC and pyg_from_networkx is not None:
            pyg_path = dataset_dir / "graph.pt"
            pyg_data = cast(Callable[[nx.Graph], Any], pyg_from_networkx)(G)
            torch.save(pyg_data, pyg_path)
        else:
            logger.debug(
                "Skipping PyG export for %s: torch_geometric unavailable (%s)",
                dataset_dir.name, _PYG_IMPORT_ERROR,
            )
        
        # Save metadata
        metadata = DatasetMetadata(
            name=f"{family}_n{n_nodes}_rep{repetition_id:02d}",
            dataset_type=DatasetType.SYNTHETIC,
            num_nodes=G.number_of_nodes(),
            num_edges=G.number_of_edges(),
            is_directed=G.is_directed(),
            has_node_features=False,
            has_node_labels=False,
            generation_params=gen_params,
            description=f"Synthetic {family} graph with {n_nodes} nodes"
        )
        
        metadata_path = dataset_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                "name": metadata.name,
                "dataset_type": metadata.dataset_type.value,
                "num_nodes": metadata.num_nodes,
                "num_edges": metadata.num_edges,
                "is_directed": metadata.is_directed,
                "has_node_features": metadata.has_node_features,
                "has_node_labels": metadata.has_node_labels,
                "generation_params": metadata.generation_params,
                "description": metadata.description,
                "seed": seed,
                "repetition_id": repetition_id
            }, f, indent=2)
        
        # Register in dataset registry
        self.registry.register(
            dataset_name=family,
            dataset_type=DatasetType.SYNTHETIC,
            repetition_id=repetition_id,
            seed=seed,
            graph_path=graph_path,
            metadata_path=metadata_path,
            available_tasks=[]  # Tasks will be added when splits are generated
        )
        
        return G, graph_path
    
    # ========================================================================
    # Graph generation functions
    # ========================================================================
    
    def _gen_modular_strong(self, n_nodes: int, seed: int) -> Tuple[nx.Graph, Dict]:
        """Generate strongly modular network."""
        from qbiocode.apps.quvine.data.random_graphs import generate_modular_network
        num_communities = 5
        nodes_per_community = n_nodes // num_communities
        p_intra, p_inter = 0.4, 0.01
        G, communities = generate_modular_network(
            num_communities=num_communities,
            nodes_per_community=nodes_per_community,
            p_intra=p_intra,
            p_inter=p_inter,
            seed=seed
        )
        params = {
            "num_communities": num_communities,
            "nodes_per_community": nodes_per_community,
            "p_intra": p_intra,
            "p_inter": p_inter
        }
        return G, params
    
    def _gen_modular_medium(self, n_nodes: int, seed: int) -> Tuple[nx.Graph, Dict]:
        """Generate medium modularity network."""
        from qbiocode.apps.quvine.data.random_graphs import generate_modular_network
        num_communities = 5
        nodes_per_community = n_nodes // num_communities
        p_intra, p_inter = 0.25, 0.05
        G, communities = generate_modular_network(
            num_communities=num_communities,
            nodes_per_community=nodes_per_community,
            p_intra=p_intra,
            p_inter=p_inter,
            seed=seed
        )
        params = {
            "num_communities": num_communities,
            "nodes_per_community": nodes_per_community,
            "p_intra": p_intra,
            "p_inter": p_inter
        }
        return G, params
    
    def _gen_core_periphery(self, n_nodes: int, seed: int) -> Tuple[nx.Graph, Dict]:
        """Generate core-periphery network."""
        from qbiocode.apps.quvine.data.random_graphs import generate_core_periphery
        n_core = n_nodes // 8
        n_periphery = 7 * n_nodes // 8
        p_core, p_core_periphery = 0.5, 0.05
        G, core_nodes, periphery_nodes = generate_core_periphery(
            n_core=n_core,
            n_periphery=n_periphery,
            p_core=p_core,
            p_core_periphery=p_core_periphery,
            seed=seed
        )
        params = {
            "n_core": n_core,
            "n_periphery": n_periphery,
            "p_core": p_core,
            "p_core_periphery": p_core_periphery
        }
        return G, params
    
    def _gen_ws_low_p(self, n_nodes: int, seed: int) -> Tuple[nx.Graph, Dict]:
        """Generate Watts-Strogatz with low rewiring probability."""
        from qbiocode.apps.quvine.data.random_graphs import generate_watts_strogatz
        k, p = 6, 0.05
        G = generate_watts_strogatz(n=n_nodes, k=k, p=p, seed=seed)
        return G, {"k": k, "p": p}
    
    def _gen_ws_high_p(self, n_nodes: int, seed: int) -> Tuple[nx.Graph, Dict]:
        """Generate Watts-Strogatz with high rewiring probability."""
        from qbiocode.apps.quvine.data.random_graphs import generate_watts_strogatz
        k, p = 6, 0.5
        G = generate_watts_strogatz(n=n_nodes, k=k, p=p, seed=seed)
        return G, {"k": k, "p": p}
    
    def _gen_erdos_renyi(self, n_nodes: int, seed: int) -> Tuple[nx.Graph, Dict]:
        """Generate Erdős-Rényi random graph."""
        from qbiocode.apps.quvine.data.random_graphs import generate_erdos_renyi
        p = 0.01
        G = generate_erdos_renyi(n=n_nodes, p=p, seed=seed)
        return G, {"p": p}
    
    def _gen_powerlaw_cluster(self, n_nodes: int, seed: int) -> Tuple[nx.Graph, Dict]:
        """Generate powerlaw cluster graph."""
        from qbiocode.apps.quvine.data.random_graphs import generate_powerlaw_cluster
        m, p = 3, 0.5
        G = generate_powerlaw_cluster(n=n_nodes, m=m, p=p, seed=seed)
        return G, {"m": m, "p": p}
    
    def _gen_random_geometric(self, n_nodes: int, seed: int) -> Tuple[nx.Graph, Dict]:
        """Generate random geometric graph."""
        from qbiocode.apps.quvine.data.random_graphs import generate_random_geometric
        radius = 0.14
        G = generate_random_geometric(n=n_nodes, radius=radius, seed=seed)
        return G, {"radius": radius}
    
    def _gen_scale_free(self, n_nodes: int, seed: int) -> Tuple[nx.Graph, Dict]:
        """Generate scale-free network (Barabási-Albert)."""
        m = 3
        G = nx.barabasi_albert_graph(n=n_nodes, m=m, seed=seed)
        return G, {"m": m}
    
    def _gen_sbm(self, n_nodes: int, seed: int) -> Tuple[nx.Graph, Dict]:
        """Generate stochastic block model."""
        rng = np.random.default_rng(seed)
        n_blocks = 5
        p_in, p_out = 0.25, 0.01
        block_size = n_nodes // n_blocks
        sizes = [block_size] * n_blocks
        p_matrix = [[p_in if i == j else p_out for j in range(n_blocks)] for i in range(n_blocks)]
        G = nx.stochastic_block_model(sizes, p_matrix, seed=int(rng.integers(1 << 31)))
        G.remove_edges_from(nx.selfloop_edges(G))
        params = {"n_blocks": n_blocks, "p_in": p_in, "p_out": p_out, "block_size": block_size}
        return G, params
    
    def _gen_configuration_model(self, n_nodes: int, seed: int) -> Tuple[nx.Graph, Dict]:
        """Generate configuration model with power-law degree distribution."""
        rng = np.random.default_rng(seed)
        gamma = 2.5
        degrees = (rng.pareto(gamma - 1, size=n_nodes) + 1).astype(int)
        degrees = np.clip(degrees, 1, n_nodes - 1)
        if degrees.sum() % 2 != 0:
            degrees[0] += 1
        G = nx.configuration_model(degrees, seed=int(rng.integers(1 << 31)))
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        return G, {"gamma": gamma}
    
    def _gen_random_regular(self, n_nodes: int, seed: int) -> Tuple[nx.Graph, Dict]:
        """Generate random regular graph."""
        d = 6
        G = nx.random_regular_graph(d=d, n=n_nodes, seed=seed)
        return G, {"d": d}
    
    def _gen_grid_torus(self, n_nodes: int, seed: int) -> Tuple[nx.Graph, Dict]:
        """Generate 2D grid torus."""
        side = int(np.sqrt(n_nodes))
        G = nx.grid_2d_graph(side, side, periodic=True)
        G = nx.convert_node_labels_to_integers(G)
        return G, {"side": side}
    
    def _gen_degree_corrected_sbm(self, n_nodes: int, seed: int) -> Tuple[nx.Graph, Dict]:
        """Generate degree-corrected stochastic block model."""
        try:
            from qbiocode.apps.quvine.data.random_graphs_extended import generate_degree_corrected_sbm
            n_blocks = 5
            target_avg_degree = 10
            G, blocks = generate_degree_corrected_sbm(n=n_nodes, n_blocks=n_blocks, target_avg_degree=target_avg_degree, seed=seed)
            params = {"n_blocks": n_blocks, "target_avg_degree": target_avg_degree}
            return G, params
        except (ImportError, TypeError, AttributeError):
            # Fallback to regular SBM if extended version not available
            return self._gen_sbm(n_nodes, seed)
    
    def _gen_heterophilic_sbm(self, n_nodes: int, seed: int) -> Tuple[nx.Graph, Dict]:
        """Generate heterophilic stochastic block model."""
        try:
            from qbiocode.apps.quvine.data.random_graphs_extended import generate_heterophilic_sbm
            n_blocks = 5
            target_avg_degree = 10
            out_in_ratio = 3.0
            G, blocks = generate_heterophilic_sbm(n=n_nodes, n_blocks=n_blocks, target_avg_degree=target_avg_degree, out_in_ratio=out_in_ratio, seed=seed)
            params = {"n_blocks": n_blocks, "target_avg_degree": target_avg_degree, "out_in_ratio": out_in_ratio}
            return G, params
        except (ImportError, TypeError, AttributeError):
            # Fallback to regular SBM if extended version not available
            return self._gen_sbm(n_nodes, seed)
    
    def load_graph(self, graph_path: Path) -> nx.Graph:
        """
        Load a pre-generated graph from disk.
        
        Parameters
        ----------
        graph_path : Path
            Path to graph file (.graphml or .pkl)
        
        Returns
        -------
        nx.Graph
            Loaded graph
        """
        graph_path = Path(graph_path)
        
        # Try GraphML first (preferred format)
        if graph_path.suffix == '.graphml' or graph_path.with_suffix('.graphml').exists():
            graphml_path = graph_path if graph_path.suffix == '.graphml' else graph_path.with_suffix('.graphml')
            return nx.read_graphml(graphml_path)
        
        # Fall back to pickle for backward compatibility
        pkl_path = graph_path if graph_path.suffix == '.pkl' else graph_path.with_suffix('.pkl')
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                return pickle.load(f)
        
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

class PPIGraphGenerator:
    """
    Centralized PPI preprocessing pipeline.

    Generates fixed disease-specific benchmark graphs per:
    - PPI source
    - disease
    - requested size
    - repetition

    Outputs:
    - graph artifact
    - metadata artifact
    - disease node artifact
    - registry records
    """

    def __init__(
        self,
        output_dir: Path,
        seed_manager: SeedManager,
        registry: DatasetRegistry,
        processed_data_dir: Path,
        registry_output_path: Optional[Path] = None,
    ):
        self.output_dir = Path(output_dir)
        self.seed_manager = seed_manager
        self.registry = registry
        self.processed_data_dir = Path(processed_data_dir)
        self.registry_output_path = registry_output_path

    def _load_ppi_graph(self, edge_path: Path) -> nx.Graph:
        df = pd.read_csv(edge_path, dtype={"node1": str, "node2": str})[["node1", "node2"]]
        G = nx.from_pandas_edgelist(df, source="node1", target="node2")
        G.remove_edges_from(nx.selfloop_edges(G))
        return nx.Graph(G)

    def _load_disease_nodes(self, disease: str) -> Dict[str, Any]:
        seed_path = self.processed_data_dir / "gene_seeds" / f"{disease}_ncbi_seeds.json"
        target_path = self.processed_data_dir / "gwas_catalog_targets" / f"{disease}_targets_gene2ncbi.json"

        with open(seed_path, "r") as f:
            seed_data = json.load(f)
        with open(target_path, "r") as f:
            target_data = json.load(f)

        seed_gene_ids = [str(x) for x in seed_data]
        target_gene_names = sorted(target_data.keys())
        target_gene_ids = [str(v) for v in target_data.values()]

        return {
            "seed_gene_ids": seed_gene_ids,
            "target_gene_names": target_gene_names,
            "target_gene_ids": target_gene_ids,
            "seed_path": seed_path,
            "target_path": target_path,
        }

    def _normalize_edges(self, edges: List[Tuple[Any, Any]]) -> List[List[str]]:
        return [[str(u), str(v)] for u, v in edges]

    def _compute_shortest_path_connectors(
        self,
        G_lcc: nx.Graph,
        selected_nodes: Set[str],
        anchor_node: str,
    ) -> Tuple[Set[str], List[List[str]]]:
        connectors: Set[str] = set()
        connector_paths: List[List[str]] = []

        for node in sorted(selected_nodes):
            if node == anchor_node:
                continue
            path = [str(x) for x in nx.shortest_path(G_lcc, source=anchor_node, target=node)]
            connector_paths.append(path)
            connectors.update(path)

        return connectors - selected_nodes, connector_paths

    def _repair_selection_to_connected(
        self,
        G_lcc: nx.Graph,
        preserved_nodes: Set[str],
        filler_nodes: Set[str],
        target_size: int,
        allow_size_expansion: bool,
    ) -> Dict[str, Any]:
        selected_nodes = {str(node) for node in preserved_nodes} | {str(node) for node in filler_nodes}
        if not selected_nodes:
            raise ValueError("Cannot build PPI subgraph with zero selected nodes.")

        G_lcc_str = nx.relabel_nodes(G_lcc, {node: str(node) for node in G_lcc.nodes()}, copy=True)

        def _induced_is_connected(nodes: Set[str]) -> bool:
            if not nodes:
                return False
            sub = G_lcc_str.subgraph(sorted(nodes))
            return sub.number_of_nodes() > 0 and nx.is_connected(sub)

        anchor_node = sorted({str(node) for node in preserved_nodes})[0] if preserved_nodes else sorted(selected_nodes)[0]

        if _induced_is_connected(selected_nodes):
            return {
                "subgraph": G_lcc_str.subgraph(sorted(selected_nodes)).copy(),
                "connector_nodes": [],
                "connector_paths": [],
                "replaced_filler_nodes": [],
                "anchor_node": anchor_node,
            }

        selected_nodes = {str(node) for node in preserved_nodes}
        connector_paths: List[List[str]] = []
        connector_nodes: Set[str] = set()
        replaced_filler_nodes: List[str] = []

        filler_queue = deque(sorted(str(node) for node in filler_nodes))
        while filler_queue:
            candidate = filler_queue.popleft()
            path = [str(x) for x in nx.shortest_path(G_lcc_str, source=anchor_node, target=candidate)]
            path_nodes = set(path)
            proposed_nodes = set(selected_nodes) | path_nodes

            if len(proposed_nodes) <= target_size or allow_size_expansion:
                selected_nodes = proposed_nodes
                connector_paths.append(path)
                connector_nodes.update(path_nodes - {str(node) for node in preserved_nodes} - {candidate})
                continue

            if candidate in path_nodes:
                path_nodes_without_candidate = path_nodes - {candidate}
                proposed_without_candidate = set(selected_nodes) | path_nodes_without_candidate
                if len(proposed_without_candidate) <= target_size:
                    selected_nodes = proposed_without_candidate
                    connector_paths.append(path)
                    connector_nodes.update(path_nodes_without_candidate - {str(node) for node in preserved_nodes})
                    replaced_filler_nodes.append(candidate)
                    continue

            # Filler node cannot be connected within the size budget — skip it.
            # Preserved (disease) nodes are never dropped; connectivity is
            # guaranteed below by the force-connect fallback.
            continue

        # Force-connect any preserved-node components that no filler path bridged.
        # This can happen when all filler paths were too long to fit in target_size.
        # We allow the subgraph to grow slightly beyond target_size here rather than
        # producing a disconnected or incomplete graph.
        subgraph_check = G_lcc_str.subgraph(selected_nodes)
        if not nx.is_connected(subgraph_check):
            for comp in list(nx.connected_components(subgraph_check)):
                if anchor_node in comp:
                    continue
                bridge = next(iter(comp))
                conn_path = [str(x) for x in nx.shortest_path(G_lcc_str, source=anchor_node, target=bridge)]
                selected_nodes.update(conn_path)
                connector_nodes.update(set(conn_path) - {str(node) for node in preserved_nodes})
                connector_paths.append(conn_path)

        subgraph = G_lcc_str.subgraph(sorted(selected_nodes)).copy()
        if not nx.is_connected(subgraph):
            raise RuntimeError("Connectivity repair failed; final induced PPI subgraph is disconnected.")

        return {
            "subgraph": subgraph,
            "connector_nodes": sorted(node for node in connector_nodes if node in subgraph.nodes()),
            "connector_paths": connector_paths,
            "replaced_filler_nodes": replaced_filler_nodes,
            "anchor_node": anchor_node,
        }

    def _sample_filler_nodes(
        self,
        dataset_name: str,
        repetition_id: int,
        G_lcc: nx.Graph,
        preserved_nodes: Set[str],
        target_size: int,
    ) -> Set[str]:
        filler_seed = self.seed_manager.get_component_seed(
            dataset_name,
            repetition_id,
            "filler_node_selection",
        )
        rng = np.random.default_rng(filler_seed)
        candidate_fillers = sorted(str(node) for node in G_lcc.nodes() if str(node) not in preserved_nodes)
        n_fillers = max(0, target_size - len(preserved_nodes))
        if n_fillers == 0 or not candidate_fillers:
            return set()

        chosen = rng.choice(candidate_fillers, size=min(n_fillers, len(candidate_fillers)), replace=False)
        return {str(x) for x in chosen.tolist()}

    def _resolve_target_size(
        self,
        requested_size: int,
        lcc_size: int,
        preserved_count: int,
        auto_increase_size: bool,
    ) -> Dict[str, int]:
        adjusted_request = requested_size
        if preserved_count > adjusted_request:
            if auto_increase_size:
                adjusted_request = preserved_count
            else:
                raise ValueError(
                    f"Requested size {requested_size} is smaller than preserved disease node count {preserved_count}."
                )

        feasible_target_size = min(adjusted_request, lcc_size)
        if feasible_target_size < preserved_count:
            raise ValueError(
                f"LCC size {lcc_size} is smaller than preserved disease node count {preserved_count}."
            )

        return {
            "requested_size": requested_size,
            "adjusted_requested_size": adjusted_request,
            "feasible_target_size": feasible_target_size,
        }

    def _build_dataset_dir(
        self,
        ppi_source: str,
        disease: str,
        requested_size: int,
        repetition_id: int,
    ) -> Path:
        network_name = f"{ppi_source}_{disease}"
        return (
            self.output_dir
            / network_name
            / f"n{requested_size}"
            / f"rep_{repetition_id:02d}"
        )

    def _save_graph_artifacts(
        self,
        dataset_dir: Path,
        G_sub: nx.Graph,
    ) -> Path:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Save graph as pickle (handles all data types without issues)
        graph_path = dataset_dir / "graph.pkl"
        with open(graph_path, "wb") as f:
            pickle.dump(G_sub, f)

        if HAS_TORCH_GEOMETRIC and pyg_from_networkx is not None:
            pyg_path = dataset_dir / "graph.pt"
            pyg_data = cast(Callable[[nx.Graph], Any], pyg_from_networkx)(G_sub)
            torch.save(pyg_data, pyg_path)
        else:
            logger.debug(
                "Skipping PyG export for %s: torch_geometric unavailable (%s)",
                dataset_dir.name, _PYG_IMPORT_ERROR,
            )

        return graph_path

    def _register_dataset(
        self,
        dataset_name: str,
        repetition_id: int,
        canonical_seed: int,
        graph_path: Path,
        metadata_path: Path,
        disease_node_path: Path,
        ppi_source: str,
        disease: str,
        requested_size: int,
        actual_size: int,
    ) -> None:
        self.registry.register(
            dataset_name=dataset_name,
            dataset_type=DatasetType.PPI,
            repetition_id=repetition_id,
            seed=canonical_seed,
            graph_path=graph_path,
            metadata_path=metadata_path,
            available_tasks=[],
            disease_node_path=disease_node_path,
            ppi_source=ppi_source,
            disease=disease,
            requested_size=requested_size,
            actual_size=actual_size,
            registry_group="ppi",
        )

    def _write_registry_outputs(self) -> None:
        if self.registry_output_path is not None:
            self.registry.save(self.registry_output_path)

    def generate_single(
        self,
        ppi_source: str,
        disease: str,
        edge_path: Path,
        requested_size: int,
        repetition_id: int,
        allow_size_expansion: bool = False,
        auto_increase_size: bool = True,
    ) -> Tuple[nx.Graph, Path]:
        dataset_name = f"{ppi_source}_{disease}_n{requested_size}"
        canonical_seed = self.seed_manager.get_seed(dataset_name, repetition_id)

        G_full = self._load_ppi_graph(edge_path)
        original_num_nodes = G_full.number_of_nodes()
        original_num_edges = G_full.number_of_edges()
        G_lcc = _largest_component_subgraph(G_full)
        lcc_size = G_lcc.number_of_nodes()
        lcc_edges = G_lcc.number_of_edges()

        disease_info = self._load_disease_nodes(disease)
        seed_gene_ids = disease_info["seed_gene_ids"]
        target_gene_ids = disease_info["target_gene_ids"]
        target_gene_names = disease_info["target_gene_names"]

        lcc_nodes = {str(node) for node in G_lcc.nodes()}
        mapped_seed_nodes = sorted(node for node in seed_gene_ids if node in lcc_nodes)
        mapped_target_nodes = sorted(node for node in target_gene_ids if node in lcc_nodes)
        unmapped_seed_genes = sorted(node for node in seed_gene_ids if node not in lcc_nodes)
        unmapped_target_genes = sorted(node for node in target_gene_ids if node not in lcc_nodes)

        preserved_nodes = set(mapped_seed_nodes) | set(mapped_target_nodes)
        overlap = sorted(set(mapped_seed_nodes) & set(mapped_target_nodes))

        size_info = self._resolve_target_size(
            requested_size=requested_size,
            lcc_size=lcc_size,
            preserved_count=len(preserved_nodes),
            auto_increase_size=auto_increase_size,
        )
        feasible_target_size = size_info["feasible_target_size"]

        filler_nodes = self._sample_filler_nodes(
            dataset_name=dataset_name,
            repetition_id=repetition_id,
            G_lcc=G_lcc,
            preserved_nodes=preserved_nodes,
            target_size=feasible_target_size,
        )

        repaired = self._repair_selection_to_connected(
            G_lcc=G_lcc,
            preserved_nodes=preserved_nodes,
            filler_nodes=filler_nodes,
            target_size=feasible_target_size,
            allow_size_expansion=allow_size_expansion,
        )
        G_sub = repaired["subgraph"]

        actual_size = G_sub.number_of_nodes()
        final_nodes_sorted = sorted(str(node) for node in G_sub.nodes())
        final_node_set = set(final_nodes_sorted)
        retained_filler_nodes = sorted(node for node in filler_nodes if node in final_node_set)
        connector_nodes = repaired["connector_nodes"]
        connector_node_set = set(connector_nodes)

        if not set(mapped_seed_nodes).issubset(final_node_set):
            raise RuntimeError("Seed preservation failed during PPI preprocessing.")
        if not set(mapped_target_nodes).issubset(final_node_set):
            raise RuntimeError("Target preservation failed during PPI preprocessing.")

        node_id_mapping = {node_id: idx for idx, node_id in enumerate(final_nodes_sorted)}
        dataset_dir = self._build_dataset_dir(ppi_source, disease, requested_size, repetition_id)
        graph_path = self._save_graph_artifacts(dataset_dir, G_sub)

        metadata = {
            "dataset_name": dataset_name,
            "ppi_source": ppi_source,
            "disease": disease,
            "requested_graph_size": requested_size,
            "adjusted_requested_graph_size": size_info["adjusted_requested_size"],
            "actual_graph_size": actual_size,
            "maximum_feasible_graph_size": lcc_size,
            "repetition_id": repetition_id,
            "canonical_seed": canonical_seed,
            "original_graph_path": str(edge_path),
            "original_number_of_nodes": original_num_nodes,
            "original_number_of_edges": original_num_edges,
            "largest_connected_component_size": lcc_size,
            "largest_connected_component_edges": lcc_edges,
            "number_of_preserved_disease_seed_nodes": len(mapped_seed_nodes),
            "number_of_preserved_disease_target_nodes": len(mapped_target_nodes),
            "number_of_preserved_disease_nodes_total": len(preserved_nodes),
            "number_of_filler_nodes": len(retained_filler_nodes),
            "number_of_connector_nodes_added": len(connector_nodes),
            "connector_nodes_added": len(connector_nodes) > 0,
            "connector_nodes": connector_nodes,
            "number_of_replaced_filler_nodes": len(repaired["replaced_filler_nodes"]),
            "replaced_filler_nodes": repaired["replaced_filler_nodes"],
            "final_number_of_nodes": G_sub.number_of_nodes(),
            "final_number_of_edges": G_sub.number_of_edges(),
            "final_graph_connected": nx.is_connected(G_sub),
            "is_directed": G_sub.is_directed(),
            "node_id_mapping": node_id_mapping,
            "graph_generation_parameters": {
                "allow_size_expansion": allow_size_expansion,
                "auto_increase_size": auto_increase_size,
                "seed_policy": "canonical_seed = base_seed + repetition_id",
                "component_seed_plan": self.seed_manager.describe_seed_plan(dataset_name, repetition_id),
                "selection_anchor_node": repaired["anchor_node"],
                "connector_paths": repaired["connector_paths"],
            },
        }

        metadata_path = dataset_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(_json_safe(metadata), f, indent=2)

        disease_nodes = {
            "disease": disease,
            "seed_node_ids": mapped_seed_nodes,
            "target_node_ids": mapped_target_nodes,
            "seed_gene_names_or_ids": seed_gene_ids,
            "target_gene_names_or_ids": target_gene_names,
            "target_gene_ids": target_gene_ids,
            "unmapped_seed_genes": unmapped_seed_genes,
            "unmapped_target_genes": unmapped_target_genes,
            "seed_target_overlap": overlap,
            "preserved_disease_node_ids": sorted(preserved_nodes),
            "connector_node_ids": connector_nodes,
            "filler_node_ids": retained_filler_nodes,
        }

        disease_node_path = dataset_dir / "disease_nodes.json"
        with open(disease_node_path, "w") as f:
            json.dump(_json_safe(disease_nodes), f, indent=2)

        self._register_dataset(
            dataset_name=dataset_name,
            repetition_id=repetition_id,
            canonical_seed=canonical_seed,
            graph_path=graph_path,
            metadata_path=metadata_path,
            disease_node_path=disease_node_path,
            ppi_source=ppi_source,
            disease=disease,
            requested_size=requested_size,
            actual_size=actual_size,
        )
        self._write_registry_outputs()

        return G_sub, graph_path

    def generate_all(
        self,
        ppi_sources: Dict[str, Path],
        diseases: List[str],
        requested_sizes: List[int],
        n_repetitions: int,
        allow_size_expansion: bool = False,
        auto_increase_size: bool = True,
    ) -> None:
        for ppi_source, edge_path in ppi_sources.items():
            for disease in diseases:
                source_graph = self._load_ppi_graph(Path(edge_path))
                source_lcc = _largest_component_subgraph(source_graph)
                source_max_size = source_lcc.number_of_nodes()

                for requested_size in requested_sizes:
                    effective_size = min(requested_size, source_max_size)
                    for repetition_id in range(n_repetitions):
                        self.generate_single(
                            ppi_source=ppi_source,
                            disease=disease,
                            edge_path=Path(edge_path),
                            requested_size=effective_size,
                            repetition_id=repetition_id,
                            allow_size_expansion=allow_size_expansion,
                            auto_increase_size=auto_increase_size,
                        )
        self._write_registry_outputs()

