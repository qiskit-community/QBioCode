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
Method Runner for Reproducible Benchmarking

Provides a unified interface for running all 42 methods with pre-generated data.
"""

import time
import json
import pickle
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, Optional
import networkx as nx
import numpy as np

from .dataset_registry import DatasetRegistry
from .seed_manager import SeedManager


class MethodRunner:
    """
    Unified runner for all QuVINE methods using pre-generated data.
    
    Ensures all methods:
    1. Load the same pre-generated graph
    2. Use the same pre-generated split
    3. Use the same canonical seed
    4. Return results with full provenance
    """
    
    def __init__(
        self,
        registry: DatasetRegistry,
        seed_manager: SeedManager
    ):
        """
        Initialize method runner.
        
        Parameters
        ----------
        registry : DatasetRegistry
            Dataset registry
        seed_manager : SeedManager
            Seed manager
        """
        self.registry = registry
        self.seed_manager = seed_manager
    
    def load_graph(self, dataset_name: str, repetition_id: int) -> nx.Graph:
        """
        Load pre-generated graph from registry.
        
        Parameters
        ----------
        dataset_name : str
            Dataset name
        repetition_id : int
            Repetition ID
        
        Returns
        -------
        nx.Graph
            Pre-generated graph
        """
        entry = self.registry.get(dataset_name, repetition_id)
        if entry is None:
            raise ValueError(f"Dataset {dataset_name} rep {repetition_id} not found in registry")
        
        if not entry.graph_path.exists():
            raise FileNotFoundError(f"Graph file not found: {entry.graph_path}")
        
        with open(entry.graph_path, 'rb') as f:
            G = pickle.load(f)
        
        return G
    
    def load_split(self, dataset_name: str, repetition_id: int, task: str) -> Dict[str, Any]:
        """
        Load pre-generated split from registry.
        
        Parameters
        ----------
        dataset_name : str
            Dataset name
        repetition_id : int
            Repetition ID
        task : str
            Task name
        
        Returns
        -------
        dict
            Pre-generated split data
        """
        entry = self.registry.get(dataset_name, repetition_id)
        if entry is None:
            raise ValueError(f"Dataset {dataset_name} rep {repetition_id} not found in registry")
        
        if task not in entry.split_paths:
            raise ValueError(f"Task {task} not available for {dataset_name} rep {repetition_id}")
        
        split_path = entry.split_paths[task]
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        
        # Try pickle first (faster)
        pkl_path = split_path.with_suffix('.pkl')
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                return pickle.load(f)
        
        # Fall back to JSON
        with open(split_path, 'r') as f:
            return json.load(f)
    
    def load_train_graph(self, dataset_name: str, repetition_id: int, task: str) -> Optional[nx.Graph]:
        """
        Load training graph for link prediction tasks.
        
        For link prediction, we need a graph with test/val edges removed to prevent
        data leakage. This is especially critical for methods like GAT/GraphGPS that
        train on edges.
        
        Parameters
        ----------
        dataset_name : str
            Dataset name
        repetition_id : int
            Repetition ID
        task : str
            Task name
        
        Returns
        -------
        nx.Graph or None
            Training graph if available (link prediction), None otherwise
        """
        if task != "link_prediction":
            return None
        
        entry = self.registry.get(dataset_name, repetition_id)
        if entry is None:
            raise ValueError(f"Dataset {dataset_name} rep {repetition_id} not found in registry")
        
        if task not in entry.split_paths:
            raise ValueError(f"Task {task} not available for {dataset_name} rep {repetition_id}")
        
        # The training graph (test/val edges removed) is produced by
        # SplitGenerator.generate_link_prediction_split. It is stored under the
        # "train_graph" key inside the split pickle; older runs may instead have
        # written a sibling train_graph.pkl. Prefer the sibling file, then fall
        # back to the in-split copy.
        split_path = entry.split_paths[task]
        train_graph_path = split_path.parent / "train_graph.pkl"

        if train_graph_path.exists():
            with open(train_graph_path, 'rb') as f:
                return pickle.load(f)

        with open(split_path, 'rb') as f:
            split = pickle.load(f)
        G_train = split.get("train_graph")
        if G_train is None:
            raise FileNotFoundError(
                f"Training graph not found: neither {train_graph_path} exists nor "
                f"does {split_path} contain a 'train_graph' key. This is required "
                f"for link prediction to prevent data leakage."
            )
        return G_train
    
    def compute_graph_complexity(
        self,
        G: nx.Graph,
        labels: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute graph complexity metrics.
        
        Parameters
        ----------
        G : nx.Graph
            Graph to analyze
        labels : np.ndarray, optional
            Node labels for homophily computation
        features : np.ndarray, optional
            Node features for Dirichlet energy computation
        
        Returns
        -------
        dict
            Dictionary of complexity metrics (all float values)
        """
        try:
            from qbiocode.evaluation.graph_evaluation import compute_enhanced_complexity_metrics
            
            # Compute complexity metrics
            complexity_metrics = compute_enhanced_complexity_metrics(
                G,
                labels=labels,
                features=features,
                sanitize=True
            )
            
            return complexity_metrics
            
        except Exception as e:
            warnings.warn(f"Failed to compute complexity metrics: {e}")
            # Return empty dict on failure to maintain type consistency
            return {}
    
    def run_method(
        self,
        method_name: str,
        dataset_name: str,
        repetition_id: int,
        task: str,
        config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Run a method with pre-generated data.
        
        Parameters
        ----------
        method_name : str
            Name of the method to run
        dataset_name : str
            Dataset name
        repetition_id : int
            Repetition ID
        task : str
            Task name
        config : dict, optional
            Method configuration/hyperparameters
        output_dir : Path, optional
            Directory to save results
        
        Returns
        -------
        dict
            Results with metrics and provenance
        """
        # Get canonical seed
        seed = self.seed_manager.get_seed(dataset_name, repetition_id)
        task_seed = self.seed_manager.get_task_seed(dataset_name, repetition_id, task)

        # Load pre-generated data
        G = self.load_graph(dataset_name, repetition_id)
        split = self.load_split(dataset_name, repetition_id, task)
        
        # For link prediction, load training graph (with test/val edges removed)
        # This prevents data leakage for methods that train on edges (e.g., GAT, GraphGPS)
        # ALL methods must use the same graph structure for fair comparison
        if task == "link_prediction":
            G_train = self.load_train_graph(dataset_name, repetition_id, task)
            if G_train is None:
                raise RuntimeError(f"Training graph not available for link prediction task")
            # Use training graph for embedding generation
            G_for_method = G_train
        else:
            G_for_method = G
        
        # Get registry entry for provenance
        entry = self.registry.get(dataset_name, repetition_id)
        if entry is None:
            raise ValueError(f"Dataset {dataset_name} rep {repetition_id} not found in registry")

        metadata = {}
        with open(entry.metadata_path, 'r') as f:
            metadata = json.load(f)

        disease_nodes = None
        if entry.disease_node_path is not None and entry.disease_node_path.exists():
            with open(entry.disease_node_path, 'r') as f:
                disease_nodes = json.load(f)

        self._validate_fixed_inputs(
            G=G,
            split=split,
            task=task,
            metadata=metadata,
            disease_nodes=disease_nodes,
            seed=seed,
            task_seed=task_seed,
            dataset_name=dataset_name,
            repetition_id=repetition_id,
        )
        
        # Extract node labels and features for complexity computation
        node_labels = None
        node_features = None
        if G_for_method.number_of_nodes() > 0:
            first_node = list(G_for_method.nodes())[0]
            if 'label' in G_for_method.nodes[first_node]:
                node_labels = np.array([G_for_method.nodes[n].get('label', -1) for n in G_for_method.nodes()])
            if 'features' in G_for_method.nodes[first_node]:
                node_features = np.array([G_for_method.nodes[n].get('features', []) for n in G_for_method.nodes()])
        
        # Compute graph complexity metrics
        complexity_metrics = self.compute_graph_complexity(
            G_for_method,
            labels=node_labels,
            features=node_features
        )
        
        # Set random seeds for reproducibility
        np.random.seed(task_seed)
        try:
            import torch
            torch.manual_seed(task_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(task_seed)
        except ImportError:
            pass
        
        # Run method based on task
        start_time = time.time()
        graph_snapshot = pickle.dumps(G_for_method, protocol=pickle.HIGHEST_PROTOCOL)
        split_for_method = deepcopy(split)
        graph_for_method = pickle.loads(graph_snapshot)

        if task == "node_classification":
            metrics = self._run_node_classification(
                method_name, graph_for_method, split_for_method, config, task_seed
            )
        elif task == "link_prediction":
            metrics = self._run_link_prediction(
                method_name, graph_for_method, split_for_method, config, task_seed
            )
        elif task == "node_ranking":
            metrics = self._run_node_ranking(
                method_name, graph_for_method, split_for_method, config, task_seed
            )
        else:
            raise ValueError(f"Unknown task: {task}")

        if pickle.dumps(G_for_method, protocol=pickle.HIGHEST_PROTOCOL) != graph_snapshot:
            raise RuntimeError(f"Method {method_name} modified the input graph in-place, which is forbidden.")
        
        runtime = time.time() - start_time
        
        # Compile results with full provenance
        results = {
            "method_name": method_name,
            "dataset_name": dataset_name,
            "dataset_type": entry.dataset_type.value,
            "ppi_source": entry.ppi_source,
            "disease": entry.disease,
            "requested_graph_size": entry.requested_size,
            "actual_graph_size": entry.actual_size,
            "repetition_id": repetition_id,
            "task": task,
            "seed": seed,
            "task_seed": task_seed,
            "graph_path": str(entry.graph_path),
            "split_path": str(entry.split_paths[task]),
            "metadata_path": str(entry.metadata_path),
            "disease_node_path": str(entry.disease_node_path) if entry.disease_node_path else None,
            "num_nodes": G_for_method.number_of_nodes(),
            "num_edges": G_for_method.number_of_edges(),
            "used_training_graph": task == "link_prediction",
            "config": config or {},
            "metrics": metrics,
            "complexity_metrics": complexity_metrics,
            "runtime_seconds": runtime,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        
        # Save results if output directory provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            result_file = output_dir / f"{method_name}_{task}_rep{repetition_id:02d}.json"
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
        
        return results
    def _validate_fixed_inputs(
        self,
        G: nx.Graph,
        split: Dict[str, Any],
        task: str,
        metadata: Dict[str, Any],
        disease_nodes: Optional[Dict[str, Any]],
        seed: int,
        task_seed: int,
        dataset_name: str,
        repetition_id: int,
    ) -> None:
        """Validate that runner inputs satisfy reproducibility and fairness constraints."""
        if split.get("canonical_seed") is not None and split.get("canonical_seed") != seed:
            raise ValueError(
                f"Split canonical seed mismatch: split has {split.get('canonical_seed')} but runner derived {seed}"
            )

        # split["seed"] records the seed used to DRAW the split, which uses the
        # task's split component (see SeedManager.TASK_TO_SPLIT_COMPONENT) — not the
        # model-run task seed. Validate against the split component accordingly.
        expected_split_seed = self.seed_manager.get_split_seed(dataset_name, repetition_id, task)
        if split.get("seed") != expected_split_seed:
            raise ValueError(
                f"Split seed mismatch: split has {split.get('seed')} but runner derived {expected_split_seed}"
            )

        if split.get("dataset_name") != dataset_name:
            raise ValueError(
                f"Split dataset mismatch: split has {split.get('dataset_name')} but runner expected {dataset_name}"
            )

        if split.get("repetition_id") != repetition_id:
            raise ValueError(
                f"Split repetition mismatch: split has {split.get('repetition_id')} but runner expected {repetition_id}"
            )

        if split.get("task") != task:
            raise ValueError(
                f"Split task mismatch: split has {split.get('task')} but runner expected {task}"
            )

        if metadata.get("canonical_seed") is not None and metadata.get("canonical_seed") != seed:
            raise ValueError(
                f"Metadata canonical seed mismatch: metadata has {metadata.get('canonical_seed')} but runner derived {seed}"
            )

        if metadata.get("final_graph_connected") is False:
            raise ValueError("Final saved graph is marked disconnected in metadata.")

        if G.number_of_nodes() == 0:
            raise ValueError("Input graph is empty.")

        if task == "node_ranking":
            expected_target_seed = self.seed_manager.get_component_seed(
                dataset_name, repetition_id, "node_ranking_target_selection"
            )
            if split.get("target_selection_seed") != expected_target_seed:
                raise ValueError(
                    f"Node-ranking target selection seed mismatch: split has {split.get('target_selection_seed')} "
                    f"but expected {expected_target_seed}"
                )

        elif task == "node_classification":
            expected_split_seed = self.seed_manager.get_component_seed(
                dataset_name, repetition_id, "node_classification_split"
            )
            if split.get("split_seed") != expected_split_seed:
                raise ValueError(
                    f"Node-classification split seed mismatch: split has {split.get('split_seed')} "
                    f"but expected {expected_split_seed}"
                )

        elif task == "link_prediction":
            expected_split_seed = self.seed_manager.get_component_seed(
                dataset_name, repetition_id, "link_prediction_edge_split"
            )
            expected_negative_seed = self.seed_manager.get_component_seed(
                dataset_name, repetition_id, "negative_edge_sampling"
            )
            if split.get("split_seed") != expected_split_seed:
                raise ValueError(
                    f"Link-prediction split seed mismatch: split has {split.get('split_seed')} "
                    f"but expected {expected_split_seed}"
                )
            if split.get("negative_sampling_seed") != expected_negative_seed:
                raise ValueError(
                    f"Negative-sampling seed mismatch: split has {split.get('negative_sampling_seed')} "
                    f"but expected {expected_negative_seed}"
                )

        if disease_nodes is not None:
            missing_seed_nodes = [
                node for node in disease_nodes.get("seed_node_ids", [])
                if node not in G.nodes()
            ]
            missing_target_nodes = [
                node for node in disease_nodes.get("target_node_ids", [])
                if node not in G.nodes()
            ]
            if missing_seed_nodes or missing_target_nodes:
                raise ValueError(
                    f"Preserved disease nodes missing from graph. "
                    f"Seeds: {missing_seed_nodes}, Targets: {missing_target_nodes}"
                )

            if task == "node_ranking":
                if split.get("seed_nodes") != disease_nodes.get("seed_node_ids", []):
                    raise ValueError("Node-ranking split seed nodes differ from preserved disease seed nodes.")
                if split.get("target_nodes") != disease_nodes.get("target_node_ids", []):
                    raise ValueError("Node-ranking split target nodes differ from preserved disease target nodes.")
    
    def _run_node_classification(
        self,
        method_name: str,
        G: nx.Graph,
        split: Dict[str, Any],
        config: Optional[Dict],
        seed: int
    ) -> Dict[str, float]:
        """Run node classification task using method adapter."""
        from .method_adapters import get_method_adapter
        
        adapter = get_method_adapter(method_name)
        return adapter(G, split, "node_classification", config, seed)
    
    def _run_link_prediction(
        self,
        method_name: str,
        G: nx.Graph,
        split: Dict[str, Any],
        config: Optional[Dict],
        seed: int
    ) -> Dict[str, float]:
        """Run link prediction task using method adapter."""
        from .method_adapters import get_method_adapter
        
        adapter = get_method_adapter(method_name)
        return adapter(G, split, "link_prediction", config, seed)
    
    def _run_node_ranking(
        self,
        method_name: str,
        G: nx.Graph,
        split: Dict[str, Any],
        config: Optional[Dict],
        seed: int
    ) -> Dict[str, float]:
        """Run node ranking task using method adapter."""
        from .method_adapters import get_method_adapter
        
        adapter = get_method_adapter(method_name)
        return adapter(G, split, "node_ranking", config, seed)
        # Extract split data
        seed_nodes = split["seed_nodes"]
        target_nodes = split["target_nodes"]
        
        # TODO: Call actual method implementation
        # For now, return dummy metrics
        metrics = {
            "recall_at_10": 0.0,
            "recall_at_50": 0.0,
            "recall_at_100": 0.0,
            "precision_at_10": 0.0,
            "precision_at_50": 0.0,
            "precision_at_100": 0.0,
            "num_seeds": len(seed_nodes),
            "num_targets": len(target_nodes)
        }
        
        return metrics


def create_method_adapter(method_name: str):
    """
    Create an adapter function that wraps an existing method to work with
    the reproducible pipeline.
    
    This is a helper for migrating existing methods.
    
    Parameters
    ----------
    method_name : str
        Name of the method
    
    Returns
    -------
    callable
        Adapter function
    """
    def adapter(
        G: nx.Graph,
        split: Dict[str, Any],
        task: str,
        seed: int,
        config: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Adapter that calls the original method with pre-generated data.
        
        Parameters
        ----------
        G : nx.Graph
            Pre-generated graph (DO NOT MODIFY)
        split : dict
            Pre-generated split
        task : str
            Task name
        seed : int
            Canonical seed
        config : dict, optional
            Method configuration
        
        Returns
        -------
        dict
            Metrics
        """
        # Import the actual method
        # This is where you would import and call the real implementation
        
        # Example for a hypothetical method:
        # from qbiocode.apps.quvine.methods import run_graphsage
        # return run_graphsage(G, split, task, seed, config)
        
        raise NotImplementedError(
            f"Method {method_name} not yet adapted to reproducible pipeline. "
            f"Please implement the adapter in method_runner.py"
        )
    
    return adapter

