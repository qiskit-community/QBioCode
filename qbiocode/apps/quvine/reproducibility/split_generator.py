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
Task Split Generator for QuVINE

Generates and saves task-specific data splits to ensure all methods use identical splits.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any, cast
import networkx as nx
import numpy as np
import torch


def _to_degree_dict(G: nx.Graph) -> Dict[Any, int]:
    """Create a plain degree dictionary compatible with static type checking."""
    graph_any = cast(Any, G)
    return {node: int(graph_any.degree[node]) for node in G.nodes()}


def _to_index_list(values: Any) -> List[int]:
    """Convert numpy arrays or iterables to plain Python int lists."""
    if hasattr(values, "tolist"):
        return [int(v) for v in values.tolist()]
    return [int(v) for v in values]

from .seed_manager import SeedManager
from .dataset_registry import DatasetRegistry


class SplitGenerator:
    """
    Generates and saves task-specific splits for reproducible benchmarking.
    
    All splits are pre-generated once and saved to disk.
    Methods then load these pre-generated splits instead of creating their own.
    
    Supports three tasks:
    1. Node Classification: train/val/test node indices + labels
    2. Link Prediction: train/val/test edge splits + negative samples
    3. Node Ranking: target nodes for ranking evaluation
    """
    
    def __init__(
        self,
        output_dir: Path,
        seed_manager: SeedManager,
        registry: DatasetRegistry
    ):
        """
        Initialize split generator.
        
        Parameters
        ----------
        output_dir : Path
            Root directory for saving splits
        seed_manager : SeedManager
            Seed manager for reproducible split generation
        registry : DatasetRegistry
            Dataset registry to update with split paths
        """
        self.output_dir = Path(output_dir)
        self.seed_manager = seed_manager
        self.registry = registry
    
    def generate_all_splits(
        self,
        dataset_name: str,
        repetition_id: int,
        G: nx.Graph,
        tasks: List[str] = ["node_classification", "link_prediction", "node_ranking"],
        disease_nodes: Optional[Dict[str, Any]] = None,
        task_config: Optional[Dict[str, Any]] = None,
        split_base_dir: Optional[Path] = None,
    ) -> Dict[str, Path]:
        """
        Generate all task splits for a dataset.
        
        Parameters
        ----------
        dataset_name : str
            Dataset name
        repetition_id : int
            Repetition ID
        G : nx.Graph
            Graph to generate splits for
        tasks : List[str]
            Tasks to generate splits for
        
        Returns
        -------
        Dict[str, Path]
            Mapping from task name to split file path
        """
        split_paths = {}
        
        task_config = task_config or {}
        split_base_dir = Path(split_base_dir) if split_base_dir is not None else None

        for task in tasks:
            if task == "node_classification":
                split_path = self.generate_node_classification_split(
                    dataset_name,
                    repetition_id,
                    G,
                    disease_nodes=disease_nodes,
                    split_base_dir=split_base_dir,
                    **task_config.get("node_classification", {})
                )
            elif task == "link_prediction":
                split_path = self.generate_link_prediction_split(
                    dataset_name,
                    repetition_id,
                    G,
                    disease_nodes=disease_nodes,
                    split_base_dir=split_base_dir,
                    **task_config.get("link_prediction", {})
                )
            elif task == "node_ranking":
                split_path = self.generate_node_ranking_split(
                    dataset_name,
                    repetition_id,
                    G,
                    disease_nodes=disease_nodes,
                    split_base_dir=split_base_dir,
                    **task_config.get("node_ranking", {})
                )
            else:
                raise ValueError(f"Unknown task: {task}")
            
            split_paths[task] = split_path
            
            # Update registry
            self.registry.add_split(dataset_name, repetition_id, task, split_path)
        
        return split_paths
    
    def generate_node_classification_split(
        self,
        dataset_name: str,
        repetition_id: int,
        G: nx.Graph,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        min_class_size: int = 5,
        disease_nodes: Optional[Dict[str, Any]] = None,
        positive_class_definition: str = "disease_seed_or_target",
        negative_class_definition: str = "non_disease_nodes",
        split_strategy: str = "stratified",
        split_base_dir: Optional[Path] = None,
    ) -> Path:
        """
        Generate node classification split with synthetic labels.
        
        Labels are generated using community detection (Louvain).
        
        Parameters
        ----------
        dataset_name : str
            Dataset name
        repetition_id : int
            Repetition ID
        G : nx.Graph
            Graph
        train_ratio : float
            Training set ratio
        val_ratio : float
            Validation set ratio
        test_ratio : float
            Test set ratio
        min_class_size : int
            Minimum nodes per class
        
        Returns
        -------
        Path
            Path to saved split file
        """
        canonical_seed = self.seed_manager.get_seed(dataset_name, repetition_id)
        seed = self.seed_manager.get_component_seed(
            dataset_name, repetition_id, "node_classification_split"
        )

        nodes = sorted(G.nodes())

        if disease_nodes is not None:
            seed_nodes = set(map(str, disease_nodes.get("seed_node_ids", [])))
            target_nodes = set(map(str, disease_nodes.get("target_node_ids", [])))
            positive_nodes = seed_nodes | target_nodes
            labels = np.array([1 if str(node) in positive_nodes else 0 for node in nodes], dtype=int)

            pos_idx = np.where(labels == 1)[0]
            neg_idx = np.where(labels == 0)[0]
            if len(pos_idx) < 2 or len(neg_idx) < 2:
                raise ValueError("Insufficient class support for PPI node classification split generation.")

            from sklearn.model_selection import train_test_split

            train_idx, temp_idx = train_test_split(
                np.arange(len(nodes)),
                train_size=train_ratio,
                stratify=labels,
                random_state=seed
            )
            val_size = val_ratio / (val_ratio + test_ratio)
            val_idx, test_idx = train_test_split(
                temp_idx,
                train_size=val_size,
                stratify=labels[temp_idx],
                random_state=seed + 1
            )

            split_data = {
                "dataset_name": dataset_name,
                "repetition_id": repetition_id,
                "task": "node_classification",
                "seed": seed,
                "canonical_seed": canonical_seed,
                "nodes": nodes,
                "labels": [int(v) for v in labels.tolist()],
                "train_idx": _to_index_list(train_idx),
                "val_idx": _to_index_list(val_idx),
                "test_idx": _to_index_list(test_idx),
                "num_classes": 2,
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                "positive_class_definition": positive_class_definition,
                "negative_class_definition": negative_class_definition,
                "split_seed": seed,
                "split_strategy": split_strategy,
                "class_distribution": {
                    "train": {
                        "positive": int(labels[train_idx].sum()),
                        "negative": int(len(train_idx) - labels[train_idx].sum()),
                    },
                    "validation": {
                        "positive": int(labels[val_idx].sum()),
                        "negative": int(len(val_idx) - labels[val_idx].sum()),
                    },
                    "test": {
                        "positive": int(labels[test_idx].sum()),
                        "negative": int(len(test_idx) - labels[test_idx].sum()),
                    },
                },
                "disease_seed_nodes": sorted(seed_nodes),
                "disease_target_nodes": sorted(target_nodes),
                "label_generation_method": "ppi_disease_labeling",
            }
        else:
            rng = np.random.default_rng(seed)

            from qbiocode.apps.quvine.evaluation.classification import generate_community_labels

            labels_dict = generate_community_labels(
                G,
                method='louvain',
                min_community_size=min_class_size,
                resolution=1.0
            )

            labels = np.array([labels_dict[node] for node in nodes])

            from sklearn.model_selection import train_test_split

            train_idx, temp_idx = train_test_split(
                np.arange(len(nodes)),
                train_size=train_ratio,
                stratify=labels,
                random_state=seed
            )

            val_size = val_ratio / (val_ratio + test_ratio)
            val_idx, test_idx = train_test_split(
                temp_idx,
                train_size=val_size,
                stratify=labels[temp_idx],
                random_state=seed + 1
            )

            split_data = {
                "dataset_name": dataset_name,
                "repetition_id": repetition_id,
                "task": "node_classification",
                "seed": seed,
                "canonical_seed": canonical_seed,
                "nodes": nodes,
                "labels": [int(v) for v in (labels.tolist() if hasattr(labels, 'tolist') else list(labels))],
                "train_idx": _to_index_list(train_idx),
                "val_idx": _to_index_list(val_idx),
                "test_idx": _to_index_list(test_idx),
                "num_classes": int(labels.max() + 1),
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                "split_seed": seed,
                "split_strategy": split_strategy,
                "label_generation_method": "louvain_community_detection"
            }
        
        # Save split with task-specific filename
        if split_base_dir is not None:
            # split_base_dir already includes the full path, don't add dataset_name/rep again
            split_dir = split_base_dir
        else:
            # Default: output_dir / dataset_name / rep_XX
            split_dir = self.output_dir / dataset_name / f"rep_{repetition_id:02d}"
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle (primary format)
        split_path_pkl = split_dir / "node_classification_split.pkl"
        with open(split_path_pkl, 'wb') as f:
            pickle.dump(split_data, f)
        
        # Also save as JSON for inspection
        split_path_json = split_dir / "node_classification_split.json"
        with open(split_path_json, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        return split_path_pkl
    
    def generate_link_prediction_split(
        self,
        dataset_name: str,
        repetition_id: int,
        G: nx.Graph,
        train_ratio: float = 0.85,
        val_ratio: float = 0.05,
        test_ratio: float = 0.10,
        negative_sampling_ratio: float = 1.0,
        disease_nodes: Optional[Dict[str, Any]] = None,
        protect_disease_edges: bool = True,
        split_strategy: str = "random_edge_split",
        negative_sampling_strategy: str = "random",
        split_base_dir: Optional[Path] = None,
    ) -> Path:
        """
        Generate link prediction split with held-out edges and negative samples.
        
        Parameters
        ----------
        dataset_name : str
            Dataset name
        repetition_id : int
            Repetition ID
        G : nx.Graph
            Graph
        train_ratio : float
            Training edges ratio
        val_ratio : float
            Validation edges ratio
        test_ratio : float
            Test edges ratio
        negative_sampling_ratio : float
            Ratio of negative to positive samples
        
        Returns
        -------
        Path
            Path to saved split file
        """
        canonical_seed = self.seed_manager.get_seed(dataset_name, repetition_id)
        seed = self.seed_manager.get_component_seed(
            dataset_name, repetition_id, "link_prediction_edge_split"
        )
        negative_seed = self.seed_manager.get_component_seed(
            dataset_name, repetition_id, "negative_edge_sampling"
        )
        rng = np.random.default_rng(seed)

        edges = list(G.edges())
        protected_edges = set()
        if disease_nodes is not None and protect_disease_edges:
            disease_set = set(map(str, disease_nodes.get("seed_node_ids", []))) | set(map(str, disease_nodes.get("target_node_ids", [])))
            protected_edges = {
                tuple(sorted((str(u), str(v))))
                for u, v in edges
                if str(u) in disease_set and str(v) in disease_set
            }

        removable_edges = []
        retained_train_edges = []
        for u, v in edges:
            normalized = tuple(sorted((str(u), str(v))))
            if normalized in protected_edges:
                retained_train_edges.append((u, v))
            else:
                removable_edges.append((u, v))

        rng.shuffle(removable_edges)
        n_edges = len(removable_edges)
        n_train = int(train_ratio * n_edges)
        n_val = int(val_ratio * n_edges)

        train_edges = retained_train_edges + removable_edges[:n_train]
        val_edges = removable_edges[n_train:n_train + n_val]
        test_edges = removable_edges[n_train + n_val:]

        from qbiocode.apps.quvine.evaluation.link_prediction import sample_negative_edges

        existing_edges = set(edges)

        n_neg_train = int(len(train_edges) * negative_sampling_ratio)
        n_neg_val = int(len(val_edges) * negative_sampling_ratio)
        n_neg_test = int(len(test_edges) * negative_sampling_ratio)

        # Sample the three negative sets so they are mutually DISJOINT: each
        # draw excludes the true edges plus any negative already assigned to an
        # earlier split. Without this, the same non-edge could appear as both a
        # train negative and a test negative, leaking test labels into training.
        neg_train_edges = sample_negative_edges(
            G, n_neg_train, existing_edges, strategy=negative_sampling_strategy, seed=negative_seed
        )
        neg_val_edges = sample_negative_edges(
            G, n_neg_val, existing_edges | set(neg_train_edges),
            strategy=negative_sampling_strategy, seed=negative_seed + 1
        )
        neg_test_edges = sample_negative_edges(
            G, n_neg_test, existing_edges | set(neg_train_edges) | set(neg_val_edges),
            strategy=negative_sampling_strategy, seed=negative_seed + 2
        )

        G_train = G.copy()
        G_train.remove_edges_from(val_edges + test_edges)

        split_data = {
            "dataset_name": dataset_name,
            "repetition_id": repetition_id,
            "task": "link_prediction",
            "seed": seed,
            "canonical_seed": canonical_seed,
            "train_edges": train_edges,
            "val_edges": val_edges,
            "test_edges": test_edges,
            "neg_train_edges": neg_train_edges,
            "neg_val_edges": neg_val_edges,
            "neg_test_edges": neg_test_edges,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "negative_sampling_ratio": negative_sampling_ratio,
            "split_seed": seed,
            "negative_sampling_seed": negative_seed,
            "split_strategy": split_strategy,
            "negative_sampling_strategy": negative_sampling_strategy,
            "is_directed": G.is_directed(),
            "protected_edge_count": len(protected_edges),
            "protect_disease_edges": protect_disease_edges,
            "num_train_edges": len(train_edges),
            "num_val_edges": len(val_edges),
            "num_test_edges": len(test_edges),
            "num_train_nodes": G_train.number_of_nodes(),
            "num_train_graph_edges": G_train.number_of_edges()
        }
        
        # Save split with task-specific filename
        if split_base_dir is not None:
            # split_base_dir already includes the full path, don't add dataset_name/rep again
            split_dir = split_base_dir
        else:
            # Default: output_dir / dataset_name / rep_XX
            split_dir = self.output_dir / dataset_name / f"rep_{repetition_id:02d}"
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle (primary format) - includes G_train in split_data
        split_data['train_graph'] = G_train  # Include training graph in split data
        split_path_pkl = split_dir / "link_prediction_split.pkl"
        with open(split_path_pkl, 'wb') as f:
            pickle.dump(split_data, f)
        
        # Also save as JSON for inspection (without graph)
        split_data_json = {k: v for k, v in split_data.items() if k != 'train_graph'}
        split_path_json = split_dir / "link_prediction_split.json"
        with open(split_path_json, 'w') as f:
            json.dump(split_data_json, f, indent=2)
        
        return split_path_pkl
    
    def generate_node_ranking_split(
        self,
        dataset_name: str,
        repetition_id: int,
        G: nx.Graph,
        n_targets: Optional[int] = None,
        target_selection: str = "high_degree",
        n_seeds: Optional[int] = None,
        seed_selection: str = "random",
        disease_nodes: Optional[Dict[str, Any]] = None,
        excluded_nodes: Optional[List[Any]] = None,
        selection_rule: Optional[str] = None,
        split_base_dir: Optional[Path] = None,
    ) -> Path:
        """
        Generate node ranking split with seed and target nodes.
        
        For PPI networks, this should use disease-specific seeds/targets.
        For synthetic networks, generates synthetic seeds/targets.
        
        Parameters
        ----------
        dataset_name : str
            Dataset name
        repetition_id : int
            Repetition ID
        G : nx.Graph
            Graph
        n_targets : int, optional
            Number of target nodes (default: 10% of nodes)
        target_selection : str
            Target selection strategy ('high_degree', 'random', 'central')
        n_seeds : int, optional
            Number of seed nodes (default: 5% of nodes)
        seed_selection : str
            Seed selection strategy ('random', 'high_degree')
        
        Returns
        -------
        Path
            Path to saved split file
        """
        canonical_seed = self.seed_manager.get_seed(dataset_name, repetition_id)
        seed = self.seed_manager.get_component_seed(
            dataset_name, repetition_id, "node_ranking"
        )
        target_seed = self.seed_manager.get_component_seed(
            dataset_name, repetition_id, "node_ranking_target_selection"
        )
        rng = np.random.default_rng(seed)
        target_rng = np.random.default_rng(target_seed)

        nodes = list(G.nodes())
        n_nodes = len(nodes)
        excluded_nodes = excluded_nodes or []

        if disease_nodes is not None:
            seed_nodes = [node for node in disease_nodes.get("seed_node_ids", []) if node in G.nodes()]
            target_nodes = [node for node in disease_nodes.get("target_node_ids", []) if node in G.nodes()]

            missing_seed_nodes = sorted(set(disease_nodes.get("seed_node_ids", [])) - set(seed_nodes))
            missing_target_nodes = sorted(set(disease_nodes.get("target_node_ids", [])) - set(target_nodes))
            if missing_seed_nodes or missing_target_nodes:
                raise ValueError(
                    f"Disease nodes missing from graph. Missing seeds: {missing_seed_nodes}; "
                    f"missing targets: {missing_target_nodes}"
                )

            candidate_universe = [node for node in sorted(nodes) if node not in excluded_nodes]
            split_data = {
                "dataset_name": dataset_name,
                "repetition_id": repetition_id,
                "task": "node_ranking",
                "seed": seed,
                "canonical_seed": canonical_seed,
                "seed_nodes": seed_nodes,
                "target_nodes": target_nodes,
                "candidate_node_universe": candidate_universe,
                "excluded_nodes": excluded_nodes,
                "target_selection_seed": target_seed,
                "selection_rule": selection_rule or "use_preserved_disease_seed_and_target_nodes",
                "n_seeds": len(seed_nodes),
                "n_targets": len(target_nodes),
                "seed_selection": "disease_seed_nodes",
                "target_selection": "disease_target_nodes",
                "all_nodes": nodes
            }
        else:
            if n_targets is None:
                n_targets = max(10, int(0.1 * n_nodes))
            if n_seeds is None:
                n_seeds = max(5, int(0.05 * n_nodes))

            if seed_selection == "random":
                seed_nodes = rng.choice(nodes, size=n_seeds, replace=False).tolist()
            elif seed_selection == "high_degree":
                degrees = _to_degree_dict(G)
                sorted_nodes = sorted(nodes, key=lambda n: degrees[n], reverse=True)
                seed_nodes = sorted_nodes[:n_seeds]
            else:
                raise ValueError(f"Unknown seed selection strategy: {seed_selection}")

            candidate_targets = [n for n in nodes if n not in seed_nodes]

            if target_selection == "random":
                target_nodes = target_rng.choice(candidate_targets, size=min(n_targets, len(candidate_targets)), replace=False).tolist()
            elif target_selection == "high_degree":
                degrees = _to_degree_dict(G)
                sorted_candidates = sorted(candidate_targets, key=lambda n: degrees[n], reverse=True)
                target_nodes = sorted_candidates[:n_targets]
            elif target_selection == "central":
                centrality = nx.betweenness_centrality(G, k=min(100, n_nodes))
                sorted_candidates = sorted(candidate_targets, key=lambda n: centrality.get(n, 0), reverse=True)
                target_nodes = sorted_candidates[:n_targets]
            else:
                raise ValueError(f"Unknown target selection strategy: {target_selection}")

            split_data = {
                "dataset_name": dataset_name,
                "repetition_id": repetition_id,
                "task": "node_ranking",
                "seed": seed,
                "canonical_seed": canonical_seed,
                "seed_nodes": seed_nodes,
                "target_nodes": target_nodes,
                "candidate_node_universe": sorted(nodes),
                "excluded_nodes": excluded_nodes,
                "target_selection_seed": target_seed,
                "selection_rule": selection_rule or "synthetic_selection",
                "n_seeds": len(seed_nodes),
                "n_targets": len(target_nodes),
                "seed_selection": seed_selection,
                "target_selection": target_selection,
                "all_nodes": nodes
            }
        
        # Save split with task-specific filename
        if split_base_dir is not None:
            # split_base_dir already includes the full path, don't add dataset_name/rep again
            split_dir = split_base_dir
        else:
            # Default: output_dir / dataset_name / rep_XX
            split_dir = self.output_dir / dataset_name / f"rep_{repetition_id:02d}"
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle (primary format)
        split_path_pkl = split_dir / "node_ranking_split.pkl"
        with open(split_path_pkl, 'wb') as f:
            pickle.dump(split_data, f)
        
        # Also save as JSON for inspection
        split_path_json = split_dir / "node_ranking_split.json"
        with open(split_path_json, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        return split_path_pkl
    
    def load_split(self, split_path: Path) -> Dict[str, Any]:
        """
        Load a pre-generated split from disk.
        
        Parameters
        ----------
        split_path : Path
            Path to split file
        
        Returns
        -------
        dict
            Split data
        """
        # Try pickle first (faster)
        pkl_path = split_path.with_suffix('.pkl')
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                return pickle.load(f)
        
        # Fall back to JSON
        with open(split_path, 'r') as f:
            return json.load(f)

