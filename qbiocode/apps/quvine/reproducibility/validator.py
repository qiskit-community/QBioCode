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
Reproducibility Validator for QuVINE

Validates that all methods use identical experimental conditions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import json
import logging
import pickle
import networkx as nx

from .dataset_registry import DatasetRegistry
from .seed_manager import SeedManager

logger = logging.getLogger(__name__)


class ReproducibilityValidator:
    """
    Validates reproducibility constraints for QuVINE experiments.
    
    Ensures that:
    1. All methods for a dataset/repetition/task use the same graph
    2. All methods use the same task split
    3. All methods use the same seed
    4. No method regenerates graphs or splits during evaluation
    """
    
    def __init__(
        self,
        registry: DatasetRegistry,
        seed_manager: SeedManager
    ):
        """
        Initialize validator.
        
        Parameters
        ----------
        registry : DatasetRegistry
            Dataset registry
        seed_manager : SeedManager
            Seed manager
        """
        self.registry = registry
        self.seed_manager = seed_manager
        self.validation_errors: List[str] = []
    
    def validate_dataset(
        self,
        dataset_name: str,
        repetition_id: int,
        task: str
    ) -> bool:
        """
        Validate that a dataset-repetition-task combination is ready for experiments.
        
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
        bool
            True if valid, False otherwise
        """
        self.validation_errors = []

        # Check dataset is registered
        entry = self.registry.get(dataset_name, repetition_id)
        if entry is None:
            self.validation_errors.append(
                f"Dataset {dataset_name} rep {repetition_id} not registered"
            )
            return False
        
        # Check graph file exists
        if not entry.graph_path.exists():
            self.validation_errors.append(
                f"Graph file not found: {entry.graph_path}"
            )
        
        # Check metadata file exists
        if not entry.metadata_path.exists():
            self.validation_errors.append(
                f"Metadata file not found: {entry.metadata_path}"
            )
        
        # Check task is available
        if task not in entry.available_tasks:
            self.validation_errors.append(
                f"Task {task} not available for {dataset_name} rep {repetition_id}. "
                f"Available tasks: {entry.available_tasks}"
            )
        
        # Check split file exists
        if task in entry.split_paths:
            if not entry.split_paths[task].exists():
                self.validation_errors.append(
                    f"Split file not found: {entry.split_paths[task]}"
                )
        else:
            self.validation_errors.append(
                f"No split path registered for task {task}"
            )
        
        # Validate seed consistency
        expected_seed = self.seed_manager.get_seed(dataset_name, repetition_id)
        if entry.seed != expected_seed:
            self.validation_errors.append(
                f"Seed mismatch: registry has {entry.seed}, "
                f"seed manager expects {expected_seed}"
            )

        metadata = self._safe_load_json(entry.metadata_path)
        if metadata:
            if metadata.get("canonical_seed") is not None and metadata.get("canonical_seed") != expected_seed:
                self.validation_errors.append(
                    f"Metadata canonical seed mismatch: {metadata.get('canonical_seed')} vs {expected_seed}"
                )
            if metadata.get("final_graph_connected") is False:
                self.validation_errors.append("Saved graph metadata reports disconnected final graph")

        disease_nodes = None
        if entry.disease_node_path is not None:
            disease_nodes = self._safe_load_json(entry.disease_node_path)
            if disease_nodes is None:
                self.validation_errors.append(f"Could not load disease node file: {entry.disease_node_path}")

        if entry.graph_path.exists():
            try:
                with open(entry.graph_path, "rb") as f:
                    G = pickle.load(f)
                self._validate_graph_fairness(G, task, metadata or {}, disease_nodes)
            except Exception as e:
                # Deliberately broad: a validator exists to report every problem it
                # finds, so an unexpected failure has to become a recorded finding
                # rather than abort the run and hide the findings already collected.
                # The type is included so the report is diagnosable.
                self.validation_errors.append(
                    f"Failed to load/validate graph {entry.graph_path}: "
                    f"{type(e).__name__}: {e}"
                )
                logger.debug("Graph validation failed for %s", entry.graph_path, exc_info=True)

        return len(self.validation_errors) == 0
    
    def validate_all_datasets(
        self,
        tasks: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Validate all datasets in the registry.
        
        Parameters
        ----------
        tasks : List[str], optional
            Tasks to validate. If None, validates all available tasks.
        
        Returns
        -------
        Dict[str, List[str]]
            Mapping from dataset_rep_task to list of errors
        """
        all_errors = {}
        
        for dataset_name in self.registry.get_all_datasets():
            for repetition_id in self.registry.get_repetitions(dataset_name):
                entry = self.registry.get(dataset_name, repetition_id)
                if entry is None:
                    continue
                
                tasks_to_check = tasks if tasks else entry.available_tasks
                
                for task in tasks_to_check:
                    key = f"{dataset_name}_rep{repetition_id:02d}_{task}"
                    is_valid = self.validate_dataset(dataset_name, repetition_id, task)
                    
                    if not is_valid:
                        all_errors[key] = self.validation_errors.copy()
        
        return all_errors
    
    def validate_experiment_consistency(
        self,
        dataset_name: str,
        repetition_id: int,
        task: str,
        method_results: List[Dict]
    ) -> bool:
        """
        Validate that all methods in an experiment used consistent inputs.
        
        Parameters
        ----------
        dataset_name : str
            Dataset name
        repetition_id : int
            Repetition ID
        task : str
            Task name
        method_results : List[Dict]
            List of result dictionaries from different methods
        
        Returns
        -------
        bool
            True if all methods used consistent inputs
        """
        self.validation_errors = []

        if not method_results:
            return True

        # Extract paths and seeds from first result
        first_result = method_results[0]
        expected_graph_path = first_result.get("graph_path")
        expected_split_path = first_result.get("split_path")
        expected_seed = first_result.get("seed")
        expected_disease_node_path = first_result.get("disease_node_path")

        # Check all other results match
        for i, result in enumerate(method_results[1:], 1):
            method_name = result.get("method_name", f"method_{i}")

            if result.get("graph_path") != expected_graph_path:
                self.validation_errors.append(
                    f"Method {method_name} used different graph: "
                    f"{result.get('graph_path')} vs {expected_graph_path}"
                )

            if result.get("split_path") != expected_split_path:
                self.validation_errors.append(
                    f"Method {method_name} used different split: "
                    f"{result.get('split_path')} vs {expected_split_path}"
                )

            if result.get("seed") != expected_seed:
                self.validation_errors.append(
                    f"Method {method_name} used different seed: "
                    f"{result.get('seed')} vs {expected_seed}"
                )

            if result.get("disease_node_path") != expected_disease_node_path:
                self.validation_errors.append(
                    f"Method {method_name} used different disease node file: "
                    f"{result.get('disease_node_path')} vs {expected_disease_node_path}"
                )

        return len(self.validation_errors) == 0
    
    def get_validation_report(self) -> str:
        """
        Get a formatted validation report.
        
        Returns
        -------
        str
            Validation report
        """
        if not self.validation_errors:
            return "✓ All validation checks passed"
        
        report = "✗ Validation errors found:\n"
        for i, error in enumerate(self.validation_errors, 1):
            report += f"  {i}. {error}\n"
        
        return report
    
    def validate_split_consistency(
        self,
        dataset_name: str,
        repetition_id: int,
        task: str
    ) -> bool:
        """
        Validate that a split file contains consistent data.
        
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
        bool
            True if split is consistent
        """
        self.validation_errors = []
        
        entry = self.registry.get(dataset_name, repetition_id)
        if entry is None or task not in entry.split_paths:
            self.validation_errors.append("Split not found in registry")
            return False
        
        split_path = entry.split_paths[task]
        if not split_path.exists():
            self.validation_errors.append(f"Split file not found: {split_path}")
            return False
        
        # Load and validate split
        try:
            with open(split_path, 'r') as f:
                split_data = json.load(f)

            expected_canonical_seed = self.seed_manager.get_seed(dataset_name, repetition_id)

            # Check required fields
            required_fields = ["dataset_name", "repetition_id", "task", "seed"]
            for field in required_fields:
                if field not in split_data:
                    self.validation_errors.append(f"Missing field in split: {field}")

            # Validate values match
            if split_data.get("dataset_name") != dataset_name:
                self.validation_errors.append(
                    f"Dataset name mismatch in split: "
                    f"{split_data.get('dataset_name')} vs {dataset_name}"
                )

            if split_data.get("repetition_id") != repetition_id:
                self.validation_errors.append(
                    f"Repetition ID mismatch in split: "
                    f"{split_data.get('repetition_id')} vs {repetition_id}"
                )

            if split_data.get("task") != task:
                self.validation_errors.append(
                    f"Task mismatch in split: "
                    f"{split_data.get('task')} vs {task}"
                )

            if split_data.get("canonical_seed") is not None and split_data.get("canonical_seed") != expected_canonical_seed:
                self.validation_errors.append(
                    f"Canonical seed mismatch in split: "
                    f"{split_data.get('canonical_seed')} vs {expected_canonical_seed}"
                )

            # split["seed"] is the seed used to DRAW the split (task split component),
            # not the model-run task seed — validate against the split component.
            expected_split_seed = self.seed_manager.get_split_seed(dataset_name, repetition_id, task)
            if split_data.get("seed") != expected_split_seed:
                self.validation_errors.append(
                    f"Split seed mismatch in split: "
                    f"{split_data.get('seed')} vs {expected_split_seed}"
                )

            # Task-specific validation
            if task == "node_classification":
                required = ["train_idx", "val_idx", "test_idx", "labels", "split_seed"]
                for field in required:
                    if field not in split_data:
                        self.validation_errors.append(
                            f"Missing field for node classification: {field}"
                        )

                expected_split_seed = self.seed_manager.get_component_seed(
                    dataset_name, repetition_id, "node_classification_split"
                )
                if split_data.get("split_seed") != expected_split_seed:
                    self.validation_errors.append(
                        f"Node-classification split seed mismatch: "
                        f"{split_data.get('split_seed')} vs {expected_split_seed}"
                    )

            elif task == "link_prediction":
                required = [
                    "train_edges", "val_edges", "test_edges",
                    "neg_train_edges", "neg_val_edges", "neg_test_edges",
                    "split_seed", "negative_sampling_seed"
                ]
                for field in required:
                    if field not in split_data:
                        self.validation_errors.append(
                            f"Missing field for link prediction: {field}"
                        )

                expected_split_seed = self.seed_manager.get_component_seed(
                    dataset_name, repetition_id, "link_prediction_edge_split"
                )
                expected_negative_seed = self.seed_manager.get_component_seed(
                    dataset_name, repetition_id, "negative_edge_sampling"
                )
                if split_data.get("split_seed") != expected_split_seed:
                    self.validation_errors.append(
                        f"Link-prediction split seed mismatch: "
                        f"{split_data.get('split_seed')} vs {expected_split_seed}"
                    )
                if split_data.get("negative_sampling_seed") != expected_negative_seed:
                    self.validation_errors.append(
                        f"Negative-sampling seed mismatch: "
                        f"{split_data.get('negative_sampling_seed')} vs {expected_negative_seed}"
                    )

            elif task == "node_ranking":
                required = ["seed_nodes", "target_nodes", "target_selection_seed"]
                for field in required:
                    if field not in split_data:
                        self.validation_errors.append(
                            f"Missing field for node ranking: {field}"
                        )

                expected_target_seed = self.seed_manager.get_component_seed(
                    dataset_name, repetition_id, "node_ranking_target_selection"
                )
                if split_data.get("target_selection_seed") != expected_target_seed:
                    self.validation_errors.append(
                        f"Node-ranking target selection seed mismatch: "
                        f"{split_data.get('target_selection_seed')} vs {expected_target_seed}"
                    )

        except Exception as e:
            # Broad for the same reason as above: recorded, not raised, so one bad
            # split file does not discard the findings for every other dataset.
            self.validation_errors.append(f"Error loading split: {type(e).__name__}: {e}")
            logger.debug("Split validation failed", exc_info=True)
        
        return len(self.validation_errors) == 0
    def _safe_load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        """Load JSON, returning None if it cannot be read.

        Returns None rather than raising because callers turn a missing or
        malformed file into a recorded validation finding. The reason is logged
        so it is recoverable -- the caller only knows *that* the load failed.
        """
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning("Could not read JSON at %s: %s", path, exc)
            return None

    def _validate_graph_fairness(
        self,
        G: nx.Graph,
        task: str,
        metadata: Dict[str, Any],
        disease_nodes: Optional[Dict[str, Any]],
    ) -> None:
        if G.number_of_nodes() == 0:
            self.validation_errors.append("Saved graph is empty")
            return

        if not nx.is_connected(G):
            self.validation_errors.append("Saved graph is not connected")

        if disease_nodes is not None:
            missing_seed_nodes = [
                node for node in disease_nodes.get("seed_node_ids", [])
                if node not in G.nodes()
            ]
            missing_target_nodes = [
                node for node in disease_nodes.get("target_node_ids", [])
                if node not in G.nodes()
            ]
            if missing_seed_nodes:
                self.validation_errors.append(
                    f"Preserved disease seed nodes missing from graph: {missing_seed_nodes}"
                )
            if missing_target_nodes:
                self.validation_errors.append(
                    f"Preserved disease target nodes missing from graph: {missing_target_nodes}"
                )

