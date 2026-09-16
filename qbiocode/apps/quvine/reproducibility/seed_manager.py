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
Centralized Seed Manager for QuVINE

Ensures explicit, auditable seed usage across all experiments for reproducibility.
"""

import json
from pathlib import Path
from typing import Dict, Optional


class SeedManager:
    """
    Manages canonical and derived random seeds for reproducible experiments.

    Canonical seed policy required for the PPI benchmark:
        canonical_seed = base_seed + repetition_id

    The canonical seed is intentionally independent of dataset identity so that
    all methods for the same repetition operate under the exact same top-level
    seed policy. Component/task-specific seeds are derived from the canonical
    seed via fixed offsets to avoid untracked randomness.
    """

    COMPONENT_OFFSETS = {
        "graph_generation": 0,
        "ppi_subsampling": 10,
        "filler_node_selection": 11,
        "connector_repair": 12,
        "node_ranking": 100,
        "node_ranking_target_selection": 101,
        "node_classification": 200,
        "node_classification_split": 201,
        "link_prediction": 300,
        "link_prediction_edge_split": 301,
        "negative_edge_sampling": 302,
        "model_initialization": 400,
        "dataloader_shuffling": 500,
    }

    TASK_TO_COMPONENT = {
        "node_ranking": "node_ranking",
        "node_classification": "node_classification",
        "link_prediction": "link_prediction",
    }

    # Component used to seed the train/val/test SPLIT for each task. Distinct from
    # TASK_TO_COMPONENT (which seeds the model run): SplitGenerator draws splits
    # with these components, so a recorded split["seed"] must be validated against
    # these, not against the model-run task seed.
    TASK_TO_SPLIT_COMPONENT = {
        "node_ranking": "node_ranking",
        "node_classification": "node_classification_split",
        "link_prediction": "link_prediction_edge_split",
    }

    def __init__(self, base_seed: int = 42):
        """
        Initialize seed manager.

        Parameters
        ----------
        base_seed : int
            Global base seed for all experiments
        """
        self.base_seed = int(base_seed)
        self._seed_cache: Dict[tuple, int] = {}

    def get_seed(self, dataset_name: str, repetition_id: int) -> int:
        """
        Get canonical seed for a dataset and repetition.

        Parameters
        ----------
        dataset_name : str
            Dataset name (retained for API compatibility and auditing)
        repetition_id : int
            Repetition index (0-based)

        Returns
        -------
        int
            Canonical seed for this repetition
        """
        key = (dataset_name, repetition_id)
        if key not in self._seed_cache:
            self._seed_cache[key] = int(self.base_seed + repetition_id)
        return self._seed_cache[key]

    def get_component_seed(
        self,
        dataset_name: str,
        repetition_id: int,
        component: str,
        extra_offset: int = 0
    ) -> int:
        """
        Get a deterministic seed for a specific pipeline component.

        Parameters
        ----------
        dataset_name : str
            Dataset name
        repetition_id : int
            Repetition index
        component : str
            Named component from COMPONENT_OFFSETS
        extra_offset : int
            Additional explicit offset for sub-operations

        Returns
        -------
        int
            Deterministic derived seed
        """
        canonical_seed = self.get_seed(dataset_name, repetition_id)
        component_offset = self.COMPONENT_OFFSETS.get(component)
        if component_offset is None:
            raise ValueError(f"Unknown seed component: {component}")
        return int(canonical_seed + component_offset + extra_offset)

    def get_task_seed(
        self,
        dataset_name: str,
        repetition_id: int,
        task: str
    ) -> int:
        """
        Get task-specific seed derived from canonical seed.

        Parameters
        ----------
        dataset_name : str
            Dataset name
        repetition_id : int
            Repetition index
        task : str
            Task name

        Returns
        -------
        int
            Task-specific seed
        """
        component = self.TASK_TO_COMPONENT.get(task)
        if component is None:
            raise ValueError(f"Unknown task for seed derivation: {task}")
        return self.get_component_seed(dataset_name, repetition_id, component)

    def get_split_seed(
        self,
        dataset_name: str,
        repetition_id: int,
        task: str
    ) -> int:
        """
        Get the seed used to draw the train/val/test split for a task.

        This is the component SplitGenerator uses when generating the split (e.g.
        ``node_classification_split``), which differs from the model-run task seed
        (``get_task_seed``). Use this to validate a recorded ``split["seed"]``.
        """
        component = self.TASK_TO_SPLIT_COMPONENT.get(task)
        if component is None:
            raise ValueError(f"Unknown task for split-seed derivation: {task}")
        return self.get_component_seed(dataset_name, repetition_id, component)

    def describe_seed_plan(self, dataset_name: str, repetition_id: int) -> Dict[str, int]:
        """Return the full seed plan for auditing."""
        plan = {
            "canonical_seed": self.get_seed(dataset_name, repetition_id)
        }
        for component, offset in sorted(self.COMPONENT_OFFSETS.items(), key=lambda x: x[1]):
            plan[component] = self.get_component_seed(dataset_name, repetition_id, component)
        return plan

    def save_seed_registry(self, output_path: Path) -> None:
        """
        Save current seed registry to disk for auditing.

        Parameters
        ----------
        output_path : Path
            Path to save seed registry JSON
        """
        registry = {
            "base_seed": self.base_seed,
            "policy": "canonical_seed = base_seed + repetition_id",
            "component_offsets": self.COMPONENT_OFFSETS,
            "seeds": {
                f"{dataset}_{rep}": {
                    "canonical_seed": seed,
                    "seed_plan": self.describe_seed_plan(dataset, rep),
                }
                for (dataset, rep), seed in self._seed_cache.items()
            }
        }

        with open(output_path, 'w') as f:
            json.dump(registry, f, indent=2)

    def load_seed_registry(self, input_path: Path) -> None:
        """
        Load seed registry from disk.

        Parameters
        ----------
        input_path : Path
            Path to seed registry JSON
        """
        with open(input_path, 'r') as f:
            registry = json.load(f)

        self.base_seed = int(registry["base_seed"])
        self._seed_cache = {}

        for key_str, seed_info in registry["seeds"].items():
            dataset, rep_str = key_str.rsplit('_', 1)
            rep = int(rep_str)
            if isinstance(seed_info, dict):
                seed = int(seed_info["canonical_seed"])
            else:
                seed = int(seed_info)
            self._seed_cache[(dataset, rep)] = seed

    def validate_seed_consistency(
        self,
        dataset_name: str,
        repetition_id: int,
        expected_seed: int
    ) -> bool:
        """
        Validate that computed canonical seed matches expected value.
        """
        actual_seed = self.get_seed(dataset_name, repetition_id)
        return actual_seed == int(expected_seed)

