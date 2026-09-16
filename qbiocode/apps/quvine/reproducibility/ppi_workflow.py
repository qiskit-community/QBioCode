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
PPI reproducibility workflow integration for QuVINE.

Builds fixed disease-specific PPI benchmark graphs, generates fixed task splits,
registers all artifacts, exports registry files, and validates readiness before
methods are run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .dataset_registry import DatasetRegistry
from .graph_generator import PPIGraphGenerator
from .seed_manager import SeedManager
from .split_generator import SplitGenerator
from .validator import ReproducibilityValidator


class PPIWorkflow:
    """
    End-to-end workflow for centralized PPI preprocessing and split generation.

    This workflow is the orchestration layer that connects:
    - PPI graph preprocessing
    - disease seed/target preservation
    - connectivity repair
    - fixed split generation
    - registry export
    - pre-run validation
    """

    DEFAULT_TASKS = ["node_ranking", "node_classification", "link_prediction"]

    def __init__(
        self,
        seed_manager: SeedManager,
        registry: DatasetRegistry,
        processed_data_dir: Path,
        data_root: Path,
        splits_root: Path,
        registry_json_path: Path,
        registry_csv_path: Optional[Path] = None,
    ):
        self.seed_manager = seed_manager
        self.registry = registry
        self.processed_data_dir = Path(processed_data_dir)
        self.data_root = Path(data_root)
        self.splits_root = Path(splits_root)
        self.registry_json_path = Path(registry_json_path)
        self.registry_csv_path = Path(registry_csv_path) if registry_csv_path is not None else None

        self.graph_generator = PPIGraphGenerator(
            output_dir=self.data_root,
            seed_manager=self.seed_manager,
            registry=self.registry,
            processed_data_dir=self.processed_data_dir,
            registry_output_path=self.registry_json_path,
        )
        self.split_generator = SplitGenerator(
            output_dir=self.splits_root,
            seed_manager=self.seed_manager,
            registry=self.registry,
        )
        self.validator = ReproducibilityValidator(
            registry=self.registry,
            seed_manager=self.seed_manager,
        )

    def _load_disease_nodes(self, disease_node_path: Path) -> Dict[str, Any]:
        with open(disease_node_path, "r") as f:
            return json.load(f)

    def _export_registry(self) -> None:
        self.registry.export_registry_json(self.registry_json_path)
        if self.registry_csv_path is not None:
            self.registry.export_registry_csv(self.registry_csv_path)

    def _build_task_split_root(
        self,
        task: str,
        ppi_source: str,
        disease: str,
        requested_size: int,
        repetition_id: int,
    ) -> Path:
        # splits_root is already .../splits/ppi — do not add "ppi" again
        network_name = f"{ppi_source}_{disease}"
        return self.splits_root / network_name / f"n{requested_size}" / f"rep_{repetition_id:02d}"

    def preprocess_single(
        self,
        ppi_source: str,
        disease: str,
        edge_path: Path,
        requested_size: int,
        repetition_id: int,
        tasks: Optional[List[str]] = None,
        task_config: Optional[Dict[str, Any]] = None,
        allow_size_expansion: bool = False,
        auto_increase_size: bool = True,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate one fixed PPI graph instance and all requested task splits.
        """
        tasks = tasks or list(self.DEFAULT_TASKS)
        task_config = task_config or {}

        G_sub, graph_path = self.graph_generator.generate_single(
            ppi_source=ppi_source,
            disease=disease,
            edge_path=Path(edge_path),
            requested_size=requested_size,
            repetition_id=repetition_id,
            allow_size_expansion=allow_size_expansion,
            auto_increase_size=auto_increase_size,
        )

        dataset_name = f"{ppi_source}_{disease}_n{requested_size}"
        entry = self.registry.get(dataset_name, repetition_id)
        if entry is None:
            raise RuntimeError(
                f"PPI dataset registration missing after graph generation for {dataset_name} rep {repetition_id}"
            )
        if entry.disease_node_path is None:
            raise RuntimeError(
                f"Disease node artifact missing for {dataset_name} rep {repetition_id}"
            )
        if entry.actual_size is None:
            raise RuntimeError(
                f"Actual graph size missing in registry for {dataset_name} rep {repetition_id}"
            )

        disease_nodes = self._load_disease_nodes(entry.disease_node_path)

        split_paths: Dict[str, Path] = {}
        for task in tasks:
            split_root = self._build_task_split_root(
                task=task,
                ppi_source=ppi_source,
                disease=disease,
                requested_size=requested_size,
                repetition_id=repetition_id,
            )
            if task == "node_ranking":
                split_path = self.split_generator.generate_node_ranking_split(
                    dataset_name=dataset_name,
                    repetition_id=repetition_id,
                    G=G_sub,
                    disease_nodes=disease_nodes,
                    split_base_dir=split_root,
                    **task_config.get("node_ranking", {}),
                )
            elif task == "node_classification":
                split_path = self.split_generator.generate_node_classification_split(
                    dataset_name=dataset_name,
                    repetition_id=repetition_id,
                    G=G_sub,
                    disease_nodes=disease_nodes,
                    split_base_dir=split_root,
                    **task_config.get("node_classification", {}),
                )
            elif task == "link_prediction":
                split_path = self.split_generator.generate_link_prediction_split(
                    dataset_name=dataset_name,
                    repetition_id=repetition_id,
                    G=G_sub,
                    disease_nodes=disease_nodes,
                    split_base_dir=split_root,
                    **task_config.get("link_prediction", {}),
                )
            else:
                raise ValueError(f"Unsupported PPI workflow task: {task}")

            split_paths[task] = split_path

        # Register task splits in-memory before validation and export.
        runtime_entry = self.registry.get(dataset_name, repetition_id)
        if runtime_entry is None:
            raise RuntimeError(
                f"PPI dataset registry entry missing before validation for {dataset_name} rep {repetition_id}"
            )
        for task in tasks:
            split_path = split_paths.get(task)
            if split_path is None:
                raise RuntimeError(f"Split path missing for task {task} in workflow output.")
            if task not in runtime_entry.available_tasks:
                runtime_entry.available_tasks.append(task)
            runtime_entry.split_paths[task] = split_path

        # Export after task info is registered so the JSON is complete.
        self._export_registry()

        validation_results: Dict[str, bool] = {}
        if validate:
            for task in tasks:
                is_valid = self.validator.validate_dataset(dataset_name, repetition_id, task)
                validation_results[task] = is_valid
                if not is_valid:
                    raise RuntimeError(
                        f"PPI workflow validation failed for {dataset_name} rep {repetition_id} task {task}: "
                        f"{self.validator.get_validation_report()}"
                    )

        return {
            "dataset_name": dataset_name,
            "repetition_id": repetition_id,
            "graph_path": graph_path,
            "split_paths": split_paths,
            "validation_results": validation_results,
        }

    def preprocess_all(
        self,
        ppi_sources: Dict[str, Path],
        diseases: List[str],
        requested_sizes: List[int],
        n_repetitions: int,
        tasks: Optional[List[str]] = None,
        task_config: Optional[Dict[str, Any]] = None,
        allow_size_expansion: bool = False,
        auto_increase_size: bool = True,
        validate: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Build all configured PPI benchmark datasets and fixed task splits.
        """
        outputs: List[Dict[str, Any]] = []

        for ppi_source, edge_path in ppi_sources.items():
            for disease in diseases:
                for requested_size in requested_sizes:
                    for repetition_id in range(n_repetitions):
                        result = self.preprocess_single(
                            ppi_source=ppi_source,
                            disease=disease,
                            edge_path=Path(edge_path),
                            requested_size=requested_size,
                            repetition_id=repetition_id,
                            tasks=tasks,
                            task_config=task_config,
                            allow_size_expansion=allow_size_expansion,
                            auto_increase_size=auto_increase_size,
                            validate=validate,
                        )
                        outputs.append(result)

        self._export_registry()
        return outputs


def build_default_ppi_workflow(
    base_seed: int,
    processed_data_dir: Path,
    data_root: Path = Path("data/ppi"),
    splits_root: Path = Path("splits"),
    registry_json_path: Path = Path("registries/ppi_dataset_registry.json"),
    registry_csv_path: Path = Path("registries/ppi_dataset_registry.csv"),
) -> PPIWorkflow:
    """
    Convenience constructor for the centralized PPI preprocessing workflow.
    """
    seed_manager = SeedManager(base_seed=base_seed)
    registry = DatasetRegistry()
    return PPIWorkflow(
        seed_manager=seed_manager,
        registry=registry,
        processed_data_dir=processed_data_dir,
        data_root=data_root,
        splits_root=splits_root,
        registry_json_path=registry_json_path,
        registry_csv_path=registry_csv_path,
    )

