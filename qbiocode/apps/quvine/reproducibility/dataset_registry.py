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
Dataset Registry for QuVINE

Central registry mapping dataset names to their locations, metadata, and task splits.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict
from enum import Enum


class DatasetType(Enum):
    """Type of dataset."""
    SYNTHETIC = "synthetic"
    REAL_WORLD = "real_world"
    PPI = "ppi"


@dataclass
class DatasetMetadata:
    """Metadata for a dataset."""
    name: str
    dataset_type: DatasetType
    num_nodes: int
    num_edges: int
    is_directed: bool
    has_node_features: bool
    has_node_labels: bool
    generation_params: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    ppi_source: Optional[str] = None
    disease: Optional[str] = None
    requested_size: Optional[int] = None
    actual_size: Optional[int] = None
    largest_connected_component_size: Optional[int] = None


@dataclass
class DatasetEntry:
    """
    Registry entry for a single dataset instance.

    Each entry represents one graph instance for a fixed
    dataset/source/disease/size/repetition combination.
    """
    dataset_name: str
    dataset_type: DatasetType
    repetition_id: int
    seed: int
    graph_path: Path
    metadata_path: Path
    available_tasks: List[str]
    split_paths: Dict[str, Path]  # task -> split file path
    disease_node_path: Optional[Path] = None
    ppi_source: Optional[str] = None
    disease: Optional[str] = None
    requested_size: Optional[int] = None
    actual_size: Optional[int] = None
    registry_group: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "dataset_name": self.dataset_name,
            "dataset_type": self.dataset_type.value,
            "repetition_id": self.repetition_id,
            "seed": self.seed,
            "graph_path": str(self.graph_path),
            "metadata_path": str(self.metadata_path),
            "available_tasks": self.available_tasks,
            "split_paths": {task: str(path) for task, path in self.split_paths.items()},
            "disease_node_path": str(self.disease_node_path) if self.disease_node_path else None,
            "ppi_source": self.ppi_source,
            "disease": self.disease,
            "requested_size": self.requested_size,
            "actual_size": self.actual_size,
            "registry_group": self.registry_group,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'DatasetEntry':
        """Create from dictionary."""
        return cls(
            dataset_name=data["dataset_name"],
            dataset_type=DatasetType(data["dataset_type"]),
            repetition_id=data["repetition_id"],
            seed=data["seed"],
            graph_path=Path(data["graph_path"]),
            metadata_path=Path(data["metadata_path"]),
            available_tasks=data["available_tasks"],
            split_paths={task: Path(path) for task, path in data["split_paths"].items()},
            disease_node_path=Path(data["disease_node_path"]) if data.get("disease_node_path") else None,
            ppi_source=data.get("ppi_source"),
            disease=data.get("disease"),
            requested_size=data.get("requested_size"),
            actual_size=data.get("actual_size"),
            registry_group=data.get("registry_group"),
        )


class DatasetRegistry:
    """
    Central registry for all datasets used in QuVINE experiments.
    
    Maintains a mapping from (dataset_name, repetition_id) to dataset locations
    and metadata. Ensures all methods access the same data files.
    """
    
    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize dataset registry.
        
        Parameters
        ----------
        registry_path : Path, optional
            Path to registry JSON file. If provided, loads existing registry.
        """
        self.entries: Dict[tuple, DatasetEntry] = {}
        self.registry_path = registry_path
        
        if registry_path and registry_path.exists():
            self.load(registry_path)
    
    def register(
        self,
        dataset_name: str,
        dataset_type: DatasetType,
        repetition_id: int,
        seed: int,
        graph_path: Path,
        metadata_path: Path,
        available_tasks: List[str],
        split_paths: Optional[Dict[str, Path]] = None,
        disease_node_path: Optional[Path] = None,
        ppi_source: Optional[str] = None,
        disease: Optional[str] = None,
        requested_size: Optional[int] = None,
        actual_size: Optional[int] = None,
        registry_group: Optional[str] = None
    ) -> None:
        """
        Register a dataset instance.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset
        dataset_type : DatasetType
            Type of dataset
        repetition_id : int
            Repetition index
        seed : int
            Seed used for generation/sampling
        graph_path : Path
            Path to saved graph file
        metadata_path : Path
            Path to metadata JSON
        available_tasks : List[str]
            Tasks available for this dataset
        split_paths : Dict[str, Path], optional
            Mapping from task name to split file path
        """
        key = (dataset_name, repetition_id)
        
        entry = DatasetEntry(
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            repetition_id=repetition_id,
            seed=seed,
            graph_path=graph_path,
            metadata_path=metadata_path,
            available_tasks=available_tasks,
            split_paths=split_paths or {},
            disease_node_path=disease_node_path,
            ppi_source=ppi_source,
            disease=disease,
            requested_size=requested_size,
            actual_size=actual_size,
            registry_group=registry_group,
        )
        
        self.entries[key] = entry
    
    def get(self, dataset_name: str, repetition_id: int) -> Optional[DatasetEntry]:
        """
        Get dataset entry.
        
        Parameters
        ----------
        dataset_name : str
            Dataset name
        repetition_id : int
            Repetition ID
        
        Returns
        -------
        DatasetEntry or None
            Dataset entry if found, None otherwise
        """
        return self.entries.get((dataset_name, repetition_id))
    
    def add_split(
        self,
        dataset_name: str,
        repetition_id: int,
        task: str,
        split_path: Path
    ) -> None:
        """
        Add a task split to an existing dataset entry.
        
        Parameters
        ----------
        dataset_name : str
            Dataset name
        repetition_id : int
            Repetition ID
        task : str
            Task name
        split_path : Path
            Path to split file
        """
        key = (dataset_name, repetition_id)
        if key not in self.entries:
            raise ValueError(f"Dataset {dataset_name} rep {repetition_id} not registered")
        
        self.entries[key].split_paths[task] = split_path
        
        if task not in self.entries[key].available_tasks:
            self.entries[key].available_tasks.append(task)
    
    def get_all_datasets(self) -> List[str]:
        """Get list of all unique dataset names."""
        return sorted(set(name for name, _ in self.entries.keys()))
    
    def get_repetitions(self, dataset_name: str) -> List[int]:
        """Get all repetition IDs for a dataset."""
        return sorted([rep for name, rep in self.entries.keys() if name == dataset_name])
    
    def get_entries_for_dataset(self, dataset_name: str) -> List[DatasetEntry]:
        """Get all entries for a dataset across all repetitions."""
        return [
            entry for (name, _), entry in self.entries.items() 
            if name == dataset_name
        ]
    
    def validate_consistency(
        self,
        dataset_name: str,
        repetition_id: int,
        task: str
    ) -> bool:
        """
        Validate that all required files exist for a dataset-task combination.
        
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
            True if all files exist, False otherwise
        """
        entry = self.get(dataset_name, repetition_id)
        if entry is None:
            return False
        
        # Check graph and metadata exist
        if not entry.graph_path.exists():
            return False
        if not entry.metadata_path.exists():
            return False
        if entry.disease_node_path is not None and not entry.disease_node_path.exists():
            return False
        
        # Check task split exists
        if task not in entry.split_paths:
            return False
        if not entry.split_paths[task].exists():
            return False
        
        return True
    
    def save(self, output_path: Optional[Path] = None) -> None:
        """
        Save registry to JSON file.
        
        Parameters
        ----------
        output_path : Path, optional
            Output path. If None, uses self.registry_path.
        """
        path = output_path or self.registry_path
        if path is None:
            raise ValueError("No output path specified")
        
        data = {
            "entries": [
                {
                    "key": f"{name}_{rep}",
                    "entry": entry.to_dict()
                }
                for (name, rep), entry in self.entries.items()
            ]
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, input_path: Path) -> None:
        """
        Load registry from JSON file.
        
        Parameters
        ----------
        input_path : Path
            Input path
        """
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        self.entries = {}
        for item in data["entries"]:
            entry = DatasetEntry.from_dict(item["entry"])
            key = (entry.dataset_name, entry.repetition_id)
            self.entries[key] = entry
        
        self.registry_path = input_path
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the registry.
        
        Returns
        -------
        dict
            Summary statistics
        """
        datasets = self.get_all_datasets()
        
        summary = {
            "total_datasets": len(datasets),
            "total_entries": len(self.entries),
            "datasets": {}
        }
        
        for dataset in datasets:
            reps = self.get_repetitions(dataset)
            entries = self.get_entries_for_dataset(dataset)
            
            tasks = set()
            for entry in entries:
                tasks.update(entry.available_tasks)
            
            summary["datasets"][dataset] = {
                "num_repetitions": len(reps),
                "repetition_ids": reps,
                "available_tasks": sorted(tasks),
                "dataset_type": entries[0].dataset_type.value if entries else None
            }
        
        return summary

    def export_registry_json(self, output_path: Path) -> None:
        """Explicit helper for writing a registry JSON artifact."""
        self.save(output_path)

    def export_registry_csv(self, output_path: Path) -> None:
        """Export flattened registry records to CSV."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "dataset_name",
            "dataset_type",
            "repetition_id",
            "seed",
            "graph_path",
            "metadata_path",
            "disease_node_path",
            "ppi_source",
            "disease",
            "requested_size",
            "actual_size",
            "registry_group",
            "available_tasks",
            "split_paths",
        ]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for (_, _), entry in sorted(self.entries.items(), key=lambda item: (item[0][0], item[0][1])):
                writer.writerow({
                    "dataset_name": entry.dataset_name,
                    "dataset_type": entry.dataset_type.value,
                    "repetition_id": entry.repetition_id,
                    "seed": entry.seed,
                    "graph_path": str(entry.graph_path),
                    "metadata_path": str(entry.metadata_path),
                    "disease_node_path": str(entry.disease_node_path) if entry.disease_node_path else "",
                    "ppi_source": entry.ppi_source or "",
                    "disease": entry.disease or "",
                    "requested_size": entry.requested_size if entry.requested_size is not None else "",
                    "actual_size": entry.actual_size if entry.actual_size is not None else "",
                    "registry_group": entry.registry_group or "",
                    "available_tasks": json.dumps(sorted(entry.available_tasks)),
                    "split_paths": json.dumps({task: str(path) for task, path in entry.split_paths.items()}),
                })

