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
Reproducibility Module for QuVINE

This module provides infrastructure for reproducible benchmarking:
- Centralized seed management
- Dataset registry
- Pre-generated graph storage
- Task-specific split generation and storage
- Validation checks for experimental consistency
"""

from .seed_manager import SeedManager
from .dataset_registry import DatasetRegistry
from .graph_generator import SyntheticGraphGenerator, PPIGraphGenerator
from .split_generator import SplitGenerator
from .validator import ReproducibilityValidator
from .ppi_workflow import PPIWorkflow, build_default_ppi_workflow
from .method_runner import MethodRunner

# GraphGenerator is an alias for SyntheticGraphGenerator for backwards compatibility
GraphGenerator = SyntheticGraphGenerator

__all__ = [
    'SeedManager',
    'DatasetRegistry',
    'SyntheticGraphGenerator',
    'PPIGraphGenerator',
    'GraphGenerator',
    'SplitGenerator',
    'ReproducibilityValidator',
    'PPIWorkflow',
    'build_default_ppi_workflow',
    'MethodRunner',
]

