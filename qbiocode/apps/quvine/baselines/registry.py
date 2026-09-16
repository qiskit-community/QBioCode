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
Method Registry for Baseline Methods

This module provides a centralized registry for managing and executing all baseline
embedding methods. It eliminates code duplication and provides consistent error handling,
timing, and logging across all methods.
"""

import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MethodMetadata:
    """
    Metadata for a registered baseline method.
    
    Attributes:
        name: Unique identifier for the method
        config_builder: Function that builds config from OmegaConf
        executor: Function that executes the method
        requires_q_targets: Whether method needs quantum targets
        requires_graph: Whether method needs graph data
        category: Method category (baseline, quantum, fusion)
        description: Human-readable description
    """
    name: str
    config_builder: Callable
    executor: Callable
    requires_q_targets: bool = False
    requires_graph: bool = True
    category: str = "baseline"
    description: str = ""


@dataclass
class MethodResult:
    """
    Result from executing a method.
    
    Attributes:
        name: Method name
        embedding: Resulting embedding matrix
        execution_time: Time taken in seconds
        success: Whether execution succeeded
        error: Error message if failed
        exception: The exception that caused the failure, so callers can chain
            (``raise ... from result.exception``) instead of losing the
            traceback to a log line
        metadata: Additional metadata
    """
    name: str
    embedding: Optional[np.ndarray] = None
    execution_time: float = 0.0
    success: bool = True
    error: Optional[str] = None
    exception: Optional[BaseException] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MethodRegistry:
    """
    Central registry for managing baseline embedding methods.
    
    This class provides:
    - Method registration and discovery
    - Consistent execution with timing
    - Error handling and logging
    - Result storage and retrieval
    
    Example:
        >>> registry = MethodRegistry(cfg, base_seed)
        >>> registry.register(method_metadata)
        >>> results = registry.run_all(graph_data, q_targets, store)
    """
    
    def __init__(self, cfg, base_seed: int, verbose: bool = False):
        """
        Initialize the method registry.
        
        Args:
            cfg: OmegaConf configuration object
            base_seed: Base random seed for reproducibility
            verbose: Whether to print verbose output
        """
        self.cfg = cfg
        self.base_seed = base_seed
        self.verbose = verbose or getattr(cfg, 'verbose', False)
        self.methods: Dict[str, MethodMetadata] = {}
        self.results: List[MethodResult] = []
        
        logger.info(f"Initialized MethodRegistry with base_seed={base_seed}")
    
    def register(self, metadata: MethodMetadata) -> None:
        """
        Register a method with the registry.
        
        Args:
            metadata: Method metadata including name, config builder, and executor
            
        Raises:
            ValueError: If method name already registered
        """
        if metadata.name in self.methods:
            raise ValueError(f"Method '{metadata.name}' already registered")
        
        self.methods[metadata.name] = metadata
        logger.debug(f"Registered method: {metadata.name} ({metadata.category})")
    
    def register_multiple(self, metadata_list: List[MethodMetadata]) -> None:
        """
        Register multiple methods at once.
        
        Args:
            metadata_list: List of method metadata objects
        """
        for metadata in metadata_list:
            self.register(metadata)
    
    def is_enabled(self, method_name: str) -> bool:
        """
        Check if a method is enabled in the configuration.
        
        Args:
            method_name: Name of the method
            
        Returns:
            True if method is enabled, False otherwise
        """
        if method_name not in self.methods:
            return False
        
        metadata = self.methods[method_name]
        
        try:
            # Build config to check if enabled
            config = metadata.config_builder(self.cfg, self.base_seed)
            return getattr(config, 'enabled', False)
        except Exception as e:
            logger.warning(f"Error checking if {method_name} is enabled: {e}")
            return False
    
    def run_method(
        self,
        method_name: str,
        graph_data,
        q_targets: Optional[List] = None,
        **kwargs
    ) -> MethodResult:
        """
        Execute a single method.
        
        Args:
            method_name: Name of the method to execute
            graph_data: NetworkX graph
            q_targets: Quantum targets (if required)
            **kwargs: Additional arguments passed to executor
            
        Returns:
            MethodResult with embedding and execution info
        """
        if method_name not in self.methods:
            return MethodResult(
                name=method_name,
                success=False,
                error=f"Method '{method_name}' not registered"
            )
        
        metadata = self.methods[method_name]
        
        # Check if method is enabled
        if not self.is_enabled(method_name):
            logger.debug(f"Method {method_name} is disabled, skipping")
            return MethodResult(
                name=method_name,
                success=False,
                error="Method disabled in configuration"
            )
        
        # Check if quantum targets are required but not provided
        if metadata.requires_q_targets and q_targets is None:
            logger.warning(f"Method {method_name} requires quantum targets but none provided, skipping")
            return MethodResult(
                name=method_name,
                success=False,
                error="Quantum targets required but not provided"
            )
        
        # Build configuration
        try:
            config = metadata.config_builder(self.cfg, self.base_seed)
        except Exception as e:
            logger.error(f"Error building config for {method_name}: {e}")
            return MethodResult(
                name=method_name,
                success=False,
                error=f"Config building failed: {str(e)}"
            )
        
        # Execute method with timing
        start_time = time.time()
        
        try:
            logger.info(f"Executing method: {method_name}")
            
            # Prepare arguments for executor
            exec_kwargs = {
                'graph_data': graph_data,
                'config': config,
                **kwargs
            }
            
            if metadata.requires_q_targets:
                exec_kwargs['q_targets'] = q_targets
            
            # Execute
            embedding = metadata.executor(**exec_kwargs)
            
            execution_time = time.time() - start_time
            
            # A library must not write to stdout: get_embeddings("netmf", ...) has to
            # be as quiet as get_embeddings("pca", ...), and QProfiler calls it once
            # per iteration per method. Route verbosity through logging so the caller
            # decides where it goes.
            if self.verbose:
                logger.info(
                    "Method %s completed in %.2f minutes", method_name, execution_time / 60
                )
            else:
                logger.debug("Method %s completed in %.2fs", method_name, execution_time)
            
            return MethodResult(
                name=method_name,
                embedding=embedding,
                execution_time=execution_time,
                success=True,
                metadata={'config': config}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            # One warning line, not a traceback: the exception travels on the result
            # so api.core can chain it (`raise QuvineMethodError(...) from exc`) and
            # the caller sees the real cause without the library dumping to stderr.
            # The full traceback stays available at DEBUG.
            logger.warning("Method %s failed: %s", method_name, e)
            logger.debug("Traceback for %s", method_name, exc_info=True)
            
            return MethodResult(
                name=method_name,
                execution_time=execution_time,
                success=False,
                error=str(e),
                exception=e,
            )
    
    def run_all(
        self,
        graph_data,
        q_targets: Optional[List] = None,
        store=None,
        method_filter: Optional[Callable[[str], bool]] = None
    ) -> List[MethodResult]:
        """
        Execute all registered and enabled methods.
        
        Args:
            graph_data: NetworkX graph
            q_targets: Quantum targets (optional)
            store: EmbeddingStore to add results to (optional)
            method_filter: Optional function to filter which methods to run
            
        Returns:
            List of MethodResult objects
        """
        results = []
        
        # Filter methods if filter provided
        methods_to_run = self.methods.keys()
        if method_filter:
            methods_to_run = [name for name in methods_to_run if method_filter(name)]
        
        logger.info(f"Running {len(methods_to_run)} methods")
        
        for method_name in methods_to_run:
            result = self.run_method(method_name, graph_data, q_targets)
            results.append(result)
            
            # Add to store if provided and successful
            if store is not None and result.success and result.embedding is not None:
                store.add(method_name, result.embedding)
        
        self.results.extend(results)
        
        # Summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        total_time = sum(r.execution_time for r in results)
        
        logger.info(
            f"Completed {len(results)} methods: "
            f"{successful} successful, {failed} failed, "
            f"total time: {total_time/60:.2f} minutes"
        )
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Registry Summary: {successful}/{len(results)} methods successful")
            print(f"Total execution time: {total_time/60:.2f} minutes")
            print(f"{'='*60}\n")
        
        return results
    
    def run_category(
        self,
        category: str,
        graph_data,
        q_targets: Optional[List] = None,
        store=None
    ) -> List[MethodResult]:
        """
        Execute all methods in a specific category.
        
        Args:
            category: Category name (e.g., 'baseline', 'quantum')
            graph_data: NetworkX graph
            q_targets: Quantum targets (optional)
            store: EmbeddingStore to add results to (optional)
            
        Returns:
            List of MethodResult objects
        """
        def category_filter(name: str) -> bool:
            return self.methods[name].category == category
        
        return self.run_all(graph_data, q_targets, store, method_filter=category_filter)
    
    def get_results(self, successful_only: bool = False) -> List[MethodResult]:
        """
        Get all execution results.
        
        Args:
            successful_only: If True, return only successful results
            
        Returns:
            List of MethodResult objects
        """
        if successful_only:
            return [r for r in self.results if r.success]
        return self.results
    
    def get_result(self, method_name: str) -> Optional[MethodResult]:
        """
        Get result for a specific method.
        
        Args:
            method_name: Name of the method
            
        Returns:
            MethodResult if found, None otherwise
        """
        for result in reversed(self.results):  # Get most recent
            if result.name == method_name:
                return result
        return None
    
    def list_methods(self, category: Optional[str] = None) -> List[str]:
        """
        List all registered methods.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of method names
        """
        if category:
            return [
                name for name, meta in self.methods.items()
                if meta.category == category
            ]
        return list(self.methods.keys())
    
    def get_metadata(self, method_name: str) -> Optional[MethodMetadata]:
        """
        Get metadata for a method.
        
        Args:
            method_name: Name of the method
            
        Returns:
            MethodMetadata if found, None otherwise
        """
        return self.methods.get(method_name)
    
    def clear_results(self) -> None:
        """Clear all stored results."""
        self.results.clear()
        logger.debug("Cleared all results")
    
    def __len__(self) -> int:
        """Return number of registered methods."""
        return len(self.methods)
    
    def __contains__(self, method_name: str) -> bool:
        """Check if method is registered."""
        return method_name in self.methods
    
    def __repr__(self) -> str:
        """String representation."""
        return f"MethodRegistry(methods={len(self.methods)}, results={len(self.results)})"

