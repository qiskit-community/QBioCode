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
Hyperparameter Loader for Tuned Methods

This module loads pre-tuned hyperparameters from JSON files and overrides
config defaults. Each dataset has its own tuning file with task-specific
parameters (node_classification, link_prediction, node_ranking).
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import fields, replace

logger = logging.getLogger(__name__)


# Map method names to their hyperparameter keys in JSON files
METHOD_TO_HYPERPARAM_KEY = {
    # 5 Baselines - direct mappings
    'node2vec': 'node2vec',
    'appnp': 'appnp',
    'graphsage': 'graphsage',
    'netmf': 'netmf',
    'baseline_gcnmf': 'baseline_gcnmf',
    
    # 10 Filter variants - map to baseline_filter
    'baseline_filter_heat': 'baseline_filter_heat',
    'baseline_filter_poly': 'baseline_filter_poly',
    'filter_rwr_heat': 'baseline_filter_heat',  # Use baseline_filter_heat tuning
    'filter_rwr_poly': 'baseline_filter_poly',  # Use baseline_filter_poly tuning
    'filter_ctqw_heat': 'baseline_filter_heat',  # Use baseline_filter_heat tuning
    'filter_ctqw_poly': 'baseline_filter_poly',  # Use baseline_filter_poly tuning
    'filter_dtqw_heat': 'baseline_filter_heat',  # Use baseline_filter_heat tuning
    'filter_dtqw_poly': 'baseline_filter_poly',  # Use baseline_filter_poly tuning
    
    # 10 GAT variants - all map to gat_baseline
    'gat_baseline': 'gat_baseline',  # Map baseline to itself
    'baseline_gat': 'gat_baseline',
    'gat_heat': 'gat_baseline',
    'gat_poly': 'gat_baseline',
    'gat_rwr': 'gat_baseline',
    'gat_ctqw_heat': 'gat_baseline',
    'gat_ctqw_poly': 'gat_baseline',
    'gat_dtqw_heat': 'gat_baseline',
    'gat_dtqw_poly': 'gat_baseline',
    'gat_rwr_heat': 'gat_baseline',
    'gat_rwr_poly': 'gat_baseline',

    # 10 GraphGPS variants - all map to graphgps_baseline
    'graphgps_baseline': 'graphgps_baseline',  # Map baseline to itself
    'baseline_graphgps': 'graphgps_baseline',
    'graphgps_heat': 'graphgps_baseline',
    'graphgps_poly': 'graphgps_baseline',
    'graphgps_rwr': 'graphgps_baseline',
    'graphgps_ctqw_heat': 'graphgps_baseline',
    'graphgps_ctqw_poly': 'graphgps_baseline',
    'graphgps_dtqw_heat': 'graphgps_baseline',
    'graphgps_dtqw_poly': 'graphgps_baseline',
    'graphgps_rwr_heat': 'graphgps_baseline',
    'graphgps_rwr_poly': 'graphgps_baseline',
    
    # 3 SGNS (QuVINE walk-based methods)
    'quvine_ctqw': 'quvine_ctqw',
    'quvine_dtqw': 'quvine_dtqw',
    'quvine_rwr': 'quvine_rwr',
}


class HyperparameterLoader:
    """
    Loads and manages tuned hyperparameters from JSON files.
    
    Example JSON structure::

        {
            "node2vec": {
                "node2vec": {
                    "node_classification": {"best_params": {...}, "best_score": 0.5},
                    "link_prediction": {"best_params": {...}, "best_score": 0.9},
                    "node_ranking": {"best_params": {...}, "best_score": 0.4}
                }
            },
            ...
        }
    """
    
    def __init__(self, tuning_dir: Optional[str] = None, dataset_name: Optional[str] = None):
        """
        Initialize hyperparameter loader.
        
        Args:
            tuning_dir: Directory containing tuning JSON files
            dataset_name: Name of the dataset (e.g., 'scale_free', 'BioPlex3_autism')
        """
        self.tuning_dir = Path(tuning_dir) if tuning_dir else None
        self.dataset_name = dataset_name
        self.hyperparams: Dict[str, Any] = {}
        self.loaded = False
        
        if self.tuning_dir and self.dataset_name:
            self.load_hyperparameters()
    
    def load_hyperparameters(self) -> bool:
        """
        Load hyperparameters from JSON file.
        
        Tries multiple filename patterns:
        1. {dataset_name}_tuning_by_task.json (new format)
        2. {dataset_name}_aggregated.json (legacy format)
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.tuning_dir or not self.dataset_name:
            logger.warning("Tuning directory or dataset name not provided")
            return False
        
        # Try multiple filename patterns
        filenames = [
            f"{self.dataset_name}_tuning_by_task.json",  # New format
            f"{self.dataset_name}_aggregated.json",       # Legacy format
        ]
        
        for filename in filenames:
            filepath = self.tuning_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, 'r') as f:
                        self.hyperparams = json.load(f)
                    self.loaded = True
                    logger.info(f"Loaded hyperparameters from {filepath}")
                    return True
                except Exception as e:
                    logger.error(f"Error loading hyperparameters from {filepath}: {e}")
                    continue
        
        logger.warning(f"Tuning file not found for dataset '{self.dataset_name}' in {self.tuning_dir}")
        logger.warning(f"Tried: {', '.join(filenames)}")
        return False
    
    def get_method_params(
        self,
        method_name: str,
        task: str = "node_classification"
    ) -> Optional[Dict[str, Any]]:
        """
        Get tuned parameters for a method and task.
        
        Args:
            method_name: Name of the method (e.g., 'node2vec', 'gat_ctqw_heat')
            task: Task type ('node_classification', 'link_prediction', 'node_ranking')
            
        Returns:
            Dictionary of best parameters, or None if not found
        """
        if not self.loaded:
            return None
        
        # Map method name to hyperparameter key
        hyperparam_key = METHOD_TO_HYPERPARAM_KEY.get(method_name)
        if not hyperparam_key:
            logger.debug(f"No hyperparameter mapping for method: {method_name}")
            return None
        
        # Navigate JSON structure: hyperparams[key][task]['best_params']
        try:
            method_data = self.hyperparams.get(hyperparam_key, {})
            task_data = method_data.get(task, {})
            best_params = task_data.get('best_params')
            
            if best_params:
                logger.debug(f"Found tuned params for {method_name} ({task})")
                return best_params
            else:
                logger.debug(f"No tuned params for {method_name} ({task})")
                return None
                
        except Exception as e:
            logger.warning(f"Error extracting params for {method_name}: {e}")
            return None
    
    def override_config(self, config, method_name: str, task: str = "node_classification"):
        """
        Override config object with tuned hyperparameters.
        
        Args:
            config: Config dataclass object
            method_name: Name of the method
            task: Task type
            
        Returns:
            Updated config object (new instance via dataclass replace)
        """
        tuned_params = self.get_method_params(method_name, task)
        if not tuned_params:
            return config
        
        # Map JSON parameter names to config attribute names
        param_mapping = self._get_param_mapping(method_name)
        
        # Build kwargs for dataclass replace
        update_kwargs = {}
        for json_key, config_attr in param_mapping.items():
            if json_key in tuned_params:
                value = tuned_params[json_key]
                # Handle string values that should be floats
                if isinstance(value, str) and 'e-' in value:
                    value = float(value)
                update_kwargs[config_attr] = value

        valid_fields = {f.name for f in fields(config)}

        # Dimension field name varies across configs: node2vec/appnp/graphsage/netmf
        # use `dimensions`, while filter/gcnmf/gat/graphgps/quvine use `embedding_dim`.
        # Route the tuned `embedding_dim` to whichever this config actually has so the
        # tuned dimension is applied rather than dropped.
        if 'embedding_dim' in tuned_params and 'embedding_dim' not in valid_fields and 'dimensions' in valid_fields:
            update_kwargs.pop('embedding_dim', None)
            update_kwargs['dimensions'] = tuned_params['embedding_dim']

        # Drop any param the config doesn't accept (e.g. a method-agnostic mapping
        # lists `epochs` but Node2VecConfig has none) so replace() never crashes.
        dropped = {k: v for k, v in update_kwargs.items() if k not in valid_fields}
        if dropped:
            logger.warning(
                f"Dropping {len(dropped)} tuned param(s) not on {type(config).__name__}: "
                f"{sorted(dropped)}"
            )
        update_kwargs = {k: v for k, v in update_kwargs.items() if k in valid_fields}

        if update_kwargs:
            logger.info(f"Overriding {len(update_kwargs)} params for {method_name}")
            return replace(config, **update_kwargs)

        return config
    
    def _get_param_mapping(self, method_name: str) -> Dict[str, str]:
        """
        Get mapping from JSON parameter names to config attribute names.
        
        Args:
            method_name: Name of the method
            
        Returns:
            Dictionary mapping JSON keys to config attributes
        """
        # Common mappings
        common = {
            'embedding_dim': 'embedding_dim',
            'hidden_dim': 'hidden_dim',
            'n_layers': 'n_layers',
            'learning_rate': 'lr',
            'weight_decay': 'weight_decay',
            'epochs': 'epochs',
            'dropout': 'dropout',
        }
        
        # Method-specific mappings
        if method_name == 'node2vec':
            return {
                **common,
                'walk_length': 'walk_length',
                'num_walks': 'num_walks',
                'p': 'p',
                'q': 'q',
                'window_size': 'window',
                'negative_samples': 'min_count',
            }
        
        elif method_name == 'appnp':
            return {
                **common,
                'alpha': 'alpha',
                'k_hops': 'K',
            }
        
        elif method_name == 'graphsage':
            return {
                **common,
                'aggregator': 'aggregator',  # May need custom handling
                'batch_size': 'batch_size',  # May need custom handling
            }
        
        elif 'filter' in method_name:
            return {
                'embedding_dim': 'embedding_dim',
                'tau': 't',
                'filter_order': 'K',
                'alpha': 'alpha',  # For poly filter
            }
        
        elif 'gat' in method_name or 'graphgps' in method_name:
            # These have nested configs, handle in override_config
            return {
                'embedding_dim': 'embedding_dim',
                'hidden_dim': 'hidden_dim',  # Will go to model config
                'n_layers': 'n_layers',  # Will go to model config
                'n_heads': 'heads',  # Will go to model config
                'dropout': 'dropout',  # Will go to model config
                'attn_dropout': 'attention_dropout',  # Will go to model config
                'learning_rate': 'lr',  # Will go to train config
                'weight_decay': 'weight_decay',  # Will go to train config
                'epochs': 'epochs',  # Will go to train config
            }
        
        elif 'gcnmf' in method_name:
            return {
                **common,
                'mf_dim': 'mf_dim',
            }
        
        elif 'netmf' in method_name:
            return {
                'embedding_dim': 'dimensions',
                'window_size': 'window_size',
                'rank': 'rank',
                'negative_samples': 'negative',
            }
        
        # Default: return common mappings
        return common
    
    def override_nested_config(self, config, method_name: str, task: str = "node_classification"):
        """
        Override nested config objects (for GAT/GraphGPS with model and train configs).
        
        Args:
            config: Config dataclass with nested model and train configs
            method_name: Name of the method
            task: Task type
            
        Returns:
            Updated config object
        """
        tuned_params = self.get_method_params(method_name, task)
        if not tuned_params:
            return config
        
        # Update model config
        model_updates = {}
        if 'hidden_dim' in tuned_params:
            model_updates['hidden_dim'] = tuned_params['hidden_dim']
        if 'n_layers' in tuned_params:
            model_updates['num_layers'] = tuned_params['n_layers']
        if 'n_heads' in tuned_params:
            model_updates['heads'] = tuned_params['n_heads']
        if 'dropout' in tuned_params:
            model_updates['dropout'] = tuned_params['dropout']
        if 'attn_dropout' in tuned_params:
            if hasattr(config.model, 'attention_dropout'):
                model_updates['attention_dropout'] = tuned_params['attn_dropout']
            elif hasattr(config.model, 'attn_dropout'):
                model_updates['attn_dropout'] = tuned_params['attn_dropout']
        
        # Update train config
        train_updates = {}
        if 'learning_rate' in tuned_params:
            train_updates['lr'] = tuned_params['learning_rate']
        if 'weight_decay' in tuned_params:
            value = tuned_params['weight_decay']
            if isinstance(value, str):
                value = float(value)
            train_updates['weight_decay'] = value
        if 'epochs' in tuned_params:
            train_updates['epochs'] = tuned_params['epochs']
        
        # Update top-level config
        top_updates = {}
        if 'embedding_dim' in tuned_params:
            top_updates['embedding_dim'] = tuned_params['embedding_dim']
            # Also update model output_dim
            model_updates['output_dim'] = tuned_params['embedding_dim']
        
        # Apply updates
        new_config = config
        if model_updates:
            new_config = replace(new_config, model=replace(config.model, **model_updates))
        if train_updates:
            new_config = replace(new_config, train=replace(new_config.train, **train_updates))
        if top_updates:
            new_config = replace(new_config, **top_updates)
        
        if model_updates or train_updates or top_updates:
            logger.info(f"Overriding nested config for {method_name}: "
                       f"{len(model_updates)} model, {len(train_updates)} train, {len(top_updates)} top")
        
        return new_config


# Global loader instance (can be set by pipeline)
_global_loader: Optional[HyperparameterLoader] = None


def set_global_loader(loader: HyperparameterLoader):
    """Set the global hyperparameter loader."""
    global _global_loader
    _global_loader = loader


def get_global_loader() -> Optional[HyperparameterLoader]:
    """Get the global hyperparameter loader."""
    return _global_loader


def load_and_override_config(config, method_name: str, task: str = "node_classification"):
    """
    Convenience function to override config with tuned hyperparameters.
    
    Args:
        config: Config dataclass object
        method_name: Name of the method
        task: Task type
        
    Returns:
        Updated config object
    """
    loader = get_global_loader()
    if not loader:
        return config
    
    # Check if config has nested structure (GAT/GraphGPS)
    if hasattr(config, 'model') and hasattr(config, 'train'):
        return loader.override_nested_config(config, method_name, task)
    else:
        return loader.override_config(config, method_name, task)

