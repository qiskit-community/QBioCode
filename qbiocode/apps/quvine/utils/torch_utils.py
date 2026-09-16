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
PyTorch Utilities and Wrappers

This module provides utilities for handling PyTorch dependencies gracefully,
including availability checks, error handling, and fallback mechanisms.
"""

import logging
from typing import Optional, Callable, Any
from functools import wraps

logger = logging.getLogger(__name__)

# Check PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    TORCH_VERSION = torch.__version__
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_VERSION = None
    torch = None
    nn = None
    F = None
    logger.warning(
        "PyTorch not available. Neural network-based methods (GAT, GraphGPS, APPNP, GraphSAGE) "
        "will not work. Install with: pip install torch"
    )


def check_torch_available() -> bool:
    """
    Check if PyTorch is available.
    
    Returns:
        bool: True if PyTorch is available, False otherwise
    """
    return TORCH_AVAILABLE


def require_torch(func: Callable) -> Callable:
    """
    Decorator to require PyTorch for a function.
    
    Raises ImportError with helpful message if PyTorch is not available.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function that checks for PyTorch
        
    Example:
        >>> @require_torch
        ... def train_model(data):
        ...     # PyTorch code here
        ...     pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not TORCH_AVAILABLE:
            raise ImportError(
                f"PyTorch is required for {func.__name__} but is not available. "
                "Install with: pip install torch>=2.0.0"
            )
        return func(*args, **kwargs)
    return wrapper


def torch_optional(fallback_value: Any = None, log_warning: bool = True) -> Callable:
    """
    Decorator for functions that optionally use PyTorch.
    
    If PyTorch is not available, returns fallback_value instead of raising error.
    
    Args:
        fallback_value: Value to return if PyTorch is not available
        log_warning: Whether to log a warning when falling back
        
    Returns:
        Decorator function
        
    Example:
        >>> @torch_optional(fallback_value=None)
        ... def optional_gpu_computation(data):
        ...     # PyTorch code here
        ...     pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not TORCH_AVAILABLE:
                if log_warning:
                    logger.warning(
                        f"{func.__name__} requires PyTorch but it's not available. "
                        f"Returning fallback value: {fallback_value}"
                    )
                return fallback_value
            return func(*args, **kwargs)
        return wrapper
    return decorator


def get_device(prefer_gpu: bool = True) -> Optional[str]:
    """
    Get the best available device for PyTorch.
    
    Args:
        prefer_gpu: Whether to prefer GPU if available
        
    Returns:
        Device string ('cuda', 'mps', 'cpu') or None if PyTorch not available
        
    Example:
        >>> device = get_device()
        >>> if device:
        ...     tensor = torch.tensor([1, 2, 3], device=device)
    """
    if not TORCH_AVAILABLE:
        return None
    
    if prefer_gpu:
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
    
    return 'cpu'


def safe_torch_import(module_name: str, class_name: Optional[str] = None) -> Optional[Any]:
    """
    Safely import a PyTorch module or class.
    
    Args:
        module_name: Name of the module to import (e.g., 'torch.nn')
        class_name: Optional class name to import from module
        
    Returns:
        Imported module/class or None if not available
        
    Example:
        >>> nn = safe_torch_import('torch.nn')
        >>> if nn:
        ...     model = nn.Linear(10, 5)
    """
    if not TORCH_AVAILABLE:
        return None
    
    try:
        if class_name:
            module = __import__(module_name, fromlist=[class_name])
            return getattr(module, class_name)
        else:
            return __import__(module_name)
    except (ImportError, AttributeError) as e:
        logger.warning(f"Failed to import {module_name}.{class_name or ''}: {e}")
        return None


class TorchNotAvailableError(ImportError):
    """Custom exception for when PyTorch is required but not available."""
    
    def __init__(self, method_name: str = "This method"):
        super().__init__(
            f"{method_name} requires PyTorch but it is not available. "
            "Install with: pip install torch>=2.0.0\n"
            "For GPU support, see: https://pytorch.org/get-started/locally/"
        )


def validate_torch_tensor(tensor: Any, name: str = "tensor") -> None:
    """
    Validate that an object is a PyTorch tensor.
    
    Args:
        tensor: Object to validate
        name: Name of the tensor for error messages
        
    Raises:
        TorchNotAvailableError: If PyTorch is not available
        TypeError: If object is not a tensor
        
    Example:
        >>> validate_torch_tensor(my_tensor, "input")
    """
    if not TORCH_AVAILABLE:
        raise TorchNotAvailableError("Tensor validation")
    
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}")


def get_torch_info() -> dict:
    """
    Get information about PyTorch installation.
    
    Returns:
        Dictionary with PyTorch information
        
    Example:
        >>> info = get_torch_info()
        >>> print(f"PyTorch version: {info['version']}")
    """
    info = {
        'available': TORCH_AVAILABLE,
        'version': TORCH_VERSION,
        'cuda_available': False,
        'cuda_version': None,
        'mps_available': False,
        'device': None,
    }
    
    if TORCH_AVAILABLE:
        info['cuda_available'] = torch.cuda.is_available()
        if info['cuda_available']:
            info['cuda_version'] = torch.version.cuda
        
        if hasattr(torch.backends, 'mps'):
            info['mps_available'] = torch.backends.mps.is_available()
        
        info['device'] = get_device()
    
    return info


# Convenience function for logging torch info
def log_torch_info():
    """Log PyTorch availability and configuration."""
    info = get_torch_info()
    
    if info['available']:
        logger.info(f"PyTorch {info['version']} is available")
        logger.info(f"Device: {info['device']}")
        if info['cuda_available']:
            logger.info(f"CUDA {info['cuda_version']} is available")
        if info['mps_available']:
            logger.info("MPS (Apple Silicon) is available")
    else:
        logger.warning("PyTorch is not available")


# Export key components
__all__ = [
    'TORCH_AVAILABLE',
    'TORCH_VERSION',
    'check_torch_available',
    'require_torch',
    'torch_optional',
    'get_device',
    'safe_torch_import',
    'TorchNotAvailableError',
    'validate_torch_tensor',
    'get_torch_info',
    'log_torch_info',
]

