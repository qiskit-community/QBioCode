"""
Generate synthetic blob (Gaussian cluster) datasets.

This module creates multiple configurations of blob datasets with varying
numbers of samples, features, centers, and cluster standard deviations,
useful for testing clustering and classification algorithms.
"""

from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
import json
import itertools
import os


def generate_blobs_datasets(
    n_samples,
    n_features,
    centers,
    cluster_std,
    save_path=None,
    random_state=42,
):
    """
    Generate multiple blob (Gaussian cluster) datasets with varying parameters.
    
    Creates a series of synthetic datasets consisting of isotropic Gaussian blobs
    for clustering and classification tasks. Each configuration varies the number
    of samples, features, cluster centers, and cluster spread.
    
    Parameters
    ----------
    n_samples : list of int
        List of sample sizes to generate for each configuration.
        Example: [100, 200, 300]
    n_features : list of int
        List of feature dimensions to generate.
        Example: [2, 4, 8]
    centers : list of int
        List of numbers of cluster centers (classes).
        Example: [2, 3, 4]
    cluster_std : list of float
        List of standard deviations of the clusters.
        Example: [0.5, 1.0, 1.5, 2.0]
    save_path : str, optional
        Directory path to save generated datasets. If None, datasets are not saved.
        Default: None
    random_state : int, optional
        Random seed for reproducibility.
        Default: 42
    
    Returns
    -------
    dict
        Dictionary containing generated datasets with keys as configuration
        strings and values as ``(X, y)`` tuples, where

        - ``X`` is the feature matrix, a ``pd.DataFrame`` of shape
          ``(n_samples, n_features)``
        - ``y`` holds the target labels, a ``pd.Series`` of shape
          ``(n_samples,)``
    
    Notes
    -----
    - Generates all combinations of input parameters
    - Each blob is an isotropic Gaussian distribution
    - Useful for testing classification and clustering algorithms
    - Blobs are well-separated when cluster_std is small relative to center distances
    
    Examples
    --------
    >>> from qbiocode.data_generation import generate_blobs_datasets
    >>> 
    >>> # Generate simple blob datasets
    >>> datasets = generate_blobs_datasets(
    ...     n_samples=[100, 200],
    ...     n_features=[2, 4],
    ...     centers=[2, 3],
    ...     cluster_std=[1.0, 1.5]
    ... )
    >>> 
    >>> # Access a specific configuration
    >>> X, y = datasets['n_samples_100_n_features_2_centers_2_cluster_std_1.0']
    >>> print(f"Shape: {X.shape}, Classes: {y.nunique()}")
    Shape: (100, 2), Classes: 2
    
    >>> # Save datasets to disk
    >>> datasets = generate_blobs_datasets(
    ...     n_samples=[100],
    ...     n_features=[2],
    ...     centers=[3],
    ...     cluster_std=[1.0],
    ...     save_path='./data/blobs'
    ... )
    
    See Also
    --------
    generate_circles_datasets : Generate concentric circles
    generate_moons_datasets : Generate interleaving half-circles
    generate_classification_datasets : Generate high-dimensional classification data
    
    References
    ----------
    Pedregosa et al., "Scikit-learn: Machine Learning in Python",
    JMLR 12, pp. 2825-2830, 2011.
    """
    dataset_config = {}
    
    # Generate all combinations of parameters
    param_combinations = list(itertools.product(
        n_samples, n_features, centers, cluster_std
    ))
    
    for n_samp, n_feat, n_cent, c_std in param_combinations:
        # Generate dataset
        X, y = make_blobs(
            n_samples=n_samp,
            n_features=n_feat,
            centers=n_cent,
            cluster_std=c_std,
            random_state=random_state,
            shuffle=True
        )
        
        # Convert to pandas for consistency with other QBioCode functions
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_feat)])
        y_series = pd.Series(y, name='target')
        
        # Create configuration key
        config_key = f'n_samples_{n_samp}_n_features_{n_feat}_centers_{n_cent}_cluster_std_{c_std}'
        
        # Store in dictionary
        dataset_config[config_key] = (X_df, y_series)
        
        # Save if path provided
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            
            # Save features
            X_df.to_csv(
                os.path.join(save_path, f'{config_key}_X.csv'),
                index=False
            )
            
            # Save labels
            y_series.to_csv(
                os.path.join(save_path, f'{config_key}_y.csv'),
                index=False,
                header=True
            )
            
            # Save configuration metadata
            metadata = {
                'n_samples': n_samp,
                'n_features': n_feat,
                'centers': n_cent,
                'cluster_std': c_std,
                'random_state': random_state,
                'dataset_type': 'blobs'
            }
            
            with open(os.path.join(save_path, f'{config_key}_config.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
    
    return dataset_config


# Default parameter configurations for quick generation
DEFAULT_N_SAMPLES = [100, 200, 300]
DEFAULT_N_FEATURES = [2, 4, 8]
DEFAULT_CENTERS = [2, 3, 4]
DEFAULT_CLUSTER_STD = [0.5, 1.0, 1.5, 2.0]


def generate_default_blobs_datasets(save_path=None, random_state=42):
    """
    Generate blob datasets with default parameter configurations.
    
    Convenience function that generates a standard set of blob datasets
    using predefined parameter ranges suitable for most testing scenarios.
    
    Parameters
    ----------
    save_path : str, optional
        Directory path to save generated datasets. If None, datasets are not saved.
        Default: None
    random_state : int, optional
        Random seed for reproducibility.
        Default: 42
    
    Returns
    -------
    dict
        Dictionary containing generated datasets.
    
    Examples
    --------
    >>> from qbiocode.data_generation import generate_default_blobs_datasets
    >>> datasets = generate_default_blobs_datasets()
    >>> print(f"Generated {len(datasets)} dataset configurations")
    """
    return generate_blobs_datasets(
        n_samples=DEFAULT_N_SAMPLES,
        n_features=DEFAULT_N_FEATURES,
        centers=DEFAULT_CENTERS,
        cluster_std=DEFAULT_CLUSTER_STD,
        save_path=save_path,
        random_state=random_state
    )

# Made with Bob
