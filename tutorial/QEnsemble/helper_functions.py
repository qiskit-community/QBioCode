"""
Helper functions for Quantum Ensemble Tutorial
===============================================

This module provides helper functions for running classical baseline comparisons
in the tutorial notebook. These functions maintain compatibility with the original
tutorial structure while being standalone.
"""

import os
import re
from collections import Counter
from typing import Any
import numpy as np
import pandas as pd
import pickle
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score

# Import from QBioCode API
from qbiocode.evaluation import evaluation_metrics

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lazypredict.Supervised import LazyClassifier
    LAZY_AVAILABLE = True
except ImportError:
    LAZY_AVAILABLE = False


def needs_run(predictions, dataset_name, method, rerun=False):
    """
    Decide whether a method still has to be computed for a dataset.

    The notebook caches every arm in ``experiments/predictions.pkl`` and skips a
    cell whose method is already a key of that cache. A method that *ran but
    produced nothing* is also a key, though, so a result recorded in an
    environment that lacked an optional dependency was cached as an empty frame
    and never retried -- which is how the committed cache came to hold
    ``xgb_gs`` with 0 rows, making the notebook print "Skipping XGBoost" on an
    install where XGBoost is present. An empty cached result therefore counts as
    absent.

    Parameters
    ----------
    predictions : dict
        The nested ``{dataset_name: {method: DataFrame}}`` cache.
    dataset_name : str
        Dataset key.
    method : str
        Method key.
    rerun : bool
        Force recomputation, ignoring anything cached.

    Returns
    -------
    bool
        True if the caller should compute this arm.
    """
    if rerun or dataset_name not in predictions:
        return True
    cached = predictions[dataset_name].get(method)
    if cached is None:
        return True
    return len(cached) == 0


def run_random_forest(predictions, dataset, method, dataset_name, seed, test_size, file_predictions, select_features=[],
                      params={}, pca_embed=False, umap_embed=False, n_features=0):
    """
    Run Random Forest classifier with optional hyperparameter search.
    
    Parameters
    ----------
    predictions : dict
        Dictionary to store prediction results
    dataset : dict
        Dictionary containing train/test splits
    method : str
        Method name (e.g., 'random_forest', 'random_forest_gs')
    dataset_name : str
        Name of the dataset
    seed : int
        Random seed for reproducibility
    test_size : float
        Test set size (not used, kept for compatibility)
    file_predictions : str
        Path to save predictions
    select_features : list, optional
        List of features to select (default: [])
    params : dict, optional
        Fixed hyperparameters (default: {})
    pca_embed : bool, optional
        Whether to use PCA embedding (default: False)
    umap_embed : bool, optional
        Whether to use UMAP embedding (default: False)
    n_features : int, optional
        Number of features for dimensionality reduction (default: 0)
    
    Returns
    -------
    dict
        Updated predictions dictionary
    """
    if (dataset_name not in predictions.keys()) or (method not in predictions[dataset_name].keys()):
        results_df = pd.DataFrame(columns=['dataset', 'method', 'dataset_params', 'seed', 'n_feature',
                                           'select_features', 'accuracy', 'brier', 'predictions', 'y_test', 'best_params'])
        if dataset_name not in predictions.keys():
            predictions[dataset_name] = {}
        predictions[dataset_name][method] = results_df
    else:
        results_df = predictions[dataset_name][method]

    # Define the hyperparameter grid
    param_distributions = {
        'n_estimators': np.arange(100, 1000, 100),
        'max_depth': np.arange(5, 20),
        'min_samples_split': np.arange(2, 10),
        'min_samples_leaf': np.arange(1, 5),
        'max_features': ['sqrt', 'log2']
    }

    for k, v in dataset[dataset_name].items():
        (X_train, X_test, y_train, y_test) = v
        if len(select_features) > 0:
            X_train = X_train.loc[:, select_features]
            X_test = X_test.loc[:, select_features]
            
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        embed = 'none'

        if pca_embed:
            embed = 'pca'
            embedder = PCA(n_features)
            X_train = embedder.fit_transform(X_train)
            X_test = embedder.transform(X_test)
        elif umap_embed:
            if not UMAP_AVAILABLE:
                raise ImportError("UMAP not available. Install with: pip install umap-learn")
            embed = 'umap'
            reducer = umap.UMAP(n_features)
            X_train = reducer.fit_transform(X_train)
            X_test = reducer.transform(X_test)

        if len(params) > 0:
            # Initialize the Random Forest Classifier
            rf = RandomForestClassifier(random_state=seed, 
                                        n_estimators=params['n_estimators'],
                                        min_samples_split=params['min_samples_split'],
                                        max_features=params['max_features'],
                                        max_depth=params['max_depth'],
                                        min_samples_leaf=params['min_samples_leaf']
                                        )

            rf.fit(X_train, y_train)
            preds = rf.predict_proba(X_test)
        else:
            # Initialize the Random Forest Classifier
            rf = RandomForestClassifier(random_state=seed)

            # Initialize RandomizedSearchCV
            rf_random = RandomizedSearchCV(estimator=rf, 
                                        param_distributions=param_distributions, 
                                        n_iter=10, 
                                        cv=3, 
                                        random_state=seed,
                                        n_jobs=-1)
            rf_random.fit(X_train, y_train)
            preds = rf_random.predict_proba(X_test)
            params = rf_random.best_params_

        a, b = evaluation_metrics(preds, y_test, save=False)    

        res = pd.DataFrame([dataset_name, method, k, seed, X_train.shape[1], ','.join(select_features), a, b, preds, y_test, params],
                        index=['dataset', 'method', 'dataset_params', 'seed', 'n_feature', 'select_features', 'accuracy', 'brier', 'predictions', 'y_test', 'best_params']).transpose()
        results_df = pd.concat([results_df, res])
            
        predictions[dataset_name][method] = results_df
        
        # save
        pickle.dump(predictions, open(file_predictions, 'wb'))
        
    return predictions


def run_xgboost(predictions, dataset, method, dataset_name, seed, test_size, file_predictions, select_features=[],
                params={}, pca_embed=False, umap_embed=False, n_features=0):
    """
    Run XGBoost classifier with optional hyperparameter search.
    
    Parameters
    ----------
    predictions : dict
        Dictionary to store prediction results
    dataset : dict
        Dictionary containing train/test splits
    method : str
        Method name (e.g., 'xgb', 'xgb_gs')
    dataset_name : str
        Name of the dataset
    seed : int
        Random seed for reproducibility
    test_size : float
        Test set size (not used, kept for compatibility)
    file_predictions : str
        Path to save predictions
    select_features : list, optional
        List of features to select (default: [])
    params : dict, optional
        Fixed hyperparameters (default: {})
    pca_embed : bool, optional
        Whether to use PCA embedding (default: False)
    umap_embed : bool, optional
        Whether to use UMAP embedding (default: False)
    n_features : int, optional
        Number of features for dimensionality reduction (default: 0)
    
    Returns
    -------
    dict
        Updated predictions dictionary
    """
    if not XGB_AVAILABLE:
        raise ImportError("XGBoost not available. Install with: pip install xgboost")
    
    if (dataset_name not in predictions.keys()) or (method not in predictions[dataset_name].keys()):
        results_df = pd.DataFrame(columns=['dataset', 'method', 'dataset_params', 'seed', 'n_feature',
                                           'select_features', 'accuracy', 'brier', 'predictions', 'y_test', 'best_params'])
        if dataset_name not in predictions.keys():
            predictions[dataset_name] = {}
        predictions[dataset_name][method] = results_df
    else:
        results_df = predictions[dataset_name][method]

    for k, v in dataset[dataset_name].items():
        (X_train, X_test, y_train, y_test) = v
        if len(select_features) > 0:
            X_train = X_train.loc[:, select_features]
            X_test = X_test.loc[:, select_features]
            
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        embed = 'none'

        if pca_embed:
            embed = 'pca'
            embedder = PCA(n_features)
            X_train = embedder.fit_transform(X_train)
            X_test = embedder.transform(X_test)
        elif umap_embed:
            if not UMAP_AVAILABLE:
                raise ImportError("UMAP not available. Install with: pip install umap-learn")
            embed = 'umap'
            reducer = umap.UMAP(n_features)
            X_train = reducer.fit_transform(X_train)
            X_test = reducer.transform(X_test)

        ##XGB
        if len(params) > 0:
            # Initialize XGB
            xgb = XGBClassifier(
                random_state=seed,
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                min_child_weight=params['min_child_weight'],
                eval_metric='logloss'
            )
            xgb.fit(X_train, y_train)
            preds = xgb.predict_proba(X_test)
        else:
            xgb = XGBClassifier(
                random_state=seed,
                eval_metric='logloss'
            )

            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
                    
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.7, 0.8, 1.0],
                'colsample_bytree': [0.7, 0.8, 1.0],
                'min_child_weight': [1, 3, 5]
            }
                    
            xgb_grid = GridSearchCV(
                estimator=xgb,
                param_grid=param_grid,
                scoring='f1_weighted',
                n_jobs=-1,
                cv=cv,
                verbose=1
            )

            xgb_grid.fit(X_train, y_train)
            preds = xgb_grid.predict_proba(X_test)
            params = xgb_grid.best_params_

        a, b = evaluation_metrics(preds, y_test, save=False)    

        res = pd.DataFrame([dataset_name, method, k, seed, X_train.shape[1], ','.join(select_features), a, b, preds, y_test, params],
                        index=['dataset', 'method', 'dataset_params', 'seed', 'n_feature', 'select_features', 'accuracy', 'brier', 'predictions', 'y_test', 'best_params']).transpose()
        results_df = pd.concat([results_df, res])
            
        predictions[dataset_name][method] = results_df
        
        # save
        pickle.dump(predictions, open(file_predictions, 'wb'))
        
    return predictions


def run_lazy_predict(predictions, dataset, method, dataset_name, seed, test_size, file_predictions, select_features=[]):
    """
    Run LazyPredict for automated model comparison.
    
    Parameters
    ----------
    predictions : dict
        Dictionary to store prediction results
    dataset : dict
        Dictionary containing train/test splits
    method : str
        Method name
    dataset_name : str
        Name of the dataset
    seed : int
        Random seed for reproducibility
    test_size : float
        Test set size (not used, kept for compatibility)
    file_predictions : str
        Path to save predictions
    select_features : list, optional
        List of features to select (default: [])
    
    Returns
    -------
    dict
        Updated predictions dictionary
    """
    if not LAZY_AVAILABLE:
        raise ImportError("LazyPredict not available. Install with: pip install lazypredict")
    
    if dataset_name not in predictions.keys():
        predictions[dataset_name] = {}
    predictions[dataset_name][method] = {}

    for k, v in dataset[dataset_name].items():
        (X_train, X_test, y_train, y_test) = v
        if len(select_features) > 0:
            X_train = X_train.loc[:, select_features]
            X_test = X_test.loc[:, select_features]
            
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        models, preds = clf.fit(X_train, X_test, y_train, y_test)
        
        predictions[dataset_name][method][k] = {}
        predictions[dataset_name][method][k]['seed'] = seed
        predictions[dataset_name][method][k]['models'] = models
        predictions[dataset_name][method][k]['preds'] = preds
        predictions[dataset_name][method][k]['y_test'] = y_test
        predictions[dataset_name][method][k]['select_features'] = select_features

    # save
    pickle.dump(predictions, open(file_predictions, 'wb'))
    
    return predictions


def run_quantum_cosine(predictions, dataset, method, dataset_name, seed, test_size, file_predictions, n_features,
                       n_trains, n_shots, pca_embed=False, umap_embed=False, select_features=[]):
    """
    Run quantum cosine classifier experiments.
    
    This function runs a simple quantum cosine similarity classifier across
    multiple parameter configurations. It uses the SWAP test to measure
    cosine similarity between training and test quantum states.
    
    Parameters
    ----------
    predictions : dict
        Dictionary to store prediction results
    dataset : dict
        Dictionary containing dataset splits
    method : str
        Method name for results tracking
    dataset_name : str
        Name of the dataset
    seed : int
        Random seed for reproducibility
    test_size : float
        Fraction of data for testing
    file_predictions : str
        Path to save predictions
    n_features : list of int
        List of feature counts to test
    n_trains : list of int
        List of training sample sizes to test
    n_shots : int
        Number of measurement shots
    pca_embed : bool, optional
        Use PCA for dimensionality reduction (default: False)
    umap_embed : bool, optional
        Use UMAP for dimensionality reduction (default: False)
    select_features : list, optional
        Specific features to select (default: [])
    
    Returns
    -------
    dict
        Updated predictions dictionary with results
    """
    from qbiocode.learning.compute_qensemble import build_cosine_classifier
    from qbiocode.utils import normalize_data, label_to_array, execute_circuit, retrieve_probabilities
    
    epsilon = 1e-15
    if (dataset_name not in predictions.keys()) or (method not in predictions[dataset_name].keys()):
        results_df = pd.DataFrame()
        if dataset_name not in predictions.keys():
            predictions[dataset_name] = {}
        predictions[dataset_name][method] = results_df
    else:
        results_df = predictions[dataset_name][method]
        
    for k, v in dataset[dataset_name].items():
        for f in n_features:
            (X_train, X_test, y_train, y_test) = v
            if len(select_features) > 0:
                X_train = X_train.loc[:, select_features]
                X_test = X_test.loc[:, select_features]
                
            Y_vector_train = label_to_array(y_train)
            Y_vector_test = label_to_array(y_test)
            test_size = Y_vector_test.shape[0]
            train_size = Y_vector_train.shape[0]
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            embed = 'none'

            # adding epsilon to avoid zero division
            X_train = X_train + epsilon
            X_test = X_test + epsilon

            if pca_embed:
                embed = 'pca'
                embedder = PCA(f)
                X_train = embedder.fit_transform(X_train)
                X_test = embedder.transform(X_test)
            elif umap_embed:
                if not UMAP_AVAILABLE:
                    raise ImportError("UMAP not available. Install with: pip install umap-learn")
                embed = 'umap'
                reducer = umap.UMAP(f)
                X_train = reducer.fit_transform(X_train)
                X_test = reducer.transform(X_test)
                    
            for n_train in n_trains:
                preds = []
                
                if (len(results_df) == 0) or (len(results_df[(results_df['dataset'] == dataset_name) &
                                                             (results_df['method'] == method) &
                                                             (results_df['dataset_params'] == k) &
                                                             (results_df['n_feature'] == f) &
                                                             (results_df['embed'] == embed) &
                                                             (results_df['select_features'] == ','.join(select_features)) &
                                                             (results_df['n_train'] == n_train)
                                                            ]) == 0):

                    for x_test, y_ts in zip(X_test, Y_vector_test):
                        ix = np.random.choice(train_size, n_train)[0]
                        x_train = X_train[ix]
                        x_tr = normalize_data(x_train)
                        y_tr = Y_vector_train[ix]
                        x_ts = normalize_data(x_test)
                        qc = build_cosine_classifier(x_tr, x_ts, y_tr)

                        r = execute_circuit(qc, n_shots=n_shots)

                        if '0' not in r.keys():
                            r['0'] = 0
                        elif '1' not in r.keys():
                            r['1'] = 0

                        preds.append(retrieve_probabilities(r))

                    a, b = evaluation_metrics(np.array(preds), y_test, save=False)
                    res = pd.DataFrame([dataset_name, method, k, seed, X_train.shape[1], qc.num_qubits, n_train, embed, ','.join(select_features), a, b, preds, y_test],
                                 index=['dataset', 'method', 'dataset_params', 'seed', 'n_feature', 'qubits', 'n_train', 'embed', 'select_features', 'accuracy', 'brier', 'predictions', 'y_test']).transpose()
                    results_df = pd.concat([results_df, res])
                    
                    predictions[dataset_name][method] = results_df
                    
                    # save
                    pickle.dump(predictions, open(file_predictions, 'wb'))
    
    return predictions


def run_quantum_ensemble(predictions, dataset, method, dataset_name, seed, test_size, file_predictions, ds, n_swaps, n_features,
                         n_trains, n_shots, pca_embed=False, umap_embed=False, device='CPU', instance='', random_unitary=False, select_features=[]):
    """
    Run quantum ensemble experiments across multiple parameter configurations.
    
    This is a comprehensive workflow function that performs a grid search over
    quantum ensemble hyperparameters (d, n_swap, n_features, n_train) and
    evaluates performance on test data. Results are automatically saved to
    disk after each configuration.
    
    Parameters
    ----------
    predictions : dict
        Dictionary to store prediction results
    dataset : dict
        Dictionary containing dataset splits
    method : str
        Method name for results tracking
    dataset_name : str
        Name of the dataset being processed
    seed : int
        Random seed for reproducibility
    test_size : float
        Fraction of data for testing
    file_predictions : str
        Path to pickle file for saving predictions
    ds : list of int
        List of ensemble depths (control qubits) to test
    n_swaps : list of int
        List of swap operation counts to test
    n_features : list of int
        List of feature counts to test (must be powers of 2)
    n_trains : list of int
        List of training sample sizes to test
    n_shots : int
        Number of measurement shots per circuit
    pca_embed : bool, optional
        Use PCA for dimensionality reduction (default: False)
    umap_embed : bool, optional
        Use UMAP for dimensionality reduction (default: False)
    device : str, optional
        Execution device: 'CPU' or 'GPU' (default: 'CPU')
    instance : str, optional
        IBM Quantum instance string (default: '')
    random_unitary : bool, optional
        Use random unitary ensemble variant (default: False)
    select_features : list, optional
        Specific feature names to select (default: [])
    
    Returns
    -------
    dict
        Updated predictions dictionary with new results
    """
    from qbiocode.learning import compute_qensemble
    
    if (dataset_name not in predictions.keys()) or (method not in predictions[dataset_name].keys()):
        results_df = pd.DataFrame()
        if dataset_name not in predictions.keys():
            predictions[dataset_name] = {}
        predictions[dataset_name][method] = results_df
    else:
        results_df = predictions[dataset_name][method]
        
    for k, v in dataset[dataset_name].items():
        for f in n_features:
            (X_train_orig, X_test_orig, y_train, y_test) = v

            X_train = X_train_orig.copy()
            X_test = X_test_orig.copy()
            
            if len(select_features) > 0:
                X_train = X_train.loc[:, select_features]
                X_test = X_test.loc[:, select_features]
            
            scaler = MinMaxScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
            X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
            embed = 'none'

            if pca_embed:
                embed = 'pca'
                embedder = PCA(f)
                X_train = embedder.fit_transform(X_train)
                X_test = embedder.transform(X_test)
            elif umap_embed:
                if not UMAP_AVAILABLE:
                    raise ImportError("UMAP not available. Install with: pip install umap-learn")
                embed = 'umap'
                reducer = umap.UMAP(f)
                X_train = reducer.fit_transform(X_train)
                X_test = reducer.transform(X_test)
            else:
                vr = X_train.apply(np.var, axis=0)
                vr = vr.sort_values(ascending=False)
                X_train = X_train[list(vr[0:f].index)].to_numpy()
                X_test = X_test[list(vr[0:f].index)].to_numpy()

            for d in ds:
                for n_train in n_trains:
                    if n_train > d:
                        for n_swap in n_swaps:
                            if (len(results_df) == 0) or (len(results_df[(results_df['dataset'] == dataset_name) &
                                                                        (results_df['method'] == method) &
                                                                        (results_df['dataset_params'] == k) &
                                                                        (results_df['n_feature'] == f) &
                                                                        (results_df['n_swap'] == n_swap) &
                                                                        (results_df['n_train'] == n_train) &
                                                                        (results_df['embed'] == embed) &
                                                                        (results_df['select_features'] == ','.join(select_features)) &
                                                                        (results_df['d'] == d)
                                                                        ]) == 0):
                                
                                # Use QBioCode API
                                ensemble_method = 'random_unitary' if random_unitary else 'swap'
                                args = {'grid_search': False}  # Required by modeleval
                                
                                res = compute_qensemble(
                                    X_train, X_test, y_train, y_test,
                                    args=args,
                                    model='QEnsemble',
                                    data_key=str(k),
                                    n_train=n_train,
                                    n_swap=n_swap,
                                    d=d,
                                    mode="balanced",
                                    ensemble_method=ensemble_method,
                                    n_shots=n_shots,
                                    seed=seed,
                                    device=device,
                                    verbose=False
                                )
                                
                                # Extract predictions and y_test from modeleval result
                                # modeleval returns DataFrame with columns like 'y_test_QEnsemble', 'y_predicted_QEnsemble', 'results_QEnsemble'
                                y_test_col = f'y_test_{res.columns[0].split("_", 2)[-1]}' if len(res.columns) > 0 else 'y_test_QEnsemble'
                                y_pred_col = f'y_predicted_{res.columns[0].split("_", 2)[-1]}' if len(res.columns) > 0 else 'y_predicted_QEnsemble'
                                results_col = f'results_{res.columns[0].split("_", 2)[-1]}' if len(res.columns) > 0 else 'results_QEnsemble'
                                
                                # Find the actual column names
                                y_test_col = [col for col in res.columns if col.startswith('y_test_')][0]
                                y_pred_col = [col for col in res.columns if col.startswith('y_predicted_')][0]
                                results_col = [col for col in res.columns if col.startswith('results_')][0]
                                
                                # Extract values
                                y_test_val = res[y_test_col].iloc[0]
                                y_pred_val = res[y_pred_col].iloc[0]
                                results_dict = res[results_col].iloc[0]
                                
                                # Convert result to DataFrame format expected by notebook
                                res_df = pd.DataFrame({
                                    'dataset': [dataset_name],
                                    'method': [method],
                                    'dataset_params': [k],
                                    'n_feature': [f],
                                    'n_swap': [n_swap],
                                    'n_train': [n_train],
                                    'embed': [embed],
                                    'select_features': [','.join(select_features)],
                                    'd': [d],
                                    'seed': [seed],
                                    'accuracy': [results_dict.get('accuracy', np.nan)],
                                    'brier': [results_dict.get('brier_score', np.nan)],
                                    'qubits': [results_dict.get('Model_Parameters', {}).get('n_qubits', np.nan)],
                                    'runtime': [results_dict.get('time', np.nan)],
                                    'y_test': [y_test_val],
                                    'predictions': [y_pred_val]
                                })
                                
                                results_df = pd.concat([results_df, res_df])
        
                                predictions[dataset_name][method] = results_df
                                        
                                # save
                                pickle.dump(predictions, open(file_predictions, 'wb'))
    
    return predictions


# Made with Bob

import os
import re
from collections import Counter
from typing import Any
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_number_predicted_classes(predictions):
    """Calculate the number of unique predicted classes."""
    # Convert to numpy array if needed
    preds = np.asarray(predictions)
    
    # If predictions are probabilities (2D array), convert to class labels
    if preds.ndim > 1:
        preds = np.argmax(preds, axis=1)
    
    # Flatten and get unique values
    preds = preds.flatten()
    return len(np.unique(preds))


def post_process_results(predictions, dir_output, datasets, metrics=['Accuracy', 'F1 Score', 'brier']):
    """
    Post-process and visualize results from quantum ensemble experiments.
    
    Parameters
    ----------
    predictions : dict
        Dictionary containing prediction results for all methods and datasets
    dir_output : str
        Directory path for saving output files
    datasets : list
        List of dataset names to process
    metrics : list, optional
        List of metrics to evaluate (default: ['Accuracy', 'F1 Score', 'brier'])
    
    Returns
    -------
    total_results_full : pd.DataFrame
        Full results DataFrame with all experiments
    sig_bestmethods_df : pd.DataFrame
        Statistical significance results comparing methods
    """
    total_results = pd.DataFrame()
    for dataset_name in datasets:
        print(f"Dataset: {dataset_name}")
        methods = list(predictions[dataset_name].keys())

        all_results = pd.DataFrame()

        for method in methods:
            if method not in ['random_forest_gs', 'xgb_gs']:  # Ignore the parameter search experiments
                print(f"Method: {method}")
                results_df = predictions[dataset_name][method]
                results_df = results_df.reset_index(drop=True)

                results_df['num_pred_classes'] = [calculate_number_predicted_classes(x) for x in results_df['predictions']]
            
                results_df.columns = [re.sub('dataset_params', 'split', re.sub('accuracy', 'Accuracy', re.sub('method', 'Model', x))) for x in results_df.columns]
                
                # Calculate F1 scores
                f1_scores = []
                for idx, row in results_df.iterrows():
                    # Get y_true and predictions
                    y_true = np.asarray(row.y_test).flatten()
                    y_pred = np.asarray(row.predictions)
                    
                    # If predictions are probabilities (2D array), convert to class labels
                    if y_pred.ndim > 1:
                        y_pred = np.argmax(y_pred, axis=1)
                    else:
                        y_pred = y_pred.flatten()
                    
                    # Validate that arrays have the same length
                    if len(y_true) != len(y_pred):
                        print(f"Warning: Length mismatch at index {idx} - y_true: {len(y_true)}, y_pred: {len(y_pred)}")
                        print(f"  y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
                        f1_scores.append(np.nan)
                        continue
                    
                    f1_scores.append(f1_score(y_true, y_pred, average='weighted'))
                results_df['F1 Score'] = f1_scores

                if method == 'qcosine':
                    results_df = results_df[(results_df['n_train']==1) & (results_df['n_feature']==2)]
                    results_df['Model'] = [':'.join([row['Model'], str(row['n_train']), str(row['n_feature']), row['embed']]) for idx, row in results_df.iterrows()]
                elif method in ['random_forest', 'xgb']:
                    results_df['Model'] = [':'.join([row['Model'], str(row['n_feature'])]) for idx, row in results_df.iterrows()]
                else:
                    results_df['Model'] = [':'.join([row['Model'], str(row['d']), str(row['n_train']), str(row['n_swap']), str(row['n_feature']), row['embed']]) for idx, row in results_df.iterrows()]
                    
                all_results = pd.concat([all_results, results_df])
                
                if method in ['random_forest', 'xgb']:
                    results_df = results_df.drop(['predictions', 'y_test', 'best_params', 'select_features'], axis=1)
                else:
                    results_df = results_df.drop(['predictions', 'y_test', 'embed', 'select_features'], axis=1)

        total_results = pd.concat([total_results, all_results])
    
    total_results_full = total_results.reset_index().copy()

    methods = ['random_forest', 'xgb', 'qcosine', 'qensemble', 'qensemble_random_unitary']
    methods_cmap = dict(zip(methods[0:2], sns.color_palette('Greys', n_colors=2)))
    methods_cmap.update(dict(zip(methods[2:], sns.color_palette(n_colors=len(methods[2:])))))

    total_results_full['key'] = ['-'.join([row['Model'], row['dataset']]) for idx, row in total_results_full.iterrows()]
    total_results_full['method'] = [re.sub(':.*', '', x) for x in total_results['Model']]

    total_results = total_results_full[['Model', 'Accuracy', 'F1 Score', 'brier', 'dataset', 'split', 'num_pred_classes']].copy()
    # Extract first element from split if it's a list/tuple, otherwise keep as is
    def extract_split_value(x):
        try:
            if isinstance(x, (list, tuple)) and len(x) > 0:
                return x[0]
            elif isinstance(x, pd.Series):
                if len(x) > 0:
                    try:
                        return x.iloc[0]
                    except (IndexError, KeyError):
                        return x.values[0] if len(x.values) > 0 else x
                else:
                    return x
            elif isinstance(x, np.ndarray) and x.size > 0:
                return x.flat[0]
            else:
                return x
        except (IndexError, KeyError, AttributeError, TypeError):
            return x
    
    try:
        total_results['split'] = total_results['split'].apply(extract_split_value)
    except Exception as e:
        print(f"Warning: Could not process split column with apply: {e}")
        # If apply fails, try manual iteration
        try:
            split_values = []
            for val in total_results['split']:
                split_values.append(extract_split_value(val))
            total_results['split'] = split_values
        except Exception as e2:
            print(f"Warning: Could not process split column manually: {e2}")
            # Last resort: convert to string
            try:
                # Use .values to avoid pandas indexing issues
                total_results['split'] = [str(x) for x in total_results['split'].values]
            except Exception as e3:
                print(f"Warning: Could not convert split to string: {e3}")
                # If even that fails, just use a placeholder
                total_results['split'] = ['unknown'] * len(total_results)

    total_results['key'] = ['-'.join([row['Model'], row['dataset']]) for idx, row in total_results.iterrows()]
    total_results['method'] = [re.sub(':.*', '', x) for x in total_results['Model']]

    total_results = total_results.groupby(['Model', 'key', 'method', 'dataset']).median()
    total_results = total_results.reset_index()
    total_results = total_results[total_results.method.isin(methods)]

    blob_names = list(set(total_results['dataset']))

    sig_bestmethods: list[Any] = []

    for metric in metrics:
        max_df = []
        for method in methods:
            for bn in blob_names:
                b = total_results[total_results['dataset'] == bn]
                m = b[b['method'] == method]
                if len(m) > 0:
                    if metric == 'brier':
                        mm = m[m[metric] == min(m[metric])].sort_values('Model')
                    else:
                        mm = m[m[metric] == max(m[metric])].sort_values('Model')
                    # Only append if mm is not empty
                    if len(mm) > 0:
                        max_df.append(total_results_full[total_results_full['key']==mm['key'].iloc[-1]])
        max_df = pd.concat(max_df)

        for method in methods:
            for d in blob_names:
                a = max_df[(max_df.dataset == d) & (max_df.method == method)][metric].apply(float)
                b_r = max_df[(max_df.dataset == d) & (max_df.method == 'random_forest')][metric].apply(float)
                b_x = max_df[(max_df.dataset == d) & (max_df.method == 'xgb')][metric].apply(float)
            
                if metric != 'brier':
                    if len(a) > 1 and len(b_r) > 1:  # Need at least 2 samples for t-test
                        t_statistic, p_value = stats.ttest_ind(a, b_r, alternative='greater')
                        sig_bestmethods.append([method, 'random_forest', d, metric, round(float(a.median()),3), round(float(b_r.median()),3),
                        round(float(a.std()), 3), round(float(b_r.std()), 3), round(float(t_statistic),3), round(float(p_value),3)])
                        if p_value < 0.05:
                            print(f"RF: {d} : {method} (n={max_df[(max_df.dataset == d) & (max_df.method == method)].shape[0]}) : t={round(t_statistic, 3)}; p={round(p_value, 3)}")

                    if len(a) > 1 and len(b_x) > 1:  # Need at least 2 samples for t-test
                        t_statistic, p_value = stats.ttest_ind(a, b_x, alternative='greater')
                        sig_bestmethods.append([method, 'xgb', d, metric, round(float(a.median()),3), round(float(b_x.median()),3),
                        round(float(a.std()), 3), round(float(b_x.std()), 3), round(float(t_statistic),3), round(float(p_value),3)])
                        if p_value < 0.05:
                            print(f"XGB: {d} : {method} (n={max_df[(max_df.dataset == d) & (max_df.method == method)].shape[0]}) : t={round(t_statistic, 3)}; p={round(p_value, 3)}")
                else:
                    if len(a) > 1 and len(b_r) > 1:  # Need at least 2 samples for t-test
                        t_statistic, p_value = stats.ttest_ind(a, b_r, alternative='less')
                        sig_bestmethods.append([method, 'random_forest', d, metric, round(float(a.median()),3), round(float(b_r.median()),3),
                        round(float(a.std()), 3), round(float(b_r.std()), 3), round(float(t_statistic),3), round(float(p_value),3)])
                        if p_value < 0.05:
                            print(f"RF: {d} : {method} (n={max_df[(max_df.dataset == d) & (max_df.method == method)].shape[0]}) : t={round(t_statistic, 3)}; p={round(p_value, 3)}")

                    if len(a) > 1 and len(b_x) > 1:  # Need at least 2 samples for t-test
                        t_statistic, p_value = stats.ttest_ind(a, b_x, alternative='less')
                        sig_bestmethods.append([method, 'xgb', d, metric, round(float(a.median()),3), round(float(b_x.median()),3),
                        round(float(a.std()), 3), round(float(b_x.std()), 3), round(float(t_statistic),3), round(float(p_value),3)])
                        if p_value < 0.05:
                            print(f"XGB: {d} : {method} (n={max_df[(max_df.dataset == d) & (max_df.method == method)].shape[0]}) : t={round(t_statistic, 3)}; p={round(p_value, 3)}")

        def reformat_dataset(x):
            xs = [str(i) for i in x]
            new_x = [xs[1], '(' + xs[2] + ',' + xs[3] + ')', '(' + xs[3] + ',' + xs[2] + ')']
            return ' | '.join(new_x)

        max_df['Blob Config'] = [reformat_dataset(x) for x in max_df['split']]

        max_df = max_df[['Blob Config', 'method'] + metrics]
        max_df = max_df.drop_duplicates()
        
        max_df = max_df.sort_values('method')
        max_df = max_df.sort_values('Blob Config')

        plt.figure(figsize=(7,5))
        sns.barplot(data=max_df, y='Blob Config', x=metric, hue='method', hue_order=methods, errorbar='se', palette=methods_cmap.values())
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(dir_output, 'Blob_max_median_'+ re.sub(' ', '_', metric) + '.pdf'))
        plt.show()
        plt.close()

    sig_bestmethods_df = pd.DataFrame(sig_bestmethods, columns=['Method', 'Baseline', 'Dataset', 'Metric', 'Median method', 'Median baseline', 
        'Std. dev. method', 'Std. dev. baseline', 't statistic', 'p-value'])
    sig_bestmethods_df = sig_bestmethods_df[sig_bestmethods_df['Method'] != sig_bestmethods_df['Baseline']]
    sig_bestmethods_df.to_csv(os.path.join(dir_output, 'Blobs_best_stats.csv'), index=False) 

    return total_results_full, sig_bestmethods_df

