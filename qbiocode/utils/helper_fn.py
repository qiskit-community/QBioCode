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
Helper Functions for Data Preprocessing and Model Evaluation
============================================================

This module provides utility functions for data preprocessing, feature encoding,
and result presentation in machine learning workflows.
"""

# ====== Base class imports ======

import time
from typing import Literal

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler

# ====== Scikit-learn imports ======


def scaler_fn(X, scaling: Literal["None", "StandardScaler", "MinMaxScaler"] = "None"):
    """
    Apply scaling transformation to input data.

    Scales the input data using one of three methods: no scaling, standard scaling
    (z-score normalization), or min-max scaling to [0, 1] range.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data to be scaled.
    scaling : {'None', 'StandardScaler', 'MinMaxScaler'}, default='None'
        Scaling method to apply:

        - 'None': No scaling, returns original data
        - 'StandardScaler': Standardize features by removing mean and scaling to unit variance
        - 'MinMaxScaler': Scale features to [0, 1] range

    Returns
    -------
    X_scaled : array-like of shape (n_samples, n_features)
        Scaled data. If scaling='None', returns original data unchanged.

    Notes
    -----
    StandardScaler transforms data to have mean=0 and variance=1:

    .. math::
        z = \\frac{x - \\mu}{\\sigma}

    MinMaxScaler transforms data to [0, 1] range:

    .. math::
        x_{scaled} = \\frac{x - x_{min}}{x_{max} - x_{min}}

    Examples
    --------
    >>> import numpy as np
    >>> from qbiocode.utils import scaler_fn
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> X_scaled = scaler_fn(X, scaling='StandardScaler')
    >>> X_minmax = scaler_fn(X, scaling='MinMaxScaler')

    See Also
    --------
    sklearn.preprocessing.StandardScaler : Standardize features
    sklearn.preprocessing.MinMaxScaler : Scale features to a range
    """
    if scaling == "MinMaxScaler":
        scaler = MinMaxScaler()
        return scaler.fit_transform(X)
    elif scaling == "StandardScaler":
        scaler = StandardScaler()
        return scaler.fit_transform(X)
    else:  # scaling == 'None'
        return X


def scale_train_test(
    X_train, X_test, scaling: Literal["None", "StandardScaler", "MinMaxScaler"] = "None"
):
    """Scale train and test with a single scaler fit on the training set only.

    Unlike calling :func:`scaler_fn` separately on each split (which fits a fresh scaler on
    the test set using test-set statistics), this fits one scaler on ``X_train`` and applies
    it to both. That is the correct protocol: the test set is transformed with train-derived
    statistics, exactly as a deployed model would see unseen data.

    Args:
        X_train, X_test (array-like): train / test feature matrices.
        scaling ({'None', 'StandardScaler', 'MinMaxScaler'}): scaler to fit on ``X_train``.

    Returns:
        tuple: ``(X_train_scaled, X_test_scaled)``. When ``scaling='None'`` the inputs are
        returned unchanged.

    Raises:
        ValueError: if ``scaling`` is not one of the three accepted names. A
            mistyped name used to fall through to the ``'None'`` branch, so
            ``scaling='minmaxscaler'`` returned unscaled data and reported success.
    """
    if scaling == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaling == "StandardScaler":
        scaler = StandardScaler()
    elif scaling == "None":
        return X_train, X_test
    else:
        raise ValueError(
            f"Unrecognized scaling {scaling!r}. Expected one of 'None', "
            f"'MinMaxScaler', 'StandardScaler' (case-sensitive)."
        )
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def feature_encoding(
    feature1,
    sparse_output=False,
    feature_encoding: Literal["None", "OneHotEncoder", "OrdinalEncoder"] = "None",
):
    """
    Encode categorical features using various encoding strategies.

    Transforms categorical features into numerical representations suitable for
    machine learning algorithms. Supports one-hot encoding, ordinal encoding,
    or no encoding.

    Parameters
    ----------
    feature1 : array-like of shape (n_samples,)
        Input categorical feature to be encoded. Should be a 1D array.
    sparse_output : bool, default=False
        If True and feature_encoding='OneHotEncoder', returns a sparse matrix.
        If False, returns a dense array. Ignored for other encoding methods.
    feature_encoding : {'None', 'OneHotEncoder', 'OrdinalEncoder'}, default='None'
        Encoding method to apply:

        - 'None': No encoding, returns original feature
        - 'OneHotEncoder': Create binary columns for each category
        - 'OrdinalEncoder': Map categories to integer values

    Returns
    -------
    feature1_encoded : array-like
        Encoded feature. Shape depends on encoding method:

        - 'None': shape (n_samples, 1)
        - 'OrdinalEncoder': shape (n_samples, 1)
        - 'OneHotEncoder': shape (n_samples, n_categories)

    Notes
    -----
    One-hot encoding creates a binary column for each unique category, useful
    when categories have no ordinal relationship. Ordinal encoding assigns
    integer values, suitable when categories have a natural order.

    The function automatically reshapes the input to (-1, 1) format required
    by scikit-learn encoders.

    Examples
    --------
    >>> import numpy as np
    >>> from qbiocode.utils import feature_encoding
    >>> categories = np.array(['A', 'B', 'C', 'A', 'B'])
    >>> # One-hot encoding
    >>> encoded_onehot = feature_encoding(categories, feature_encoding='OneHotEncoder')
    >>> # Ordinal encoding
    >>> encoded_ordinal = feature_encoding(categories, feature_encoding='OrdinalEncoder')

    See Also
    --------
    sklearn.preprocessing.OneHotEncoder : Encode categorical features as one-hot
    sklearn.preprocessing.OrdinalEncoder : Encode categorical features as integers
    """
    if feature_encoding == "OrdinalEncoder":
        encoder = OrdinalEncoder()
        return encoder.fit_transform(feature1.reshape(-1, 1))
    elif feature_encoding == "OneHotEncoder":
        encoder = OneHotEncoder(sparse_output=sparse_output)
        return encoder.fit_transform(feature1.reshape(-1, 1))
    else:  # feature_encoding == 'None'
        return feature1


def print_results(model, accuracy, f1, compile_time, params):
    """
    Print formatted machine learning model evaluation results.

    Displays model performance metrics and parameters in a consistent,
    readable format. Useful for comparing multiple models during
    experimentation and benchmarking.

    Parameters
    ----------
    model : str
        Name or identifier of the machine learning model.
    accuracy : float
        Accuracy score of the model, typically in range [0, 1].
    f1 : float
        F1 score of the model, harmonic mean of precision and recall.
    compile_time : float
        Time taken to train/compile the model, in seconds.
    params : dict
        Dictionary of model hyperparameters and configuration settings.

    Returns
    -------
    None
        Prints results to stdout.

    Notes
    -----
    The function formats floating-point numbers to 4 decimal places for
    consistency. All metrics are printed with descriptive labels.

    Examples
    --------
    >>> from qbiocode.utils import print_results
    >>> params = {'n_estimators': 100, 'max_depth': 10}
    >>> print_results('RandomForest', 0.9234, 0.9156, 2.345, params)
    RandomForest Model Accuracy score: 0.9234
    RandomForest Model F1 score: 0.9156
    Time taken for RandomForest Model (secs): 2.3450
    RandomForest Model Params:  {'n_estimators': 100, 'max_depth': 10}

    See Also
    --------
    sklearn.metrics.accuracy_score : Compute accuracy
    sklearn.metrics.f1_score : Compute F1 score
    """
    print(f"{model} Model Accuracy score: {accuracy:.4f}")
    print(f"{model} Model F1 score: {f1:.4f}")
    print(f"Time taken for {model} Model (secs): {compile_time:.4f}")
    print(f"{model} Model Params: ", params)


# Made with Bob
