"""
Embeddings Module for QBioCode
==============================

This module provides feature embedding and encoding methods for quantum
machine learning. It includes functions for computing various embeddings
and quantum feature maps.

Available Functions
-------------------
- get_embeddings: Reduce a train/test pair with any named method -- classical
  (``pca``, ``nmf``, ``lle``, ``isomap``, ``spectral``, ``umap``, ``none``) or
  QuVINE graph embeddings (``quvine_rwr``, ``quvine_fused``, ``node2vec``, ...)
- pqk: Projected Quantum Kernel embedding
- is_transductive: Whether a method sees test *features* at embed time
- check_embedding_name: Validate a method name up front, before doing any work

Available Constants
-------------------
- SKLEARN_METHODS: the classical method names, always available
- QUVINE_HEADLINE_METHODS: the QuVINE names worth trying first
- QUVINE_METHODS: every QuVINE name. Listed even without the ``[quvine]`` extra --
  resolving a name is stdlib-only, so discovery works and only *running* a method
  raises. Empty only if the QuVINE subpackage itself cannot be imported.

Available Classes
-----------------
- ConvAutoencoder: Convolutional autoencoder for dimensionality reduction.
  Imported lazily -- it is the only part of this package that needs torch.

Usage
-----
>>> from qbiocode.embeddings import get_embeddings, pqk
>>> # Reduce a train/test pair -- the scaler-style split is the point: the
>>> # transform is fitted on train only for every inductive method.
>>> X_train_emb, X_test_emb = get_embeddings("pca", X_train, X_test, n_components=2)
>>> # A QuVINE graph embedding, same call shape (needs pip install "qbiocode[quvine]")
>>> X_train_emb, X_test_emb = get_embeddings("quvine_rwr", X_train, X_test, n_components=8)
>>> # Projected Quantum Kernel embedding
>>> X_pqk = pqk(X, n_components=4)
"""

from .embed import (
    QUVINE_HEADLINE_METHODS,
    QUVINE_METHODS,
    SKLEARN_METHODS,
    check_embedding_name,
    get_embeddings,
    is_transductive,
    pqk,
)

__all__ = [
    "get_embeddings",
    "is_transductive",
    "pqk",
    "ConvAutoencoder",
    "SKLEARN_METHODS",
    "check_embedding_name",
    "QUVINE_HEADLINE_METHODS",
    "QUVINE_METHODS",
]


def __getattr__(name):
    """Resolve ``ConvAutoencoder`` on first use rather than at import time.

    It is the one name here that needs torch, and importing torch eagerly put
    torch's OpenMP runtime into every process that touched ``qbiocode`` -- which
    made every later XGBoost fit segfault on macOS. See
    ``qbiocode.utils._openmp`` for the diagnosis. Nothing else in the package
    uses this class, so the eager import bought nothing.
    """
    if name == "ConvAutoencoder":
        from .compute_autoencoder import ConvAutoencoder

        globals()[name] = ConvAutoencoder
        return ConvAutoencoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
