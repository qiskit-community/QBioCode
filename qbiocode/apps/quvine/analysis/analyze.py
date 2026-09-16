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

import logging
import os

import pandas as pd 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.linalg import orthogonal_procrustes
from numpy.linalg import svd, eigh
from sklearn.cross_decomposition import CCA 
from scipy.spatial.distance import pdist, squareform 
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


def _can_show() -> bool:
    """Whether ``plt.show()`` would display anything (see visualization.visualize_correlation)."""
    return mpl.get_backend().lower() not in ("agg", "pdf", "ps", "svg", "cairo", "template")

def normalize(embedding, eps=1e-8): 
    mean = embedding.mean(axis=0, keepdims=True)
    std = embedding.std(axis=0, keepdims=True) + eps
    return (embedding - mean) / std

def procrustes_residual(Z1,Z2): 
    scaledZ1 = Z1 - Z1.mean(0)
    scaledZ2 = Z2 - Z2.mean(0)
    R, _ = orthogonal_procrustes(scaledZ1, scaledZ2)
    residual = np.linalg.norm(scaledZ1 - scaledZ2@R)/np.linalg.norm(scaledZ1)
    
    return residual

def cca_correlation(Z1, Z2, n_components=10):
    scaledZ1 = normalize(Z1)
    scaledZ2 = normalize(Z2)
    cca = CCA(n_components=n_components)
    Z1_c, Z2_c = cca.fit_transform(scaledZ1, scaledZ2)
    corrs = [np.corrcoef(Z1_c[:,i], Z2_c[:,i])[0,1] for i in range(n_components)]
    return np.mean(corrs)

def rsa_corr(Z1, Z2):
    D1 = pdist(Z1)
    D2 = pdist(Z2)
    rsa_corr = spearmanr(D1, D2).correlation
    return rsa_corr

def knn_sets(Z, k=10): 
    neighbors = NearestNeighbors(n_neighbors=k+1).fit(Z)
    idx = neighbors.kneighbors(Z, return_distance=False)
    return [set(row[1:]) for row in idx]

def knn_overlap(Z1, Z2, k=10):
    K1 = knn_sets(Z1, k)
    K2 = knn_sets(Z2, k)
    overlap = np.mean([len(K1[i] & K2[i])/k for i in range(len(K1))])
    return overlap

def effective_rank(s):
    p = s / s.sum()
    return np.exp(-np.sum(p * np.log(p + 1e-12)))

def plot_singular_values(singular_values, label='concatenate', filename=None, show=False):
    """Plot the log singular-value spectrum, optionally writing it to ``filename``.

    Args:
        singular_values: Singular values in descending order.
        label: Legend label for the series.
        filename: Where to write the figure; nothing is written when ``None``.
        show: Whether to call ``plt.show()``. Defaults to ``False`` because this is
            library code -- under a GUI backend ``show`` blocks until a human closes
            the window, and it is ignored under a non-interactive one.

    Returns:
        matplotlib.figure.Figure: the figure, already closed but still savable.
    """
    fig, ax = plt.subplots()
    ax.plot(list(range(1, len(singular_values)+1)), [np.log(x) for x in singular_values],
            marker='o', color='blue', label=label)
    ax.set_xlabel('Singular Value Index')
    ax.set_ylabel('Log(Singular Value)')
    ax.legend()
    # Save before showing. The previous order showed first, so under a GUI backend
    # the file was not written until someone closed the window -- a batch run
    # stalled indefinitely with nothing on disk to show it had got this far.
    if filename is not None:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info("Singular-value spectrum saved to: %s", filename)
    if show and _can_show():
        plt.show()
    plt.close(fig)
    return fig
    
def spectral_info(embeddings, labels, plot_flag=False, outdir=".", show=False):
    """Singular-value spectra and effective ranks for a list of embeddings.

    Args:
        embeddings: Embedding matrices to compare.
        labels: One label per embedding, used in the legend and the result keys.
        plot_flag: Whether to draw and write the three spectrum figures.
        outdir: Directory the figures are written to. Previously they went to the
            process's current working directory under fixed names, so two runs from
            the same directory silently overwrote each other's plots.
        show: Whether to call ``plt.show()``; see :func:`plot_singular_values`.

    Returns:
        dict: ``{label: effective_rank}``.
    """
    singular_values = []
    ks = []
    for z in embeddings:
        scaledZ = normalize(z)
        s = np.linalg.svd(scaledZ, compute_uv=False)
        singular_values.append(s)
        ks.append(len(s))
    k = min(ks)
    sk = []
    for s in singular_values:
        sk.append(s[:k])

    if plot_flag:
        os.makedirs(outdir, exist_ok=True)
        skn = [s / np.sum(s) for s in sk]
        # (filename, plotting method, y-label, series) -- one figure each, drawn on
        # its own axes rather than through the pyplot state machine, saved before
        # any show, and closed so repeated calls do not accumulate figures.
        panels = [
            ('log_spectrum.png', 'plot', 'log($s_i$)', [np.log(s) for s in sk]),
            ('loglog_spectrum.png', 'loglog', 'loglog($s_i$)', sk),
            ('log_normalized_spectrum.png', 'plot', 'log(normalized $s_i$)',
             [np.log(s) for s in skn]),
        ]
        for filename, method, ylabel, series in panels:
            fig, ax = plt.subplots()
            for i, s in enumerate(series):
                getattr(ax, method)(s, label=labels[i])
            ax.set_xlabel('singular value index i')
            ax.set_ylabel(ylabel)
            ax.legend()
            path = os.path.join(outdir, filename)
            fig.savefig(path, dpi=300, bbox_inches='tight')
            logger.info("Spectrum figure saved to: %s", path)
            if show and _can_show():
                plt.show()
            plt.close(fig)
    effective_ranks = {} 
    for i,label in enumerate(labels): 
        effective_ranks[label] = effective_rank(sk[i]) 
    
    return effective_ranks  
