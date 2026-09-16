"""
Visualization Module for QBioCode
=================================

This module provides visualization tools for analyzing and presenting
machine learning results, including correlation analysis and performance
comparisons between classical and quantum models.

Available Functions
-------------------
- compute_results_correlation: Compute Spearman correlation between metrics
- plot_results_correlation: Create correlation plots and visualizations
- publication_style: The journal-figure rcParams, to opt into deliberately
- PUBLICATION_STYLE: the same settings as a plain dict
- CorrelationFigures: what plot_results_correlation returns

Usage
-----
>>> from qbiocode.visualization import plot_results_correlation
>>> figs = plot_results_correlation(correlations_df, save_file_path='plots/corr.pdf')
>>> figs.scatter.savefig('plots/corr_600dpi.png', dpi=600)

Importing this module does not modify ``matplotlib.rcParams``. The journal styling
is applied per figure; adopt it for your own figures with:

>>> import matplotlib.pyplot as plt
>>> from qbiocode.visualization import publication_style
>>> with plt.rc_context(publication_style()):
...     fig, ax = plt.subplots()
"""

from .visualize_correlation import (
    PUBLICATION_STYLE,
    CorrelationFigures,
    compute_results_correlation,
    plot_results_correlation,
    publication_style,
)

__all__ = [
    "PUBLICATION_STYLE",
    "CorrelationFigures",
    "compute_results_correlation",
    "plot_results_correlation",
    "publication_style",
]
