Applications
============

QBioCode provides standalone applications for common quantum machine learning workflows. These apps offer user-friendly interfaces and configuration-based workflows for complex analyses.

QProfiler
---------

QProfiler is an automated benchmarking tool for comparing quantum and classical machine learning models. It provides:

* Systematic model evaluation across multiple algorithms
* YAML-based configuration for reproducible experiments
* Automated performance metrics collection (accuracy, F1-score, AUC)
* Statistical analysis and visualization tools
* Support for custom datasets and embeddings

See the :doc:`QProfiler documentation <apps/profiler>` for detailed usage instructions.

QSage
-----

QSage is an intelligent meta-learning system that predicts which machine learning models will perform best on your dataset *before* you run them. By learning from data complexity patterns across multiple datasets, QSage provides data-driven model recommendations.

* Learns from History: Trains on data complexity metrics and model performance from previous experiments
* Predicts Performance: Estimates how well each model will perform on new, unseen datasets
* Ranks Models: Provides confidence-weighted rankings of classical and quantum models
* Saves Time: Helps you focus computational resources on the most promising models


See the :doc:`QSage documentation <apps/sage>` for detailed usage instructions.

QuVINE
------

QuVINE (Quantum View-based Network Embeddings) turns a graph into low-dimensional node embeddings, combining classical and quantum random walks with SGNS-based representation learning. It is provided as an in-tree app (``qbiocode.apps.quvine``) plus a ``quvine`` command-line tool.

* Multi-view graph construction from a single input graph
* Classical and quantum random walks (RWR, discrete- and continuous-time quantum walks)
* SGNS-based embedding learning, with quantum-calibrated filter / GAT / GraphGPS variants
* Classical baselines (node2vec, NetMF, APPNP) for comparison
* Reproducible, iterative evaluation pipelines
* 83 named methods selectable by a single ``method`` string
* Usable through :func:`qbiocode.get_embeddings` alongside ``pca``, ``nmf`` and ``umap``

QuVINE's dependencies are optional, so install it with ``pip install "qbiocode[quvine]"``.

Graph-complexity metrics are intentionally *not* part of the embedding app; they live in :func:`qbiocode.evaluate_graph` (``qbiocode.evaluation.graph_evaluation``). All 88 of them are grouped and described under :ref:`Graph-Complexity Measures <graph-complexity-measures>`.

See the :doc:`QuVINE documentation <apps/quvine>` for detailed usage instructions, and the :doc:`QuVINE Configuration Guide <apps/quvine_config>` for the shipped YAML config.

.. toctree::
   :hidden:
   :maxdepth: 1

   QProfiler <apps/profiler>
   QSage <apps/sage>
   QuVINE <apps/quvine>

.. Made with Bob
