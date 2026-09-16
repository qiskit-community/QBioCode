
#######################################
QuVINE
#######################################

**Quantum View-based Network Embeddings**

QuVINE turns a graph into low-dimensional node embeddings by combining
classical and quantum random walks with SGNS-based representation learning. It
is vendored into QBioCode as an in-tree app (``qbiocode.apps.quvine``).

.. _quvine-installation:

Installation
============

QuVINE's Python modules ship with QBioCode, but its third-party dependencies
(gensim, hiperwalk, node2vec, torch-geometric, python-louvain, ripser,
omegaconf) are heavy, so they are behind an optional extra:

.. code-block:: bash

    pip install "qbiocode[quvine]"

A plain ``pip install qbiocode`` leaves QuVINE's dependencies out. ``import
qbiocode`` and every classical embedding still work; requesting a QuVINE method
then raises a :class:`~qbiocode.apps.quvine.QuvineDependencyError` that names
the extra, the missing module, and the command above. To check an existing
environment:

.. code-block:: python

    from qbiocode.apps.quvine import describe_environment
    print(describe_environment())

🔬 **What QuVINE Does**
   1. **Builds multiple views** of a single input graph
   2. **Runs walks**: random walk with restart (RWR) and discrete-/continuous-time quantum walks (DTQW/CTQW)
   3. **Learns embeddings** via skip-gram negative sampling (SGNS), with quantum-calibrated filter / GAT / GraphGPS variants
   4. **Compares against classical baselines**: node2vec, NetMF, APPNP
   5. **Fuses views** into a single embedding when requested

.. note::
    Before you start, make sure that you have installed QBioCode correctly by following the  `Installation <https://qiskit-community.github.io/QBioCode/installation.html>`_ guide.

.. important::
    Graph-complexity metrics are **not** part of the embedding app. They live in
    QBioCode's own :func:`qbiocode.evaluate_graph`
    (``qbiocode.evaluation.graph_evaluation``), which you can run on the same
    graph to characterize it.

How QuVINE Works
================

QuVINE does not walk the input graph directly. It builds a family of bounded
**views** around each node, walks each view with a mix of classical and quantum
walks, treats the resulting node sequences as a corpus, and learns one embedding
per walk kind with skip-gram negative sampling -- optionally fusing them into a
single matrix.

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: 🌐 Construction
      :class-header: bg-primary text-white

      **Preprocess** (research pipeline only)

      - Radius-bounded subsample around the seed set
      - Local-score sparsification, degree-capped

      **Views**

      - ``views.num_views`` constrained subgraphs per root
      - Bounded by ``max_nodes`` / ``max_edges`` / ``max_degree``
      - Hubs down-weighted by ``degree_alpha``

   .. grid-item-card:: 🎲 Representation
      :class-header: bg-success text-white

      **Walks** -- one corpus per kind in ``walks.kinds``

      - ``rwr`` classical random walk with restart
      - ``ctqw`` continuous-time quantum walk
      - ``dtqw`` discrete-time quantum walk

      **SGNS**

      - Skip-gram over the walk token sequences
      - One ``(n_nodes, dim)`` matrix per walk kind
      - Fused across kinds by ``fusion.method``

The Embedding Pipeline
----------------------

.. code-block:: text

   networkx.Graph  (or a 2-/3-column edge list)
            ↓
   [Preprocess]  subsample → sparsify          (Hydra pipeline only)
            ↓
   Views:  num_views constrained subgraphs per root
            ↓
   Walks:  rwr ┄┄┄┐   ctqw ┄┄┄┐   dtqw ┄┄┄┐    (each kind, independently)
            ↓     │           │           │
   Corpus: node-sequence tokens per walk kind
            ↓     │           │           │
   SGNS:   E_rwr  │   E_ctqw  │   E_dtqw  │    (n_nodes × dim each)
            ↓─────┴───────────┴───────────┘
   Fusion: fusion.method (svd | graphreg | attention | hybrid | ...)
            ↓
   EmbedResult.embedding  →  (n_nodes, dim)

Stages in detail
----------------

**1. Views instead of the whole graph.** For each root node QuVINE samples
``views.num_views`` bounded subgraphs, each capped at ``max_nodes`` nodes,
``max_edges`` edges and ``max_degree`` degree. This is what makes quantum walks
tractable: a CTQW or DTQW needs an operator over the walked subgraph, so the caps
bound that cost per root rather than by the size of the whole network. Sampling
is degree-normalized (``degree_norm``, ``degree_alpha``) so a view is not
swallowed by whichever hub it happened to touch.

**2. Walks, one corpus per kind.** Every entry in ``walks.kinds`` produces its
own token corpus over the views. ``rwr`` is the classical control;
``ctqw`` evolves for ``walks.time``; ``dtqw`` takes ``walks.steps`` steps with the
``walks.coin`` operator. Walks are seeded per root and derived from
``experiment.base_seed``, so a run reproduces regardless of how the roots were
scheduled across workers.

**3. SGNS.** Each corpus is trained with skip-gram negative sampling
(``train.sg``, ``train.negative``, ``train.window``, ``train.epochs``) into a
``(n_nodes, train.embedding_dim)`` matrix.

**4. Fusion.** The ``*_fused`` methods combine the per-kind matrices through
``fusion.method`` at rank ``fusion.k``. The single-kind methods
(``quvine_rwr``, ``quvine_ctqw``, ``quvine_dtqw``) skip this and return one
matrix directly.

**5. Quantum-calibrated baselines.** The ``gat_*``, ``graphgps_*`` and
``filter_*`` families are classical architectures whose spectral filter is fitted
against a quantum walk's profile (``heat_qcal_ctqw``, ``poly_qcal_dtqw``, ...).
They exist so a quantum walk can be compared against a classical model that has
been given the *same* spectral information -- which is a much harder baseline
than an unmodified GAT.

.. seealso::
   Every knob named above is documented in the
   :doc:`QuVINE Configuration Guide <quvine_config>`, including which of the
   three entry points reads it.


Usage
=====

QuVINE can be used as a **command-line tool** or as a **Python library**.

Command-Line Interface
----------------------

After installing QBioCode, the ``quvine`` command embeds the nodes of a graph
given as a 2- or 3-column edge list (``source,target[,weight]``). A conventional
header row is recognized and skipped; ``#`` comment lines are ignored:

.. code-block:: bash

   # List all available methods
   quvine --list-methods

   # Embed a graph with the default fused method
   quvine --edgelist edges.csv --method quvine_fused --output out/

   # Use a classical baseline on a tab-separated, weighted edge list
   quvine --edgelist edges.tsv --sep '\t' --weighted --method node2vec --output out/

Command-Line Options
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Flag
     - Default
     - Description
   * - ``--edgelist``, ``-i``
     - *required*
     - Path to the edge-list file: ``source,target`` or
       ``source,target,weight``.
   * - ``--output``, ``-o``
     - *required*
     - Output directory. Created if it does not exist.
   * - ``--method``, ``-m``
     - ``quvine_fused``
     - Embedding method. Any name from ``--list-methods``; an unrecognized name
       is rejected with suggestions drawn from the real dispatch table.
   * - ``--sep``
     - ``,``
     - Field separator of the edge list. Use ``--sep '\t'`` for TSV.
   * - ``--weighted``
     - off
     - Read a third column as the edge weight. Without this the third column is
       ignored and every edge weighs 1.
   * - ``--header``
     - ``auto``
     - Whether the first row is column names. ``auto`` recognizes a row whose
       first two fields are both conventional column names (``source``/``target``,
       ``gene1``/``gene2``, ``from``/``to``, ...) and reports on stderr that it
       skipped it. ``yes`` and ``no`` force the decision.
   * - ``--base-seed``
     - from config
     - Integer seed, overriding ``experiment.base_seed``. Set it for anything you
       intend to reproduce.
   * - ``--config``
     - packaged default
     - Path to a YAML config. See the
       :doc:`Configuration Guide <quvine_config>`.
   * - ``--npy``
     - off
     - Additionally write the raw array as ``embedding.npy``.
   * - ``--list-methods``
     - off
     - Print every available method and exit. Methods whose dependency is
       missing are reported as unavailable rather than omitted silently.

The tool validates before it computes: a missing, nonexistent or directory
``--edgelist``, a ``--config`` that is not a file, an empty ``--sep``, an
unknown ``--method``, an edge list that is only a header, or a graph that parses
to zero nodes each exit non-zero with a message naming the problem. In particular a mistyped ``--config`` path is
an error -- earlier versions silently fell back to the packaged default and
produced a run configured by something other than what you asked for.

Outputs
^^^^^^^

A ``quvine`` run writes three things into ``--output``:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - File
     - Contents
   * - ``embedding.csv``
     - The embedding matrix. Index column ``node`` carries the original graph
       node labels; columns are ``dim_0`` … ``dim_{d-1}``.
   * - ``embedding.npy``
     - The same matrix as a raw NumPy array, written only with ``--npy``.
   * - ``embedding_meta.json``
     - Run metadata: ``requested_method``, the ``method`` it resolved to,
       ``kind`` (``sgns`` / ``sgns_fused`` / ``registry``), ``dim``, ``n_nodes``,
       ``execution_time``, and ``used_quantum_targets``.

``requested_method`` and ``method`` differ whenever an alias or a config choice
resolves: ``quvine_fused`` runs as ``fused:svd`` when ``fusion.method`` is
``svd``. Both are recorded, so a result file says exactly what produced it.

.. note::
   For a fused method ``dim`` is ``fusion.k``, not ``train.embedding_dim`` --
   fusion reduces the stacked per-walk-kind embeddings to its own rank. With the
   shipped defaults that is 10 rather than 64. Single-kind methods return
   ``train.embedding_dim``.

.. note::
   The research pipeline (``python -m qbiocode.apps.quvine.main``) writes a
   larger result set -- ``ranking_results.csv``, ``embedding_comparison.csv``,
   the resolved ``config.yaml``, ``summary.json``, per-iteration
   ``embeddings/embeddings_iter_{n}.npz`` and plots -- into
   ``runtime.output_dir``. See the
   :doc:`Configuration Guide <quvine_config>`.

Python Library Usage
--------------------

.. code-block:: python

   import networkx as nx
   from qbiocode.apps.quvine import embed, list_methods

   # Inspect the available methods (83 in total)
   print(list_methods())

   # Build (or load) a graph
   G = nx.karate_club_graph()

   # Embed its nodes; result.embedding has shape (n_nodes, dim)
   result = embed(G, "quvine_fused", base_seed=0)
   print(result.embedding.shape)

The :func:`embed` function accepts the graph, a ``method`` name, and optional
configuration/override arguments (``config``, ``overrides``, ``base_seed``,
``fuse``, ``n_jobs``, ...). It returns an ``EmbedResult`` whose ``embedding``
attribute is the ``(n_nodes, dim)`` embedding matrix.

Evaluating the input graph:

.. code-block:: python

   from qbiocode import evaluate_graph

   metrics = evaluate_graph(G, name="karate")   # -> pandas.DataFrame

As a QBioCode Embedding
-----------------------

QuVINE is also wired into QBioCode's embedding layer, so any of its method names
is usable wherever ``pca``, ``nmf`` or ``umap`` is -- including QProfiler's
``embeddings:`` config list. The call shape is identical:

.. code-block:: python

   from qbiocode import get_embeddings

   X_train_emb, X_test_emb = get_embeddings(
       "quvine_rwr", X_train, X_test, n_components=8
   )

A symmetric k-nearest-neighbour graph is built over the rows of
``vstack([X_train, X_test])`` with edge weight ``1/(1+d)``, embedded with the
requested method, reduced to ``n_components`` if the method returned a different
width, and split back into the train and test blocks.

.. warning::

   QuVINE methods are **transductive**: no QuVINE method has an out-of-sample
   ``transform``, so the test rows take part in building the graph. Test
   *features* therefore influence the geometry; test *labels* never enter at any
   point. ``get_embeddings`` emits one :class:`UserWarning` per call saying so,
   and :func:`qbiocode.is_transductive` lets downstream code branch on it. The
   classical ``spectral`` mode has exactly the same property.

Discovering what is available:

.. code-block:: python

   import qbiocode

   qbiocode.SKLEARN_METHODS           # the classical modes, always present
   qbiocode.QUVINE_HEADLINE_METHODS   # the QuVINE names worth trying first
   qbiocode.QUVINE_METHODS            # all 83; empty if the extra is absent

``netmf``, ``appnp`` and the ``gat_*`` family need nothing beyond the base
install. The rest raise a message naming the missing dependency and the exact
install command -- see :ref:`the installation note above <quvine-installation>`.

Available Methods
=================

``list_methods()`` returns 83 method names spanning several families. Pass
``kind="sgns"``, ``kind="registry"`` or ``kind="fused"`` to narrow the list:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Family
     - Examples
   * - Fused (all walk kinds combined)
     - ``quvine_fused``, ``quvine``, ``fused``, ``quvine_sgns_fused``
   * - QuVINE (quantum-calibrated)
     - ``quvine_rwr``, ``quvine_ctqw_heat``, ``quvine_dtqw_poly``, ...
   * - Filter variants
     - ``filter_rwr_heat``, ``filter_ctqw_poly``, ``filter_dtqw_heat``, ...
   * - GAT variants
     - ``gat_heat``, ``gat_ctqw_poly``, ``gat_dtqw_heat``, ...
   * - GraphGPS variants
     - ``graphgps_*``
   * - Walks
     - ``rwr``, ``ctqw``, ``dtqw``, ``sgns_*``
   * - Classical baselines
     - ``node2vec``, ``netmf``, ``appnp``, ``graphsage``

.. tip::
    Small quantum-walk step counts often work best: high step counts can
    over-mix the walk, so a modest number of steps tends to give comparable or
    better embeddings.

Configuration
=============

Every stage above is driven by a YAML config. QuVINE ships a working default and
finds it automatically, so nothing below is required to get started -- but the
three entry points read *different parts* of it, which is the one thing worth
knowing before you edit it:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Entry point
     - Reads
   * - :func:`embed` and the ``quvine`` CLI
     - ``views``, ``walks``, ``train``, ``min_count``, ``fusion``,
       ``experiment.base_seed``
   * - ``python -m qbiocode.apps.quvine.main``
     - all of the above **plus** ``data_path``, ``graph``, ``disease``,
       ``runtime``, ``preprocess``, ``baselines``, ``analysis``, ``evaluation``

Point QuVINE at your own file with ``embed(..., config=path)``, the CLI's
``--config``, or the ``QUVINE_DEFAULT_CONFIG`` environment variable; override
individual keys with ``overrides={"train": {"epochs": 20}}`` or the dotlist form
``overrides=["train.epochs=20"]``.

.. toctree::
   :maxdepth: 2

   Configure YAML <quvine_config.md>

.. _graph-complexity-measures:

Graph-Complexity Measures
=========================

:func:`qbiocode.evaluate_graph` is the graph analogue of QProfiler's
:ref:`data-complexity measures <data-complexity-measures>`: it summarizes a graph
as a single-row :class:`pandas.DataFrame` you can correlate against embedding
quality, use to choose a method, or report alongside a result.

.. code-block:: python

   from qbiocode import evaluate_graph

   metrics = evaluate_graph(G, name="karate")   # 1 x 88 DataFrame

.. important::
   ``evaluate_graph`` is **core QBioCode**, not part of the QuVINE app -- it is
   documented here because this is where you are most likely to want it. It needs
   only the base install; QuVINE's optional extra is irrelevant to it.

It merges two families of metrics, and each degrades independently: if one
family cannot be computed for a graph, that family's columns are dropped with a
:class:`UserWarning` and the rest are still returned. An empty graph yields a
size-only summary. Passing ``None`` raises :class:`TypeError`.

The 88 columns
--------------

.. list-table::
   :header-rows: 1
   :widths: 24 8 68

   * - Family
     - Count
     - Metrics
   * - Size and density
     - 8
     - ``num_nodes``, ``num_edges``, ``num_nodes_raw``, ``num_edges_raw``,
       ``log_num_nodes``, ``log_num_edges``, ``density``, ``avg_degree``.
       The ``_raw`` pair is the graph as given; the unsuffixed pair is after
       QuVINE's own normalization.
   * - Laplacian spectrum
     - 16
     - ``spectral_gap``, ``normalized_spectral_gap``, ``algebraic_connectivity``,
       ``algebraic_connectivity_ratio``, ``spectral_entropy``,
       ``spectral_entropy_partial``, ``von_neumann_entropy``, ``estrada_index``,
       ``laplacian_effective_rank_partial``, ``spectral_degeneracy_fraction``,
       ``spectral_dimension``, ``bipartite_proximity``, ``eigenvalue_mean``,
       ``eigenvalue_std``, ``eigenvalue_max``, ``eigenvalue_min``.
       The spectral gap is the single most useful predictor here -- a small gap
       means slow diffusion and weakly separated communities.
   * - Eigenvector localization
     - 8
     - ``ipr_low_mean``, ``ipr_high_mean``, ``adjacency_ipr_low_mean``,
       ``adjacency_ipr_high_mean``, ``inverse_participation_ratio``,
       ``participation_ratio``, ``spectral_concentration``,
       ``dominant_eigenvector_centrality``.
       High inverse participation ratio (IPR) means the eigenvectors are
       concentrated on few nodes, which walk-based embeddings find hard.
   * - Diffusion and walk spectra
     - 4
     - ``heat_kernel_trace_t1``, ``heat_kernel_trace_t10``,
       ``nonbacktracking_spectral_radius``, ``wl_compression_ratio``.
       The two heat-kernel traces bracket short- and long-time diffusion;
       ``wl_compression_ratio`` is how far Weisfeiler-Lehman refinement collapses
       the node set, i.e. how much structural symmetry there is.
   * - Quantum composites
     - 5
     - ``quantum_complexity``, ``quantum_advantage_score``,
       ``quantum_advantage_arithmetic``, ``quantum_advantage_geometric``,
       ``quantum_advantage_harmonic``.
       Composite scores built from the spectral columns above, meant as a
       screening heuristic for where a quantum walk may help -- not a guarantee.
   * - Paths and connectivity
     - 5
     - ``approx_avg_path_length``, ``path_length_ratio``,
       ``largest_cc_fraction``, ``log_odd_girth``, ``approx_conductance``.
       ``largest_cc_fraction`` below 1 means the graph is disconnected, which
       every walk method notices.
   * - Degree and centrality concentration
     - 12
     - ``degree_gini``, ``max_degree_fraction``, ``degree_assortativity``,
       ``degree_heterogeneity``, ``pagerank_gini``,
       ``betweenness_gini_approx``, ``closeness_gini_approx``,
       ``core_number_gini``, ``centrality_entropy``, ``centrality_variance``,
       ``centrality_gini``, ``centrality_range``.
       Gini coefficients near 1 mark a hub-dominated graph, where a ranking
       result needs degree-matched null controls to mean anything.
   * - Community and clustering
     - 6
     - ``modularity``, ``transitivity``, ``clustering_mean``,
       ``clustering_std``, ``cycle_density``, ``cyclomatic_number``.
   * - Ollivier-Ricci curvature
     - 10
     - ``orc_gJC_mean``, ``orc_gJC_min``, ``orc_gJC_max``, ``orc_gJC_std``,
       ``orc_kLB_mean``, ``orc_kLB_min``, ``orc_kLB_max``, ``orc_kLB_std``,
       ``orc_negative_fraction``, ``orc_num_edges``.
       Two curvature estimators -- a Jaccard-based approximation (``gJC``) and a
       Lin-Lu-Yau lower bound (``kLB``). Negative curvature marks bridging
       edges between communities.
   * - Effective resistance
     - 3
     - ``kirchhoff_index``, ``kirchhoff_per_pair``, ``kirchhoff_normalised``.
   * - Persistent homology
     - 8
     - ``betti_0``, ``betti_1``, ``betti_2``, ``betti_sum``,
       ``euler_characteristic``, ``persistence_entropy_H0``,
       ``persistence_entropy_H1``, ``persistence_entropy_H2``.
       Needs ``ripser`` (part of the ``[quvine]`` extra); dropped with a warning
       if it is absent.
   * - Label- and feature-aware
     - 2
     - ``label_homophily``, ``feature_dirichlet_energy``. Computed only when the
       graph carries node labels or features; a low homophily graph is one where
       neighbouring nodes disagree, and embedding-then-classify does poorly.
   * - Identifier
     - 1
     - ``Graph`` -- the ``name`` you passed, so several summaries concatenate
       into a comparable table.

.. tip::
   To compare graphs, concatenate the single-row frames:

   .. code-block:: python

      import pandas as pd
      from qbiocode import evaluate_graph

      table = pd.concat([evaluate_graph(g, name=n) for n, g in graphs.items()])

   That is exactly what the :doc:`Getting Started notebook
   <../tutorials/QuVINE/example_quvine>` does before correlating complexity
   against macro-F1.

The individual functions behind these columns are documented in
:py:mod:`qbiocode.evaluation.graph_evaluation`.

Tutorials
=========

Four notebooks, in reading order -- the synthetic walkthrough introduces the API
and the method registry, two single-cell notebooks apply it to real data, and the
last drives QuVINE through QProfiler:

- :doc:`QuVINE - Getting Started <../tutorials/QuVINE/example_quvine>` -
  benchmark 12 methods across three stochastic block models of increasing
  difficulty, then correlate each graph's complexity against node-classification
  macro-F1. Fully synthetic, so no data files are needed.
- :doc:`QuVINE on Single-Cell Data <../tutorials/QuVINE/quvine_sc_cd4_vs_cd8>` -
  build a kNN graph from PBMC single-cell data, inspect it with
  :py:func:`qbiocode.evaluate_graph`, and compare classical against
  quantum-calibrated walk embeddings on a CD4-vs-CD8 task.
- :doc:`QuVINE on T vs. Monocyte <../tutorials/QuVINE/quvine_sc_t_vs_mono>` - a
  transductive, semi-supervised task on an 800-cell two-view graph with soft seed
  labels, evaluated both by training on the seed embeddings and by ranking
  non-seed nodes against degree- and distance-matched null controls.
- :doc:`QuVINE Embeddings in QProfiler <../tutorials/QProfiler/sc_binary_quvine_2x2_qprofiler>` -
  drive QuVINE through :py:func:`qbiocode.get_embeddings` like any other embedding
  and let QProfiler benchmark the 2x2 classical/quantum design.
- The :doc:`Tutorials <../tutorials>` page for the full gallery.

.. seealso::
   - :doc:`QuVINE Configuration Guide <quvine_config>` - the shipped config, section by section
   - :py:mod:`qbiocode.apps.quvine` - App package
   - :py:func:`qbiocode.evaluate_graph` - Graph-complexity metrics
