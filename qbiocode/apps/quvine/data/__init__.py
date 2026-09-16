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

"""Graph and dataset preparation utilities for QuVINE.

Loading graphs and GWAS tables from disk, preprocessing them (sparsification,
largest-connected-component extraction, node subsampling), bounded-radius
ego-net expansion, and a library of synthetic graph generators used by
``qbiocode.apps.quvine.reproducibility`` to build reproducible benchmarks.

Everything here needs only the base install -- numpy, pandas, networkx -- so
this package is imported eagerly and is unaffected by the ``[quvine]`` extra.

.. note::

   Graph-complexity metrics are **not** here. They live in QBioCode's own
   :func:`qbiocode.evaluate_graph` (``qbiocode.evaluation.graph_evaluation``),
   which supersedes the ``graph_complexity`` module that used to sit alongside
   these files in the standalone QuVINE distribution.
"""

from qbiocode.apps.quvine.data.data_loader import (
    load_graph,
    load_gwas_data,
    load_pegasus_results,
    load_seeds_and_targets,
)
from qbiocode.apps.quvine.data.prepare import (
    PrepareGraphConfig,
    keep_largest_connected_component,
    prepare_graph,
)
from qbiocode.apps.quvine.data.random_graphs import (
    add_hub_nodes,
    generate_barabasi_albert,
    generate_bipartite_random,
    generate_core_periphery,
    generate_erdos_renyi,
    generate_graph_with_seeds_and_targets,
    generate_hierarchical_network,
    generate_modular_network,
    generate_powerlaw_cluster,
    generate_random_geometric,
    generate_stochastic_block_model,
    generate_watts_strogatz,
    get_graph_statistics,
)
from qbiocode.apps.quvine.data.random_graphs_extended import (
    generate_configuration_model_graph,
    generate_degree_corrected_sbm,
    generate_grid_torus_lattice,
    generate_heterophilic_sbm,
    generate_random_regular_expander_like,
    sample_degree_sequence,
    sweep_configuration_model_graphs,
    sweep_degree_corrected_sbm,
    sweep_grid_torus_lattice,
    sweep_heterophilic_sbm,
    sweep_random_regular_expander_like,
)
from qbiocode.apps.quvine.data.sparsify import (
    materialize_undirected_simple_graph,
    sparsify_edges_biological,
)
from qbiocode.apps.quvine.data.subgraph import (
    expand_neighborhood,
    induce_subgraph_by_nodes,
    subsample_nodes,
    subsample_nodes_with_protected,
)

__all__ = [
    # Data loading
    "load_graph",
    "load_gwas_data",
    "load_pegasus_results",
    "load_seeds_and_targets",
    # Graph preparation
    "prepare_graph",
    "PrepareGraphConfig",
    "keep_largest_connected_component",
    "materialize_undirected_simple_graph",
    "sparsify_edges_biological",
    # Subgraphs
    "expand_neighborhood",
    "induce_subgraph_by_nodes",
    "subsample_nodes",
    "subsample_nodes_with_protected",
    # Random graph generators
    "add_hub_nodes",
    "generate_barabasi_albert",
    "generate_bipartite_random",
    "generate_core_periphery",
    "generate_erdos_renyi",
    "generate_graph_with_seeds_and_targets",
    "generate_hierarchical_network",
    "generate_modular_network",
    "generate_powerlaw_cluster",
    "generate_random_geometric",
    "generate_stochastic_block_model",
    "generate_watts_strogatz",
    "get_graph_statistics",
    # Extended random graph generators
    "generate_configuration_model_graph",
    "generate_degree_corrected_sbm",
    "generate_grid_torus_lattice",
    "generate_heterophilic_sbm",
    "generate_random_regular_expander_like",
    "sample_degree_sequence",
    "sweep_configuration_model_graphs",
    "sweep_degree_corrected_sbm",
    "sweep_grid_torus_lattice",
    "sweep_heterophilic_sbm",
    "sweep_random_regular_expander_like",
]
