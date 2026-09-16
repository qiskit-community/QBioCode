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
qbiocode.apps.quvine

QuVINE (Quantum View-based Network Embeddings) vendored into QBioCode as an
in-tree app.

The Python modules ship with QBioCode itself, but their third-party dependencies
(gensim, hiperwalk, node2vec, torch-geometric, python-louvain, ripser, omegaconf)
are behind an optional extra::

    pip install "qbiocode[quvine]"

Without it, ``import qbiocode`` and every classical embedding keep working, and
requesting a QuVINE method raises
:class:`qbiocode.apps.quvine._deps.QuvineDependencyError` naming the extra and
the install command.

    from qbiocode.apps.quvine import embed
    result = embed(graph, "quvine_rwr")   # result.embedding -> (n_nodes, dim)

Core features:
- Multi-view graph construction
- Classical and quantum random walks
- SGNS-based embedding learning
- Iterative, reproducible evaluation pipelines

Graph-complexity metrics are intentionally NOT part of this app; they live in
QBioCode's own module ``qbiocode.evaluation.graph_evaluation``.
"""

__version__ = "0.1.0"


def __getattr__(name: str):
    """Lazy-load heavy submodules.

    Keeps ``import qbiocode.apps.quvine`` free of the [quvine] extra's
    dependencies, so the package is importable on a bare install.
    """
    if name == "Pipeline":
        from qbiocode.apps.quvine.pipeline import Pipeline
        return Pipeline
    if name in ("embed", "EmbedResult", "QuvineMethodError"):
        from qbiocode.apps.quvine.api.core import EmbedResult, QuvineMethodError, embed
        return {"embed": embed, "EmbedResult": EmbedResult,
                "QuvineMethodError": QuvineMethodError}[name]
    if name == "load_config":
        from qbiocode.apps.quvine.api.config import load_config
        return load_config
    if name in ("list_methods", "resolve_method"):
        from qbiocode.apps.quvine.api.aliases import list_methods, resolve_method
        return {"list_methods": list_methods, "resolve_method": resolve_method}[name]
    if name in ("QuvineDependencyError", "describe_environment", "missing_dependencies"):
        from qbiocode.apps.quvine import _deps
        return getattr(_deps, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Pipeline",
    "embed",
    "EmbedResult",
    "QuvineMethodError",
    "load_config",
    "list_methods",
    "resolve_method",
    "QuvineDependencyError",
    "describe_environment",
    "missing_dependencies",
]
