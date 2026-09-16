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

"""Optional baseline embedding methods.

Every name below is re-exported for convenience only. The guards keep
``import qbiocode`` working when a baseline's dependency is absent, but they do
not make the method registry resilient -- :mod:`.adapters` imports from each
module directly, so a genuinely broken baseline still fails loudly there, which
is the correct outcome. What the guards must not do is fail *silently*: a
shorter ``__all__`` with no explanation is undiagnosable, so each failure is
recorded in :data:`UNAVAILABLE` and logged at DEBUG.
"""

import logging

logger = logging.getLogger(__name__)

__all__ = ["UNAVAILABLE"]

#: Maps a baseline module name to the reason it could not be imported. Empty on a
#: complete install. Inspect it when an expected method is missing from
#: :func:`qbiocode.apps.quvine.list_methods`.
UNAVAILABLE: "dict[str, str]" = {}


def _guard(module: str, names: "list[str]") -> None:
    """Re-export ``names`` from ``.<module>``, recording the reason on failure."""
    import importlib

    try:
        mod = importlib.import_module(f".{module}", __name__)
    except ImportError as exc:
        UNAVAILABLE[module] = str(exc)
        logger.debug("Baseline %r unavailable: %s", module, exc)
        return
    for name in names:
        globals()[name] = getattr(mod, name)
        __all__.append(name)


_guard("node2vec", ["run_node2vec"])
_guard("netmf", ["run_netmf"])
_guard("graphsage", ["run_graphsage"])
_guard("appnp", ["run_appnp", "generate_appnp_embedding"])
_guard("gcn_mf", [
    "GCNMF",
    "GCNLayer",
    "QuVINEGCNMF",
    "normalize_adjacency",
    "train_gcn_mf",
    "precompute_quantum_diffusion",
    "generate_baseline_gcnmf_embedding",
    "generate_baseline_filter_embedding_wrapper",
])
_guard("graphgps", [
    "GraphGPSConfig",
    "generate_graphgps_embedding",
    "generate_multiple_graphgps_embeddings",
])

# GraphGPS's TrainConfig is re-exported under a disambiguated name (gat.py has one
# too), so it cannot go through the loop above.
if "graphgps" not in UNAVAILABLE:
    from .graphgps import TrainConfig as GraphGPSTrainConfig

    __all__.append("GraphGPSTrainConfig")
