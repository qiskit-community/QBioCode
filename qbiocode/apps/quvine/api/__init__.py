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
QuVINE programmatic embedding API.

    from qbiocode.apps.quvine.api import embed, load_config, list_methods, resolve_method
"""

# Resolved lazily. Importing `core`, `config`, `sgns` or `targets` eagerly would
# pull in omegaconf (and, transitively, the rest of the [quvine] extra) merely to
# reach `resolve_method`, which needs nothing but the standard library. Since
# `qbiocode.embeddings` probes method names through `resolve_method` on every
# `get_embeddings` call, that probe must stay cheap and dependency-free.
_LAZY_ATTRS = {
    "list_methods": "qbiocode.apps.quvine.api.aliases",
    "resolve_method": "qbiocode.apps.quvine.api.aliases",
    "load_config": "qbiocode.apps.quvine.api.config",
    "embed": "qbiocode.apps.quvine.api.core",
    "EmbedResult": "qbiocode.apps.quvine.api.core",
    "QuvineMethodError": "qbiocode.apps.quvine.api.core",
    "run_sgns": "qbiocode.apps.quvine.api.sgns",
    "build_quantum_targets": "qbiocode.apps.quvine.api.targets",
}


def __getattr__(name: str):
    module_path = _LAZY_ATTRS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))

__all__ = [
    "embed",
    "EmbedResult",
    "QuvineMethodError",
    "load_config",
    "list_methods",
    "resolve_method",
    "run_sgns",
    "build_quantum_targets",
]
