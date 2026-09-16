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

"""The public surface of ``import qbiocode``.

Two properties are load-bearing for the ``[quvine]`` extra:

* every name in ``__all__`` actually resolves -- a stale entry is a documented
  import that raises;
* importing the package pulls in none of the extra's dependencies, so a bare
  install works. The check runs in a subprocess, because once any other test in
  the session has imported gensim, ``sys.modules`` can no longer answer the
  question.
"""

from __future__ import annotations

import pytest

import qbiocode

from .conftest import REPO_ROOT, run_python

# Modules that only the [quvine] extra installs. torch and omegaconf are
# deliberately absent from this list: torch is a base dependency (the embedding
# layer imports it eagerly) and omegaconf arrives with hydra.
QUVINE_ONLY_MODULES = [
    "gensim",
    "hiperwalk",
    "node2vec",
    "community",
    "ripser",
    "torch_geometric",
]

# Names the docs and notebooks tell users to import from the top level.
DOCUMENTED_NAMES = [
    "get_embeddings",
    "evaluate_graph",
    "scale_train_test",
    "evaluate",
]


def test_all_is_declared_and_sorted_into_something_usable():
    assert isinstance(qbiocode.__all__, list)
    assert len(qbiocode.__all__) == len(set(qbiocode.__all__)), "duplicates in __all__"


@pytest.mark.parametrize("name", sorted(qbiocode.__all__))
def test_every_advertised_name_resolves(name):
    assert hasattr(qbiocode, name), (
        f"__all__ advertises {name!r} but the attribute does not exist"
    )


@pytest.mark.parametrize("name", DOCUMENTED_NAMES)
def test_the_documented_helpers_are_importable_from_the_top_level(name):
    assert hasattr(qbiocode, name)
    assert name in qbiocode.__all__, f"{name} is importable but missing from __all__"


def test_star_import_matches_all():
    namespace: dict = {}
    exec("from qbiocode import *", namespace)  # noqa: S102 - that is the thing under test
    # exec injects __builtins__ and nothing else; every other name came from
    # the star import, dunders included -- __version__ is a legitimate export.
    exported = {key for key in namespace if key != "__builtins__"}
    assert exported == set(qbiocode.__all__)


def test_importing_qbiocode_does_not_require_the_quvine_extra():
    """A bare install must import cleanly; QuVINE's deps load only when used."""
    code = (
        "import sys\n"
        "import qbiocode\n"
        f"leaked = [m for m in {QUVINE_ONLY_MODULES!r} if m in sys.modules]\n"
        "print('LEAKED:' + ','.join(leaked))\n"
    )
    completed = run_python(code, cwd=REPO_ROOT, timeout=300)
    assert completed.returncode == 0, completed.stderr[-3000:]
    leaked = completed.stdout.strip().split("LEAKED:")[-1].strip()
    assert not leaked, (
        f"import qbiocode eagerly imported {leaked}; those live behind the "
        f"[quvine] extra, so a bare install would fail at import time"
    )


def test_the_quvine_method_names_are_discoverable():
    """Discovery has to work like pca/nmf/umap: a listing, not a docstring."""
    from qbiocode.embeddings import QUVINE_METHODS, is_transductive

    assert isinstance(QUVINE_METHODS, (list, tuple))
    assert "quvine_rwr" in QUVINE_METHODS
    assert is_transductive("pca") is False
    assert is_transductive("spectral") is True
