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

"""Guards on the test suite itself.

``pytest.importorskip`` is the right tool for exactly one situation: a genuinely
optional dependency, absent in a supported install. Used anywhere else it is a
silent off switch, because a skip and a pass are the same colour in a summary
line.

Eighteen call sites in this suite guarded *first-party* modules --
``pytest.importorskip("qbiocode.utils.qutils")`` and friends. Injecting a single
``ImportError`` at the top of ``qbiocode/utils/qutils.py`` turned 32 assertions
across ``test_error_contracts.py`` into skips and the suite still reported
green: ``compute_pqk``, ``qprofiler`` and ``gat`` all import that module
transitively, so one fault cascaded through four unrelated test classes. Five
more guarded ``matplotlib``, ``networkx``, ``pyyaml``, ``nbformat`` and
``nbclient``, none of which is optional here.

These two tests keep that from coming back.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover -- exercised only on 3.10
    tomllib = pytest.importorskip("tomli", reason="needs tomllib (3.11+) or tomli")

TESTS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

#: Import name -> distribution name, for the cases where they differ. Only the
#: packages this suite actually guards need an entry.
IMPORT_TO_DISTRIBUTION = {
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "skdim": "scikit-dimension",
    "umap": "umap-learn",
    "community": "python-louvain",
    "torch_geometric": "torch-geometric",
    "igraph": "igraph",
}

#: Modules that may be guarded even though no extra declares them, each with
#: the reason. A version-conditional shim is a legitimate skip: on Python 3.11+
#: the guarded branch is never reached at all.
CONDITIONAL_SHIMS = {
    "tomli": "the Python 3.10 backport of the stdlib tomllib",
}


def _requirement_names(path):
    """Distribution names declared in a requirements file, lowercased."""
    names = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        for separator in ("==", ">=", "<=", "~=", "!=", ">", "<", "["):
            line = line.split(separator, 1)[0]
        names.append(line.strip().lower())
    return names


def _declared_optional():
    """Every distribution any extra declares, plus the tiered requirements files.

    Both sources matter and neither subsumes the other: ``quvine`` and ``docs``
    are mirrored between ``pyproject.toml`` and ``requirements/``
    (test_requirements_consistency.py pins them together), while ``dev`` and
    ``apps`` exist only as extras. Reading just the files made ``build`` -- a
    genuinely optional build frontend that only the sdist test needs -- look like
    an undeclared guard.
    """
    optional = set()
    for filename in ("requirements-quvine.txt", "requirements-docs.txt"):
        optional |= set(_requirement_names(REPO_ROOT / "requirements" / filename))

    with open(REPO_ROOT / "pyproject.toml", "rb") as handle:
        config = tomllib.load(handle)
    for requirements in config["project"].get("optional-dependencies", {}).values():
        for requirement in requirements:
            # "qbiocode[apps,quvine]" in the `all` extra is a self-reference, not a
            # third-party package; the extras it names are read on their own pass.
            name = re.split(r"[\[<>=!~; ]", requirement, maxsplit=1)[0].strip().lower()
            if name and name != "qbiocode":
                optional.add(name)
    return optional


def _importorskip_targets():
    """Every literal ``pytest.importorskip("...")`` target, with its location."""
    for path in sorted(TESTS_DIR.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "importorskip"):
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue  # a computed target; nothing to check statically
            yield path.relative_to(REPO_ROOT), node.lineno, node.args[0].value


def test_no_first_party_module_is_reached_through_importorskip():
    """A broken module of ours must fail the suite, not switch it off."""
    offenders = [
        f"{path}:{line} importorskip({target!r})"
        for path, line, target in _importorskip_targets()
        if target == "qbiocode" or target.startswith("qbiocode.")
    ]
    assert not offenders, (
        "first-party modules guarded by pytest.importorskip -- an ImportError in "
        "any of them, including a real regression, becomes a skip instead of a "
        "failure. Import them directly at module scope:\n  " + "\n  ".join(offenders)
    )


def test_every_guarded_module_is_actually_optional():
    """Guarding a mandatory dependency is the same off switch, one step removed.

    A package in ``requirements-base.txt`` is present in every supported
    install, so its guard can only ever fire when the environment is already
    broken -- and then it hides the breakage.
    """
    mandatory = set(_requirement_names(REPO_ROOT / "requirements" / "requirements-base.txt"))
    optional = _declared_optional()

    wrong = []
    for path, line, target in _importorskip_targets():
        if target.startswith("qbiocode"):
            continue  # the test above owns this case
        root = target.split(".", 1)[0]
        distribution = IMPORT_TO_DISTRIBUTION.get(root, root)
        if root in CONDITIONAL_SHIMS or distribution in optional:
            continue
        why = (
            "declared in requirements-base.txt, so it is present in every install"
            if distribution in mandatory
            else "declared by no extra and no requirements file"
        )
        wrong.append(f"{path}:{line} importorskip({target!r}) -- {distribution} is {why}")

    assert not wrong, (
        "pytest.importorskip on a module that is not optional. Import it "
        "directly, or -- if it really is optional -- declare it in an extra in "
        "pyproject.toml (or add it to CONDITIONAL_SHIMS with the reason):\n  "
        + "\n  ".join(wrong)
    )


@pytest.mark.parametrize("module", sorted(CONDITIONAL_SHIMS))
def test_the_documented_shims_are_still_referenced(module):
    """Stop the allowlist above from outliving the code it excuses."""
    guarded = {target for _, _, target in _importorskip_targets()}
    assert module in guarded, (
        f"{module} is allowlisted in CONDITIONAL_SHIMS but nothing guards it any "
        f"more; drop the entry ({CONDITIONAL_SHIMS[module]})."
    )
