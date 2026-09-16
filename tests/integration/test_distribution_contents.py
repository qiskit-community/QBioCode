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

"""What ``pip install qbiocode`` actually receives.

The build is where a repository's untracked state leaks into a published
artifact, and nothing else in the suite looks at it. ``MANIFEST.in`` used to say
``recursive-include tutorial *.csv``, which is a glob over the working tree, not
over the index -- so on a checkout where the data-generation notebook had been
run once it swept 239 gitignored CSVs into the sdist: 62 MB compressed, 123 MB
unpacked, and a size that was a function of the builder's shell history rather
than of the repository. Nobody would have noticed before it was on PyPI.

Hence the invariant below, which is stronger than a size cap and does not need
updating as the tree grows: **the sdist contains only files git tracks.** Run
artifacts, editor backups, a stray virtualenv, a downloaded dataset -- all are
untracked, so all are caught by one assertion.

The wheel gets the complementary check. It ships a subset of the sdist and is
built by a different mechanism (``[tool.setuptools.package-data]`` plus
``include-package-data``), so a data file can be present in one and missing from
the other. Two config YAMLs are the whole runtime data payload, and the apps
that read them fail at startup without them.

Both build with ``--no-isolation`` so the test needs no network: it uses the
setuptools already installed rather than provisioning a fresh build environment.
Skipped when ``build`` is absent -- that is a missing tool, not a broken package.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from .conftest import REPO_ROOT

pytest.importorskip("build", reason="the [dev] extra provides `build`; without it there is nothing to inspect")

# setuptools writes these into the sdist; they exist in no checkout.
GENERATED_AT_BUILD_TIME = {"PKG-INFO", "setup.cfg"}

# The runtime data payload. Each is read at startup by the app that owns it, and
# a wheel without them installs cleanly and then fails on first use.
EXPECTED_WHEEL_DATA = {
    "qbiocode/apps/qprofiler/configs/config.yaml",
    "qbiocode/apps/quvine/configs/config.yaml",
}


#: Directories that belong to the developer's machine, not to the commit. Copied
#: past when staging a build tree: each one either cannot appear in a released
#: artifact or actively distorts what does.
NOT_PART_OF_THE_SOURCE = (
    ".git", ".venv", ".venv-quvine", ".pytest_cache", ".mypy_cache",
    "__pycache__", "dist", "build", "node_modules",
)


def _stage_source_tree(destination: Path) -> Path:
    """Copy the working tree, minus the builder's own leftovers.

    Deliberately *not* a clean ``git archive``: the point of this module is that
    MANIFEST.in globs the working tree, so run output sitting in ``tutorial/**/data``
    has to be visible here or the test proves nothing.

    What is excluded is the build machinery's own residue, and one item in it is
    load-bearing: ``qbiocode.egg-info/SOURCES.txt``. setuptools *unions* the
    previous SOURCES.txt into each new sdist instead of recomputing it, so a
    checkout that ever built with a broader MANIFEST.in keeps shipping the files
    that MANIFEST.in no longer names -- 231 untracked CSVs, in the case that
    prompted this. That is builder-dependence too, but of a different kind than
    this module tests, and it would mask the working-tree channel entirely. It is
    also why a release must build from a fresh checkout, or delete ``*.egg-info``
    first; ``.github/workflows/release.yml`` gets this right by construction,
    since ``actions/checkout`` has no egg-info to be stale.
    """
    root = destination / "repo"
    shutil.copytree(
        REPO_ROOT,
        root,
        ignore=lambda directory, names: {
            name for name in names
            if name in NOT_PART_OF_THE_SOURCE or name.endswith(".egg-info")
        },
        symlinks=True,
    )
    return root


@pytest.fixture(scope="module")
def distributions(tmp_path_factory):
    """Build a real sdist and wheel into a scratch directory.

    Built into ``tmp_path`` rather than ``dist/``, so a stale artifact from a
    previous manual build cannot be mistaken for this one's output.
    """
    outdir = tmp_path_factory.mktemp("dist")
    source = _stage_source_tree(tmp_path_factory.mktemp("src"))
    completed = subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(outdir)],
        cwd=str(source),
        capture_output=True,
        text=True,
        timeout=900,
    )
    if completed.returncode != 0:
        pytest.skip(
            "python -m build --no-isolation failed; this environment cannot build "
            f"the package:\n{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
        )
    sdists = list(outdir.glob("*.tar.gz"))
    wheels = list(outdir.glob("*.whl"))
    assert len(sdists) == 1, f"expected one sdist, got {sdists}"
    assert len(wheels) == 1, f"expected one wheel, got {wheels}"
    return sdists[0], wheels[0]


def _tracked_files() -> set[str]:
    """Every path in the git index.

    ``-z`` matters: this repository contains ``tutorial/PQK - OV.ipynb``, and
    splitting git's output on whitespace reports it as untracked, which reads as
    exactly the bug this module exists to catch.
    """
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    return {path for path in completed.stdout.split("\0") if path}


def _sdist_members(sdist: Path) -> list[str]:
    """Paths inside the sdist, with its ``qbiocode-<version>/`` prefix stripped."""
    with tarfile.open(sdist) as archive:
        return [
            member.name.split("/", 1)[1]
            for member in archive.getmembers()
            if member.isfile() and "/" in member.name
        ]


def test_the_sdist_ships_only_files_git_tracks(distributions):
    """An untracked file in the sdist is a build that depends on the builder.

    The failure this pins was not a wrong file list -- it was a *variable* one:
    the same commit produced a 9 MB or a 62 MB sdist depending on whether the
    notebook had been run in that checkout.
    """
    sdist, _ = distributions
    tracked = _tracked_files()
    strays = sorted(
        name
        for name in _sdist_members(sdist)
        if name not in tracked
        and name not in GENERATED_AT_BUILD_TIME
        and not name.startswith("qbiocode.egg-info/")
    )
    assert not strays, (
        f"{len(strays)} file(s) in the sdist that git does not track, so this "
        f"artifact reflects the build machine rather than the commit:\n"
        + "\n".join(strays[:30])
    )


def test_the_sdist_carries_the_files_the_build_itself_needs(distributions):
    """``pyproject.toml`` reads its dependencies from ``requirements/`` at build time.

    So an sdist missing that file is not merely incomplete -- it cannot be built
    from, and the failure surfaces only when someone installs from source.
    """
    sdist, _ = distributions
    members = set(_sdist_members(sdist))
    assert "requirements/requirements-base.txt" in members, (
        "the sdist omits the requirements file `[tool.setuptools.dynamic]` reads "
        "dependencies from; building from this sdist would produce a package with "
        "no dependencies at all"
    )


def test_the_wheel_carries_the_configs_the_apps_read(distributions):
    """A wheel without these installs fine and fails on first run."""
    _, wheel = distributions
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    missing = sorted(EXPECTED_WHEEL_DATA - names)
    assert not missing, (
        f"the wheel omits {missing}; the app that reads each one fails at startup, "
        "and `pip install qbiocode` is the only way anyone would find out"
    )


def test_the_wheel_ships_no_tests_docs_or_notebooks(distributions):
    """A wheel is what gets imported; the tutorial tree does not belong in site-packages.

    ``packages.find`` excludes ``tests*``/``docs*``/``archive*``, but
    ``include-package-data = true`` means anything MANIFEST.in reaches inside a
    *package* directory ships too -- so this checks the outcome, not the config.
    """
    _, wheel = distributions
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    unwanted = sorted(
        name
        for name in names
        if name.startswith(("tests/", "docs/", "archive/", "tutorial/"))
        or name.endswith(".ipynb")
    )
    assert not unwanted, f"the wheel ships files that are not part of the library:\n{unwanted[:30]}"


def test_the_declared_dependencies_survive_the_build(distributions):
    """Dependency metadata is computed at build time, so it can silently come out empty.

    It did: the old 189-line ``setup.py`` re-declared metadata that
    ``pyproject.toml`` also declared, and the sdist it produced listed no runtime
    dependencies while leaking the *build-time* setuptools pin into them. Both
    halves are asserted here, because both shipped.
    """
    import re

    sdist, _ = distributions
    with tarfile.open(sdist) as archive:
        name = next(n for n in archive.getnames() if n.endswith("PKG-INFO") and n.count("/") == 1)
        metadata = archive.extractfile(name).read().decode()

    requirements = re.findall(r"^Requires-Dist: (.+)$", metadata, re.MULTILINE)
    assert requirements, (
        "the sdist declares no dependencies; `pip install qbiocode` from it would "
        "install an unusable package"
    )
    base = [r for r in requirements if "extra ==" not in r]
    assert base, "every dependency is behind an extra, so a plain install gets nothing"

    def distribution_name(requirement):
        """`numpy>=2.0` -> `numpy`; the name ends at the first specifier or marker."""
        return re.split(r"[<>=!~;\[\s]", requirement, maxsplit=1)[0].strip().lower()

    leaked = [r for r in base if distribution_name(r) == "setuptools"]
    assert not leaked, (
        f"build-time setuptools leaked into the runtime dependencies as {leaked}; "
        "it belongs in [build-system] requires, and the `[quvine]` extra pins it "
        "separately for node2vec's sake"
    )
