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

"""The Sphinx site builds, and builds without a single warning.

This started as a structural check with about thirty docstring warnings written
off as noise, because turning every warning into an error would have failed the
docs job on an unrelated nit in a module nobody touched. Those thirty are fixed:
malformed lists and formulae that rendered as block quotes, a heading whose
underline was one character short, ``|u - v|`` read as an RST substitution, a
JSON sample parsed as section titles. So the exemption is gone and the bar is
zero warnings, enforced here and by ``-W`` in ``docs/Makefile``.

Zero is the only threshold that holds. Any other number is a budget, and a
budget is spent: the warning that means the site is genuinely broken -- a
toctree pointing at a document that does not exist, a page reachable from
nothing, an unresolvable reference -- arrives indistinguishable from the thirty
already being ignored.

Skipped without the ``[docs]`` extra, without the pandoc binary that nbsphinx
shells out to for notebook pages, and without nbconvert's ``rst`` template --
those three are missing toolchain, not broken documentation, and reporting them
as failures would train people to ignore this test.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from .conftest import REPO_ROOT, subprocess_env

pytest.importorskip("sphinx", reason="the [docs] extra is not installed")

DOCS_SOURCE = REPO_ROOT / "docs" / "source"

def _nbconvert_rst_template_available() -> bool:
    """nbsphinx renders notebook cells through nbconvert's ``rst`` template.

    The template ships as data under a prefix's ``share/jupyter``, and jupyter
    only searches ``sys.prefix``. In a virtualenv built with
    ``--system-site-packages``, nbconvert is importable from the base
    environment while its templates are not on the search path, and the build
    dies in extension setup with "No template sub-directory with name 'rst'".
    That is an environment defect, not a documentation defect, so skip rather
    than report the docs as broken.
    """
    try:
        from jupyter_core.paths import jupyter_path
    except ImportError:
        return False
    return any(
        (Path(directory) / "nbconvert" / "templates" / "rst").is_dir()
        for directory in jupyter_path()
    )


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    """One build, shared by every assertion below -- Sphinx is not cheap."""
    if shutil.which("pandoc") is None:
        pytest.skip("nbsphinx needs the pandoc binary to render notebook pages")
    if not _nbconvert_rst_template_available():
        pytest.skip(
            "nbconvert's rst template is not on the jupyter search path; set "
            "JUPYTER_PATH to the prefix that owns share/jupyter"
        )
    out_dir = tmp_path_factory.mktemp("html")
    doctrees = tmp_path_factory.mktemp("doctrees")
    completed = subprocess.run(
        [
            sys.executable, "-m", "sphinx",
            "-b", "html",
            "-d", str(doctrees),
            str(DOCS_SOURCE), str(out_dir),
        ],
        cwd=str(REPO_ROOT),
        env=subprocess_env(),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    return completed, out_dir


def test_the_build_succeeds(build):
    completed, _ = build
    assert completed.returncode == 0, (
        f"sphinx-build exited {completed.returncode}\n"
        f"--- stdout ---\n{completed.stdout[-4000:]}\n"
        f"--- stderr ---\n{completed.stderr[-4000:]}"
    )


def test_it_emits_no_warnings_at_all(build):
    """Every warning, wherever it comes from -- page markup or docstring.

    A docstring warning is not cosmetic: docutils drops or mangles the construct
    it could not parse, so the published API page shows a run-together paragraph
    or a stray block quote where a list or a formula was meant. The location is
    ``qbiocode/...`` but the damage is on the site.

    The one warning that is genuinely not ours -- sphinx-autodoc-typehints
    importing ``_typeshed`` through pydantic's dataclass internals -- is
    suppressed by subtype in ``conf.py``, where the reason is recorded next to
    it, rather than filtered out here where it would shade every future
    ``guarded_import`` failure too.
    """
    completed, _ = build
    text = completed.stdout + completed.stderr
    offenders = [
        line
        for line in text.splitlines()
        if re.search(r"\b(WARNING|ERROR)\b", line)
    ]
    assert not offenders, (
        f"{len(offenders)} warning(s) from a build that must emit none:\n"
        + "\n".join(offenders[:40])
    )


def test_the_makefile_treats_warnings_as_errors():
    """The gate lives in the Makefile, so CI's plain ``make html`` inherits it.

    Without this, ``-W`` can be dropped from ``SPHINXOPTS`` and nothing fails:
    the test above would still pass locally while CI quietly went back to
    building a warning-ridden site successfully.
    """
    makefile = (REPO_ROOT / "docs" / "Makefile").read_text()
    match = re.search(r"^SPHINXOPTS\s*\?*=\s*(.*)$", makefile, re.MULTILINE)
    assert match, "docs/Makefile does not define SPHINXOPTS"
    assert "-W" in match.group(1).split(), (
        f"docs/Makefile SPHINXOPTS is {match.group(1)!r}; a docs build that "
        f"warns must fail"
    )


@pytest.mark.parametrize(
    "page",
    [
        "index.html",
        "installation.html",
        "tutorials.html",
        # A generated page: conf.py copies tutorial/ into docs/source/tutorials/
        # at builder-inited. If that hook breaks, every tutorial link on the site
        # 404s, and without this the build still "succeeds".
        "tutorials/QuVINE/example_quvine.html",
        "apps/quvine.html",
        "api/qbiocode.evaluation.html",
    ],
)
def test_the_expected_pages_are_written(build, page):
    _, out_dir = build
    assert (out_dir / page).is_file(), f"{page} was not produced"


def test_the_sphinx_output_lands_where_ci_uploads_from():
    """``docs/Makefile`` and the CI upload path must name the same directory.

    They disagreed once: the Makefile built into ``docs/build`` while a
    hand-made ``docs/_build/html`` tree was committed and published.
    """
    makefile = (REPO_ROOT / "docs" / "Makefile").read_text()
    match = re.search(r"^BUILDDIR\s*=\s*(\S+)", makefile, re.MULTILINE)
    assert match, "docs/Makefile does not define BUILDDIR"
    build_dir = match.group(1)
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert f"docs/{build_dir}/html" in workflow, (
        f"docs/Makefile builds into docs/{build_dir}, which the CI workflow "
        f"never uploads from"
    )
