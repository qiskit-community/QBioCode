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

"""Every declared console script imports and answers ``--help``.

This is the regression guard for the broken ``cli.py``: it did
``from apps.qprofiler.qprofiler import main``, and no top-level ``apps`` package
exists in the tree, so ``qprofiler`` was an unconditional ``ImportError`` for
anyone who installed the package. Nothing in the test suite noticed, because
nothing imported the entry point.

The targets are read out of ``pyproject.toml`` rather than listed here, so a
console script added later is covered without editing this file.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from .conftest import REPO_ROOT, subprocess_env


def _console_scripts() -> dict[str, str]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)
    scripts = config["project"]["scripts"]
    assert scripts, "pyproject declares no console scripts"
    return scripts


SCRIPTS = _console_scripts()


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_the_console_script_imports_and_answers_help(name):
    """Resolve ``module:function`` and run ``--help``, from *outside* the checkout.

    The cwd matters: the broken import worked by accident for anyone whose shell
    happened to sit in a directory where ``apps`` was importable, and failed for
    every real installation.

    Both halves share one subprocess -- importing qbiocode costs about ten
    seconds, and there are four scripts to cover.
    """
    target = SCRIPTS[name]
    module, _, function = target.partition(":")
    code = (
        "import sys, importlib\n"
        f"module = importlib.import_module({module!r})\n"
        f"entry = getattr(module, {function!r})\n"
        "assert callable(entry), 'entry point is not callable'\n"
        f"sys.argv = [{name!r}, '--help']\n"
        "try:\n"
        "    entry()\n"
        "except SystemExit as exit_request:\n"
        "    code = exit_request.code or 0\n"
        "    assert code == 0, f'--help exited {code}'\n"
        "print('HELP-OK', file=sys.stderr)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(Path(REPO_ROOT.anchor)),
        env=subprocess_env(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, (
        f"console script {name} -> {target} failed\n"
        f"--- stdout ---\n{completed.stdout[-2000:]}\n"
        f"--- stderr ---\n{completed.stderr[-3000:]}"
    )
    assert "HELP-OK" in completed.stderr, "the entry point never returned from --help"
    assert completed.stdout.strip(), f"{name} --help printed nothing"
