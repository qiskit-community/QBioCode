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

"""Shared machinery for the integration tier.

These tests run QBioCode the way a user does -- a real subprocess, a real config,
a real dataset on disk -- because that is the only way to catch defects that live
in the seams. The first one it found: ``_resolve_scaling`` accepted a plain
``['True']`` in every unit test and rejected Hydra's ``ListConfig(['True'])``,
which is what the shipped config actually produces.

Every subprocess pins ``PYTHONPATH`` to this checkout. Without it a machine that
also has qbiocode installed elsewhere (an editable install in a base environment,
say) imports *that* copy, and the tests silently validate the wrong tree.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "qbiocode" / "apps" / "qprofiler" / "configs"

# Long enough that a train/test split leaves both classes on both sides, small
# enough that a full QProfiler run finishes in seconds.
N_SAMPLES = 60
N_FEATURES = 5


def subprocess_env() -> dict:
    """Environment for a child process that must import *this* checkout."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{existing}" if existing else str(REPO_ROOT)
    )
    # Headless: a child that pops a window blocks CI forever.
    env["MPLBACKEND"] = "Agg"
    return env


def run_python(code: str, cwd: Path, timeout: int = 900) -> subprocess.CompletedProcess:
    """Run ``code`` in a child interpreter rooted at this checkout."""
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(cwd),
        env=subprocess_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def write_dataset(directory: Path, name: str = "tiny.csv", seed: int = 0) -> Path:
    """A small, learnable binary dataset: label depends on feature 0 plus noise.

    Learnable matters. On pure noise every model scores near chance, and a test
    that compares metrics across seeds cannot tell a real difference from the
    variance of a coin flip.
    """
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N_SAMPLES, N_FEATURES))
    y = (X[:, 0] + 0.3 * rng.normal(size=N_SAMPLES) > 0).astype(int)
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(N_FEATURES)])
    frame["label"] = y
    path = directory / name
    frame.to_csv(path, index=False)
    return path


def run_qprofiler(work_dir: Path, overrides: list[str], timeout: int = 900) -> pd.DataFrame:
    """Run QProfiler end to end in ``work_dir`` and return its ModelResults frame.

    Raises:
        AssertionError: if the run exits non-zero, or produces no results file --
            a QProfiler run that writes nothing while exiting 0 is the exact
            failure mode ``_validate_config`` was added to prevent.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    write_dataset(work_dir / "data")
    argv = [
        "qprofiler",
        f"--config-dir={CONFIG_DIR}",
        "--config-name=config",
        "folder_path=data",
        "file_dataset=ALL",
        *overrides,
    ]
    code = (
        "import sys\n"
        f"sys.argv = {argv!r}\n"
        "from qbiocode.apps.qprofiler.qprofiler import main\n"
        "main()\n"
    )
    completed = run_python(code, cwd=work_dir, timeout=timeout)
    assert completed.returncode == 0, (
        f"qprofiler exited {completed.returncode}\n"
        f"--- stdout ---\n{completed.stdout[-4000:]}\n"
        f"--- stderr ---\n{completed.stderr[-4000:]}"
    )
    results = sorted(work_dir.glob("results/**/ModelResults.csv"))
    assert results, (
        "qprofiler exited 0 but wrote no ModelResults.csv; a successful run that "
        f"produces nothing is indistinguishable from total failure.\n"
        f"--- stdout ---\n{completed.stdout[-2000:]}"
    )
    assert len(results) == 1, f"expected one results file, found {results}"
    return pd.read_csv(results[0])


# Metrics only. 'time' is wall-clock and 'Model_Parameters' is a repr; neither is
# part of the reproducibility contract.
METRIC_COLUMNS = ["accuracy", "f1_score", "auc"]
KEY_COLUMNS = ["Dataset", "embeddings", "iteration", "model"]


def metric_signature(frame: pd.DataFrame) -> pd.DataFrame:
    """The part of a results frame that two runs at the same seed must match."""
    ordered = frame.sort_values(KEY_COLUMNS).reset_index(drop=True)
    return ordered[KEY_COLUMNS + METRIC_COLUMNS]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT
