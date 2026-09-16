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

"""The PQK projection cache must key on the feature map that produced it.

The cache filename used to be
``pqk_projection_{data_key}_{embedding}_{n_components}_{iter}_{train,test}.npy``,
which names the *dataset* but not the *feature map*. Changing ``encoding``,
``entanglement``, ``reps`` or ``primitive`` and rerunning into the same
``pqk_projection_dir`` therefore reloaded the previous run's projections and
reported them as the new result -- a silently wrong comparison between feature
maps, which is precisely what the single-cell tutorials were doing.

These tests are black-box: they run ``compute_pqk`` on the statevector simulator
and look at which files appear on disk. They deliberately do not recompute the
fingerprint, because a test that reimplements the key would pass even if the key
were wrong.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from qbiocode.learning.compute_pqk import compute_pqk

#: Small enough to keep the simulator work negligible, large enough for the
#: 5-fold stratified search inside modeleval (5 samples per class).
N_TRAIN, N_TEST, N_FEATURES = 10, 4, 2

#: The feature-map parameters the cache key must cover.
BASELINE = {"encoding": "Z", "entanglement": "linear", "reps": 2}


def _dataset():
    rng = np.random.default_rng(0)
    return (
        rng.normal(size=(N_TRAIN, N_FEATURES)),
        rng.normal(size=(N_TEST, N_FEATURES)),
        np.array([0, 1] * (N_TRAIN // 2)),
        np.array([0, 1] * (N_TEST // 2)),
    )


def _args(projection_dir):
    return {
        "backend": "simulator",  # StatevectorEstimator: offline and fast
        "seed": 42,
        "shots": 100,
        "grid_search": False,
        "pqk_projection_dir": str(projection_dir),
    }


def _projections(projection_dir):
    """The projection files in a directory, ignoring the checkpoints subdirectory."""
    return {
        name
        for name in os.listdir(projection_dir)
        if name.startswith("pqk_projection_") and name.endswith(".npy")
    }


def _run(projection_dir, **overrides):
    X_train, X_test, y_train, y_test = _dataset()
    params = {**BASELINE, **overrides}
    return compute_pqk(
        X_train, X_test, y_train, y_test, _args(projection_dir), data_key="ds", **params
    )


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    """A directory holding one completed baseline run, shared across the module.

    Module-scoped because the first ``compute_pqk`` call pays the qiskit warm-up;
    the projections themselves take milliseconds at this size.
    """
    directory = tmp_path_factory.mktemp("pqk_projections")
    _run(directory)
    assert len(_projections(directory)) == 2, "baseline run wrote no projection pair"
    return directory


@pytest.fixture(scope="module")
def pristine_cache(tmp_path_factory):
    """A directory holding *only* the baseline pair, for the corruption tests.

    Separate from ``cache_dir`` on purpose: the parametrized tests below add further
    fingerprints to that one, and corrupting a file belonging to some other feature
    map proves nothing -- a baseline run never validates it.
    """
    directory = tmp_path_factory.mktemp("pqk_pristine")
    _run(directory)
    return directory


@pytest.mark.parametrize(
    "overrides",
    [
        {"encoding": "ZZ"},
        {"reps": 3},
        {"entanglement": "full"},
    ],
    ids=["encoding", "reps", "entanglement"],
)
def test_changing_the_feature_map_writes_a_separate_cache_entry(cache_dir, overrides):
    """Each of these produces different projections, so each needs its own files."""
    before = _projections(cache_dir)
    _run(cache_dir, **overrides)
    after = _projections(cache_dir)

    assert before < after, (
        f"changing {overrides} reused the existing cache: the projections on disk were "
        "computed with a different feature map"
    )
    assert len(after - before) == 2, f"expected a new train/test pair, got {after - before}"


def test_identical_parameters_reuse_the_cache_without_touching_a_backend(
    cache_dir, monkeypatch
):
    """A cache hit must short-circuit before any session is opened."""
    import qbiocode.utils.qutils as qutils

    def _fail(*args, **kwargs):
        raise AssertionError("a backend session was opened despite a full cache hit")

    monkeypatch.setattr(qutils, "get_backend_session", _fail)

    before = _projections(cache_dir)
    _run(cache_dir)  # exactly the baseline parameters
    assert _projections(cache_dir) == before, "a cache hit wrote new files"


class TestAStaleCacheFileIsRefused:
    """A file whose shape cannot match the current dataset must raise, not load.

    Validation matters because the fingerprint covers the feature map but not the
    dataset size: pointing a differently-shaped dataset at the same
    ``pqk_projection_dir`` and ``data_key`` reaches a file that is wrong for it.
    """

    @staticmethod
    def _the_train_projection(directory):
        train = [n for n in _projections(directory) if n.endswith("_train.npy")]
        assert len(train) == 1, (
            f"expected exactly one train projection to corrupt, found {sorted(train)}"
        )
        return os.path.join(directory, train[0])

    def test_a_wrong_row_count_is_reported_with_both_numbers(self, pristine_cache, tmp_path):
        import shutil

        directory = tmp_path / "rows"
        shutil.copytree(pristine_cache, directory)
        path = self._the_train_projection(directory)
        np.save(path, np.load(path, allow_pickle=False)[:3])  # 3 rows, not N_TRAIN

        with pytest.raises(ValueError, match=r"has 3 rows.*expects 10 rows"):
            _run(directory)

    def test_a_wrong_feature_width_is_reported_with_both_numbers(
        self, pristine_cache, tmp_path
    ):
        import shutil

        directory = tmp_path / "width"
        shutil.copytree(pristine_cache, directory)
        path = self._the_train_projection(directory)
        rows = np.load(path, allow_pickle=False)
        # Right number of rows, wrong width: what a run at a different feature
        # dimension leaves behind. 3 observables x 2 qubits = 6 is expected.
        np.save(path, np.zeros((len(rows), 4)))

        with pytest.raises(ValueError, match=r"4 features per row.*produces 6"):
            _run(directory)
