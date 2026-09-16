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

"""QuVINE reached through QProfiler, exactly like pca or umap.

The point of the whole embedding hookup is that a QuVINE method is selectable by
name from the config -- ``embeddings: ['quvine_rwr']`` -- with no other change.
That path crosses the config validator, the kNN graph builder, the SGNS
embedder and the dimension-safety net, so only a real run exercises it.

Skipped without the ``[quvine]`` extra: ``gensim`` backs the SGNS methods.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gensim", reason="the [quvine] extra is not installed")

from .conftest import METRIC_COLUMNS, run_qprofiler  # noqa: E402

METHOD = "quvine_rwr"
N_COMPONENTS = 2


@pytest.fixture(scope="module")
def quvine_run(tmp_path_factory):
    return run_qprofiler(
        tmp_path_factory.mktemp("quvine"),
        [
            f"embeddings=[{METHOD}]",
            "model=[lr]",
            "iter=1",
            f"n_components={N_COMPONENTS}",
            "n_jobs=1",
            "seed=7",
        ],
    )


class TestTheConfigDrivenPath:
    def test_the_quvine_embedding_ran(self, quvine_run):
        assert list(quvine_run["embeddings"].unique()) == [METHOD]
        assert len(quvine_run) == 1

    def test_the_embedding_width_is_the_requested_dimension(self, quvine_run):
        """The safety net must deliver exactly ``n_components``, whatever SGNS returned."""
        assert list(quvine_run["# Features"].unique()) == [N_COMPONENTS]

    @pytest.mark.parametrize("column", METRIC_COLUMNS)
    def test_the_metrics_are_finite(self, quvine_run, column):
        """Not *good* -- finite. A degenerate embedding shows up as nan, not as chance."""
        values = pd.to_numeric(quvine_run[column])
        assert values.notna().all(), f"{column} is nan; the embedding collapsed"
        assert np.isfinite(values).all()


class TestTheDirectEmbeddingCall:
    """``get_embeddings`` must behave like the sklearn methods it sits beside."""

    @pytest.fixture(scope="class")
    def matrices(self):
        rng = np.random.default_rng(0)
        return rng.normal(size=(20, 5)), rng.normal(size=(8, 5))

    def test_it_returns_train_and_test_blocks_of_the_requested_width(self, matrices):
        from qbiocode import get_embeddings

        X_train, X_test = matrices
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z_train, Z_test = get_embeddings(METHOD, X_train, X_test, n_components=3)
        assert Z_train.shape == (len(X_train), 3)
        assert Z_test.shape == (len(X_test), 3)
        assert np.isfinite(Z_train).all() and np.isfinite(Z_test).all()

    def test_it_declares_itself_transductive(self, matrices):
        """Silence here is the defect: test features do participate in the graph."""
        from qbiocode import get_embeddings

        X_train, X_test = matrices
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            get_embeddings(METHOD, X_train, X_test, n_components=2)
        messages = [str(entry.message) for entry in caught]
        assert any("transductive" in message for message in messages), (
            f"no transductivity warning was emitted; got {messages}"
        )
