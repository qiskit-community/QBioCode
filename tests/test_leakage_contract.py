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

"""The train/test contamination contract.

QProfiler used to scale by fitting a scaler on ``vstack([X_train, X_test])``, so the
test distribution shaped the transform applied to the training data. Every number
it reported was therefore optimistic by an unknown amount.

The invariant that rules that out is stronger than "we call fit on train": the
*training* output must not depend on the test rows at all. These tests assert it by
holding the training set fixed and varying the test set wildly -- if a single
training value moves, something fitted on test data.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest

from qbiocode.embeddings.embed import get_embeddings
from qbiocode.utils.helper_fn import scale_train_test

SCALERS = ["StandardScaler", "MinMaxScaler"]


@pytest.fixture
def train():
    rng = np.random.default_rng(0)
    return rng.normal(size=(40, 6))


@pytest.mark.parametrize("scaling", SCALERS)
@pytest.mark.parametrize(
    "test_shift",
    [0.0, 50.0, -50.0, 1e4],
    ids=["identical", "shifted-up", "shifted-down", "extreme"],
)
def test_the_training_transform_ignores_the_test_set(train, scaling, test_shift):
    """Fit statistics must come from train alone, whatever the test set looks like."""
    rng = np.random.default_rng(1)
    test = rng.normal(size=(15, 6)) + test_shift

    baseline, _ = scale_train_test(train, train[:1], scaling=scaling)
    varied, _ = scale_train_test(train, test, scaling=scaling)

    np.testing.assert_array_equal(
        np.asarray(baseline),
        np.asarray(varied),
        err_msg=(
            f"the scaled training set changed when the test set shifted by {test_shift}: "
            "the scaler saw test data"
        ),
    )


@pytest.mark.parametrize("scaling", SCALERS)
def test_the_test_set_is_transformed_not_refitted(scaling):
    """A scaler fit per split would map each split onto the same range, hiding drift."""
    train = np.array([[0.0], [2.0], [4.0], [6.0], [8.0]])
    # Deliberately asymmetric about the train mean (4.0): a test set symmetric about it
    # standardizes to mean exactly zero by construction, which would make the
    # StandardScaler branch below pass for the wrong reason.
    test = np.array([[16.0], [-8.0], [20.0]])

    train_scaled, test_scaled = scale_train_test(train, test, scaling=scaling)
    train_scaled = np.asarray(train_scaled)
    test_scaled = np.asarray(test_scaled)

    if scaling == "MinMaxScaler":
        # Train spans exactly [0, 1]; test lies outside it because train's min/max
        # were used. A refit-per-split would have squashed test into [0, 1] too.
        assert train_scaled.min() == pytest.approx(0.0)
        assert train_scaled.max() == pytest.approx(1.0)
        assert test_scaled.max() > 1.0, "test was rescaled onto the train range"
        assert test_scaled.min() < 0.0, "test was rescaled onto the train range"
    else:
        assert train_scaled.mean() == pytest.approx(0.0, abs=1e-12)
        # A scaler refit on test would centre test at zero.
        assert abs(test_scaled.mean()) > 0.5, "test appears to have been centred on itself"


def test_none_returns_the_inputs_untouched():
    train = np.arange(12, dtype=float).reshape(4, 3)
    test = np.full((2, 3), 99.0)
    out_train, out_test = scale_train_test(train, test, scaling="None")
    assert out_train is train and out_test is test


def test_a_mistyped_scaler_is_refused_rather_than_silently_skipped():
    """``'minmaxscaler'`` once fell through to the no-op branch and reported success."""
    with pytest.raises(ValueError, match="case-sensitive"):
        scale_train_test(np.zeros((3, 2)), np.zeros((1, 2)), scaling="minmaxscaler")


class TestLabelsNeverReachTheEmbedding:
    """Test *features* may participate in a transductive embedding; labels never do."""

    def test_get_embeddings_takes_no_label_argument(self):
        params = set(inspect.signature(get_embeddings).parameters)
        label_like = {"y", "y_train", "y_test", "labels", "target", "y_true"}
        assert not (params & label_like), (
            f"get_embeddings accepts label arguments {sorted(params & label_like)}; "
            "a supervised embedding would leak the target into the feature space"
        )

    def test_no_embedding_helper_accepts_labels(self):
        """Covers the private helpers too, so the contract cannot be reintroduced."""
        import qbiocode.embeddings.embed as embed_module

        offenders = []
        for name, obj in vars(embed_module).items():
            if not inspect.isfunction(obj) or obj.__module__ != embed_module.__name__:
                continue
            params = set(inspect.signature(obj).parameters)
            leaked = params & {"y", "y_train", "y_test", "labels", "target"}
            if leaked:
                offenders.append(f"{name}{sorted(leaked)}")
        assert not offenders, f"embedding helpers accepting labels: {offenders}"

    def test_a_transductive_embedding_is_unchanged_by_the_test_labels(self):
        """Sanity check that the embedding is a pure function of the feature matrices."""
        rng = np.random.default_rng(7)
        X_train, X_test = rng.normal(size=(30, 5)), rng.normal(size=(10, 5))
        with pytest.warns(UserWarning, match="transductive"):
            first = get_embeddings("spectral", X_train, X_test, n_components=3)
        with pytest.warns(UserWarning, match="transductive"):
            second = get_embeddings("spectral", X_train, X_test, n_components=3)
        for a, b in zip(first, second):
            np.testing.assert_allclose(np.abs(np.asarray(a)), np.abs(np.asarray(b)))


class TestNMFSurvivesALeakageFreeScaler:
    """The leakage fix broke NMF, and every shipped config runs NMF.

    Fitting the MinMaxScaler on train+test bounded *both* splits to [0, 1], so
    ``NMF.transform`` never saw a negative. Fitting it on the training split alone
    -- the correct protocol -- leaves the test split free to fall below the
    training minimum, and NMF is defined only on non-negative input. The result
    was ``ValueError: Negative values in data passed to X in NMF`` on
    ``embeddings: ['pca', 'nmf', 'none']`` with ``scaling: ['True']``, which is
    the config shipped with the QProfiler tutorial.

    The fix clips the test matrix at zero and says how much it clipped. Undoing
    the clip breaks the notebook; undoing the *report* hides an extrapolation.
    """

    @staticmethod
    def _split_with_test_below_the_training_range():
        rng = np.random.default_rng(3)
        train = rng.uniform(1.0, 2.0, size=(40, 6))
        test = rng.uniform(0.0, 1.5, size=(15, 6))  # partly below train's minimum
        return scale_train_test(train, test, scaling="MinMaxScaler")

    def test_the_scaler_really_does_put_test_rows_below_zero(self):
        """Guards the premise: without this the test below proves nothing."""
        _, test = self._split_with_test_below_the_training_range()
        assert test.min() < 0, (
            "the fixture no longer produces out-of-range test values, so the NMF "
            "case it is meant to exercise is not being reached"
        )

    def test_nmf_transforms_the_test_split_instead_of_raising(self):
        train, test = self._split_with_test_below_the_training_range()
        with pytest.warns(UserWarning, match="clipped to 0"):
            train_embedded, test_embedded = get_embeddings("nmf", train, test, n_components=3)
        assert train_embedded.shape == (40, 3)
        assert test_embedded.shape == (15, 3)
        assert np.isfinite(test_embedded).all()

    def test_the_clip_reports_how_many_entries_it_touched(self):
        """A silent clip would hide a test split outside the training range."""
        train, test = self._split_with_test_below_the_training_range()
        expected = int((test < 0).sum())
        with pytest.warns(UserWarning, match=rf"\b{expected} of {test.size} test entries"):
            get_embeddings("nmf", train, test, n_components=3)

    def test_a_test_split_inside_the_training_range_is_not_warned_about(self):
        """No excursion, no warning -- the report has to mean something."""
        rng = np.random.default_rng(11)
        train = rng.uniform(0.0, 5.0, size=(40, 6))
        test = rng.uniform(1.0, 4.0, size=(10, 6))
        train, test = scale_train_test(train, test, scaling="MinMaxScaler")
        assert test.min() >= 0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            get_embeddings("nmf", train, test, n_components=3)
        # NMF's own ConvergenceWarning is expected here and is not what this checks.
        assert not [w for w in caught if "clipped" in str(w.message)]

    def test_the_train_embedding_is_unaffected_by_the_clip(self):
        """Clipping is a test-side repair; it must not change the fitted model."""
        train, test = self._split_with_test_below_the_training_range()
        with pytest.warns(UserWarning):
            first, _ = get_embeddings("nmf", train, test, n_components=3)
        untouched_test = np.clip(test, 0.0, None)
        second, _ = get_embeddings("nmf", train, untouched_test, n_components=3)
        np.testing.assert_allclose(first, second)
