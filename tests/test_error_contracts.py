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

"""What the code does when it is given something wrong, or cannot compute a result.

Three failure shapes are covered, because each one shipped and each one is worse
than an exception:

1. **A fabricated value returned as a measurement.** ``netmf`` returned
   ``np.random.randn(...)`` when its factorization failed, and the registry
   scored that noise as an embedding. ``evaluate_graph`` reported modularity
   ``0.0`` -- "no community structure beyond chance", an achievable finding --
   for a partition it could not compute at all, and link prediction reported AUC
   ``0.5`` for an evaluation that was undefined. The tests here pin the honest
   alternatives: raise, or report ``nan`` and count how many values a summary
   statistic was actually computed over.

2. **A silent no-op reported as success.** ``scale_train_test('minmaxscaler')``
   returned unscaled data; QProfiler with ``iter: 0`` or an empty ``embeddings``
   list produced no output and exited 0; a mistyped ``--config`` fell back to the
   packaged default. Every one now raises and names the value it received.

3. **An error attributed to the wrong thing.** A mistyped ``encoding`` reached
   qiskit and came back as "'str' object has no attribute 'num_qubits'"; a
   mistyped ``entanglement`` came back as "Something went wrong in Rust space";
   an unknown model name came back as a bare ``KeyError`` from a joblib worker.
   These are validated at the boundary now, and the assertions below check the
   message names the *parameter*, not just the fact of failure.
"""

import logging
import pathlib
from importlib import import_module

import networkx as nx
import numpy as np
import pytest

# First-party modules are imported directly, never through
# ``pytest.importorskip``. Each one below needs only packages that
# requirements-base.txt makes mandatory -- qiskit, torch, scikit-learn, hydra --
# so there is no install in which they are legitimately missing. Guarding them
# meant the opposite of safety: a genuine ImportError anywhere in the package
# turned these tests *off* rather than red. Injecting one fault at the top of
# qbiocode/utils/qutils.py silently skipped 32 assertions in this file, because
# compute_pqk, qprofiler and gat all import it transitively. A broken import has
# to fail loudly, so it is a collection error now.
# ``import_module`` rather than ``from X import Y``: several of these packages
# re-export a *function* or a string under the same name as the submodule
# (``qbiocode.learning.compute_pqk`` is both a module and the function it
# defines), so ``from`` would bind whichever the parent package happened to
# expose. ``import_module`` always returns the module, and still raises.
_compute_pqk = import_module("qbiocode.learning.compute_pqk")
_gat = import_module("qbiocode.apps.quvine.baselines.gat")
_link_prediction = import_module("qbiocode.apps.quvine.evaluation.link_prediction")
_method_adapters = import_module("qbiocode.apps.quvine.reproducibility.method_adapters")
_netmf = import_module("qbiocode.apps.quvine.baselines.netmf")
_qprofiler = import_module("qbiocode.apps.qprofiler.qprofiler")
_qutils = import_module("qbiocode.utils.qutils")


# ---------------------------------------------------------------------------
# 1. Undefined results are nan or exceptions, never plausible numbers
# ---------------------------------------------------------------------------
def test_undefined_modularity_is_nan_not_zero():
    """0.0 modularity is a real finding; an uncomputable partition must not fake it."""
    from qbiocode.evaluation.graph_evaluation import compute_community_metrics

    # A graph with no edges has no community structure to detect at all.
    metrics = compute_community_metrics(nx.empty_graph(5))
    assert "modularity" in metrics
    assert np.isnan(metrics["modularity"]), (
        f"expected nan for an undetectable partition, got {metrics['modularity']!r}"
    )


def test_link_prediction_reports_nan_for_a_single_class_split(caplog):
    """Ranking metrics are undefined without both classes -- 0.5 would be a lie."""
    mod = _method_adapters

    embeddings = np.random.default_rng(0).normal(size=(6, 4))
    # This path reports through `logging`, not `warnings`: it runs inside the
    # reproducibility harness, which captures logs per method and would otherwise
    # drop the reason entirely.
    with caplog.at_level(logging.WARNING, logger=mod.__name__):
        out = mod.evaluate_link_prediction(embeddings, [(0, 1), (2, 3)], [])
    assert "single class" in caplog.text, caplog.text

    assert set(out) == {"auc_roc", "auc_pr", "f1"}
    assert all(np.isnan(v) for v in out.values()), out


def test_link_prediction_summary_is_nan_aware():
    """One undefined method must not erase every method that succeeded."""
    ev = _link_prediction

    results = {
        "good_a": {"auc_roc": 0.8, "auc_pr": 0.7, "mrr": 0.6},
        "good_b": {"auc_roc": 0.6, "auc_pr": 0.5, "mrr": 0.4},
        "undefined": {"auc_roc": float("nan"), "auc_pr": float("nan"), "mrr": float("nan")},
    }
    summary = ev.summarize_link_prediction_results(results)

    assert summary["mean_auc_roc"] == pytest.approx(0.7), summary
    assert summary["n_defined_auc_roc"] == 2
    assert summary["n_successful_methods"] == 3


def test_link_prediction_summary_of_nothing_is_nan_not_zero():
    ev = _link_prediction

    summary = ev.summarize_link_prediction_results({})
    assert np.isnan(summary["mean_auc_roc"])
    assert summary["n_defined_auc_roc"] == 0


def test_netmf_raises_rather_than_returning_random_noise():
    """A failed factorization must reach the registry as a failure, not a result."""
    netmf = _netmf
    import inspect

    # Comments are stripped first: the code that replaced the fallback *explains*
    # it, and that explanation names the call it removed.
    source = "\n".join(
        line.split("#", 1)[0] for line in inspect.getsource(netmf).splitlines()
    )
    assert "np.random.randn" not in source and "np.random.normal" not in source, (
        "netmf must not synthesize an embedding: a random matrix returned from a "
        "failed factorization is scored by the registry as if it were a result."
    )


# ---------------------------------------------------------------------------
# 2. Boundary validation: evaluate_graph
# ---------------------------------------------------------------------------
def test_evaluate_graph_rejects_none_but_accepts_an_empty_graph():
    from qbiocode import evaluate_graph

    with pytest.raises(TypeError, match="got None"):
        evaluate_graph(None)

    with pytest.warns(UserWarning, match="empty graph"):
        df = evaluate_graph(nx.Graph(), name="empty")
    assert df.shape[0] == 1
    assert int(df["num_nodes"].iloc[0]) == 0


def test_evaluate_graph_names_the_type_it_was_handed():
    from qbiocode import evaluate_graph

    with pytest.raises(TypeError, match="ndarray"):
        evaluate_graph(np.eye(3))


def test_evaluate_graph_still_summarizes_a_real_graph():
    from qbiocode import evaluate_graph

    df = evaluate_graph(nx.karate_club_graph(), name="karate")
    assert df.shape[0] == 1
    row = df.iloc[0]
    assert row["Graph"] == "karate"
    assert int(row["num_nodes"]) == 34
    for column in ("spectral_gap", "density", "modularity"):
        assert column in df.columns


# ---------------------------------------------------------------------------
# 2b. Boundary validation: embeddings
# ---------------------------------------------------------------------------
def test_check_embedding_name_normalizes_and_suggests():
    from qbiocode.embeddings import check_embedding_name

    assert check_embedding_name("  PCA ") == "pca"
    with pytest.raises(ValueError, match="Did you mean: pca"):
        check_embedding_name("pcaa")
    with pytest.raises(ValueError, match=r"got 3 \(int\)"):
        check_embedding_name(3)


@pytest.mark.parametrize(
    "X_train, X_test, expected",
    [
        (np.zeros(10), np.zeros((3, 2)), "must be 2-D"),
        (np.zeros((10, 4)), np.zeros((3, 2)), "same number of features"),
        (np.zeros((0, 4)), np.zeros((3, 4)), "X_train is empty"),
    ],
)
def test_get_embeddings_validates_matrix_shapes(X_train, X_test, expected):
    from qbiocode import get_embeddings

    with pytest.raises(ValueError, match=expected):
        get_embeddings("pca", X_train, X_test, n_components=2)


def test_get_embeddings_accepts_nested_lists():
    """A list of lists is the natural notebook input; it used to fail on .shape."""
    from qbiocode import get_embeddings

    Z_train, Z_test = get_embeddings(
        "pca", [[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]], [[2.0, 3.0]], n_components=1
    )
    assert Z_train.shape == (3, 1) and Z_test.shape == (1, 1)


# ---------------------------------------------------------------------------
# 2c. Boundary validation: scaling, qutils, compute_pqk, model_run
# ---------------------------------------------------------------------------
def test_scale_train_test_rejects_a_mistyped_scaler():
    """The old else-branch silently returned unscaled data for any typo."""
    from qbiocode import scale_train_test

    X_train, X_test = np.array([[1.0], [3.0]]), np.array([[2.0]])
    with pytest.raises(ValueError, match="Unrecognized scaling 'minmaxscaler'"):
        scale_train_test(X_train, X_test, "minmaxscaler")
    # The three documented names keep working.
    assert scale_train_test(X_train, X_test, "None")[0] is X_train
    assert scale_train_test(X_train, X_test, "MinMaxScaler")[0].ravel().tolist() == [0.0, 1.0]


def test_get_feature_map_rejects_unknown_names_and_patterns():
    qutils = _qutils

    with pytest.raises(ValueError, match=r"Unsupported feature_map 'zz'"):
        qutils.get_feature_map("zz", 4)
    with pytest.raises(ValueError, match=r"Unsupported entanglement 'chain'"):
        qutils.get_feature_map("ZZ", 4, entanglement="chain")
    with pytest.raises(ValueError, match="positive integer"):
        qutils.get_feature_map("Z", 0)


def test_every_advertised_optimizer_is_actually_constructible():
    """L_BFGS_B was built and thrown away by a `==` typo, so it never worked."""
    qutils = _qutils

    for name in qutils.SUPPORTED_OPTIMIZERS:
        assert qutils.get_optimizer(name, max_iter=3) is not None, name
    with pytest.raises(ValueError, match="Unsupported optimizer 'lbfgs'"):
        qutils.get_optimizer("lbfgs")


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"encoding": "zz"}, "encoding must be one of"),
        ({"entanglement": "chain"}, "entanglement must be one of"),
        ({"reps": 0}, "positive integer"),
        ({"primitive": "sampler"}, "primitive must be 'estimator'"),
    ],
)
def test_compute_pqk_validates_the_feature_map_before_doing_any_work(tmp_path, kwargs, expected):
    compute_pqk_mod = _compute_pqk

    rng = np.random.default_rng(0)
    args = {"backend": "simulator", "pqk_projection_dir": str(tmp_path / "proj")}
    with pytest.raises(ValueError, match=expected):
        compute_pqk_mod.compute_pqk(
            rng.random((6, 3)), rng.random((4, 3)),
            np.array([0, 1, 0, 1, 0, 1]), np.array([0, 1, 0, 1]),
            args, **kwargs,
        )
    # Nothing was created: validation runs before makedirs.
    assert not (tmp_path / "proj").exists()


def test_compute_pqk_requires_a_backend_in_args(tmp_path):
    compute_pqk_mod = _compute_pqk

    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="missing the required 'backend' key"):
        compute_pqk_mod.compute_pqk(
            rng.random((6, 3)), rng.random((4, 3)),
            np.array([0, 1, 0, 1, 0, 1]), np.array([0, 1, 0, 1]),
            {"pqk_projection_dir": str(tmp_path)},
        )


def test_model_run_names_the_available_models():
    """An unknown name used to escape as a bare KeyError from a joblib worker."""
    from qbiocode.evaluation.model_run import model_run

    rng = np.random.default_rng(0)
    X, X_test = rng.random((8, 3)), rng.random((4, 3))
    y, y_test = np.array([0, 1] * 4), np.array([0, 1] * 2)

    with pytest.raises(ValueError, match="Unknown model.*'svm'"):
        model_run(X, X_test, y, y_test, "k", {"model": ["svm"], "n_jobs": 1})
    with pytest.raises(ValueError, match="is empty"):
        model_run(X, X_test, y, y_test, "k", {"model": [], "n_jobs": 1})


# ---------------------------------------------------------------------------
# 2d. Boundary validation: the QProfiler config
# ---------------------------------------------------------------------------
@pytest.fixture
def profiler_config():
    return {
        "folder_path": "tutorial_test_data", "file_dataset": "ALL",
        "embeddings": ["pca", "none"], "n_components": 3, "model": ["svc"],
        "seed": 42, "q_seed": 42, "test_size": 0.3, "iter": 2,
        "scaling": ["True"], "backend": "simulator", "n_jobs": 4,
    }


@pytest.mark.parametrize(
    "patch, expected",
    [
        ({"iter": 0}, "must be a positive integer"),
        ({"test_size": 1.0}, "strictly between 0 and 1"),
        ({"test_size": 0.0}, "strictly between 0 and 1"),
        ({"n_components": 0}, "must be a positive integer"),
        ({"n_jobs": 0}, "non-zero integer"),
        ({"embeddings": []}, "embeddings is empty"),
        ({"embeddings": ["pca", "umapp"]}, "Unknown embedding 'umapp'"),
        ({"model": []}, "model is empty"),
        ({"scaling": "sometimes"}, "Unrecognized scaling"),
    ],
)
def test_profiler_config_is_validated_before_the_run(profiler_config, patch, expected):
    import logging

    qp = _qprofiler

    with pytest.raises(ValueError, match=expected):
        qp._validate_config({**profiler_config, **patch}, logging.getLogger("test"))


def test_profiler_config_reports_every_missing_key_at_once(profiler_config):
    import logging

    qp = _qprofiler

    del profiler_config["seed"]
    del profiler_config["iter"]
    with pytest.raises(ValueError) as excinfo:
        qp._validate_config(profiler_config, logging.getLogger("test"))
    message = str(excinfo.value)
    assert "'seed'" in message and "'iter'" in message


@pytest.mark.parametrize(
    "value, expected",
    [
        (["True"], "MinMaxScaler"),     # what the shipped config writes
        (["False"], "None"),
        (True, "MinMaxScaler"),         # the natural YAML, which used to TypeError
        (False, "None"),
        (["true"], "MinMaxScaler"),     # used to be silently ignored
        ("StandardScaler", "StandardScaler"),
        ("None", "None"),
    ],
)
def test_scaling_flag_spellings_all_resolve(value, expected):
    qp = _qprofiler

    assert qp._resolve_scaling(value) == expected


class TestFolderPathResolution:
    """``folder_path`` must resolve from a checkout that is not named ``QBioCode``.

    ``dir_home = re.sub('QBioCode.*', 'QBioCode', os.getcwd())`` only lands when the
    current directory really sits under one literally called ``QBioCode``. It does
    not for the GitHub source zip, which unpacks to ``QBioCode-main``, nor for a
    lowercase clone -- and every shipped config writes ``folder_path`` relative to
    the checkout root (``tutorial/QProfiler/data/ld_data``). So running the QProfiler
    tutorial from a source download failed with a path that had ``tutorial/QProfiler``
    in it twice, and the notebook could not be executed at all.
    """

    @staticmethod
    def _checkout(root, name):
        """A minimal checkout: <root>/<name>/tutorial/QProfiler/data/ld_data."""
        target = root / name / "tutorial" / "QProfiler" / "data" / "ld_data"
        target.mkdir(parents=True)
        return target

    def test_it_resolves_from_a_checkout_named_anything(self, tmp_path, monkeypatch):
        target = self._checkout(tmp_path, "QBioCode-main")
        monkeypatch.chdir(target.parents[2])  # .../QBioCode-main/tutorial/QProfiler
        resolved = _qprofiler._resolve_input_folder("tutorial/QProfiler/data/ld_data")
        assert resolved is not None and pathlib.Path(resolved).resolve() == target.resolve()

    def test_an_absolute_path_is_used_as_given(self, tmp_path):
        target = self._checkout(tmp_path, "anything")
        assert pathlib.Path(
            _qprofiler._resolve_input_folder(str(target))
        ).resolve() == target.resolve()

    def test_a_path_relative_to_the_current_directory_wins_first(self, tmp_path, monkeypatch):
        target = self._checkout(tmp_path, "QBioCode-main")
        monkeypatch.chdir(target.parent)  # .../data
        assert pathlib.Path(
            _qprofiler._resolve_input_folder("ld_data")
        ).resolve() == target.resolve()

    def test_a_folder_that_exists_nowhere_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert _qprofiler._resolve_input_folder("no/such/folder") is None

    def test_the_error_names_every_place_it_looked(self, tmp_path, monkeypatch, profiler_config):
        """A bare "not a directory" left users with nothing to act on."""
        monkeypatch.chdir(tmp_path)
        config = dict(profiler_config, folder_path="no/such/folder")
        with pytest.raises(ValueError) as failure:
            _qprofiler.main.__wrapped__(config)
        message = str(failure.value)
        assert "no/such/folder" in message
        assert str(tmp_path) in message or "current directory" in message
        assert "absolute path" in message


# ---------------------------------------------------------------------------
# 3. Degenerate graphs still produce correctly-shaped features
# ---------------------------------------------------------------------------
def test_nodelist_must_be_a_permutation_of_the_graph():
    gat = _gat

    G = nx.path_graph(4)
    assert gat.get_nodelist(G) == list(G.nodes())
    with pytest.raises(ValueError, match="not present in the graph"):
        gat.get_nodelist(G, [0, 1, 99])
    with pytest.raises(ValueError, match="duplicate node ids"):
        gat.get_nodelist(G, [0, 1, 1])


@pytest.mark.parametrize(
    "graph",
    [
        nx.MultiGraph([(0, 1), (0, 1), (1, 2)]),
        nx.DiGraph([(0, 1), (1, 2)]),
        nx.Graph([(0, 0), (0, 1)]),
    ],
    ids=["multigraph", "digraph", "self_loop"],
)
def test_feature_matrices_keep_their_width_on_degenerate_graphs(graph):
    """Metrics networkx cannot compute fall back to a sentinel column, not a crash."""
    gat = _gat

    features = gat.build_structural_features(graph)
    assert features.shape[0] == graph.number_of_nodes()
    assert features.shape[1] > 0
    assert np.isfinite(features).all(), "sentinel columns must be finite, not nan"
