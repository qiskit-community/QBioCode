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

"""QuVINE as a first-class embedding: ``get_embeddings("quvine_rwr", ...)``
must behave exactly like ``get_embeddings("pca", ...)``.

Three defects motivate most of what is asserted here, and each one shipped in
the internal tree:

1. **Optional dependencies resolved at import time.** ``walks/ctqw.py`` bound
   ``hiperwalk`` at module scope; ``walks/base.py`` imports it eagerly; so an
   RWR-only run -- which never touches a quantum walk -- died reporting CTQW's
   missing dependency. ``baselines/node2vec.py`` had the same shape, and because
   ``baselines/__init__.py`` swallows ``ImportError``, it left ``run_node2vec``
   unbound and took *every* registry method down with a node2vec error.
   :func:`test_no_optional_dependency_is_resolved_at_import_time` is the
   structural guard against the whole class.
2. **A subpackage that was never committed.** ``qbiocode/apps/quvine/data/``
   was matched by an unanchored ``data/`` rule in the internal ``.gitignore``,
   so ``embedding/quantum_filters.py`` imported a module that did not exist and
   the entire 69-method registry was unreachable. The directory was recovered
   from the pre-commit working tree it was written in and is tracked here.
3. **Library code writing to stdout.** The registry printed ``✓ netmf: 0.00
   minutes`` per call. QProfiler calls this once per method per iteration.
"""

import ast
import contextlib
import io
import pathlib
import warnings

import numpy as np
import pytest

import qbiocode
from qbiocode.embeddings import embed as embed_mod

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
QUVINE_ROOT = REPO_ROOT / "qbiocode" / "apps" / "quvine"


@pytest.fixture
def train_test():
    """A train/test pair large enough for a kNN graph to be connected."""
    rng = np.random.default_rng(0)
    return rng.random((30, 8)), rng.random((10, 8))


# --------------------------------------------------------------------------
# Structural guards
# --------------------------------------------------------------------------


def test_no_optional_dependency_is_resolved_at_import_time():
    """No module under ``apps/quvine`` may call ``require_module`` at module scope.

    Resolving an optional dependency while the module body executes attributes
    the failure to whichever module happened to be imported first rather than to
    the feature the user actually asked for. Every such call must sit inside a
    function so it fires at the point of use.
    """
    def module_scope_nodes(node):
        """Yield nodes that execute when the module body runs.

        A class body executes at import time, so it is included; a function body
        does not, so the walk stops there -- that is exactly the distinction this
        test exists to enforce.
        """
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            yield child
            yield from module_scope_nodes(child)

    offenders = []
    for path in sorted(QUVINE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for sub in module_scope_nodes(tree):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id == "require_module"
            ):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{sub.lineno}")
    assert not offenders, (
        "require_module must be called at use time, not import time: "
        + ", ".join(offenders)
    )


def test_the_quvine_data_subpackage_is_present_and_importable():
    """The registry is unreachable without it -- see defect 2 in the module docstring."""
    from qbiocode.apps.quvine.data import expand_neighborhood

    assert (QUVINE_ROOT / "data" / "__init__.py").is_file()
    assert callable(expand_neighborhood)


def test_the_method_registry_is_reachable():
    """Importing the adapters builds the registry; a missing ``data`` module broke it."""
    from qbiocode.apps.quvine.baselines import adapters

    assert hasattr(adapters, "run_node2vec_adapter")


# --------------------------------------------------------------------------
# expand_neighborhood
# --------------------------------------------------------------------------


def test_expand_neighborhood_is_radius_bounded_and_includes_its_roots():
    import networkx as nx

    from qbiocode.apps.quvine.data import expand_neighborhood

    path = nx.path_graph(5)  # 0-1-2-3-4
    assert expand_neighborhood(path, {2}, radius=0) == {2}
    assert expand_neighborhood(path, {2}, radius=1) == {1, 2, 3}
    assert expand_neighborhood(path, {0, 4}, radius=1) == {0, 1, 3, 4}
    # Saturating the graph must not loop forever or drop nodes.
    assert expand_neighborhood(path, {0}, radius=99) == set(path.nodes())


def test_expand_neighborhood_filters_absent_roots_at_every_radius():
    """Callers sample roots then prune the graph, so an absent root is not an error.

    It used to be filtered only for ``radius >= 1``: ``radius=0`` returned the
    roots verbatim, so it could hand back a node the graph does not contain.
    """
    import networkx as nx

    from qbiocode.apps.quvine.data import expand_neighborhood

    G = nx.path_graph(3)
    for radius in (0, 1, 2):
        assert expand_neighborhood(G, {"not-a-node"}, radius=radius) == set(), radius
    assert expand_neighborhood(G, {0, "not-a-node"}, radius=1) == {0, 1}


def test_the_recovered_data_modules_all_import_and_work():
    """The four modules the unanchored gitignore rule dropped.

    Not a smoke test for its own sake: ``pipeline.py`` imports two of them at
    module scope and ``reproducibility.graph_generator`` the other two, so their
    absence is what made ``Pipeline`` unimportable.
    """
    from qbiocode.apps.quvine.data import (
        PrepareGraphConfig,
        generate_erdos_renyi,
        keep_largest_connected_component,
        load_graph,
        load_gwas_data,
        prepare_graph,
    )

    assert all(callable(f) for f in (load_graph, load_gwas_data, prepare_graph))
    assert callable(keep_largest_connected_component)

    G = generate_erdos_renyi(n=40, p=0.12, seed=0)
    assert G.number_of_nodes() == 40
    # Seeded generation must be reproducible -- the reproducibility package
    # exists to guarantee every method sees the same graph instance.
    assert sorted(generate_erdos_renyi(n=40, p=0.12, seed=0).edges()) == sorted(G.edges())

    largest = keep_largest_connected_component(G)
    assert largest.number_of_nodes() <= G.number_of_nodes()
    assert PrepareGraphConfig is not None


def test_the_pipeline_is_importable():
    """It imports ``data.data_loader`` and ``data.prepare`` at module scope."""
    from qbiocode.apps.quvine import Pipeline

    assert Pipeline.__name__ == "Pipeline"


# --------------------------------------------------------------------------
# Public surface
# --------------------------------------------------------------------------


def test_the_method_name_constants_are_exported_and_consistent():
    assert qbiocode.SKLEARN_METHODS == ("none", "pca", "nmf", "lle", "isomap", "spectral", "umap")
    # The headline names exist to make QuVINE as discoverable as pca/nmf/umap.
    for name in qbiocode.QUVINE_HEADLINE_METHODS:
        assert name in qbiocode.QUVINE_METHODS, name
    assert not set(qbiocode.SKLEARN_METHODS) & set(qbiocode.QUVINE_METHODS)


def test_is_transductive_separates_the_two_protocols():
    for name in ("none", "pca", "nmf", "lle", "isomap", "umap"):
        assert qbiocode.is_transductive(name) is False, name
    # spectral has no out-of-sample transform, so it is fitted on the stacked
    # matrix and sliced -- test features participate, test labels never do.
    assert qbiocode.is_transductive("spectral") is True
    for name in qbiocode.QUVINE_HEADLINE_METHODS:
        assert qbiocode.is_transductive(name) is True, name


# --------------------------------------------------------------------------
# get_embeddings behaviour
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["pca", "nmf", "spectral"])
def test_a_classical_method_returns_the_requested_width(method, train_test):
    X_train, X_test = train_test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Z_train, Z_test = qbiocode.get_embeddings(method, X_train, X_test, n_components=4)
    assert Z_train.shape == (30, 4)
    assert Z_test.shape == (10, 4)


def test_none_is_an_identity_passthrough(train_test):
    X_train, X_test = train_test
    Z_train, Z_test = qbiocode.get_embeddings("none", X_train, X_test)
    assert np.array_equal(Z_train, X_train)
    assert np.array_equal(Z_test, X_test)


def test_n_components_defaults_to_the_feature_count(train_test):
    X_train, X_test = train_test
    Z_train, _ = qbiocode.get_embeddings("pca", X_train, X_test)
    assert Z_train.shape[1] == X_train.shape[1]


def test_a_transductive_method_warns_once_per_call(train_test):
    X_train, X_test = train_test
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        qbiocode.get_embeddings("spectral", X_train, X_test, n_components=3)
    transductive = [w for w in caught if "transductive" in str(w.message)]
    assert len(transductive) == 1
    # The warning has to say what does and does not leak, or it is just noise.
    message = str(transductive[0].message)
    assert "labels" in message


@pytest.mark.parametrize("method", ["none", "pca"])
def test_an_inductive_method_does_not_warn(method, train_test):
    X_train, X_test = train_test
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        qbiocode.get_embeddings(method, X_train, X_test, n_components=3)
    assert not [w for w in caught if "transductive" in str(w.message)]


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        ({"embedding": "definitely-not-a-method"}, "definitely-not-a-method"),
        ({"embedding": "pcaa"}, "pca"),          # close-name suggestion
        ({"embedding": 42}, "42"),               # non-str
        ({"n_components": 0}, "0"),
        ({"n_components": -3}, "-3"),
        ({"n_components": 2.5}, "2.5"),
        ({"n_components": True}, "True"),        # bool is an int subclass
        ({"n_components": 99}, "99"),            # wider than the input
    ],
)
def test_bad_arguments_raise_value_error_naming_the_value(kwargs, fragment, train_test):
    """Boundary validation, not a downstream sklearn assertion."""
    X_train, X_test = train_test
    call = {"embedding": "pca", "X_train": X_train, "X_test": X_test, **kwargs}
    with pytest.raises(ValueError) as excinfo:
        qbiocode.get_embeddings(**call)
    assert fragment in str(excinfo.value)


# --------------------------------------------------------------------------
# QuVINE routing
# --------------------------------------------------------------------------


def test_a_registry_method_needing_no_extra_returns_the_requested_width(train_test):
    """netmf is pure numpy/networkx, so it must work on a bare install."""
    X_train, X_test = train_test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Z_train, Z_test = qbiocode.get_embeddings("netmf", X_train, X_test, n_components=4)
    assert Z_train.shape == (30, 4)
    assert Z_test.shape == (10, 4)
    assert np.isfinite(Z_train).all() and np.isfinite(Z_test).all()


def test_get_embeddings_writes_nothing_to_stdout(train_test):
    X_train, X_test = train_test
    stdout = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(stdout):
        warnings.simplefilter("ignore")
        qbiocode.get_embeddings("netmf", X_train, X_test, n_components=4)
    assert stdout.getvalue() == ""


def test_a_missing_dependency_names_the_method_that_needs_it(train_test):
    """The regression test for the misattribution bug (defect 1).

    Skipped once the extra is installed -- there is nothing to misattribute then.
    """
    from qbiocode.apps.quvine._deps import missing_dependencies

    missing = missing_dependencies()
    cases = {"node2vec": "node2vec", "quvine_rwr": "gensim", "quvine_ctqw": "hiperwalk"}
    tested = 0
    X_train, X_test = train_test
    for method, module in cases.items():
        if module not in missing:
            continue
        tested += 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(Exception) as excinfo:
                qbiocode.get_embeddings(method, X_train, X_test, n_components=4)
        message = str(excinfo.value)
        assert module in message, f"{method} should blame {module}, said: {message}"
        assert 'pip install "qbiocode[quvine]"' in message
    if tested == 0:
        pytest.skip("the [quvine] extra is fully installed; nothing to misattribute")


def test_a_quvine_failure_keeps_its_cause_chained(train_test):
    """``raise ... from exc`` -- the traceback must survive, not end up in a log line."""
    from qbiocode.apps.quvine._deps import missing_dependencies

    if "node2vec" not in missing_dependencies():
        pytest.skip("node2vec is installed")
    X_train, X_test = train_test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(Exception) as excinfo:
            qbiocode.get_embeddings("node2vec", X_train, X_test, n_components=4)
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_quvine_method_detection_survives_the_extra_being_absent():
    """``_is_quvine_method`` returning False on ImportError is what keeps the
    classical modes working when QuVINE cannot be imported at all."""
    assert embed_mod._is_quvine_method("quvine_rwr") is True
    assert embed_mod._is_quvine_method("pca") is False
    assert embed_mod._is_quvine_method("definitely-not-a-method") is False


def test_every_synthetic_graph_family_generates():
    """The 15 reproducibility families were all dead while ``data/`` was missing.

    ``SyntheticGraphGenerator`` imports ``data.random_graphs`` and
    ``random_graphs_extended`` inside ``__init__``, so constructing it raised.
    """
    import tempfile
    from pathlib import Path

    from qbiocode.apps.quvine.reproducibility import DatasetRegistry, SeedManager
    from qbiocode.apps.quvine.reproducibility.graph_generator import SyntheticGraphGenerator

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        generator = SyntheticGraphGenerator(
            output_dir=tmp / "graphs",
            seed_manager=SeedManager(base_seed=0),
            registry=DatasetRegistry(tmp / "registry"),
        )
        for family in SyntheticGraphGenerator.SYNTHETIC_FAMILIES:
            G, path = generator.generate_single(family, n_nodes=40, repetition_id=0)
            assert G.number_of_nodes() > 0, family
            assert G.number_of_edges() > 0, family
            assert path.exists(), family


def test_ego_net_helpers_that_depend_on_subgraph_work():
    """Both callers of ``expand_neighborhood``, exercised through their own API."""
    import networkx as nx

    from qbiocode.apps.quvine.embedding.quantum_filters import get_ego_net_nodes_quvine

    G = nx.karate_club_graph()
    nodes = get_ego_net_nodes_quvine(G, center=0, k=1)
    assert 0 in nodes
    assert set(nodes) <= set(G.nodes())
    # The truncation branch keeps the center and fills with the highest degrees.
    truncated = get_ego_net_nodes_quvine(G, center=0, k=2, max_nodes=5)
    assert len(truncated) == 5
    assert truncated[0] == 0


def test_build_walk_targets_produces_calibration_targets():
    """``api.targets`` imports ``data.subgraph`` inside the function body.

    RWR is the walk kind that needs no optional dependency, so this runs on a
    bare install; ctqw/dtqw would need hiperwalk.
    """
    import networkx as nx

    from qbiocode.apps.quvine.api.targets import build_walk_targets

    G = nx.karate_club_graph()
    targets = build_walk_targets(G, seeds=[0, 33], walk_type="rwr", num_subgraphs=2)
    assert targets, "expected at least one target"
    for target in targets:
        # The contract gat.calibrate_heat_kernel validates.
        assert {"nodes", "center", "pQ"} <= set(target)
        assert target["center"] in target["nodes"]
        assert len(target["pQ"]) == len(target["nodes"])
