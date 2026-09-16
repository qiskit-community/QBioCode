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

import difflib
import os
import warnings
from functools import lru_cache

import numpy as np

# ====== Qiskit imports ======
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli

# ====== Embedding functions imports ======
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, SpectralEmbedding
from umap import UMAP

import qbiocode.utils.qutils as qutils


def pqk(
    X_train,
    X_test,
    args,
    store=False,
    data_key="",
    encoding="Z",
    data_map=True,
    primitive="estimator",
    entanglement="linear",
    reps=2,
):
    """
    This function generates quantum circuits, computes projections of the data onto these circuits.
    It uses a feature map to encode the data into quantum states and then measures the expectation values
    of Pauli operators to obtain the features.
    This function requires a quantum backend (simulator or real quantum hardware) for execution.
    It supports various configurations such as encoding methods, entanglement strategies, and repetitions
    of the feature map. Optionally the results are saved to files for training and test projections.

    Args:
        X_train (np.ndarray): Training data features.
        X_test (np.ndarray): Test data features.
        args (dict): Backend configuration. Requires ``backend`` (``'simulator'``,
            ``'simulator_aer'``, or an ``'ibm_*'`` device) and, for the
            simulator, ``seed`` so the statevector primitive is reproducible.
        store (bool): If true projections are stored, using data_key as indefitier
        data_key (str): Key for the dataset, default is ''.
        encoding (str): Encoding method for the quantum circuit, default is 'Z'.
        data_map (bool): If true ensures that all multiplicative factors of data features inside single qubit gates are 1.0. Not applicable for Hejsemberg feature maps
        primitive (str): Primitive type to use, default is 'estimator'.
        entanglement (str): Entanglement strategy, default is 'linear'.
        reps (int): Number of repetitions for the feature map, default is 2.

    Returns:
        modeleval (dict): A dictionary containing evaluation metrics and model parameters.
    """

    feat_dimension = X_train.shape[1]

    # Shared with compute_pqk so the two PQK paths cannot drift again: the copy
    # that used to live here coerced its result with float(), which qiskit's
    # symbolic ParameterVector cannot satisfy, so every data_map=True call raised
    # "Parameter expression with unbound parameters ... is not numeric".
    data_map_func = qutils.unit_coefficient_data_map if data_map else None

    # choose a method for mapping your features onto the circuit
    feature_map, _ = qutils.get_feature_map(
        feature_map=encoding,
        feat_dimension=X_train.shape[1],
        reps=reps,
        entanglement=entanglement,
        data_map_func=data_map_func,
    )

    # Build quantum circuit
    circuit = QuantumCircuit(feature_map.num_qubits)
    circuit.compose(feature_map, inplace=True)
    num_qubits = circuit.num_qubits

    #  Generate the backend, session and primitive
    backend, session, prim = qutils.get_backend_session(args, "estimator", num_qubits=num_qubits)
    try:

        # Transpile
        if args["backend"] != "simulator":
            circuit = qutils.transpile_circuit(
                circuit, opt_level=3, backend=backend, PT=True, initial_layout=None
            )

        for f_tr in ["train", "test"]:

            if "train" in f_tr:
                dat = X_train.copy()
            else:
                dat = X_test.copy()

            # Identity operator on all qubits
            id = "I" * feat_dimension

            # We group all commuting observables
            # These groups are the Pauli X, Y and Z operators on individual qubits
            # Apply the circuit layout to the observable if mapped to device
            if args["backend"] != "simulator":
                observables_x = []
                observables_y = []
                observables_z = []
                for i in range(feat_dimension):
                    observables_x.append(
                        Pauli(id[:i] + "X" + id[(i + 1) :]).apply_layout(
                            circuit.layout, num_qubits=backend.num_qubits
                        )
                    )
                    observables_y.append(
                        Pauli(id[:i] + "Y" + id[(i + 1) :]).apply_layout(
                            circuit.layout, num_qubits=backend.num_qubits
                        )
                    )
                    observables_z.append(
                        Pauli(id[:i] + "Z" + id[(i + 1) :]).apply_layout(
                            circuit.layout, num_qubits=backend.num_qubits
                        )
                    )
            else:
                observables_x = [Pauli(id[:i] + "X" + id[(i + 1) :]) for i in range(feat_dimension)]
                observables_y = [Pauli(id[:i] + "Y" + id[(i + 1) :]) for i in range(feat_dimension)]
                observables_z = [Pauli(id[:i] + "Z" + id[(i + 1) :]) for i in range(feat_dimension)]

            # projections[i][j][k] will be the expectation value of the j-th Pauli operator (0: X, 1: Y, 2: Z)
            # of datapoint i on qubit k
            projections = []

            for i in range(len(dat)):

                # Get training sample
                parameters = dat[i]

                # We define the primitive unified blocs (PUBs) consisting of the embedding circuit,
                # set of observables and the circuit parameters
                pub_x = (circuit, observables_x, parameters)
                pub_y = (circuit, observables_y, parameters)
                pub_z = (circuit, observables_z, parameters)

                job = prim.run([pub_x, pub_y, pub_z])
                job_result_x = job.result()[0].data.evs
                job_result_y = job.result()[1].data.evs
                job_result_z = job.result()[2].data.evs

                # Record <X>, <Y> and <Z> on all qubits for the current datapoint
                projections.append([job_result_x, job_result_y, job_result_z])

            if store:
                if not os.path.exists("pqk_projections"):
                    os.makedirs("pqk_projections")

                file_projection = os.path.join(
                    "pqk_projections", "pqk_projection_" + data_key + "_" + f_tr + ".npy"
                )

                np.save(file_projection, projections)

            if "train" in f_tr:
                X_train_prj = np.array(projections.copy()).reshape(len(projections), -1)
            else:
                X_test_prj = np.array(projections.copy()).reshape(len(projections), -1)

    finally:
        if not isinstance(session, type(None)):
            session.close()

    return X_train_prj, X_test_prj


# ---------------------------------------------------------------------------
# Method catalogue
# ---------------------------------------------------------------------------
#: Feature-reduction modes backed by scikit-learn / UMAP.
SKLEARN_METHODS = ("none", "pca", "nmf", "lle", "isomap", "spectral", "umap")

#: The headline QuVINE names, for discoverability. `QUVINE_METHODS` holds all of
#: them; these are the ones worth naming next to `pca`/`nmf`/`umap`.
QUVINE_HEADLINE_METHODS = (
    "quvine_fused",
    "quvine_rwr",
    "quvine_dtqw",
    "quvine_ctqw",
    "node2vec",
    "netmf",
    "appnp",
)

# Methods that cannot follow the fit-on-train / transform-on-test contract, because
# they have no out-of-sample `transform`: the embedding is fit on the concatenated
# train+test rows and sliced back. Test *features* therefore participate; test
# *labels* never do. See `is_transductive`.
_TRANSDUCTIVE_SKLEARN_METHODS = frozenset({"spectral"})


@lru_cache(maxsize=1)
def _quvine_method_names():
    """All QuVINE method names, or ``()`` if the QuVINE app cannot be imported.

    Name resolution is stdlib-only, so this works without the ``[quvine]`` extra —
    the names are listed even when running one would raise
    :class:`~qbiocode.apps.quvine.QuvineDependencyError`.
    """
    try:
        from qbiocode.apps.quvine import list_methods

        return tuple(list_methods())
    except ImportError:
        return ()


#: Every QuVINE method name accepted by :func:`get_embeddings`, e.g.
#: ``quvine_rwr``, ``quvine_ctqw_heat``, ``node2vec``, ``graphgps_rwr_poly``.
QUVINE_METHODS = _quvine_method_names()


def _is_quvine_method(embedding: str) -> bool:
    """Return True if ``embedding`` names a QuVINE embedding method.

    QuVINE method names (``quvine_rwr``, ``quvine_ctqw``, ``node2vec``, ``graphsage``,
    ``baseline_filter_heat``, ``filter_ctqw_heat``, ...) resolve through the QuVINE app's
    ``resolve_method`` and never collide with the sklearn embedding modes handled below.

    Returning ``False`` when the QuVINE app cannot be imported is what keeps the
    ``[quvine]`` extra non-breaking: every sklearn mode still works, and a QuVINE name
    falls through to the ``ValueError`` below.
    """
    try:
        from qbiocode.apps.quvine import resolve_method
    except ImportError:
        # QuVINE app unavailable: treat as "not a QuVINE method" so the sklearn
        # embeddings still work.
        return False

    try:
        resolve_method(embedding)
        return True
    except KeyError:
        # Unknown name -> not a QuVINE method. Any other error is a real bug and propagates.
        return False


def is_transductive(embedding: str) -> bool:
    """Return True if ``embedding`` is fit on the combined train and test rows.

    Inductive methods (``pca``, ``nmf``, ``lle``, ``isomap``, ``umap``) are fit on the
    training rows alone and applied to the test rows through ``transform``. Transductive
    methods — ``spectral`` and every QuVINE method — have no out-of-sample ``transform``,
    so the embedding is computed once over ``vstack([X_train, X_test])`` and sliced back.

    Test *features* therefore participate in the embedding; test *labels* never reach it.
    This is the standard protocol for unsupervised graph and manifold embeddings, but it
    means an embedding fit this way cannot be reused on rows unseen at fit time, and a
    reported test score is a transductive score. Callers that need the distinction — e.g.
    to skip a method in a strictly inductive benchmark — should branch on this.

    Args:
        embedding (str): An embedding name accepted by :func:`get_embeddings`.

    Returns:
        bool: True for a transductive method. Unknown names return False rather than
        raising, so this is safe to call before validation.

    Examples:
        >>> is_transductive("pca")
        False
        >>> is_transductive("spectral")
        True
    """
    name = str(embedding).lower().strip()
    if name in _TRANSDUCTIVE_SKLEARN_METHODS:
        return True
    return _is_quvine_method(name)


def _warn_transductive(embedding: str) -> None:
    """Warn once per call that test features participate in the embedding."""
    warnings.warn(
        f"Embedding {embedding!r} is transductive: it has no out-of-sample transform, so "
        f"it is fit on the concatenated train and test rows and then sliced back. Test "
        f"features participate in the embedding; test labels do not. Scores obtained this "
        f"way are transductive and are not comparable to a strictly inductive protocol. "
        f"Use qbiocode.embeddings.is_transductive() to branch on this.",
        UserWarning,
        stacklevel=3,
    )


def _nonnegative_for_nmf(X_test, embedding):
    """Clip ``X_test`` at zero so NMF can transform it, and say how much was clipped.

    NMF is defined only on non-negative input. A MinMaxScaler fit on the *training*
    split alone -- which is the leakage-free protocol, and what QProfiler now does --
    maps train into [0, 1] but leaves test free to fall outside it, so a test row
    below the training minimum arrives negative and ``NMF.transform`` raises
    ``ValueError: Negative values in data passed to X in NMF``. That is what made
    every shipped QProfiler config crash: ``embeddings: ['pca', 'nmf', 'none']`` with
    ``scaling: ['True']``.

    Clipping at zero is the honest handling: the alternative -- scaling on train+test
    together, so nothing ever falls outside [0, 1] -- is the leak this release
    removed. The clip is reported rather than silent, because a large excursion means
    the test split is genuinely outside the training range and the embedding is being
    asked to extrapolate.
    """
    X_test = np.asarray(X_test, dtype=float)
    negative = X_test < 0
    count = int(negative.sum())
    if count:
        warnings.warn(
            f"{count} of {X_test.size} test entries were negative and have been "
            f"clipped to 0 for the {embedding!r} embedding (most negative: "
            f"{float(X_test.min()):.4g}). NMF is only defined on non-negative input, "
            f"and a scaler fit on the training split alone -- the leakage-free "
            f"protocol -- does not bound the test split to the training range.",
            UserWarning,
            stacklevel=4,
        )
        X_test = np.clip(X_test, 0.0, None)
    return X_test


def _unknown_embedding_error(embedding: str) -> ValueError:
    """Build the error for an unrecognized embedding name, with close matches."""
    known = list(SKLEARN_METHODS) + list(QUVINE_METHODS)
    suggestions = difflib.get_close_matches(embedding, known, n=5)
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    quvine_note = ""
    if not QUVINE_METHODS:
        quvine_note = (
            " QuVINE method names are unavailable in this environment; install them with: "
            'pip install "qbiocode[quvine]".'
        )
    return ValueError(
        f"Unknown embedding {embedding!r}.{hint} Valid sklearn modes are "
        f"{list(SKLEARN_METHODS)}; QuVINE methods are listed by "
        f"qbiocode.embeddings.QUVINE_METHODS ({len(QUVINE_METHODS)} names)."
        f"{quvine_note}"
    )


def check_embedding_name(embedding: str) -> str:
    """Normalize and validate an embedding name, raising if it is unknown.

    :func:`get_embeddings` calls this, so a caller that is about to run many
    embeddings can validate the whole list *before* doing any work and get the
    identical message. QProfiler uses it for exactly that: a typo in the sixth
    entry of ``embeddings`` used to surface only after the first five had run.

    Args:
        embedding (str): Method name, case-insensitive, surrounding space ignored.

    Returns:
        str: The normalized (lower-cased, stripped) name.

    Raises:
        ValueError: if ``embedding`` is not a string, or names no known method.
            The message lists close matches and the valid sklearn modes.
    """
    if not isinstance(embedding, str):
        raise ValueError(
            f"embedding must be a string naming an embedding method; got "
            f"{embedding!r} ({type(embedding).__name__}). Valid options: "
            f"{list(SKLEARN_METHODS)}"
            + (f" plus {len(QUVINE_METHODS)} QuVINE methods." if QUVINE_METHODS else ".")
        )
    name = embedding.lower().strip()
    if name not in SKLEARN_METHODS and not _is_quvine_method(name):
        raise _unknown_embedding_error(name)
    return name


def _quvine_embed(embedding, X_train, X_test, n_components, n_neighbors=30, quvine_args=None):
    """Embed via QuVINE, treating the (train+test) feature matrix as one cell graph.

    QuVINE is a *transductive* graph-embedding method: it embeds every node of a graph at
    once (keyed by node identity), so it cannot follow sklearn's fit-on-train /
    transform-on-test contract. A single kNN graph is built over the concatenated
    ``[X_train, X_test]`` feature rows, all nodes are embedded once, then the rows are
    sliced back into train/test by construction order. The class *label* never enters
    ``embed`` -- only feature-derived graph structure -- so this is the standard
    transductive protocol, not label leakage. See :func:`is_transductive`.

    Args:
        embedding (str): QuVINE method name (e.g. ``"quvine_ctqw"``, ``"filter_ctqw_heat"``).
        X_train, X_test (array-like): feature matrices; rows are samples.
        n_components (int): output embedding width (drives QuVINE ``dimension``).
        n_neighbors (int): neighbors for the kNN sample graph.
        quvine_args (dict, optional): extra OmegaConf overrides merged onto the QuVINE config
            (e.g. ``{"walks": {"steps": 4}}``). ``dimension`` is always set to ``n_components``.

    Returns:
        tuple: ``(Z_train, Z_test)`` each with ``n_components`` columns.

    Raises:
        QuvineDependencyError: if the ``[quvine]`` extra is not installed. The message names
            the extra and the install command.
    """
    import networkx as nx
    from sklearn.neighbors import kneighbors_graph

    from qbiocode.apps.quvine import embed

    quvine_args = dict(quvine_args or {})

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    n_train = X_train.shape[0]
    X_all = np.vstack([X_train, X_test])
    node_ids = [f"c{i}" for i in range(X_all.shape[0])]  # QuVINE SGNS requires string ids

    # Build a symmetric kNN graph over all samples; edge weight = 1/(1+distance).
    k = int(min(n_neighbors, max(1, X_all.shape[0] - 1)))
    A = kneighbors_graph(X_all, n_neighbors=k, mode="distance", include_self=False)
    A = A.maximum(A.T)  # symmetrize
    G = nx.Graph()
    G.add_nodes_from(node_ids)
    Acoo = A.tocoo()
    for i, j, d in zip(Acoo.row, Acoo.col, Acoo.data):
        if i < j:
            G.add_edge(node_ids[i], node_ids[j], weight=float(1.0 / (1.0 + d)))

    overrides = {"dimension": int(n_components)}
    overrides.update(quvine_args)
    result = embed(G, embedding, overrides=overrides)
    Z = np.asarray(result.embedding, dtype=float)  # rows aligned to node_ids order

    # Safety net: some arms may not honor `dimension` exactly -> reduce to n_components.
    if Z.shape[1] != n_components:
        Z = PCA(n_components=int(min(n_components, *Z.shape))).fit_transform(Z)

    return Z[:n_train], Z[n_train:]


def get_embeddings(
    embedding: str,
    X_train,
    X_test,
    n_neighbors=30,
    n_components=None,
    method=None,
    quvine_args=None,
):
    """Apply an embedding to the training and test datasets.

    Inductive methods are fit on ``X_train`` and applied to ``X_test`` through
    ``transform``. Transductive methods (``spectral`` and every QuVINE method) have no
    out-of-sample ``transform``, so they are fit once on the concatenated rows and sliced
    back; a ``UserWarning`` is emitted and :func:`is_transductive` reports which is which.

    Args:
        embedding (str): The embedding to use.

            * scikit-learn / UMAP modes: ``'none'``, ``'pca'``, ``'nmf'``, ``'lle'``,
              ``'isomap'``, ``'spectral'``, ``'umap'``. Listed in
              :data:`SKLEARN_METHODS`.
            * QuVINE graph embeddings, accepted on exactly the same footing:
              ``'quvine_fused'``, ``'quvine_rwr'``, ``'quvine_dtqw'``, ``'quvine_ctqw'``,
              ``'node2vec'``, ``'netmf'``, ``'appnp'`` and 76 more. The full list is
              :data:`QUVINE_METHODS`; the headline names are
              :data:`QUVINE_HEADLINE_METHODS`. These need the optional extra:
              ``pip install "qbiocode[quvine]"``.
        X_train (array-like): The training dataset.
        X_test (array-like): The test dataset.
        n_neighbors (int, optional): Number of neighbors for the neighbor-based embeddings
            and for the QuVINE kNN sample graph. Defaults to 30.
        n_components (int, optional): Width of the embedding. If None, defaults to the
            number of features in ``X_train``.
        method (str, optional): Method for Locally Linear Embedding. Defaults to None.
        quvine_args (dict, optional): Extra config overrides forwarded to QuVINE ``embed``
            when ``embedding`` is a QuVINE method, e.g.
            ``{"walks": {"steps": 4}, "train": {"epochs": 10}}``. Ignored for the
            sklearn modes.

    Returns:
        tuple: ``(X_train_embedded, X_test_embedded)``.

    Raises:
        ValueError: if ``embedding`` is not a known name (the message lists close
            matches), if ``n_components`` is not a positive integer, if
            ``n_components`` exceeds the feature count for an sklearn mode, or if
            ``X_train``/``X_test`` are not 2-D arrays with matching widths.
        QuvineDependencyError: if a QuVINE method is requested without the ``[quvine]``
            extra installed. The message names the extra and the install command.

    Warns:
        UserWarning: once per call, when ``embedding`` is transductive.

    Examples:
        >>> Z_tr, Z_te = get_embeddings("pca", X_train, X_test, n_components=8)
        >>> Z_tr, Z_te = get_embeddings("quvine_rwr", X_train, X_test, n_components=8)
    """

    embedding = check_embedding_name(embedding)

    # Coerced and shape-checked before anything reads X_train.shape: a list of
    # lists (the natural thing to pass from a notebook) otherwise failed on the
    # `n_components is None` default below with "'list' object has no attribute
    # 'shape'", which names neither the parameter nor the requirement.
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError(
            f"X_train and X_test must be 2-D (n_samples, n_features); got "
            f"{X_train.ndim}-D and {X_test.ndim}-D. Reshape a single sample with "
            f"X.reshape(1, -1)."
        )
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"X_train and X_test must have the same number of features -- one "
            f"embedding is fitted for both -- got {X_train.shape[1]} and "
            f"{X_test.shape[1]}."
        )
    if X_train.shape[0] == 0:
        raise ValueError(
            "X_train is empty; an embedding cannot be fitted on zero samples."
        )

    # Default the embedding width to the feature count. This is the documented behavior, but
    # without it `n_components=None` (the documented default!) reaches the comparison below and
    # raises TypeError: '<=' not supported between 'NoneType' and 'int'.
    if n_components is None:
        n_components = X_train.shape[1]
    # Validate at the boundary: a bad width otherwise surfaces as an sklearn assertion or,
    # for QuVINE, as a failure deep inside the walk code.
    if isinstance(n_components, bool) or not isinstance(n_components, (int, np.integer)):
        raise ValueError(
            f"n_components must be a positive integer or None; got {n_components!r}."
        )
    n_components = int(n_components)
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1; got {n_components}.")

    # check_embedding_name above already rejected unknown names, so this only
    # decides which of the two routes the (valid) name takes.
    is_quvine = _is_quvine_method(embedding)

    if embedding != "none" and (is_quvine or embedding in _TRANSDUCTIVE_SKLEARN_METHODS):
        _warn_transductive(embedding)

    # QuVINE methods are transductive graph embeddings; route them before the sklearn
    # feature-reduction path (their output width is independent of the feature count, so the
    # n_components <= n_features check below does not apply).
    if is_quvine:
        return _quvine_embed(
            embedding,
            X_train,
            X_test,
            n_components,
            n_neighbors=n_neighbors,
            quvine_args=quvine_args,
        )

    if n_components > X_train.shape[1]:
        raise ValueError(
            f"n_components={n_components} exceeds the {X_train.shape[1]} features in "
            f"X_train; {embedding!r} cannot produce more components than input features."
        )
    if "none" == embedding:
        return X_train, X_test
    else:
        embedding_model = None
        if "pca" == embedding:
            embedding_model = PCA(n_components=n_components)
        elif "nmf" == embedding:
            embedding_model = NMF(n_components=n_components)
        elif "lle" == embedding:
            if method is None:
                embedding_model = LocallyLinearEmbedding(
                    n_neighbors=n_neighbors, n_components=n_components, method="standard"
                )
            else:
                embedding_model = LocallyLinearEmbedding(
                    n_neighbors=n_neighbors, n_components=n_components, method="modified"
                )
        elif "isomap" == embedding:
            embedding_model = Isomap(
                n_neighbors=n_neighbors,
                n_components=n_components,
            )
        elif "umap" == embedding:
            embedding_model = UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
            )

        if "spectral" == embedding:
            # SpectralEmbedding has no out-of-sample `transform`, so it must be fit
            # transductively on the combined train+test rows and then sliced back. Only
            # feature structure is used (no labels), so no label leakage is introduced --
            # the UserWarning above declares it. Same protocol as the QuVINE arms.
            n_train = X_train.shape[0]
            X_all = np.vstack([X_train, X_test])
            Z = SpectralEmbedding(n_components=n_components, eigen_solver="arpack").fit_transform(X_all)
            return Z[:n_train], Z[n_train:]

        X_train = embedding_model.fit_transform(X_train)
        if "nmf" == embedding:
            X_test = _nonnegative_for_nmf(X_test, embedding)
        X_test = embedding_model.transform(X_test)

    return X_train, X_test
