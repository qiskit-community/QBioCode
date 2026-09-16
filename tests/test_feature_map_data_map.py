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

"""The ``data_map=True`` feature maps must be buildable.

Qiskit calls a ``data_map_func`` with a *symbolic* ``ParameterVector`` while it
constructs the feature-map circuit -- ``PauliFeatureMap.pauli_block`` does
``self._data_map_func(np.asarray(ParameterVector("_", length=len(pauli))))``.
The data map used by :func:`qbiocode.embeddings.embed.pqk` narrowed its result
with ``float()``, which that symbolic input cannot satisfy, so *every*
``data_map=True`` call died with

    TypeError: Parameter expression with unbound parameters {...} is not numeric

before a single circuit was executed. It went unnoticed because
:func:`~qbiocode.learning.compute_pqk.compute_pqk` carried a fixed copy of the
same function, so the QProfiler ``pqk`` model worked while calling ``pqk()``
directly -- what ``tutorial/PQK - OV.ipynb`` does -- did not.

Both paths now share :func:`qbiocode.utils.qutils.unit_coefficient_data_map`.
These tests cover the symbolic input directly and through each feature map, so a
future ``float()`` cannot come back on either path.
"""

from __future__ import annotations

import numpy as np
import pytest

from qiskit.circuit import ParameterVector
from qiskit.circuit.parameterexpression import ParameterExpression

from qbiocode.utils.qutils import get_feature_map, unit_coefficient_data_map


def test_numeric_input_maps_to_a_float():
    """The documented numeric contract: halve at every step, return a float."""
    assert unit_coefficient_data_map(np.array([1.0])) == pytest.approx(0.5)
    assert isinstance(unit_coefficient_data_map(np.array([1.0])), float)
    # (2 * 3) / 2 == 3.0
    assert unit_coefficient_data_map(np.array([2.0, 3.0])) == pytest.approx(3.0)
    # ((2 * 4) / 2 * 5) / 2 == 10.0
    assert unit_coefficient_data_map(np.array([2.0, 4.0, 5.0])) == pytest.approx(10.0)


@pytest.mark.parametrize("length", [1, 2, 3])
def test_symbolic_input_survives_unevaluated(length):
    """A ParameterVector must come back as an expression, not raise."""
    params = np.asarray(ParameterVector("_", length=length))
    mapped = unit_coefficient_data_map(params)
    assert isinstance(mapped, ParameterExpression)
    # Still symbolic: it has not been collapsed to a number.
    assert mapped.parameters


@pytest.mark.parametrize(
    ("encoding", "entanglement"),
    [("Z", "linear"), ("ZZ", "linear"), ("ZZ", "pairwise"), ("P", "full")],
)
def test_every_feature_map_builds_with_the_data_map(encoding, entanglement):
    """This is the call that used to raise, one per supported feature map."""
    feature_map, dim = get_feature_map(
        feature_map=encoding,
        feat_dimension=3,
        reps=2,
        entanglement=entanglement,
        data_map_func=unit_coefficient_data_map,
    )
    assert dim == 3
    assert feature_map.num_qubits == 3
    # Decomposing forces qiskit to build the parameterised blocks, which is
    # where the data map is consulted.
    assert feature_map.decompose().num_parameters == 3


def test_pqk_reaches_the_backend_with_data_map_enabled():
    """``pqk(..., data_map=True)`` runs end to end on the simulator.

    Guards the wiring as well as the function: ``pqk`` must pass the shared data
    map through to ``get_feature_map`` rather than define its own.
    """
    from qbiocode.embeddings.embed import pqk

    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(4, 3))
    X_test = rng.normal(size=(2, 3))

    train_proj, test_proj = pqk(
        X_train,
        X_test,
        args={"backend": "simulator", "seed": 1234},
        store=False,
        encoding="ZZ",
        data_map=True,
        entanglement="pairwise",
        reps=2,
    )

    # 3 qubits x the X/Y/Z expectation values measured per qubit.
    assert train_proj.shape == (4, 9)
    assert test_proj.shape == (2, 9)
    assert np.isfinite(train_proj).all()
    assert np.isfinite(test_proj).all()


def test_missing_seed_names_the_key():
    """``args`` without ``seed`` used to surface as a bare ``KeyError: 'seed'``."""
    from qbiocode.utils.qutils import get_backend_session

    with pytest.raises(ValueError) as excinfo:
        get_backend_session({"backend": "simulator"}, "estimator", num_qubits=2)
    message = str(excinfo.value)
    assert "seed" in message
    assert "reproducible" in message
    # The message reports what it did get, so the caller can see the typo.
    assert "['backend']" in message
