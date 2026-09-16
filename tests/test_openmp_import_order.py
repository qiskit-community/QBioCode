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

"""Two OpenMP runtimes in one process, and the import order that decides it.

``xgboost`` and ``torch`` each vendor their own ``libomp`` and each links it under
the same install name, so importing both maps two independent LLVM OpenMP
runtimes into the process. The first to initialise claims the process-wide
runtime state, and the second then crashes when it opens a parallel region:

    Segmentation fault: 11 -- EXC_BAD_ACCESS in libomp.dylib
    __kmp_suspend_initialize_thread <- __kmp_launch_worker

There is no Python traceback and no exception to catch; the interpreter is simply
gone, which is why this surfaced only as ``DeadKernelError`` from a notebook and
as a silent exit 139 from a script.

``qbiocode`` used to force the losing order. ``qbiocode/__init__.py`` imports
``.embeddings`` before ``.learning``, and ``.embeddings`` eagerly imported
``ConvAutoencoder``, whose first line is ``import torch`` -- so torch's runtime
was installed first in *every* process that imported the package, and every
XGBoost fit that followed died. That covered QPL's ``qpl_xgb`` arm and
``compute_xgb``, i.e. any QProfiler run configured with an XGBoost model.

Notably, turning parallelism down does **not** avoid it: neither ``n_jobs=1`` on
the estimator nor ``n_jobs=1`` on the surrounding search helps, because the fault
is inside the OpenMP runtime, below joblib. So these tests guard the import
order, which is the only thing that actually fixed it.

Both libraries are base dependencies, so nothing here is guarded with
``importorskip``: a missing torch or xgboost is a broken install, and these tests
should fail rather than quietly skip if the crash is back.

The tests run in subprocesses because import order is a property of a fresh
interpreter and cannot be restored once a module is in ``sys.modules``. None of
them provokes the bad order on purpose -- a deliberate SIGSEGV would leave crash
reports behind and tell us nothing we do not already know.
"""

import subprocess
import sys
import textwrap

import pytest


def _run(code):
    """Run a snippet in a fresh interpreter and hand back the finished process."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=600,
    )


def _assert_not_crashed(proc):
    """Fail with the signal name rather than an unexplained empty stdout."""
    if proc.returncode < 0:
        pytest.fail(
            f"child was killed by signal {-proc.returncode} "
            f"(SIGSEGV is 11) -- the OpenMP import-order guard has regressed.\n"
            f"stdout: {proc.stdout!r}\nstderr: {proc.stderr[-2000:]}"
        )
    assert proc.returncode == 0, f"exit {proc.returncode}\nstderr: {proc.stderr[-2000:]}"


def test_importing_qbiocode_does_not_import_torch():
    """torch is the heavier of the two runtimes and almost never needed.

    Only ``ConvAutoencoder`` uses it, nothing in the package calls that class, and
    importing it eagerly is what put torch's OpenMP runtime first.
    """
    proc = _run(
        """
        import sys
        import qbiocode
        print("torch" in sys.modules)
        """
    )
    _assert_not_crashed(proc)
    assert proc.stdout.strip() == "False", (
        "import qbiocode pulled torch in. Something imported "
        "qbiocode.embeddings.compute_autoencoder (or torch directly) at module "
        "scope again; keep it behind a lazy attribute."
    )


def test_importing_qbiocode_initialises_xgboost_first():
    """The preload has to run before any submodule import, not merely eventually.

    ``qbiocode.learning`` imports xgboost anyway, so the assertion here is about
    ordering: xgboost must already be resolved by the time the package body ends,
    so that a caller importing torch afterwards is still safe.
    """
    proc = _run(
        """
        import sys
        import qbiocode
        print("xgboost" in sys.modules)
        """
    )
    _assert_not_crashed(proc)
    assert proc.stdout.strip() == "True", (
        "qbiocode finished importing without initialising xgboost's OpenMP "
        "runtime, so a later 'import torch' can win the race and crash any "
        "subsequent XGBoost fit."
    )


def test_xgboost_can_fit_after_torch_has_been_imported():
    """The case the preload exists for: both runtimes live, xgboost still fits.

    A 20-tree fit is enough -- the crash is at the first parallel region, not a
    function of problem size.
    """
    proc = _run(
        """
        import warnings
        warnings.filterwarnings("ignore")
        import numpy as np
        import qbiocode          # orders the runtimes
        import torch             # second runtime, now safe
        from xgboost import XGBClassifier

        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 12))
        y = (X[:, 0] > 0).astype(int)
        XGBClassifier(n_estimators=20, eval_metric="logloss").fit(X, y)
        print("ok")
        """
    )
    _assert_not_crashed(proc)
    assert proc.stdout.strip().endswith("ok")


def test_the_full_qpl_model_set_fits():
    """All five QPL learners in one process, in the order compute_qpl uses them.

    ``qpl_xgb`` is fitted last, which is why the crash always appeared after the
    other four had already printed and looked like a failure in the quantum step.
    """
    proc = _run(
        """
        import sys, warnings
        warnings.filterwarnings("ignore")
        import numpy as np
        import qbiocode.learning.compute_qpl
        cq = sys.modules["qbiocode.learning.compute_qpl"]

        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 12))
        y = (X[:, 0] > 0).astype(int)
        for name in ("rf", "mlp", "svc", "lr", "xgb"):
            getattr(cq, f"create_{name}_model")(42).fit(X, y)
        print("ok")
        """
    )
    _assert_not_crashed(proc)
    assert proc.stdout.strip().endswith("ok")


def test_conv_autoencoder_is_still_reachable():
    """Making it lazy must not remove it from the package's public surface."""
    proc = _run(
        """
        import qbiocode.embeddings as e
        from qbiocode.embeddings import ConvAutoencoder
        assert ConvAutoencoder.__name__ == "ConvAutoencoder"
        assert "ConvAutoencoder" in dir(e)
        assert "ConvAutoencoder" in e.__all__
        print("ok")
        """
    )
    _assert_not_crashed(proc)
    assert proc.stdout.strip() == "ok"


def test_an_unknown_embeddings_attribute_still_raises_attribute_error():
    """The lazy ``__getattr__`` must not turn typos into something else.

    A module ``__getattr__`` that forgets to raise ``AttributeError`` breaks
    ``hasattr``, ``inspect``, and every ``from ... import`` diagnostic.
    """
    import qbiocode.embeddings as e

    with pytest.raises(AttributeError, match="no attribute 'ConvAutoencodr'"):
        e.ConvAutoencodr  # noqa: B018  -- deliberate typo
