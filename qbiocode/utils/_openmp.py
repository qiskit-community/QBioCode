"""Load the OpenMP-bearing native libraries in an order that does not crash.

The problem is macOS-specific but the guard is harmless everywhere.

``xgboost`` and ``torch`` each vendor their own copy of ``libomp.dylib`` and each
links it under the same install name (``@rpath/libomp.dylib``), so a process that
imports both ends up with two independent LLVM OpenMP runtimes mapped in.  The
first one to initialise claims the process-wide runtime state; when the second
later opens a parallel region it finds bookkeeping it did not create and dies
with no Python traceback at all::

    Segmentation fault: 11
    EXC_BAD_ACCESS (KERN_INVALID_ADDRESS)
    libomp.dylib  __kmp_suspend_initialize_thread
    libomp.dylib  __kmp_launch_worker

The *order* is what decides it, and only the order.  Measured on this venv
(torch 2.x, xgboost 3.4.1, macOS 15 / arm64), fitting an ``XGBClassifier``:

===============================  ======
import sequence                  result
===============================  ======
``xgboost`` only                 ok
``xgboost`` then ``torch``       ok
``torch`` then ``xgboost``       SIGSEGV
===============================  ======

Neither ``n_jobs=1`` on the estimator nor ``n_jobs=1`` on the surrounding
``RandomizedSearchCV`` avoids it -- the fault is inside the OpenMP runtime, below
joblib -- so this is not a thread-oversubscription problem and cannot be fixed by
turning parallelism down at the sklearn level.  Only ``OMP_NUM_THREADS=1``
(disabling OpenMP outright) or getting the import order right avoids it.

``qbiocode`` used to get the order exactly wrong: ``qbiocode/__init__.py`` imports
``.embeddings`` before ``.learning``, and ``qbiocode.embeddings`` eagerly imported
``ConvAutoencoder``, whose first line is ``import torch``.  So torch's runtime was
always installed first, and every pipeline that reached an XGBoost fit -- QPL's
``qpl_xgb`` arm, ``compute_xgb`` -- died with a bare SIGSEGV, taking any notebook
kernel with it and reporting only ``DeadKernelError``.  ``ConvAutoencoder`` is
imported lazily now, so most processes never map torch at all; this preload covers
the ones that do.
"""

import importlib.util
import sys
import warnings

_done = False


def preload_openmp_libraries() -> None:
    """Initialise xgboost's OpenMP runtime before anything can import torch.

    Idempotent and safe to call from any import path.  A missing or broken
    xgboost is not this function's problem to report -- ``compute_xgb`` and
    ``compute_qpl`` already raise actionable errors naming libomp and the exact
    reinstall command -- so failures here are swallowed rather than turned into
    an import-time error in a module that only wanted to order two libraries.
    """
    global _done
    if _done:
        return
    _done = True

    if "torch" in sys.modules and "xgboost" not in sys.modules:
        # Too late to order them: torch already owns the runtime. Say so, because
        # the alternative is an unexplained SIGSEGV at the first XGBoost fit.
        if importlib.util.find_spec("xgboost") is not None:
            warnings.warn(
                "torch was imported before qbiocode, so torch's OpenMP runtime is "
                "already active. On macOS, xgboost and torch ship separate copies of "
                "libomp and the second one to start can crash the process with "
                "SIGSEGV during model fitting. To avoid it, import qbiocode (or "
                "xgboost) before torch, or run with OMP_NUM_THREADS=1.",
                RuntimeWarning,
                stacklevel=2,
            )
        return

    try:
        import xgboost  # noqa: F401
    except Exception:
        # No xgboost, or an xgboost that cannot load its native library. Either way
        # there is no second OpenMP runtime to order against.
        pass
