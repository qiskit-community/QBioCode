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

"""Locate the fixture files the tutorial notebooks read.

The notebooks live in one tree, ``tutorial/``, but their fixtures are spread
across three directories inside it, because several are shared between notebooks
and are deliberately committed only once:

============================= ===================================================
``tutorial/QuVINE/datasets/`` the graph benchmarks (``pbmc5k_graph_*``) and the
                              balanced per-task subsets
``tutorial/QProfiler/data/``  ``pbmc5k_small_cd4_vs_cd8.h5ad`` and the
                              ``sc_binary/*.csv`` exports
``tutorial/QSage/data/``      ``qprofiler_benchmarks.csv``, the QProfiler results
                              table QSage trains on
============================= ===================================================

A notebook cannot simply read ``./data``, then: the file it needs may belong to a
sibling notebook's directory. It also cannot read the copy Sphinx renders --
``docs/source/tutorials/`` is generated from this tree at build time *without* the
fixtures, since ``nbsphinx_execute = 'never'`` means nothing runs there.

Every notebook previously derived its own path with

.. code-block:: python

    _REPO = os.path.dirname(os.path.dirname(os.path.abspath(qbiocode.__file__)))
    H5AD_DIR = os.environ.get("QBC_DATA", os.path.join(_REPO, "tutorial", "QuVINE", "datasets"))

which has three faults this module fixes. It assumes an **editable** install --
for a normal ``pip install qbiocode`` the derived root is ``site-packages`` and
the path simply does not exist. It knows about exactly one of the three fixture
directories, so a file that is committed once and shared is unreachable from the
notebook that does not own it. And when the path is wrong the failure surfaces as
whatever ``anndata.read_h5ad`` says about a missing file, naming neither the
override that would fix it nor the directories that were tried.

:func:`tutorial_data_path` is the single resolution path. In order:

1. every directory in the ``QBC_DATA`` environment variable
   (``os.pathsep``-separated, so several may be listed);
2. the three directories above, relative to a source checkout -- located from the
   installed package *and* by walking up from the current working directory, so
   it resolves whether the notebook runs from a clone with the package installed
   editable, or from a clone with the package installed normally;
3. the notebook-local conventions ``./data``, ``./datasets`` and ``.``.

If nothing matches, the raised :class:`FileNotFoundError` lists every directory
searched and how to point at the data explicitly.
"""

import logging
import os

logger = logging.getLogger(__name__)

#: Fixture directories, relative to the root of a source checkout, in the order
#: they are searched. Paths are stored as tuples of components so they build
#: correctly on Windows.
REPO_DATA_DIRS = (
    ("tutorial", "QuVINE", "datasets"),
    ("tutorial", "QProfiler", "data"),
    ("tutorial", "QSage", "data"),
)

#: Directories relative to the *current working directory*, which for a notebook
#: is the directory the notebook itself lives in.
LOCAL_DATA_DIRS = (("data",), ("datasets",), ())

#: How far up from the current working directory to look for a checkout root.
_MAX_PARENTS = 6

#: Environment variable holding one or more override directories.
DATA_ENV_VAR = "QBC_DATA"


def _package_root():
    """The directory containing the ``qbiocode`` package directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _looks_like_checkout(path):
    """Whether ``path`` is the root of a QBioCode source checkout.

    Checked structurally rather than by name so a renamed clone still resolves.
    """
    return os.path.isdir(os.path.join(path, "qbiocode")) and (
        os.path.isdir(os.path.join(path, "tutorial"))
        or os.path.isdir(os.path.join(path, "docs"))
    )


def _candidate_roots():
    """Plausible checkout roots, nearest first, without duplicates."""
    roots = []
    seen = set()

    def add(path):
        real = os.path.realpath(path)
        if real not in seen:
            seen.add(real)
            roots.append(real)

    # The package's own parent: correct for an editable install, harmless
    # (it is site-packages, and _looks_like_checkout rejects it) otherwise.
    add(_package_root())

    # Walking up from the cwd covers a normal install used from inside a clone,
    # which is the case the old notebook path could not handle at all.
    here = os.path.realpath(os.getcwd())
    for _ in range(_MAX_PARENTS + 1):
        add(here)
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent

    return [root for root in roots if _looks_like_checkout(root)]


def _env_dirs():
    """Directories listed in ``QBC_DATA``, in order, ignoring empty entries."""
    raw = os.environ.get(DATA_ENV_VAR, "")
    return [part for part in raw.split(os.pathsep) if part]


def tutorial_data_dirs():
    """Return the directories :func:`tutorial_data_path` searches, in order.

    Returns
    -------
    list of str
        Absolute paths. Directories that do not exist are kept, so callers and
        error messages can report exactly where the search looked.

    Examples
    --------
    >>> from qbiocode.utils import tutorial_data_dirs
    >>> any("QuVINE" in d for d in tutorial_data_dirs())      # doctest: +SKIP
    True
    """
    dirs = []
    seen = set()

    def add(path):
        absolute = os.path.abspath(path)
        if absolute not in seen:
            seen.add(absolute)
            dirs.append(absolute)

    for directory in _env_dirs():
        add(directory)
    for root in _candidate_roots():
        for parts in REPO_DATA_DIRS:
            add(os.path.join(root, *parts))
    for parts in LOCAL_DATA_DIRS:
        add(os.path.join(os.getcwd(), *parts))

    return dirs


def tutorial_data_path(filename, search_dirs=None):
    """Return the absolute path of a tutorial fixture.

    Parameters
    ----------
    filename : str
        Fixture file name, e.g. ``"pbmc5k_small_cd4_vs_cd8.h5ad"``. An absolute
        path, or a relative path that already resolves from the current working
        directory, is returned unchanged -- so a notebook may hard-code a path
        when it has one and still call this function.
    search_dirs : sequence of str, optional
        Directories to search instead of :func:`tutorial_data_dirs`.

    Returns
    -------
    str
        Absolute path to an existing file.

    Raises
    ------
    FileNotFoundError
        If no directory holds ``filename``. The message lists every directory
        searched and how to override the location.

    Notes
    -----
    When ``QBC_DATA`` is set but does not contain ``filename`` and a repository
    directory does, the fixture is still returned -- and the substitution is
    logged at ``WARNING``. Resolving silently against a directory the caller did
    not ask for is how a stale fixture gets read for an entire session without
    anyone noticing.

    Examples
    --------
    >>> import anndata as ad                                  # doctest: +SKIP
    >>> from qbiocode.utils import tutorial_data_path         # doctest: +SKIP
    >>> adata = ad.read_h5ad(tutorial_data_path("pbmc5k_small_cd4_vs_cd8.h5ad"))
    """
    if os.path.isfile(filename):
        return os.path.abspath(filename)

    dirs = list(tutorial_data_dirs()) if search_dirs is None else [
        os.path.abspath(d) for d in search_dirs
    ]
    env_dirs = {os.path.abspath(d) for d in _env_dirs()} if search_dirs is None else set()

    for directory in dirs:
        candidate = os.path.join(directory, filename)
        if os.path.isfile(candidate):
            if env_dirs and directory not in env_dirs:
                logger.warning(
                    "%s is set (%s) but does not contain %r; using %s instead. "
                    "Set %s to the directory that holds the fixture if this is "
                    "not the file you meant to read.",
                    DATA_ENV_VAR,
                    os.environ.get(DATA_ENV_VAR, ""),
                    filename,
                    candidate,
                    DATA_ENV_VAR,
                )
            return candidate

    searched = "\n".join(f"  - {d}" for d in dirs) or "  (no candidate directories)"
    raise FileNotFoundError(
        f"Tutorial fixture {filename!r} was not found. Searched:\n{searched}\n"
        f"Point {DATA_ENV_VAR} at the directory holding it "
        f"(export {DATA_ENV_VAR}=/path/to/datasets), run the notebook from a "
        f"QBioCode checkout, or regenerate the fixtures with the preprocessing "
        f"tutorial (tutorial/Preprocessing/sc-qc.ipynb), which "
        f"writes every pbmc5k_* file from the raw 10x matrix."
    )
