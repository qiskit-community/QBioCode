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

"""Optional-dependency handling for the QuVINE embedding app.

QuVINE ships behind the ``quvine`` extra, so ``import qbiocode`` and every
classical embedding must keep working when none of these packages is installed.
This module is the one place that turns a missing package into an actionable
message: which method or feature was asked for, which distribution is missing,
and the exact command that fixes it.

Import this module freely -- it depends only on the standard library.

    from qbiocode.apps.quvine._deps import require_module
    hpw = require_module("hiperwalk", feature="continuous-time quantum walks")
"""

from __future__ import annotations

import importlib
from typing import Dict, List, Optional, Tuple

#: Name of the extra that provides every package listed below.
QUVINE_EXTRA = "quvine"

#: The exact command that installs it.
INSTALL_HINT = 'pip install "qbiocode[quvine]"'

#: Import name -> (distribution requirement, what it powers).
#:
#: The distribution name is not always the import name (``python-louvain``
#: installs a module called ``community``), so both are recorded; the error
#: message needs the distribution, the code needs the import name.
OPTIONAL_DEPENDENCIES: Dict[str, Tuple[str, str]] = {
    "gensim": ("gensim>=4.4", "SGNS embedding learning (word2vec)"),
    "hiperwalk": ("hiperwalk", "discrete- and continuous-time quantum walks"),
    "node2vec": ("node2vec", "the node2vec baseline"),
    "community": ("python-louvain", "Louvain modularity in the evaluation suite"),
    # torch is deliberately absent: it is a base dependency (qbiocode.embeddings
    # imports it eagerly), so its absence means a broken install, not a missing
    # extra, and pointing the user at [quvine] would be wrong advice.
    "torch_geometric": ("torch-geometric", "the GraphGPS baseline"),
    "omegaconf": ("omegaconf", "configuration loading"),
    "ripser": ("ripser", "persistent-homology graph metrics"),
}


class QuvineDependencyError(ImportError):
    """A QuVINE feature was requested but its optional dependency is absent.

    Subclasses :class:`ImportError` on purpose. ``qbiocode.embeddings`` probes
    for QuVINE method names inside ``except ImportError``, so that probe keeps
    returning ``False`` -- and every classical embedding keeps working -- when
    the extra is not installed.
    """


def _describe(module_name: str) -> Tuple[str, Optional[str]]:
    """Return ``(distribution, feature)`` for ``module_name``.

    A dotted name is resolved via its root package, so ``"gensim.models"`` is
    described by the ``gensim`` entry rather than falling through as unknown.
    """
    if module_name in OPTIONAL_DEPENDENCIES:
        return OPTIONAL_DEPENDENCIES[module_name]
    root = module_name.split(".", 1)[0]
    return OPTIONAL_DEPENDENCIES.get(root, (module_name, None))


def build_message(
    module_name: str,
    *,
    method: Optional[str] = None,
    feature: Optional[str] = None,
    cause: Optional[BaseException] = None,
) -> str:
    """Compose the user-facing message for a missing optional dependency."""
    distribution, known_feature = _describe(module_name)
    feature = feature or known_feature

    if method:
        subject = f"QuVINE method {method!r}"
    elif feature:
        subject = f"QuVINE {feature}"
    else:
        subject = "This QuVINE feature"

    message = (
        f"{subject} requires the optional {QUVINE_EXTRA!r} extra. "
        f"Install it with: {INSTALL_HINT} "
        f"(missing: {module_name}, provided by {distribution})"
    )
    if cause is not None and not isinstance(cause, ModuleNotFoundError):
        # An ImportError that is not "no such module" means the package IS
        # installed but failed to load -- a version conflict, a broken build.
        # Reinstalling the extra will not help, so say what actually happened.
        message = (
            f"{subject} could not load its dependency {module_name!r} "
            f"(provided by {distribution}): {cause}. "
            f"The package appears to be installed but is not usable; check for a "
            f"version conflict. To reinstall the extra: {INSTALL_HINT}"
        )
    return message


def require_module(
    module_name: str,
    *,
    method: Optional[str] = None,
    feature: Optional[str] = None,
):
    """Import and return ``module_name``, or raise :class:`QuvineDependencyError`.

    Parameters
    ----------
    module_name:
        Import name of the module, e.g. ``"hiperwalk"``.
    method:
        The QuVINE method name the caller was asked for, if any. Naming it makes
        the message far more useful than a bare traceback.
    feature:
        What the dependency powers, used when no specific method applies.
        Defaults to the description in :data:`OPTIONAL_DEPENDENCIES`.

    Raises
    ------
    QuvineDependencyError
        If the module cannot be imported. The message names the method or
        feature, the missing module, its distribution, and the install command.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise QuvineDependencyError(
            build_message(module_name, method=method, feature=feature, cause=exc)
        ) from exc


def is_available(module_name: str) -> bool:
    """Return whether ``module_name`` can be imported, without raising."""
    try:
        importlib.import_module(module_name)
    except ImportError:
        return False
    return True


def missing_dependencies() -> List[str]:
    """Return the import names of the QuVINE dependencies that are absent.

    Useful for diagnostics: ``quvine --help`` and the test suite report which
    parts of the app are usable in the current environment.
    """
    return [name for name in OPTIONAL_DEPENDENCIES if not is_available(name)]


def describe_environment() -> str:
    """Return a human-readable summary of which optional dependencies are present."""
    missing = missing_dependencies()
    if not missing:
        return "All QuVINE optional dependencies are installed."
    lines = [
        f"{len(missing)} of {len(OPTIONAL_DEPENDENCIES)} QuVINE optional "
        f"dependencies are missing; install them with: {INSTALL_HINT}",
    ]
    for name in missing:
        distribution, feature = _describe(name)
        lines.append(f"  - {name} (from {distribution}) -- needed for {feature}")
    return "\n".join(lines)
