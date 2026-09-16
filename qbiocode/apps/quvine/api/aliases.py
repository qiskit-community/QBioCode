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

"""
Method-name resolution for the QuVINE embedding API.

A single string passed to :func:`quvine.embed` is mapped here to one of three
execution kinds:

- ``"sgns"``     -- the true QuVINE SGNS walk embedding (views -> walks ->
                    corpus -> word2vec) for a single walk kind (rwr/ctqw/dtqw).
- ``"registry"`` -- a single method from the comparison ``MethodRegistry``
                    (node2vec, gat_*, graphgps_*, filter_*, quvine_* GCN-MF, ...).
- ``"fused"``    -- run every walk kind and fuse the results into one embedding.

No registry key is ever renamed, so the per-method tuned-hyperparameter lookups
in ``baselines/hyperparameter_loader.py`` keep working. Friendly unified aliases
are layered *on top* of the existing keys.
"""

from __future__ import annotations

import difflib
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# SGNS walk embeddings (the true QuVINE SGNS path; no quantum targets needed)
# ---------------------------------------------------------------------------
# Maps a friendly name -> walk kind understood by the SGNS core.
SGNS_ALIASES: Dict[str, str] = {
    "rwr": "rwr",
    "ctqw": "ctqw",
    "dtqw": "dtqw",
    # explicit "this is the SGNS variant" spellings
    "quvine_rwr_sgns": "rwr",
    "quvine_ctqw_sgns": "ctqw",
    "quvine_dtqw_sgns": "dtqw",
    "sgns_rwr": "rwr",
    "sgns_ctqw": "ctqw",
    "sgns_dtqw": "dtqw",
    # bare quvine_<walk> defaults to the SGNS embedding (most intuitive meaning)
    "quvine_rwr": "rwr",
    "quvine_ctqw": "ctqw",
    "quvine_dtqw": "dtqw",
}

# ---------------------------------------------------------------------------
# Fused embeddings: run all walk kinds, then fuse.
# ---------------------------------------------------------------------------
# These names are handled directly by ``core.embed`` rather than by a walk kind
# or a registry key, but they must still resolve and be listed: the ``quvine``
# CLI validates ``--method`` against ``list_methods()`` (its own default is
# ``quvine_fused``), and ``qbiocode.embeddings`` decides whether to route a name
# to QuVINE by whether ``resolve_method`` accepts it.
FUSED_ALIASES: Tuple[str, ...] = (
    "quvine",
    "quvine_fused",
    "quvine_sgns",
    "quvine_sgns_fused",
    "fused",
)

# ---------------------------------------------------------------------------
# Canonical registry keys (the 36 registered methods). Kept in sync with
# baselines/registration.py; a unit test asserts this set matches the registry.
# ---------------------------------------------------------------------------
REGISTRY_KEYS: Tuple[str, ...] = (
    # 5 classical baselines
    "node2vec", "appnp", "graphsage", "netmf", "baseline_gcnmf",
    # 8 filter variants (2 baseline + 6 walk-calibrated)
    "baseline_filter_heat", "baseline_filter_poly",
    "filter_rwr_heat", "filter_rwr_poly",
    "filter_ctqw_heat", "filter_ctqw_poly",
    "filter_dtqw_heat", "filter_dtqw_poly",
    # 10 GAT variants: baseline + 3 classical single-axis + 6 quantum-calibrated
    "gat_baseline",
    "gat_heat", "gat_poly", "gat_rwr",
    "gat_ctqw_heat", "gat_ctqw_poly",
    "gat_dtqw_heat", "gat_dtqw_poly",
    "gat_rwr_heat", "gat_rwr_poly",
    # 10 GraphGPS variants: baseline + 3 classical single-axis + 6 quantum-calibrated
    "graphgps_baseline",
    "graphgps_heat", "graphgps_poly", "graphgps_rwr",
    "graphgps_ctqw_heat", "graphgps_ctqw_poly",
    "graphgps_dtqw_heat", "graphgps_dtqw_poly",
    "graphgps_rwr_heat", "graphgps_rwr_poly",
    # 3 GCN-MF QuVINE variants
    "quvine_ctqw", "quvine_dtqw", "quvine_rwr",
)

# ---------------------------------------------------------------------------
# Registry key -> the cfg.baselines.<group> section its config builder reads.
# For most methods this is the key itself; the filter family shares a group.
# Used by core.embed() to auto-enable an explicitly requested method.
# ---------------------------------------------------------------------------
REGISTRY_CONFIG_GROUP: Dict[str, str] = {
    "baseline_filter_heat": "baseline_filter",
    "baseline_filter_poly": "baseline_filter",
    "filter_rwr_heat": "filter_rwr",
    "filter_rwr_poly": "filter_rwr",
    "filter_ctqw_heat": "filter_ctqw",
    "filter_ctqw_poly": "filter_ctqw",
    "filter_dtqw_heat": "filter_dtqw",
    "filter_dtqw_poly": "filter_dtqw",
    # Single-axis classical GAT/GPS keys share their own per-name group; included
    # explicitly so config_group_for() returns the key itself (default behavior),
    # but also so the reader knows these are intentional independent groups.
}


def config_group_for(registry_key: str) -> str:
    """Return the cfg.baselines.<group> section name for a registry key."""
    return REGISTRY_CONFIG_GROUP.get(registry_key, registry_key)


def variant_for(registry_key: str) -> Optional[str]:
    """
    Return the GAT/GraphGPS ``variant`` string implied by a registry key.

    The GAT/GraphGPS config builders default ``variant='raw'`` and do NOT infer
    it from the method name, so e.g. ``gat_rwr_heat`` would silently run as a
    plain (non-quantum) GAT. ``embed()`` uses this to set the variant from the
    method name (``gat_rwr_heat`` -> ``heat_qcal_rwr``) unless the caller's
    config already specifies one. Returns ``None`` for non-GAT/GraphGPS keys.

    Mapping:
        gat_baseline / graphgps_baseline  -> "raw"
        gat_heat     / graphgps_heat      -> "heat_fixed"
        gat_poly     / graphgps_poly      -> "poly_fixed"
        gat_rwr      / graphgps_rwr       -> "rwr"
        gat_*_heat   / graphgps_*_heat    -> "heat_qcal_<walk>"
        gat_*_poly   / graphgps_*_poly    -> "poly_qcal_<walk>"
    """
    if registry_key in ("gat_baseline", "graphgps_baseline"):
        return "raw"
    if registry_key.startswith(("gat_", "graphgps_")):
        parts = registry_key.split("_")
        # Combined quantum-calibrated variant, e.g. gat_rwr_heat -> heat_qcal_rwr.
        if len(parts) >= 3 and parts[-2] in ("rwr", "ctqw", "dtqw") \
                and parts[-1] in ("heat", "poly"):
            return f"{parts[-1]}_qcal_{parts[-2]}"
        # Single-axis classical variants (no quantum calibration).
        if parts[-1] == "heat":
            return "heat_fixed"
        if parts[-1] == "poly":
            return "poly_fixed"
        if parts[-1] == "rwr":
            return "rwr"
    return None


# ---------------------------------------------------------------------------
# Friendly unified aliases -> existing registry key.
# ---------------------------------------------------------------------------
REGISTRY_ALIASES: Dict[str, str] = {
    # GCN-MF quantum-calibrated (registry's quvine_* keys), disambiguated from
    # the SGNS path above via the explicit "gcnmf" infix.
    "quvine_gcnmf_rwr": "quvine_rwr",
    "quvine_gcnmf_ctqw": "quvine_ctqw",
    "quvine_gcnmf_dtqw": "quvine_dtqw",

    # Back-compat: the old baseline names resolve to the canonical *_baseline keys.
    # (gat_baseline / graphgps_baseline are now the canonical names in REGISTRY_KEYS.)
    "baseline_gat": "gat_baseline",
    "baseline_graphgps": "graphgps_baseline",

    # GAT family: quvine_gat_<walk>_<filter> -> gat_<walk>_<filter>.
    # NOTE: bare gat_rwr/gat_heat/gat_poly are canonical keys (real classical
    # methods), so they are NOT aliases. gat_ctqw/gat_dtqw are intentionally not
    # exposed (dropped "direct" proxies). The quvine_ prefix denotes quantum path.
    "quvine_gat_ctqw_heat": "gat_ctqw_heat", "quvine_gat_ctqw_poly": "gat_ctqw_poly",
    "quvine_gat_dtqw_heat": "gat_dtqw_heat", "quvine_gat_dtqw_poly": "gat_dtqw_poly",
    "quvine_gat_rwr_heat": "gat_rwr_heat",   "quvine_gat_rwr_poly": "gat_rwr_poly",
    # quvine_gat_<walk> (filter-less) defaults to the heat-calibrated variant
    "quvine_gat_ctqw": "gat_ctqw_heat",
    "quvine_gat_dtqw": "gat_dtqw_heat",
    "quvine_gat_rwr": "gat_rwr_heat",
    "quvine_gat": "gat_baseline",

    # GraphGPS family
    "quvine_graphgps_ctqw_heat": "graphgps_ctqw_heat", "quvine_graphgps_ctqw_poly": "graphgps_ctqw_poly",
    "quvine_graphgps_dtqw_heat": "graphgps_dtqw_heat", "quvine_graphgps_dtqw_poly": "graphgps_dtqw_poly",
    "quvine_graphgps_rwr_heat": "graphgps_rwr_heat",   "quvine_graphgps_rwr_poly": "graphgps_rwr_poly",
    "quvine_graphgps_ctqw": "graphgps_ctqw_heat",
    "quvine_graphgps_dtqw": "graphgps_dtqw_heat",
    "quvine_graphgps_rwr": "graphgps_rwr_heat",
    "quvine_graphgps": "graphgps_baseline",

    # Filter family: quvine_<walk>_<filter> -> filter_<walk>_<filter>
    "quvine_rwr_heat": "filter_rwr_heat", "quvine_rwr_poly": "filter_rwr_poly",
    "quvine_ctqw_heat": "filter_ctqw_heat", "quvine_ctqw_poly": "filter_ctqw_poly",
    "quvine_dtqw_heat": "filter_dtqw_heat", "quvine_dtqw_poly": "filter_dtqw_poly",
    "quvine_heat": "baseline_filter_heat", "quvine_poly": "baseline_filter_poly",
}


def _normalize(name: str) -> str:
    return str(name).strip().lower()


def resolve_method(method: str) -> Tuple[str, str]:
    """
    Resolve a method name to ``(kind, key)``.

    Returns:
        ``("sgns", walk_kind)`` for the SGNS walk path,
        ``("registry", registry_key)`` for a single registry method, or
        ``("fused", "fused")`` for a fused multi-walk embedding.

    Raises:
        KeyError: if the name is unknown (message includes close matches).
    """
    norm = _normalize(method)

    if norm in FUSED_ALIASES:
        return ("fused", "fused")
    if norm in SGNS_ALIASES:
        return ("sgns", SGNS_ALIASES[norm])
    if norm in REGISTRY_ALIASES:
        return ("registry", REGISTRY_ALIASES[norm])
    if norm in REGISTRY_KEYS:
        return ("registry", norm)

    suggestions = difflib.get_close_matches(norm, list_methods(), n=5)
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise KeyError(f"Unknown method {method!r}.{hint}")


def list_methods(kind: Optional[str] = None) -> List[str]:
    """
    List all callable method names.

    Args:
        kind: ``"sgns"``, ``"registry"`` or ``"fused"`` to filter, or None for
              everything (sorted, de-duplicated across aliases and canonical
              keys).
    """
    sgns_names = sorted(SGNS_ALIASES.keys())
    registry_names = sorted(set(REGISTRY_KEYS) | set(REGISTRY_ALIASES.keys()))
    fused_names = sorted(FUSED_ALIASES)

    if kind == "sgns":
        return sgns_names
    if kind == "registry":
        return registry_names
    if kind == "fused":
        return fused_names
    if kind is not None:
        raise ValueError(
            f"kind must be 'sgns', 'registry', 'fused', or None; got {kind!r}"
        )
    return sorted(set(sgns_names) | set(registry_names) | set(fused_names))
