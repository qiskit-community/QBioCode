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
Config loading/merging for the QuVINE embedding API.

The API reuses the same Hydra/OmegaConf YAML schema the pipeline uses
(``configs/config.yaml``). Unlike the CLI, the API does not need the Hydra
runtime -- it simply loads a base ``DictConfig`` and deep-merges overrides.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

from omegaconf import DictConfig, OmegaConf


def _find_default_config() -> Path:
    """
    Locate the packaged default ``config.yaml``.

    Search order:
    1. ``$QUVINE_DEFAULT_CONFIG`` if set.
    2. ``configs/config.yaml`` walking up from this file (repo / installed layout).
    """
    env_path = os.environ.get("QUVINE_DEFAULT_CONFIG")
    if env_path:
        return Path(env_path)

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "configs" / "config.yaml"
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not locate a default QuVINE config.yaml. Pass config=<path> "
        "explicitly or set the QUVINE_DEFAULT_CONFIG environment variable."
    )


def load_config(
    config: Optional[Union[DictConfig, dict, str]] = None,
    overrides: Optional[Union[dict, list, DictConfig]] = None,
    default_path: Optional[str] = None,
) -> DictConfig:
    """
    Build the OmegaConf config the API operates on.

    Args:
        config: One of
            * ``None``         -- load the packaged default config.
            * a ``str``/path   -- load that YAML file.
            * a ``dict``       -- treat as a full config.
            * a ``DictConfig`` -- used as-is (a deep copy is made).
        overrides: Deep-merged on top of ``config``. Accepts a dict, a
            ``DictConfig``, or a list of Hydra-style ``"a.b=c"`` dotlist strings.
        default_path: Explicit path to use instead of auto-discovery when
            ``config is None``.

    Returns:
        A resolved-on-access ``DictConfig``.
    """
    if config is None:
        base = OmegaConf.load(default_path or _find_default_config())
    elif isinstance(config, str):
        base = OmegaConf.load(config)
    elif isinstance(config, DictConfig):
        base = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    elif isinstance(config, dict):
        base = OmegaConf.create(config)
    else:
        raise TypeError(
            f"config must be None, str, dict, or DictConfig; got {type(config)!r}"
        )

    if overrides:
        if isinstance(overrides, list):
            override_cfg = OmegaConf.from_dotlist(list(overrides))
        else:
            override_cfg = OmegaConf.create(overrides)
        base = OmegaConf.merge(base, override_cfg)

    return base
