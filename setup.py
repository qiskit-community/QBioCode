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
QBioCode: Quantum Applications in Healthcare and Life Science Data

All project metadata (name, version, dependencies, entry points, package data)
is declared in ``pyproject.toml`` (PEP 621). This file is a thin shim kept only
so that legacy ``python setup.py`` / older tooling invocations keep working; it
intentionally passes no arguments so that setuptools reads everything from
``pyproject.toml``.

Do NOT re-add metadata here: duplicating fields that ``pyproject.toml`` already
declares (dependencies, version, ...) makes the two disagree and was the cause
of past install/version errors -- e.g. a build-time ``setuptools`` pin leaking
into the published package's runtime dependencies, or an empty dependency list
when building from an sdist.
"""

from setuptools import setup

setup()
