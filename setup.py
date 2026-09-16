"""Compatibility entry point for legacy setuptools invocations.

All project metadata and dependencies are defined in ``pyproject.toml`` so
PEP 517 builds and legacy setuptools builds share one source of truth.
"""

from setuptools import setup


setup()
