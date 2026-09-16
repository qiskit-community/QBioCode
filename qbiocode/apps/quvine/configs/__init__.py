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

"""Packaged Hydra configuration for the QuVINE app (see config.yaml).

This file makes the directory a real package. Without it
``[tool.setuptools.packages.find]`` skips the directory entirely, so the
modules here are missing from a built wheel even though they import fine from
a source checkout.
"""
