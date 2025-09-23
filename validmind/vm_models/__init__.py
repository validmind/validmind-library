# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Models entrypoint
"""

from .dataset.dataset import VMDataset
from .figure import Figure
from .input import VMInput
from .model import R_MODEL_TYPES, ModelAttributes, VMModel
from .result import ResultTable, TestResult

__all__ = [
    "VMInput",
    "VMDataset",
    "VMModel",
    "Figure",
    "ModelAttributes",
    "R_MODEL_TYPES",
    "ResultTable",
    "TestResult",
    "TestSuite",
    "TestSuiteRunner",
]


def __getattr__(name):  # Lazy access to avoid circular imports at module import time
    if name == "TestSuite":
        from .test_suite.test_suite import TestSuite as _TestSuite

        return _TestSuite
    if name == "TestSuiteRunner":
        from .test_suite.runner import TestSuiteRunner as _TestSuiteRunner

        return _TestSuiteRunner
    raise AttributeError(f"module 'validmind.vm_models' has no attribute {name!r}")
