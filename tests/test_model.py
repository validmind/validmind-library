# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Unit tests for model type detection in validmind.vm_models.model
"""

import sys
import unittest

from sklearn.linear_model import LogisticRegression

from validmind.vm_models.model import is_pytorch_model


class BrokenTorchFinder:
    """Simulates a broken torch install where importing torch raises an
    error other than ImportError (e.g. OSError WinError 1114 when c10.dll
    fails to load on Windows)."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise OSError(
                "[WinError 1114] A dynamic link library (DLL) initialization "
                "routine failed. Error loading c10.dll"
            )
        return None


class TestIsPyTorchModel(unittest.TestCase):
    def test_non_torch_model_returns_false(self):
        self.assertFalse(is_pytorch_model(LogisticRegression()))

    def test_broken_torch_install_returns_false(self):
        saved_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "torch" or name.startswith("torch.")
        }
        for name in saved_modules:
            del sys.modules[name]

        finder = BrokenTorchFinder()
        sys.meta_path.insert(0, finder)

        try:
            self.assertFalse(is_pytorch_model(LogisticRegression()))
        finally:
            sys.meta_path.remove(finder)
            sys.modules.update(saved_modules)


if __name__ == "__main__":
    unittest.main()
