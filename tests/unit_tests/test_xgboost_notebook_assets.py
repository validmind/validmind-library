# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import unittest
from pathlib import Path

try:
    import xgboost as xgb
except ImportError:
    xgb = None


@unittest.skipUnless(xgb is not None, "xgboost optional extra required")
class TestXGBoostNotebookAssets(unittest.TestCase):
    def test_json_models_load(self):
        repository_root = Path(__file__).parents[2]
        model_paths = (
            "notebooks/quickstart/xgboost_model_champion.json",
            "notebooks/use_cases/ongoing_monitoring/xgboost_model.json",
            "notebooks/use_cases/validation/xgb_model_champion.json",
        )

        for model_path in model_paths:
            with self.subTest(model_path=model_path):
                model = xgb.XGBClassifier()
                model.load_model(repository_root / model_path)


if __name__ == "__main__":
    unittest.main()
