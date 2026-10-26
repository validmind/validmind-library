import unittest

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import validmind as vm
from validmind.tests.model_validation.statsmodels.ExpectedCoefficientSigns import (
    ExpectedCoefficientSigns,
)


class TestExpectedCoefficientSigns(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 400
        df = pd.DataFrame(
            {
                "up": rng.normal(0, 1, n),
                "down": rng.normal(0, 1, n),
                "none": rng.normal(0, 1, n),
                "group": rng.choice(["a", "b"], n),
            }
        )
        df["y"] = 2 * df.up - 3 * df.down + rng.normal(0, 1, n)
        self.fit = smf.ols("y ~ up + down + none + C(group)", df).fit()
        self.model = vm.init_model(self.fit, input_id="m", __log=False)

    def test_passes_matching_signs(self):
        group_sign = 1 if self.fit.params["C(group)[T.b]"] > 0 else -1
        table, passed, _ = ExpectedCoefficientSigns(
            self.model,
            expected_signs={"up": "+", "down": "negative", "C(group)[T.b]": group_sign},
        )
        self.assertTrue(passed)
        self.assertTrue((table["Result"] == "Pass").all())

    def test_fails_significant_wrong_sign(self):
        table, passed, _ = ExpectedCoefficientSigns(
            self.model, expected_signs={"up": "-", "down": "-"}
        )
        results = table.set_index("Coefficient")["Result"]
        self.assertFalse(passed)
        self.assertEqual(results["up"], "Fail")
        self.assertEqual(results["down"], "Pass")

    def test_distinguishes_unsupported_from_contradicted(self):
        wrong = "-" if self.fit.params["none"] > 0 else "+"
        table, passed, _ = ExpectedCoefficientSigns(
            self.model, expected_signs={"none": wrong}
        )
        self.assertFalse(passed)
        self.assertEqual(table["Result"].iloc[0], "Fail (not significant)")

    def test_unknown_coefficient_raises(self):
        with self.assertRaisesRegex(ValueError, "not in the model"):
            ExpectedCoefficientSigns(self.model, expected_signs={"missing": "+"})

    def test_invalid_sign_raises(self):
        with self.assertRaisesRegex(ValueError, "must be"):
            ExpectedCoefficientSigns(self.model, expected_signs={"up": "up"})


if __name__ == "__main__":
    unittest.main()
