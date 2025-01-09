import unittest
import pandas as pd
import numpy as np
import validmind as vm
from validmind.tests.data_validation.KPSS import KPSS
from validmind import RawData


class TestKPSS(unittest.TestCase):
    def setUp(self):
        # Create test datasets
        np.random.seed(42)
        n_samples = 100

        # Create datetime index
        date_rng = pd.date_range(start="2023-01-01", periods=n_samples, freq="D")

        # Stationary series (random walk around mean)
        stationary_data = np.random.normal(loc=0, scale=1, size=n_samples)

        # Non-stationary series (random walk)
        non_stationary_data = np.cumsum(
            np.random.normal(loc=0, scale=1, size=n_samples)
        )

        # Create DataFrame with both series and datetime index
        df = pd.DataFrame(
            {"stationary": stationary_data, "non_stationary": non_stationary_data},
            index=date_rng,
        )

        self.vm_dataset = vm.init_dataset(
            input_id="test_dataset", dataset=df, __log=False
        )

    def test_kpss_structure(self):
        result, raw_data = KPSS(self.vm_dataset)

        # Check basic structure
        self.assertIsInstance(result, dict)
        self.assertIn("KPSS Test Results", result)

        # Check results structure
        kpss_results = result["KPSS Test Results"]
        self.assertIsInstance(kpss_results, list)

        for column_result in kpss_results:
            self.assertIn("Variable", column_result)
            self.assertIn("stat", column_result)
            self.assertIn("pvalue", column_result)
            self.assertIn("usedlag", column_result)
            self.assertIn("critical_values", column_result)

        # Check raw data instance
        self.assertIsInstance(raw_data, RawData)

    def test_kpss_results(self):
        result, _ = KPSS(self.vm_dataset)
        kpss_results = result["KPSS Test Results"]

        # Get results for each series
        stationary_result = next(
            r for r in kpss_results if r["Variable"] == "stationary"
        )
        non_stationary_result = next(
            r for r in kpss_results if r["Variable"] == "non_stationary"
        )

        # Check p-values (lower p-value indicates non-stationarity)
        self.assertGreater(
            stationary_result["pvalue"], 0.05
        )  # Stationary series should have high p-value
        self.assertLess(
            non_stationary_result["pvalue"], 0.05
        )  # Non-stationary series should have low p-value

        # Check test statistics
        self.assertLess(stationary_result["stat"], non_stationary_result["stat"])

    def test_critical_values(self):
        result, _ = KPSS(self.vm_dataset)
        kpss_results = result["KPSS Test Results"]

        for column_result in kpss_results:
            critical_values = column_result["critical_values"]

            # Check critical values are present
            self.assertIsInstance(critical_values, dict)

            # Check critical values are in descending order (more stringent critical values are larger)
            self.assertGreater(
                float(critical_values["1%"]), float(critical_values["5%"])
            )
            self.assertGreater(
                float(critical_values["5%"]), float(critical_values["10%"])
            )
