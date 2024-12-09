import unittest
import pandas as pd
import numpy as np
import validmind as vm
from validmind.tests.data_validation.PhillipsPerronArch import PhillipsPerronArch
from validmind.errors import SkipTestError


class TestPhillipsPerronArch(unittest.TestCase):
    def setUp(self):
        # Create test dataset
        np.random.seed(42)
        n_samples = 100

        # Create datetime index
        date_rng = pd.date_range(start="2023-01-01", periods=n_samples, freq="D")

        # Create stationary series (random walk around mean)
        stationary_data = np.random.normal(0, 1, n_samples)

        # Create non-stationary series (random walk)
        non_stationary_data = np.cumsum(np.random.normal(0, 1, n_samples))

        df = pd.DataFrame(
            {"stationary": stationary_data, "non_stationary": non_stationary_data},
            index=date_rng,
        )

        self.vm_dataset = vm.init_dataset(
            input_id="test_dataset", dataset=df, __log=False
        )

    def test_output_structure(self):
        result = PhillipsPerronArch(self.vm_dataset)

        # Check basic structure
        self.assertIsInstance(result, dict)
        self.assertIn("Phillips-Perron Test Results", result)

        # Check results structure
        pp_results = result["Phillips-Perron Test Results"]
        self.assertIsInstance(pp_results, list)

        for column_result in pp_results:
            self.assertIn("Variable", column_result)
            self.assertIn("stat", column_result)
            self.assertIn("pvalue", column_result)
            self.assertIn("usedlag", column_result)
            self.assertIn("nobs", column_result)

    def test_stationarity_results(self):
        result = PhillipsPerronArch(self.vm_dataset)
        pp_results = result["Phillips-Perron Test Results"]

        # Get results for each series
        stationary_result = next(r for r in pp_results if r["Variable"] == "stationary")
        non_stationary_result = next(
            r for r in pp_results if r["Variable"] == "non_stationary"
        )

        # Stationary series should have low p-value (reject null hypothesis of non-stationarity)
        self.assertLess(stationary_result["pvalue"], 0.05)

        # Non-stationary series should have high p-value
        self.assertGreater(non_stationary_result["pvalue"], 0.05)

    def test_invalid_index(self):
        # Create dataset without datetime index
        df = pd.DataFrame({"col1": np.random.rand(100), "col2": np.random.rand(100)})

        invalid_dataset = vm.init_dataset(
            input_id="invalid_dataset", dataset=df, __log=False
        )

        # Should raise ValueError for non-datetime index
        with self.assertRaises(ValueError):
            PhillipsPerronArch(invalid_dataset)

    def test_non_numeric_data(self):
        # Create dataset with non-numeric column
        date_rng = pd.date_range(start="2023-01-01", periods=100, freq="D")
        df = pd.DataFrame(
            {"numeric": np.random.rand(100), "non_numeric": ["a"] * 100}, index=date_rng
        )

        mixed_dataset = vm.init_dataset(
            input_id="mixed_dataset", dataset=df, __log=False
        )

        # Should handle non-numeric data gracefully
        result = PhillipsPerronArch(mixed_dataset)
        pp_results = result["Phillips-Perron Test Results"]

        # Should only include results for numeric column
        self.assertEqual(len(pp_results), 1)
        self.assertEqual(pp_results[0]["Variable"], "numeric")
