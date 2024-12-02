import unittest
import pandas as pd
import validmind as vm
import plotly.graph_objs as go
from validmind.errors import SkipTestError
from validmind.tests.data_validation.TargetRateBarPlots import TargetRateBarPlots


class TestTargetRateBarPlots(unittest.TestCase):
    def setUp(self):
        # Set consistent size for all columns
        n_samples = 100

        # Create a sample dataset with categorical features and binary target
        df = pd.DataFrame(
            {
                "cat1": ["A", "B", "C"] * (n_samples // 3) + ["A"] * (n_samples % 3),
                "cat2": ["X", "Y"] * (n_samples // 2) + ["X"] * (n_samples % 2),
                "target": [0, 1] * (n_samples // 2)
                + [0] * (n_samples % 2),  # Binary target
            }
        )

        self.vm_dataset = vm.init_dataset(
            input_id="test_dataset",
            dataset=df,
            feature_columns=["cat1", "cat2"],
            target_column="target",
            __log=False,
        )

        # Create dataset with no categorical columns
        df_no_cat = pd.DataFrame(
            {
                "num1": range(n_samples),
                "num2": range(n_samples),
                "target": [0, 1] * (n_samples // 2) + [0] * (n_samples % 2),
            }
        )

        self.vm_dataset_no_cat = vm.init_dataset(
            input_id="test_dataset_no_cat",
            dataset=df_no_cat,
            feature_columns=["num1", "num2"],
            target_column="target",
            __log=False,
        )

        # Create dataset with non-binary target
        df_non_binary = pd.DataFrame(
            {
                "cat1": ["A", "B", "C"] * (n_samples // 3) + ["A"] * (n_samples % 3),
                "target": range(n_samples),  # Non-binary target
            }
        )

        self.vm_dataset_non_binary = vm.init_dataset(
            input_id="test_dataset_non_binary",
            dataset=df_non_binary,
            feature_columns=["cat1"],
            target_column="target",
            __log=False,
        )

    def test_target_rate_bar_plots(self):
        figures = TargetRateBarPlots(self.vm_dataset)

        # Check that we get the correct number of figures (one per categorical column)
        self.assertIsInstance(figures, tuple)
        self.assertEqual(len(figures), 2)  # Should have 2 figures for cat1 and cat2

        # Check that outputs are plotly figures
        for fig in figures:
            self.assertIsInstance(fig, go.Figure)

    def test_no_categorical_columns(self):
        # Should raise SkipTestError when no categorical columns present
        with self.assertRaises(SkipTestError):
            TargetRateBarPlots(self.vm_dataset_no_cat)

    def test_non_binary_target(self):
        # Should raise SkipTestError when target is not binary
        with self.assertRaises(SkipTestError):
            TargetRateBarPlots(self.vm_dataset_non_binary)
