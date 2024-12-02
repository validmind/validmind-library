import unittest
import pandas as pd
import validmind as vm
import plotly.graph_objs as go
from validmind.errors import SkipTestError
from validmind.tests.data_validation.TabularNumericalHistograms import (
    TabularNumericalHistograms,
)


class TestTabularNumericalHistograms(unittest.TestCase):
    def setUp(self):
        # Set consistent size for all columns
        n_samples = 100

        # Create a sample dataset with numerical and non-numerical columns
        df = pd.DataFrame(
            {
                "num1": range(n_samples),
                "num2": [i * 2 for i in range(n_samples)],
                "categorical": ["A", "B", "C"] * (n_samples // 3)
                + ["A"] * (n_samples % 3),
            }
        )

        self.vm_dataset = vm.init_dataset(
            input_id="test_dataset",
            dataset=df,
            feature_columns=["num1", "num2", "categorical"],
            __log=False,
        )

        # Create dataset with no numerical columns
        df_no_numeric = pd.DataFrame(
            {
                "cat1": ["A", "B", "C"] * (n_samples // 3) + ["A"] * (n_samples % 3),
                "cat2": ["X", "Y"] * (n_samples // 2) + ["X"] * (n_samples % 2),
            }
        )

        self.vm_dataset_no_numeric = vm.init_dataset(
            input_id="test_dataset_no_numeric",
            dataset=df_no_numeric,
            feature_columns=["cat1", "cat2"],
            __log=False,
        )

    def test_numerical_histograms(self):
        figures = TabularNumericalHistograms(self.vm_dataset)

        # Check that we get the correct number of figures (one per numerical column)
        self.assertIsInstance(figures, tuple)
        self.assertEqual(len(figures), 2)  # Should have 2 figures for num1 and num2

        # Check that outputs are plotly figures
        for fig in figures:
            self.assertIsInstance(fig, go.Figure)

    def test_no_numerical_columns(self):
        # Should raise ValueError when no numerical columns present
        with self.assertRaises(ValueError):
            TabularNumericalHistograms(self.vm_dataset_no_numeric)
