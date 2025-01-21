import unittest
import pandas as pd
import validmind as vm
import plotly.graph_objs as go
from validmind import RawData
from validmind.errors import SkipTestError
from validmind.tests.data_validation.TabularCategoricalBarPlots import (
    TabularCategoricalBarPlots,
)


class TestTabularCategoricalBarPlots(unittest.TestCase):
    def setUp(self):
        # Create a sample dataset with categorical and numerical columns
        df = pd.DataFrame(
            {
                "cat1": ["A", "B", "C", "A", "B"] * 20,
                "cat2": ["X", "Y", "X", "Y", "X"] * 20,
                "numeric": range(100),
            }
        )

        self.vm_dataset = vm.init_dataset(
            input_id="test_dataset",
            dataset=df,
            feature_columns=["cat1", "cat2", "numeric"],
            __log=False,
        )

        # Create dataset with no categorical columns
        df_no_cat = pd.DataFrame({"numeric1": range(100), "numeric2": range(100, 200)})

        self.vm_dataset_no_cat = vm.init_dataset(
            input_id="test_dataset_no_cat",
            dataset=df_no_cat,
            feature_columns=["numeric1", "numeric2"],
            __log=False,
        )

    def test_categorical_bar_plots(self):
        figures = TabularCategoricalBarPlots(self.vm_dataset)

        # Check that the last element is an instance of RawData
        self.assertIsInstance(figures[-1], RawData)

        # Remove the raw data before checking figures
        figures = figures[:-1]

        # Check that we get the correct number of figures (one per categorical column)
        self.assertIsInstance(figures, tuple)
        self.assertEqual(len(figures), 2)  # Should have 2 figures for cat1 and cat2

        # Check that outputs are plotly figures
        for fig in figures:
            self.assertIsInstance(fig, go.Figure)

    def test_no_categorical_columns(self):
        # Should raise SkipTestError when no categorical columns present
        with self.assertRaises(SkipTestError):
            TabularCategoricalBarPlots(self.vm_dataset_no_cat)
