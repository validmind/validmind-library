import unittest
import pandas as pd
import validmind as vm
import plotly.graph_objs as go
from validmind.errors import SkipTestError
from validmind.tests.data_validation.WOEBinPlots import WOEBinPlots
from validmind import RawData


class TestWOEBinPlots(unittest.TestCase):
    def setUp(self):
        # Create a sample dataset with categorical features and binary target
        n_samples = 1000

        df = pd.DataFrame(
            {
                "cat1": ["A", "B", "C"] * (n_samples // 3) + ["A"] * (n_samples % 3),
                "cat2": ["X", "Y"] * (n_samples // 2) + ["X"] * (n_samples % 2),
                "numeric": range(n_samples),
                "target": [0, 1] * (n_samples // 2)
                + [0] * (n_samples % 2),  # Binary target
            }
        )

        self.vm_dataset = vm.init_dataset(
            input_id="test_dataset",
            dataset=df,
            feature_columns=["cat1", "cat2", "numeric"],  # Include all feature columns
            target_column="target",
            __log=False,
        )

        # Create dataset with no features
        df_no_features = pd.DataFrame(
            {"target": [0, 1] * (n_samples // 2) + [0] * (n_samples % 2)}
        )

        self.vm_dataset_no_features = vm.init_dataset(
            input_id="test_dataset_no_features",
            dataset=df_no_features,
            feature_columns=[],  # No features at all
            target_column="target",
            __log=False,
        )

    def test_woe_bin_plots(self):
        results = WOEBinPlots(self.vm_dataset)

        # Check that we get the correct number of figures (one per feature column)
        self.assertIsInstance(results, tuple)
        self.assertEqual(
            len(results), 4
        )  # Should have 3 figures: cat1, cat2, and numeric and a RawData object

        # Check that outputs are plotly figures and the last one is RawData
        for fig in results[:-1]:
            self.assertIsInstance(fig, go.Figure)

        # Check that the last output is an instance of RawData
        self.assertIsInstance(results[-1], RawData)

        # Verify all feature columns have corresponding plots
        titles = [fig.layout.title.text for fig in results[:-1]]
        expected_features = ["cat1", "cat2", "numeric"]
        self.assertTrue(
            all(any(feat in title for feat in expected_features) for title in titles)
        )

    def test_no_features(self):
        # Should raise SkipTestError when no features present
        with self.assertRaises(SkipTestError):
            WOEBinPlots(self.vm_dataset_no_features)
