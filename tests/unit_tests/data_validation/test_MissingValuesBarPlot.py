import unittest
import pandas as pd
import numpy as np
import validmind as vm
from validmind.tests.data_validation.MissingValuesBarPlot import MissingValuesBarPlot
import plotly.graph_objects as go
from validmind import RawData


class TestMissingValuesBarPlot(unittest.TestCase):
    def setUp(self):
        # Create test dataset with known missing values
        n_samples = 100

        # Create data with controlled missing values
        data = {
            "no_missing": np.random.rand(n_samples),  # 0% missing
            "low_missing": np.random.rand(n_samples),  # 20% missing (below threshold)
            "high_missing": np.random.rand(n_samples),  # 90% missing (above threshold)
            "complete": np.ones(n_samples),  # 0% missing
        }

        # Insert missing values
        data["low_missing"][:20] = np.nan  # 20% missing
        data["high_missing"][:90] = np.nan  # 90% missing

        df = pd.DataFrame(data)

        self.vm_dataset = vm.init_dataset(
            input_id="test_dataset", dataset=df, __log=False
        )

    def test_figure_structure(self):
        fig, raw_data = MissingValuesBarPlot(self.vm_dataset, threshold=80)

        # Check figure return type
        self.assertIsInstance(fig, go.Figure)

        # Check raw data return type
        self.assertIsInstance(raw_data, RawData)

    def test_data_traces(self):
        fig, _ = MissingValuesBarPlot(self.vm_dataset, threshold=80)

        # Should have 3 traces: below threshold, above threshold, and threshold line
        self.assertEqual(len(fig.data), 3)

        # Check trace types
        self.assertEqual(fig.data[0].type, "bar")  # Below threshold
        self.assertEqual(fig.data[1].type, "bar")  # Above threshold
        self.assertEqual(fig.data[2].type, "scatter")  # Threshold line
