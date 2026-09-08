import unittest

import pandas as pd
import validmind as vm
from plotly.graph_objects import Figure

from validmind.errors import SkipTestError
from validmind.tests.plots.BoxPlot import BoxPlot


class TestBoxPlot(unittest.TestCase):
    def setUp(self):
        df = pd.DataFrame(
            {
                "num1": [1.0, 2.0, 3.0, 4.0],
                "num2": [10, 20, 30, 40],
                "cat": ["a", "b", "a", "b"],
                "target": [0, 1, 0, 1],
            }
        )
        self.vm_dataset = vm.init_dataset(
            input_id="dataset", dataset=df, target_column="target", __log=False
        )

    def test_default_columns_uses_numeric_features(self):
        # Regression: columns=None used to raise KeyError("None of [Index([None]...")
        fig = BoxPlot(self.vm_dataset)
        self.assertIsInstance(fig, Figure)
        self.assertEqual(sorted(t.name for t in fig.data), ["num1", "num2"])

    def test_explicit_columns_filtered_to_numeric(self):
        fig = BoxPlot(self.vm_dataset, columns=["num1", "cat", "missing"])
        self.assertEqual([t.name for t in fig.data], ["num1"])

    def test_group_by(self):
        fig = BoxPlot(self.vm_dataset, columns=["num1"], group_by="cat")
        self.assertEqual(len(fig.data), 2)

    def test_no_numeric_columns_skips(self):
        with self.assertRaises(SkipTestError):
            BoxPlot(self.vm_dataset, columns=["cat"])


if __name__ == "__main__":
    unittest.main()
