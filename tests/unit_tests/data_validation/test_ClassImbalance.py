import unittest
import pandas as pd
import validmind as vm
from validmind import RawData
from validmind.errors import SkipTestError
from validmind.tests.data_validation.ClassImbalance import ClassImbalance
from plotly.graph_objs import Figure


class TestClassImbalance(unittest.TestCase):
    def setUp(self):
        # Create a simple dataset with balanced classes
        balanced_df = pd.DataFrame(
            {
                "target": ["A", "B", "A", "B", "A", "B"],  # 50-50 split
            }
        )
        self.balanced_dataset = vm.init_dataset(
            input_id="balanced",
            dataset=balanced_df,
            target_column="target",
            __log=False,
        )

        # Create a dataset with imbalanced classes
        imbalanced_df = pd.DataFrame(
            {
                "target": ["A", "A", "A", "A", "A", "B"],  # 83-17 split
            }
        )
        self.imbalanced_dataset = vm.init_dataset(
            input_id="imbalanced",
            dataset=imbalanced_df,
            target_column="target",
            __log=False,
        )

    def test_balanced_classes(self):
        results, figure, passed, raw_data = ClassImbalance(
            self.balanced_dataset, min_percent_threshold=20
        )

        # Check return types
        self.assertIsInstance(results, dict)
        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(passed, bool)
        self.assertIsInstance(raw_data, RawData)

        # Check results for balanced dataset
        imbalance_data = results["target Class Imbalance"]
        self.assertEqual(len(imbalance_data), 2)  # Two classes

        # Both classes should pass with 50%
        self.assertTrue(all(row["Pass/Fail"] == "Pass" for row in imbalance_data))
        self.assertTrue(passed)  # Overall test should pass

    def test_imbalanced_classes(self):
        results, figure, passed, raw_data = ClassImbalance(
            self.imbalanced_dataset, min_percent_threshold=20
        )

        # Check return type for raw data
        self.assertIsInstance(raw_data, RawData)

        imbalance_data = results["target Class Imbalance"]

        # Class B should fail (17% < 20%)
        b_class = next(row for row in imbalance_data if row["target"] == "B")
        self.assertEqual(b_class["Pass/Fail"], "Fail")
        self.assertFalse(passed)  # Overall test should fail

    def test_custom_threshold(self):
        # With threshold of 10%, both classes should pass even in imbalanced dataset
        results, figure, passed, raw_data = ClassImbalance(
            self.imbalanced_dataset, min_percent_threshold=10
        )
        self.assertTrue(passed)

    def test_missing_target(self):
        # Dataset without target column should raise SkipTestError
        df_no_target = pd.DataFrame({"x": [1, 2, 3]})
        dataset_no_target = vm.init_dataset(
            input_id="no_target",
            dataset=df_no_target,
            __log=False,
        )

        with self.assertRaises(SkipTestError):
            ClassImbalance(dataset_no_target)
