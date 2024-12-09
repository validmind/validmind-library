import unittest
import pandas as pd
import validmind as vm
from validmind.tests.data_validation.DatasetSplit import DatasetSplit


class TestDatasetSplit(unittest.TestCase):
    def setUp(self):
        # Create three different datasets with different sizes
        train_df = pd.DataFrame(
            {"feature1": range(60), "feature2": range(60)}  # 60 rows
        )

        test_df = pd.DataFrame(
            {"feature1": range(20), "feature2": range(20)}  # 20 rows
        )

        val_df = pd.DataFrame({"feature1": range(20), "feature2": range(20)})  # 20 rows

        # Initialize VMDatasets
        self.train_dataset = vm.init_dataset(
            input_id="train_ds", dataset=train_df, __log=False
        )

        self.test_dataset = vm.init_dataset(
            input_id="test_ds", dataset=test_df, __log=False
        )

        self.val_dataset = vm.init_dataset(
            input_id="validation_ds", dataset=val_df, __log=False
        )

    def test_dataset_split_proportions(self):
        # Run DatasetSplit
        result = DatasetSplit([self.train_dataset, self.test_dataset, self.val_dataset])

        # Verify the structure of the result
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4)  # 3 datasets + total

        # Create a dictionary for easier testing
        result_dict = {item["Dataset"]: item for item in result}

        # Test total size
        self.assertEqual(result_dict["Total"]["Size"], 100)
        self.assertEqual(result_dict["Total"]["Proportion"], "100%")

        # Test individual dataset sizes
        self.assertEqual(result_dict["train_ds"]["Size"], 60)
        self.assertEqual(result_dict["train_ds"]["Proportion"], "60.00%")

        self.assertEqual(result_dict["test_ds"]["Size"], 20)
        self.assertEqual(result_dict["test_ds"]["Proportion"], "20.00%")

        self.assertEqual(result_dict["validation_ds"]["Size"], 20)
        self.assertEqual(result_dict["validation_ds"]["Proportion"], "20.00%")

    def test_dataset_split_with_none(self):
        # Test with some datasets being None
        result = DatasetSplit([self.train_dataset, None, self.test_dataset])

        # Create a dictionary for easier testing
        result_dict = {item["Dataset"]: item for item in result}

        # Test total size
        self.assertEqual(result_dict["Total"]["Size"], 80)
        self.assertEqual(result_dict["Total"]["Proportion"], "100%")

        # Test individual dataset sizes
        self.assertEqual(result_dict["train_ds"]["Size"], 60)
        self.assertEqual(result_dict["train_ds"]["Proportion"], "75.00%")

        self.assertEqual(result_dict["test_ds"]["Size"], 20)
        self.assertEqual(result_dict["test_ds"]["Proportion"], "25.00%")
