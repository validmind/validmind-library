"""
Unit tests for VMDataset class
"""

import unittest
from unittest import TestCase
from unittest.mock import patch

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from validmind.client import init_model
from validmind.errors import MissingOrInvalidModelPredictFnError
from validmind.models import MetadataModel
from validmind.vm_models.dataset.dataset import DataFrameDataset
from validmind.vm_models.model import ModelAttributes, VMModel


class TestTabularDataset(TestCase):
    def setUp(self):
        """
        Create a sample dataset for testing
        """
        self.df = pd.DataFrame(
            {"col1": [1, 2, 3], "col2": ["a", "b", "c"], "target": [0, 1, 0]}
        )

    def test_init_dataset_pandas_no_options(self):
        """
        Test that a DataFrameDataset can be initialized with a pandas DataFrame and no options
        """
        vm_dataset = DataFrameDataset(raw_dataset=self.df)

        pd.testing.assert_frame_equal(vm_dataset.df, self.df)

    def test_init_dataset_pandas_target_column(self):
        """
        Test that a DataFrameDataset provides access to the target column
        """
        vm_dataset = DataFrameDataset(raw_dataset=self.df, target_column="target")

        self.assertEqual(vm_dataset.target_column, "target")
        np.testing.assert_array_equal(vm_dataset.y, self.df["target"].values)
        pd.testing.assert_frame_equal(vm_dataset.y_df(), self.df["target"].to_frame())

        # Feature columns should be all columns except the target column
        self.assertEqual(vm_dataset.feature_columns_numeric, ["col1"])
        self.assertEqual(vm_dataset.feature_columns_categorical, ["col2"])
        self.assertEqual(vm_dataset.feature_columns, ["col1", "col2"])

    def test_init_dataset_pandas_feature_columns(self):
        """
        Test that a DataFrameDataset allows configuring feature columns
        """
        vm_dataset = DataFrameDataset(
            raw_dataset=self.df, target_column="target", feature_columns=["col1"]
        )

        # Only one feature column "col1"
        np.testing.assert_array_equal(vm_dataset.x, self.df[["col1"]].values)

        self.assertEqual(vm_dataset.feature_columns_numeric, ["col1"])
        self.assertEqual(vm_dataset.feature_columns_categorical, [])
        self.assertEqual(vm_dataset.feature_columns, ["col1"])

    def test_dtype_preserved(self):
        """
        Test that dtype is preserved in DataFrameDataset.
        """

        test_df = pd.DataFrame({"col1": pd.Categorical(["x", "y", "z"])})

        # Verify original data is categorical
        self.assertTrue(
            pd.api.types.is_categorical_dtype(test_df["col1"]),
            "Original DataFrame should have categorical dtype",
        )

        # Verify categorical dtype is preserved
        dataset = DataFrameDataset(raw_dataset=test_df, input_id="test_dataset")

        self.assertTrue(
            pd.api.types.is_categorical_dtype(dataset.df["col1"]),
            "DataFrameDataset should preserve categorical dtype",
        )

    def test_assign_predictions_invalid_model(self):
        """
        Test assigning predictions to dataset with an invalid model
        """
        vm_dataset = DataFrameDataset(
            raw_dataset=self.df, target_column="target", feature_columns=["col1"]
        )

        vm_model = dict()
        with self.assertRaises(ValueError, msg="Model must be a VMModel instance"):
            vm_dataset.assign_predictions(model=vm_model)

        with self.assertRaises(
            TypeError,
            msg="Can't instantiate abstract class VMModel with abstract method predict",
        ):
            vm_model = VMModel(input_id="1234")

        vm_model = MetadataModel(
            input_id="1234",
            attributes=ModelAttributes.from_dict(
                {
                    "architecture": "Spark",
                    "language": "Python",
                }
            ),
        )
        with self.assertRaises(
            MissingOrInvalidModelPredictFnError,
            msg=(
                "Cannot compute predictions for model's that don't support inference. "
                "You can pass `prediction_values` or `prediction_columns` to use precomputed predictions"
            ),
        ):
            vm_dataset.assign_predictions(model=vm_model)

    def test_assign_predictions_with_classification_model(self):
        """
        Test assigning predictions to dataset with a valid model
        """
        df = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6], "y": [0, 1, 0]})
        vm_dataset = DataFrameDataset(
            raw_dataset=df, target_column="y", feature_columns=["x1", "x2"]
        )

        # Train a simple model
        model = LogisticRegression()
        model.fit(vm_dataset.x, vm_dataset.y.ravel())

        vm_model = init_model(input_id="logreg", model=model, __log=False)
        self.assertIsNone(vm_dataset.prediction_column(vm_model))

        vm_dataset.assign_predictions(model=vm_model)
        self.assertEqual(vm_dataset.prediction_column(vm_model), "logreg_prediction")

        # Check that the predictions are assigned to the dataset
        self.assertTrue("logreg_prediction" in vm_dataset._df.columns)
        self.assertIsInstance(vm_dataset.y_pred(vm_model), np.ndarray)
        self.assertIsInstance(vm_dataset.y_pred_df(vm_model), pd.DataFrame)

        # This model in particular will calculate probabilities as well
        self.assertTrue("logreg_probabilities" in vm_dataset._df.columns)
        self.assertIsInstance(vm_dataset.y_prob(vm_model), np.ndarray)
        self.assertIsInstance(vm_dataset.y_prob_df(vm_model), pd.DataFrame)

    def test_assign_predictions_with_regression_model(self):
        """
        Test assigning predictions to dataset with a valid model
        """
        # TODO "y": [0.1, 0.2, 0.3] wil trick the _is_probability() method
        df = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6], "y": [0.1, 1.2, 2.3]})
        vm_dataset = DataFrameDataset(
            raw_dataset=df, target_column="y", feature_columns=["x1", "x2"]
        )

        # Train a simple model
        model = LinearRegression()
        model.fit(vm_dataset.x, vm_dataset.y.ravel())

        vm_model = init_model(input_id="linreg", model=model, __log=False)
        self.assertIsNone(vm_dataset.prediction_column(vm_model))

        vm_dataset.assign_predictions(model=vm_model)
        self.assertEqual(vm_dataset.prediction_column(vm_model), "linreg_prediction")

        # Check that the predictions are assigned to the dataset
        self.assertTrue("linreg_prediction" in vm_dataset._df.columns)
        self.assertIsInstance(vm_dataset.y_pred(vm_model), np.ndarray)
        self.assertIsInstance(vm_dataset.y_pred_df(vm_model), pd.DataFrame)

        # Linear models do not have probabilities
        self.assertFalse("linreg_probabilities" in vm_dataset._df.columns)

    def test_assign_predictions_with_multiple_models(self):
        """
        Test assigning predictions from multiple models to a single dataset
        """
        df = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6], "y": [0, 1, 0]})
        vm_dataset = DataFrameDataset(
            raw_dataset=df, target_column="y", feature_columns=["x1", "x2"]
        )

        # Train simple models
        lr_model = LogisticRegression()
        lr_model.fit(vm_dataset.x, vm_dataset.y.ravel())

        rf_model = RandomForestClassifier()
        rf_model.fit(vm_dataset.x, vm_dataset.y.ravel())

        vm_lr_model = init_model(input_id="logreg", model=lr_model, __log=False)
        vm_rf_model = init_model(input_id="rf", model=rf_model, __log=False)

        lr_model_predictions = lr_model.predict(vm_dataset.x)
        rf_model_predictions = rf_model.predict(vm_dataset.x)

        vm_dataset.assign_predictions(model=vm_lr_model)
        vm_dataset.assign_predictions(model=vm_rf_model)

        self.assertEqual(vm_dataset.prediction_column(vm_lr_model), "logreg_prediction")
        self.assertEqual(vm_dataset.prediction_column(vm_rf_model), "rf_prediction")

        # Check that the predictions are assigned to the dataset and they match
        # their respective models
        self.assertTrue("logreg_prediction" in vm_dataset._df.columns)
        self.assertTrue("rf_prediction" in vm_dataset._df.columns)
        np.testing.assert_array_equal(
            vm_dataset.y_pred(vm_lr_model), lr_model_predictions
        )
        np.testing.assert_array_equal(
            vm_dataset.y_pred(vm_rf_model), rf_model_predictions
        )

        # This model in particular will calculate probabilities as well
        self.assertTrue("logreg_probabilities" in vm_dataset._df.columns)
        self.assertTrue("rf_probabilities" in vm_dataset._df.columns)

    def test_assign_predictions_with_model_and_prediction_values(self):
        """
        Test assigning predictions to dataset with pre-computed model predictions
        """
        df = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6], "y": [0, 1, 0]})
        vm_dataset = DataFrameDataset(
            raw_dataset=df, target_column="y", feature_columns=["x1", "x2"]
        )

        # Train a simple model
        lr_model = LogisticRegression()
        lr_model.fit(vm_dataset.x, vm_dataset.y.ravel())

        vm_lr_model = init_model(input_id="logreg", model=lr_model, __log=False)

        lr_model_predictions = lr_model.predict(vm_dataset.x)

        with patch.object(
            lr_model, "predict", return_value=lr_model_predictions
        ) as mock:
            vm_dataset.assign_predictions(
                model=vm_lr_model, prediction_values=lr_model_predictions
            )
            # The model's predict method should not be called
            mock.assert_not_called()

        self.assertEqual(vm_dataset.prediction_column(vm_lr_model), "logreg_prediction")

        self.assertTrue("logreg_prediction" in vm_dataset._df.columns)
        np.testing.assert_array_equal(
            vm_dataset.y_pred(vm_lr_model), lr_model_predictions
        )

        # Probabilities are not auto-assigned if prediction_values are provided
        self.assertTrue("logreg_probabilities" not in vm_dataset._df.columns)

    def test_assign_predictions_with_no_model_and_prediction_values(self):
        """
        Test assigning predictions to dataset with pre-computed model predictions
        """
        df = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6], "y": [0, 1, 0]})
        vm_dataset = DataFrameDataset(
            raw_dataset=df, target_column="y", feature_columns=["x1", "x2"]
        )

        # Train a simple model
        # This time let's simulate that the predictions came from a model we don't have access to
        lr_model = LogisticRegression()
        lr_model.fit(vm_dataset.x, vm_dataset.y.ravel())

        model_attributes = {
            "architecture": "spark",
            "language": "Python",
        }

        vm_lr_model = init_model(
            input_id="logreg", attributes=model_attributes, __log=False
        )

        lr_model_predictions = lr_model.predict(vm_dataset.x)

        with patch.object(
            lr_model, "predict", return_value=lr_model_predictions
        ) as mock:
            vm_dataset.assign_predictions(
                model=vm_lr_model, prediction_values=lr_model_predictions
            )
            # The model's predict method should not be called
            mock.assert_not_called()

        self.assertEqual(vm_dataset.prediction_column(vm_lr_model), "logreg_prediction")

        self.assertTrue("logreg_prediction" in vm_dataset._df.columns)
        np.testing.assert_array_equal(
            vm_dataset.y_pred(vm_lr_model), lr_model_predictions
        )

        # Probabilities are not auto-assigned if prediction_values are provided
        self.assertTrue("logreg_probabilities" not in vm_dataset._df.columns)


if __name__ == "__main__":
    unittest.main()
