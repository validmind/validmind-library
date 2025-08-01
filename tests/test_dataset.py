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

    def test_assign_predictions_with_classification_predict_fn(self):
        """
        Test assigning predictions to dataset with a model created using predict_fn for classification
        """
        df = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6], "y": [0, 1, 0]})
        vm_dataset = DataFrameDataset(
            raw_dataset=df, target_column="y", feature_columns=["x1", "x2"]
        )

        # Define a simple classification predict function
        def simple_classify_fn(input_dict):
            # Simple rule: if x1 + x2 > 5, return 1, else 0
            return 1 if input_dict["x1"] + input_dict["x2"] > 5 else 0

        vm_model = init_model(
            input_id="predict_fn_classifier", predict_fn=simple_classify_fn, __log=False
        )
        self.assertIsNone(vm_dataset.prediction_column(vm_model))

        vm_dataset.assign_predictions(model=vm_model)
        self.assertEqual(
            vm_dataset.prediction_column(vm_model), "predict_fn_classifier_prediction"
        )

        # Check that the predictions are assigned to the dataset
        self.assertTrue("predict_fn_classifier_prediction" in vm_dataset._df.columns)
        self.assertIsInstance(vm_dataset.y_pred(vm_model), np.ndarray)
        self.assertIsInstance(vm_dataset.y_pred_df(vm_model), pd.DataFrame)

        # Verify the actual predictions match our function logic
        expected_predictions = [0, 1, 1]  # [1+4=5 -> 0, 2+5=7 -> 1, 3+6=9 -> 1]
        np.testing.assert_array_equal(vm_dataset.y_pred(vm_model), expected_predictions)

    def test_assign_predictions_with_regression_predict_fn(self):
        """
        Test assigning predictions to dataset with a model created using predict_fn for regression
        """
        df = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6], "y": [0.1, 1.2, 2.3]})
        vm_dataset = DataFrameDataset(
            raw_dataset=df, target_column="y", feature_columns=["x1", "x2"]
        )

        # Define a simple regression predict function
        def simple_regression_fn(input_dict):
            # Simple linear combination: x1 * 0.5 + x2 * 0.3
            return input_dict["x1"] * 0.5 + input_dict["x2"] * 0.3

        vm_model = init_model(
            input_id="predict_fn_regressor", predict_fn=simple_regression_fn, __log=False
        )
        self.assertIsNone(vm_dataset.prediction_column(vm_model))

        vm_dataset.assign_predictions(model=vm_model)
        self.assertEqual(
            vm_dataset.prediction_column(vm_model), "predict_fn_regressor_prediction"
        )

        # Check that the predictions are assigned to the dataset
        self.assertTrue("predict_fn_regressor_prediction" in vm_dataset._df.columns)
        self.assertIsInstance(vm_dataset.y_pred(vm_model), np.ndarray)
        self.assertIsInstance(vm_dataset.y_pred_df(vm_model), pd.DataFrame)

        # Verify the actual predictions match our function logic
        expected_predictions = [
            1 * 0.5 + 4 * 0.3,  # 0.5 + 1.2 = 1.7
            2 * 0.5 + 5 * 0.3,  # 1.0 + 1.5 = 2.5
            3 * 0.5 + 6 * 0.3,  # 1.5 + 1.8 = 3.3
        ]
        np.testing.assert_array_almost_equal(
            vm_dataset.y_pred(vm_model), expected_predictions
        )

    def test_assign_predictions_with_complex_predict_fn(self):
        """
        Test assigning predictions to dataset with a predict_fn that returns complex outputs
        """
        df = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6], "y": [0, 1, 0]})
        vm_dataset = DataFrameDataset(
            raw_dataset=df, target_column="y", feature_columns=["x1", "x2"]
        )

        # Define a predict function that returns a dictionary
        def complex_predict_fn(input_dict):
            prediction = 1 if input_dict["x1"] + input_dict["x2"] > 5 else 0
            confidence = abs(input_dict["x1"] - input_dict["x2"]) / 10.0
            return {
                "prediction": prediction,
                "confidence": confidence,
                "feature_sum": input_dict["x1"] + input_dict["x2"],
            }

        vm_model = init_model(
            input_id="complex_predict_fn", predict_fn=complex_predict_fn, __log=False
        )

        vm_dataset.assign_predictions(model=vm_model)
        self.assertEqual(
            vm_dataset.prediction_column(vm_model), "complex_predict_fn_prediction"
        )

        # Check that the predictions and other columns are assigned to the dataset
        self.assertTrue("complex_predict_fn_prediction" in vm_dataset._df.columns)
        self.assertTrue("complex_predict_fn_confidence" in vm_dataset._df.columns)
        self.assertTrue("complex_predict_fn_feature_sum" in vm_dataset._df.columns)

        # Verify the prediction values (extracted from "prediction" key in dict)
        predictions = vm_dataset.y_pred(vm_model)
        expected_predictions = [0, 1, 1]  # [1+4=5 -> 0, 2+5=7 -> 1, 3+6=9 -> 1]
        np.testing.assert_array_equal(predictions, expected_predictions)

        # Verify other dictionary keys were added as separate columns
        confidence_values = vm_dataset._df["complex_predict_fn_confidence"].values
        expected_confidence = [0.3, 0.3, 0.3]  # |1-4|/10, |2-5|/10, |3-6|/10
        np.testing.assert_array_almost_equal(confidence_values, expected_confidence)

        feature_sum_values = vm_dataset._df["complex_predict_fn_feature_sum"].values
        expected_feature_sums = [5, 7, 9]  # 1+4, 2+5, 3+6
        np.testing.assert_array_equal(feature_sum_values, expected_feature_sums)

    def test_assign_predictions_with_multiple_predict_fn_models(self):
        """
        Test assigning predictions from multiple models created with predict_fn
        """
        df = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6], "y": [0, 1, 0]})
        vm_dataset = DataFrameDataset(
            raw_dataset=df, target_column="y", feature_columns=["x1", "x2"]
        )

        # Define two different predict functions
        def predict_fn_1(input_dict):
            return 1 if input_dict["x1"] > 1.5 else 0

        def predict_fn_2(input_dict):
            return 1 if input_dict["x2"] > 4.5 else 0

        vm_model_1 = init_model(
            input_id="predict_fn_model_1", predict_fn=predict_fn_1, __log=False
        )
        vm_model_2 = init_model(
            input_id="predict_fn_model_2", predict_fn=predict_fn_2, __log=False
        )

        vm_dataset.assign_predictions(model=vm_model_1)
        vm_dataset.assign_predictions(model=vm_model_2)

        self.assertEqual(
            vm_dataset.prediction_column(vm_model_1), "predict_fn_model_1_prediction"
        )
        self.assertEqual(
            vm_dataset.prediction_column(vm_model_2), "predict_fn_model_2_prediction"
        )

        # Check that both prediction columns exist
        self.assertTrue("predict_fn_model_1_prediction" in vm_dataset._df.columns)
        self.assertTrue("predict_fn_model_2_prediction" in vm_dataset._df.columns)

        # Verify predictions are different based on the different logic
        predictions_1 = vm_dataset.y_pred(vm_model_1)
        predictions_2 = vm_dataset.y_pred(vm_model_2)

        expected_predictions_1 = [0, 1, 1]  # x1 > 1.5: [1 -> 0, 2 -> 1, 3 -> 1]
        expected_predictions_2 = [0, 1, 1]  # x2 > 4.5: [4 -> 0, 5 -> 1, 6 -> 1]

        np.testing.assert_array_equal(predictions_1, expected_predictions_1)
        np.testing.assert_array_equal(predictions_2, expected_predictions_2)

    def test_assign_predictions_with_predict_fn_and_prediction_values(self):
        """
        Test assigning predictions with predict_fn model but using pre-computed prediction values
        """
        df = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6], "y": [0, 1, 0]})
        vm_dataset = DataFrameDataset(
            raw_dataset=df, target_column="y", feature_columns=["x1", "x2"]
        )

        # Define a predict function
        def predict_fn(input_dict):
            return 1 if input_dict["x1"] + input_dict["x2"] > 5 else 0

        vm_model = init_model(
            input_id="predict_fn_with_values", predict_fn=predict_fn, __log=False
        )

        # Pre-computed predictions (different from what the function would return)
        precomputed_predictions = [1, 0, 1]

        with patch.object(vm_model, "predict") as mock_predict:
            vm_dataset.assign_predictions(
                model=vm_model, prediction_values=precomputed_predictions
            )
            # The model's predict method should not be called
            mock_predict.assert_not_called()

        self.assertEqual(
            vm_dataset.prediction_column(vm_model), "predict_fn_with_values_prediction"
        )

        # Check that the precomputed predictions are used
        self.assertTrue("predict_fn_with_values_prediction" in vm_dataset._df.columns)
        np.testing.assert_array_equal(
            vm_dataset.y_pred(vm_model), precomputed_predictions
        )

    def test_assign_predictions_with_invalid_predict_fn(self):
        """
        Test assigning predictions with an invalid predict_fn (should raise error during model creation)
        """
        # Try to create a model with a non-callable predict_fn
        with self.assertRaises(ValueError) as context:
            init_model(input_id="invalid_predict_fn", predict_fn="not_a_function", __log=False)

        self.assertIn("FunctionModel requires a callable predict_fn", str(context.exception))


if __name__ == "__main__":
    unittest.main()
