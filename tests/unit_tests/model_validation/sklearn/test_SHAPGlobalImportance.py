import unittest

import numpy as np

from validmind.tests.model_validation.sklearn.SHAPGlobalImportance import (
    select_shap_values,
)


class TestSelectShapValues(unittest.TestCase):
    def test_binary_or_regression_2d_array_passthrough(self):
        # A single (samples, features) array (binary or regression) is used as-is.
        values = np.arange(6.0).reshape(3, 2)
        np.testing.assert_array_equal(select_shap_values(values), values)

    def test_list_binary_defaults_to_class_one(self):
        class0 = np.zeros((3, 2))
        class1 = np.ones((3, 2))
        np.testing.assert_array_equal(select_shap_values([class0, class1]), class1)

    def test_list_multiclass_selects_requested_class(self):
        classes = [np.full((3, 2), i, dtype=float) for i in range(5)]
        np.testing.assert_array_equal(
            select_shap_values(classes, class_of_interest=3),
            np.full((3, 2), 3.0),
        )

    def test_3d_array_multiclass_selects_trailing_class_axis(self):
        # Newer SHAP returns (samples, features, classes); regression test for the
        # IndexError raised when this array was passed straight to summary_plot.
        samples, features, classes = 4, 2, 5
        values = np.arange(samples * features * classes, dtype=float).reshape(
            samples, features, classes
        )
        selected = select_shap_values(values, class_of_interest=2)
        self.assertEqual(selected.shape, (samples, features))
        np.testing.assert_array_equal(selected, values[:, :, 2])

    def test_3d_array_multiclass_without_class_raises_clear_error(self):
        values = np.zeros((4, 2, 5))
        with self.assertRaises(ValueError) as ctx:
            select_shap_values(values)
        self.assertIn("class_of_interest", str(ctx.exception))

    def test_class_of_interest_out_of_bounds_raises(self):
        values = np.zeros((4, 2, 3))
        with self.assertRaises(ValueError):
            select_shap_values(values, class_of_interest=9)


if __name__ == "__main__":
    unittest.main()
