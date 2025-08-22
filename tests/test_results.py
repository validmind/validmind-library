import asyncio
import unittest
from unittest.mock import patch
import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import HTML, VBox

from validmind.vm_models.result import (
    TestResult,
    ErrorResult,
    TextGenerationResult,
    ResultTable,
    RawData,
    UnitMetricValue,
    RowMetricValues,
)

from validmind.vm_models.figure import Figure
from validmind.errors import InvalidParameterError

loop = asyncio.new_event_loop()


class MockAsyncResponse:
    def __init__(self, status, text=None, json_data=None):
        self.status = status
        self.status_code = status
        self._text = text
        self._json_data = json_data

    async def text(self):
        return self._text

    async def json(self):
        return self._json_data

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def __aenter__(self):
        return self


class TestResultClasses(unittest.TestCase):
    def tearDownClass():
        loop.close()

    def run_async(self, func, *args, **kwargs):
        return loop.run_until_complete(func(*args, **kwargs))

    def test_raw_data_initialization(self):
        """Test RawData initialization and methods"""
        raw_data = RawData(log=True, dataset_duplicates=pd.DataFrame({'col1': [1, 2]}))

        self.assertTrue(raw_data.log)
        self.assertIsInstance(raw_data.dataset_duplicates, pd.DataFrame)
        self.assertEqual(raw_data.__repr__(), "RawData(log, dataset_duplicates)")

    def test_result_table_initialization(self):
        """Test ResultTable initialization and methods"""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        table = ResultTable(data=df, title="Test Table")

        self.assertEqual(table.title, "Test Table")
        self.assertIsInstance(table.data, pd.DataFrame)
        self.assertEqual(table.__repr__(), 'ResultTable(title="Test Table")')

    def test_error_result(self):
        """Test ErrorResult initialization and methods"""
        error = ValueError("Test error")
        error_result = ErrorResult(
            result_id="test_error",
            error=error,
            message="Test error message"
        )

        self.assertEqual(error_result.name, "Failed Test")
        self.assertEqual(error_result.error, error)
        self.assertEqual(error_result.message, "Test error message")

        widget = error_result.to_widget()
        self.assertIsInstance(widget, HTML)

    def test_test_result_initialization(self):
        """Test TestResult initialization and basic methods"""
        test_result = TestResult(
            result_id="test_1",
            name="Test 1",
            description="Test description",
            metric=0.95,
            passed=True
        )

        self.assertEqual(test_result.result_id, "test_1")
        self.assertEqual(test_result.name, "Test 1")
        self.assertEqual(test_result.description, "Test description")
        self.assertEqual(test_result.metric, 0.95)
        self.assertTrue(test_result.passed)

    def test_test_result_add_table(self):
        """Test adding tables to TestResult"""
        test_result = TestResult(result_id="test_1")
        df = pd.DataFrame({'col1': [1, 2, 3]})

        test_result.add_table(df, title="Test Table")
        self.assertEqual(len(test_result.tables), 1)
        self.assertEqual(test_result.tables[0].title, "Test Table")

    def test_test_result_add_figure(self):
        """Test adding figures to TestResult"""
        test_result = TestResult(result_id="test_1")
        fig = plt.figure()
        plt.plot([1, 2, 3])
        
        test_result.add_figure(fig)
        self.assertEqual(len(test_result.figures), 1)
        self.assertIsInstance(test_result.figures[0], Figure)

    def test_test_result_remove_table(self):
        """Test removing tables from TestResult"""
        test_result = TestResult(result_id="test_1")
        df = pd.DataFrame({'col1': [1, 2, 3]})

        test_result.add_table(df)
        test_result.remove_table(0)
        self.assertEqual(len(test_result.tables), 0)

    def test_test_result_remove_figure(self):
        """Test removing figures from TestResult"""
        test_result = TestResult(result_id="test_1")
        fig = plt.figure()
        plt.plot([1, 2, 3])
        
        test_result.add_figure(fig)
        test_result.remove_figure(0)
        self.assertEqual(len(test_result.figures), 0)

    def test_test_result_serialize(self):
        """Test TestResult serialization"""
        test_result = TestResult(
            result_id="test_1",
            title="Test Title",
            ref_id="ref_1",
            params={"param1": 1},
            passed=True,
            inputs={}  # Initialize empty inputs dictionary
        )
        
        serialized = test_result.serialize()
        self.assertEqual(serialized["test_name"], "test_1")
        self.assertEqual(serialized["title"], "Test Title")
        self.assertEqual(serialized["ref_id"], "ref_1")
        self.assertEqual(serialized["params"], {"param1": 1})
        self.assertTrue(serialized["passed"])
        self.assertEqual(serialized["inputs"], [])  # Empty inputs list

    @patch("validmind.api_client.alog_test_result")
    @patch("validmind.api_client.alog_figure")
    @patch("validmind.api_client.alog_metric")
    async def test_test_result_log_async(self, mock_metric, mock_figure, mock_test_result):
        """Test async logging of TestResult"""
        mock_test_result.return_value = MockAsyncResponse(200, json={"cuid": "123"})
        mock_figure.return_value = MockAsyncResponse(200, json={"cuid": "456"})
        mock_metric.return_value = MockAsyncResponse(200, json={"cuid": "789"})

        test_result = TestResult(
            result_id="test_1",
            metric=0.95,
            description="Test description"
        )
       
        await test_result.log_async(section_id="section_1", position=0)
       
        mock_test_result.assert_called_once()
        mock_metric.assert_called_once()

    def test_text_generation_result(self):
        """Test TextGenerationResult initialization and methods"""
        text_result = TextGenerationResult(
            result_id="text_1",
            title="Text Test",
            description="Generated text"
        )
        
        self.assertEqual(text_result.name, "Text Generation Result")
        self.assertEqual(text_result.title, "Text Test")
        self.assertEqual(text_result.description, "Generated text")
        
        widget = text_result.to_widget()
        self.assertIsInstance(widget, VBox)

    def test_validate_log_config(self):
        """Test validation of log configuration"""
        test_result = TestResult(result_id="test_1")
        
        # Test valid config
        valid_config = {
            "hideTitle": True,
            "hideText": False,
            "hideParams": True,
            "hideTables": False,
            "hideFigures": True
        }
        test_result.validate_log_config(valid_config)  # Should not raise exception
        
        # Test invalid keys
        invalid_config = {"invalidKey": True}
        with self.assertRaises(InvalidParameterError):
            test_result.validate_log_config(invalid_config)
        
        # Test non-boolean values
        invalid_type_config = {"hideTitle": "true"}
        with self.assertRaises(InvalidParameterError):
            test_result.validate_log_config(invalid_type_config)

    @patch("validmind.api_client.update_metadata")
    async def test_metadata_update_content_id_handling(self, mock_update_metadata):
        """Test metadata update with different content_id scenarios"""
        # Test case 1: With content_id
        test_result = TestResult(
            result_id="test_1",
            description="Test description",
            _was_description_generated=False
        )
        await test_result.log_async(content_id="custom_content_id")
        mock_update_metadata.assert_called_with(
            content_id="custom_content_id::default",
            text="Test description"
        )

        # Test case 2: Without content_id
        mock_update_metadata.reset_mock()
        await test_result.log_async()
        mock_update_metadata.assert_called_with(
            content_id="test_description:test_1::default",
            text="Test description"
        )

        # Test case 3: With AI generated description
        test_result._was_description_generated = True
        mock_update_metadata.reset_mock()
        await test_result.log_async()
        mock_update_metadata.assert_called_with(
            content_id="test_description:test_1::ai",
            text="Test description"
        )

    def test_metric_values_initialization_scalar(self):
        """Test UnitMetricValue initialization with scalar values"""
        # Test integer
        mv_int = UnitMetricValue(42)
        self.assertEqual(mv_int.get_values(), 42)
        self.assertTrue(mv_int.is_scalar())
        self.assertFalse(mv_int.is_list())
        self.assertEqual(mv_int.get_metric_type(), "unit_metric")

        # Test float
        mv_float = UnitMetricValue(3.14)
        self.assertEqual(mv_float.get_values(), 3.14)
        self.assertTrue(mv_float.is_scalar())
        self.assertFalse(mv_float.is_list())
        self.assertEqual(mv_float.get_metric_type(), "unit_metric")

    def test_metric_values_initialization_list(self):
        """Test RowMetricValues initialization with list values"""
        # Test list of mixed numeric types
        mv_list = RowMetricValues([1, 2.5, 3, 4.0])
        self.assertEqual(mv_list.get_values(), [1, 2.5, 3, 4.0])
        self.assertFalse(mv_list.is_scalar())
        self.assertTrue(mv_list.is_list())
        self.assertEqual(mv_list.get_metric_type(), "row_metrics")

        # Test empty list
        mv_empty = RowMetricValues([])
        self.assertEqual(mv_empty.get_values(), [])
        self.assertFalse(mv_empty.is_scalar())
        self.assertTrue(mv_empty.is_list())
        self.assertEqual(mv_empty.get_metric_type(), "row_metrics")

    def test_metric_values_validation_valid(self):
        """Test metric values validation with valid inputs"""
        # These should not raise any exceptions
        UnitMetricValue(42)
        UnitMetricValue(3.14)
        RowMetricValues([1, 2, 3])
        RowMetricValues([1.1, 2.2, 3.3])
        RowMetricValues([])
        RowMetricValues([42])

    def test_metric_values_validation_invalid_types(self):
        """Test metric values validation with invalid types"""
        invalid_values = [
            "string",
            {"key": "value"},
            None,
            [1, 2, "invalid"],
            [1, None, 3],
            [1, {"key": "val"}, 3],
        ]

        for invalid_value in invalid_values:
            with self.assertRaises(ValueError):
                if isinstance(invalid_value, list):
                    RowMetricValues(invalid_value)
                else:
                    UnitMetricValue(invalid_value)

    def test_metric_values_validation_boolean_rejection(self):
        """Test metric values rejection of boolean values"""
        # Boolean scalars should be rejected
        with self.assertRaises(ValueError) as context:
            UnitMetricValue(True)
        self.assertIn("Boolean values are not allowed", str(context.exception))

        with self.assertRaises(ValueError) as context:
            UnitMetricValue(False)
        self.assertIn("Boolean values are not allowed", str(context.exception))

        # Boolean in lists should be rejected
        with self.assertRaises(ValueError) as context:
            RowMetricValues([1, True, 3])
        self.assertIn("Boolean values are not allowed in metric value lists", str(context.exception))

        with self.assertRaises(ValueError) as context:
            RowMetricValues([False, 1, 2])
        self.assertIn("Boolean values are not allowed in metric value lists", str(context.exception))

    def test_metric_values_string_representation(self):
        """Test metric values string representation methods"""
        # Scalar representation
        mv_scalar = UnitMetricValue(42)
        self.assertEqual(str(mv_scalar), "42")
        self.assertEqual(repr(mv_scalar), "UnitMetricValue(42)")

        # List representation
        mv_list = RowMetricValues([1, 2, 3])
        self.assertEqual(str(mv_list), "[1, 2, 3]")
        self.assertEqual(repr(mv_list), "RowMetricValues([3 values])")

        # Empty list representation
        mv_empty = RowMetricValues([])
        self.assertEqual(str(mv_empty), "[]")
        self.assertEqual(repr(mv_empty), "RowMetricValues([0 values])")

    def test_metric_values_equality(self):
        """Test metric values equality comparison"""
        # Scalar equality
        mv1 = UnitMetricValue(42)
        mv2 = UnitMetricValue(42)
        mv3 = UnitMetricValue(43)

        self.assertEqual(mv1, mv2)
        self.assertNotEqual(mv1, mv3)
        self.assertEqual(mv1, 42)  # Equality with raw value
        self.assertNotEqual(mv1, 43)

        # List equality
        mv_list1 = RowMetricValues([1, 2, 3])
        mv_list2 = RowMetricValues([1, 2, 3])
        mv_list3 = RowMetricValues([1, 2, 4])

        self.assertEqual(mv_list1, mv_list2)
        self.assertNotEqual(mv_list1, mv_list3)
        self.assertEqual(mv_list1, [1, 2, 3])  # Equality with raw list
        self.assertNotEqual(mv_list1, [1, 2, 4])

    def test_metric_values_serialization(self):
        """Test metric values serialization"""
        # Scalar serialization
        mv_scalar = UnitMetricValue(42)
        self.assertEqual(mv_scalar.serialize(), 42)

        # List serialization
        mv_list = RowMetricValues([1, 2.5, 3])
        self.assertEqual(mv_list.serialize(), [1, 2.5, 3])

        # Empty list serialization
        mv_empty = RowMetricValues([])
        self.assertEqual(mv_empty.serialize(), [])

    def test_test_result_metric_values_integration(self):
        """Test metric values integration with TestResult"""
        test_result = TestResult(result_id="test_metric_values")

        # Test setting metric with scalar using set_metric
        test_result.set_metric(0.85)
        self.assertIsInstance(test_result.metric, UnitMetricValue)
        self.assertIsNone(test_result.row_metric)
        self.assertEqual(test_result.metric.get_values(), 0.85)
        self.assertEqual(test_result._get_metric_display_value(), 0.85)
        self.assertEqual(test_result._get_metric_serialized_value(), 0.85)

        # Test setting metric with list using set_metric
        test_result.set_metric([0.1, 0.2, 0.3])
        self.assertIsInstance(test_result.row_metric, RowMetricValues)
        self.assertIsNone(test_result.metric)
        self.assertEqual(test_result.row_metric.get_values(), [0.1, 0.2, 0.3])
        self.assertEqual(test_result._get_metric_display_value(), [0.1, 0.2, 0.3])
        self.assertEqual(test_result._get_metric_serialized_value(), [0.1, 0.2, 0.3])

        # Test setting metric with MetricValues object directly
        mv = UnitMetricValue(99.9)
        test_result.set_metric(mv)
        self.assertIs(test_result.metric, mv)
        self.assertIsNone(test_result.row_metric)
        self.assertEqual(test_result._get_metric_display_value(), 99.9)
        self.assertEqual(test_result._get_metric_serialized_value(), 99.9)

        # Test setting RowMetricValues object directly
        rmv = RowMetricValues([1.0, 2.0, 3.0])
        test_result.set_metric(rmv)
        self.assertIs(test_result.row_metric, rmv)
        self.assertIsNone(test_result.metric)
        self.assertEqual(test_result._get_metric_display_value(), [1.0, 2.0, 3.0])
        self.assertEqual(test_result._get_metric_serialized_value(), [1.0, 2.0, 3.0])

    def test_test_result_metric_type_detection(self):
        """Test metric type detection for both metric and row_metric fields"""
        test_result = TestResult(result_id="test_metric_type")
        
        # Test unit metric type
        test_result.set_metric(42.0)
        self.assertEqual(test_result._get_metric_type(), "unit_metric")
        
        # Test row metric type
        test_result.set_metric([1.0, 2.0, 3.0])
        self.assertEqual(test_result._get_metric_type(), "row_metrics")
        
        # Test with MetricValues objects
        test_result.set_metric(UnitMetricValue(99.9))
        self.assertEqual(test_result._get_metric_type(), "unit_metric")
        
        test_result.set_metric(RowMetricValues([4.0, 5.0]))
        self.assertEqual(test_result._get_metric_type(), "row_metrics")
        
        # Test with no metric
        test_result.metric = None
        test_result.row_metric = None
        self.assertIsNone(test_result._get_metric_type())

    def test_test_result_backward_compatibility(self):
        """Test backward compatibility with direct metric assignment"""
        test_result = TestResult(result_id="test_backward_compat")

        # Direct assignment of raw values (old style)
        test_result.metric = 42.0
        self.assertEqual(test_result._get_metric_display_value(), 42.0)
        self.assertEqual(test_result._get_metric_serialized_value(), 42.0)

        # Direct assignment of list (old style)
        test_result.metric = [1.0, 2.0, 3.0]
        self.assertEqual(test_result._get_metric_display_value(), [1.0, 2.0, 3.0])
        self.assertEqual(test_result._get_metric_serialized_value(), [1.0, 2.0, 3.0])

        # Mixed usage - set with set_metric then access display value
        test_result.set_metric(100)
        self.assertIsInstance(test_result.metric, UnitMetricValue)
        self.assertEqual(test_result._get_metric_display_value(), 100)

    def test_test_result_metric_values_widget_display(self):
        """Test MetricValues display in TestResult widgets"""
        # Test scalar metric display
        test_result_scalar = TestResult(result_id="test_scalar_widget")
        test_result_scalar.set_metric(0.95)

        widget_scalar = test_result_scalar.to_widget()
        self.assertIsInstance(widget_scalar, HTML)
        # Check that the metric value appears in the HTML
        self.assertIn("0.95", widget_scalar.value)

        # Test list metric display
        test_result_list = TestResult(result_id="test_list_widget")
        test_result_list.set_metric([0.1, 0.2, 0.3])

        widget_list = test_result_list.to_widget()
        # Even with lists, when no tables/figures exist, it returns HTML
        self.assertIsInstance(widget_list, HTML)
        # Check that the list values appear in the HTML
        self.assertIn("[0.1, 0.2, 0.3]", widget_list.value)

    def test_metric_values_edge_cases(self):
        """Test metric values edge cases"""
        # Test with very large numbers
        large_num = 1e10
        mv_large = UnitMetricValue(large_num)
        self.assertEqual(mv_large.get_values(), large_num)

        # Test with very small numbers
        small_num = 1e-10
        mv_small = UnitMetricValue(small_num)
        self.assertEqual(mv_small.get_values(), small_num)

        # Test with negative numbers
        negative_num = -42.5
        mv_negative = UnitMetricValue(negative_num)
        self.assertEqual(mv_negative.get_values(), negative_num)

        # Test with zero
        mv_zero = UnitMetricValue(0)
        self.assertEqual(mv_zero.get_values(), 0)

        # Test with list containing zeros and negatives
        mixed_list = [0, -1, 2.5, -3.14]
        mv_mixed = RowMetricValues(mixed_list)
        self.assertEqual(mv_mixed.get_values(), mixed_list)

    def test_metric_values_type_consistency(self):
        """Test that metric values maintain type consistency"""
        # Integer input should remain integer
        mv_int = UnitMetricValue(42)
        self.assertIsInstance(mv_int.get_values(), int)
        self.assertIsInstance(mv_int.serialize(), int)

        # Float input should remain float
        mv_float = UnitMetricValue(3.14)
        self.assertIsInstance(mv_float.get_values(), float)
        self.assertIsInstance(mv_float.serialize(), float)

        # List input should remain list
        mv_list = RowMetricValues([1, 2, 3])
        self.assertIsInstance(mv_list.get_values(), list)
        self.assertIsInstance(mv_list.serialize(), list)


if __name__ == "__main__":
    unittest.main()
