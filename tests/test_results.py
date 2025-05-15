import asyncio
import json
import unittest
from unittest.mock import MagicMock, Mock, patch
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from ipywidgets import HTML, VBox

from validmind.vm_models.result import (
    Result,
    TestResult,
    ErrorResult,
    TextGenerationResult,
    ResultTable,
    RawData,
)
from validmind.vm_models.figure import Figure
from validmind.errors import InvalidParameterError
from validmind.ai.utils import DescriptionFuture

loop = asyncio.new_event_loop()

class MockAsyncResponse:
    def __init__(self, status, text=None, json=None):
        self.status = status
        self.status_code = status
        self._text = text
        self._json = json

    async def text(self):
        return self._text

    async def json(self):
        return self._json

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

if __name__ == "__main__":
    unittest.main() 