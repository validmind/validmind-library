import asyncio
import json
import os
import unittest
from unittest.mock import MagicMock, Mock, patch

import matplotlib.pyplot as plt
from aiohttp.formdata import FormData

# simluate environment variables being set
os.environ["VM_API_KEY"] = "your_api_key"
os.environ["VM_API_SECRET"] = "your_api_secret"
os.environ["VM_API_HOST"] = "your_api_host"
os.environ["VM_API_MODEL"] = "your_model"

import validmind.api_client as api_client
from validmind.errors import (
    MissingAPICredentialsError,
    MissingModelIdError,
    APIRequestError,
)
from validmind.utils import md_to_html
from validmind.vm_models.figure import Figure


loop = asyncio.new_event_loop()


def mock_figure():
    fig = plt.figure()
    plt.plot([1, 2, 3])
    return Figure(key="key", figure=fig, metadata={"asdf": 1234})


class MockResponse:
    def __init__(self, status, text=None, json=None):
        self.status = status
        self.status_code = status
        self.text = text
        self._json = json

    def json(self):
        return self._json


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


class TestAPIClient(unittest.TestCase):
    def tearDownClass():
        loop.close()

    def run_async(self, func, *args, **kwargs):
        return loop.run_until_complete(func(*args, **kwargs))

    @patch("requests.get")
    def test_init_successful(self, mock_requests_get):
        mock_data = {
            "project": {"name": "test_project", "cuid": os.environ["VM_API_MODEL"]}
        }
        mock_response = Mock(status_code=200, json=Mock(return_value=mock_data))
        mock_requests_get.return_value = mock_response

        success = api_client.init()
        self.assertIsNone(success)

        mock_requests_get.assert_called_once_with(
            url=f"{os.environ['VM_API_HOST']}/ping",
            headers={
                "X-API-KEY": os.environ["VM_API_KEY"],
                "X-API-SECRET": os.environ["VM_API_SECRET"],
                "X-MODEL-CUID": os.environ["VM_API_MODEL"],
                "X-MONITORING": "False",
            },
        )

    def test_get_api_host(self):
        host = api_client.get_api_host()
        self.assertEqual(host, "your_api_host")

    def test_get_api_model(self):
        model = api_client.get_api_model()
        self.assertEqual(model, "your_model")

    @patch("requests.get")
    def test_init_missing_model_id(self, mock_requests_get):
        mock_requests_get.return_value = Mock()

        project = os.environ.pop("VM_API_MODEL")
        with self.assertRaises(MissingModelIdError):
            api_client.init(model=None)

        os.environ["VM_API_MODEL"] = project

        mock_requests_get.assert_not_called()

    @patch("requests.get")
    def test_init_missing_api_key_secret(self, mock_get):
        mock_get.return_value = Mock()

        api_key = os.environ.pop("VM_API_KEY")
        api_secret = os.environ.pop("VM_API_SECRET")

        with self.assertRaises(MissingAPICredentialsError):
            api_client.init(model="model_id", api_key=None, api_secret=None)

        os.environ["VM_API_KEY"] = api_key
        os.environ["VM_API_SECRET"] = api_secret

        mock_get.assert_not_called()

    @patch("requests.get")
    def test_init_unsuccessful_ping(self, mock_get):
        mock_get.return_value = MockResponse(500, text="Internal Server Error")

        with self.assertRaises(Exception) as cm:
            api_client.init()

        self.assertIsInstance(cm.exception, APIRequestError)

        mock_get.assert_called_once_with(
            url=f"{os.environ['VM_API_HOST']}/ping",
            headers={
                "X-API-KEY": os.environ["VM_API_KEY"],
                "X-API-SECRET": os.environ["VM_API_SECRET"],
                "X-MODEL-CUID": os.environ["VM_API_MODEL"],
                "X-MONITORING": "False",
            },
        )

    @patch("aiohttp.ClientSession.get")
    def test_get_metadata(self, mock_get: MagicMock):
        res_json = [{"cuid": "1234"}]
        mock_get.return_value = MockAsyncResponse(200, json=res_json)

        response = self.run_async(api_client.get_metadata, "content_id")

        url = f"{os.environ['VM_API_HOST']}/get_metadata/content_id"
        url += f""
        mock_get.assert_called_with(url)

        self.assertEqual(response, res_json)

    @patch("aiohttp.ClientSession.post")
    def test_log_figure_matplot(self, mock_post: MagicMock):
        mock_post.return_value = MockAsyncResponse(200, json={"cuid": "1234"})

        self.run_async(api_client.log_figure, mock_figure())

        url = f"{os.environ['VM_API_HOST']}/log_figure"
        mock_post.assert_called_once()
        self.assertEqual(mock_post.call_args[0][0], url)
        self.assertIsInstance(mock_post.call_args[1]["data"], FormData)

    @patch("aiohttp.ClientSession.post")
    def test_log_metadata(self, mock_post: MagicMock):
        mock_post.return_value = MockAsyncResponse(200, json={"cuid": "abc1234"})

        self.run_async(
            api_client.log_metadata,
            "1234",
            text="Some Text",
            _json={"key": "value"},
        )

        url = f"{os.environ['VM_API_HOST']}/log_metadata"
        mock_post.assert_called_with(
            url,
            data=json.dumps(
                {
                    "content_id": "1234",
                    "text": "Some Text",
                    "json": {"key": "value"},
                }
            ),
        )

    @patch("aiohttp.ClientSession.post")
    def test_log_metric_result(self, mock_post):
        metric = Mock(serialize=MagicMock(return_value={"key": "value"}))

        mock_post.return_value = MockAsyncResponse(200, json={"cuid": "abc1234"})

        self.run_async(api_client.log_metric_result, metric, inputs=["input1"])

        url = f"{os.environ['VM_API_HOST']}/log_metrics"
        mock_post.assert_called_with(
            url, data=json.dumps([{"key": "value", "inputs": ["input1"]}])
        )

    @patch("aiohttp.ClientSession.post")
    def test_log_test_result(self, mock_post):
        result = Mock(serialize=MagicMock(return_value={"key": "value"}))

        mock_post.return_value = MockAsyncResponse(200, json={"cuid": "abc1234"})

        self.run_async(api_client.log_test_result, result, ["input1"])

        url = f"{os.environ['VM_API_HOST']}/log_test_results"

        mock_post.assert_called_with(
            url, data=json.dumps({"key": "value", "inputs": ["input1"]})
        )


if __name__ == "__main__":
    unittest.main()
