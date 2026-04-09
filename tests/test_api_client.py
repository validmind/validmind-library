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
from validmind.__version__ import __version__
from validmind.errors import (
    APIRequestError,
    MissingAPICredentialsError,
    MissingModelIdError,
)
from validmind.utils import md_to_html
from validmind.vm_models.figure import Figure


loop = asyncio.new_event_loop()


def mock_figure():
    fig = plt.figure()
    plt.plot([1, 2, 3])
    return Figure(key="key", figure=fig, ref_id="asdf")


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
            "model": {"name": "test_model", "cuid": os.environ["VM_API_MODEL"]}
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
                "X-LIBRARY-VERSION": __version__,
            },
        )

    @patch("validmind.api_client.logger.error")
    @patch("requests.get")
    def test_init_warns_when_document_is_missing(
        self, mock_requests_get, mock_logger_error
    ):
        mock_data = {
            "model": {"name": "test_model", "cuid": os.environ["VM_API_MODEL"]}
        }
        mock_response = Mock(status_code=200, json=Mock(return_value=mock_data))
        mock_requests_get.return_value = mock_response

        api_client.init()

        mock_logger_error.assert_called_once_with(
            "Future releases will require `document` as one of the options you must provide to `vm.init()`. "
            "To learn more, refer to https://docs.validmind.ai/developer/validmind-library.html"
        )

    @patch("validmind.api_client.logger.error")
    @patch("requests.get")
    def test_init_no_warning_when_document_is_passed(
        self, mock_requests_get, mock_logger_error
    ):
        mock_data = {
            "model": {"name": "test_model", "cuid": os.environ["VM_API_MODEL"]}
        }
        mock_response = Mock(status_code=200, json=Mock(return_value=mock_data))
        mock_requests_get.return_value = mock_response

        api_client.init(document="documentation")

        mock_logger_error.assert_not_called()
        mock_requests_get.assert_called_once_with(
            url=f"{os.environ['VM_API_HOST']}/ping",
            headers={
                "X-API-KEY": os.environ["VM_API_KEY"],
                "X-API-SECRET": os.environ["VM_API_SECRET"],
                "X-MODEL-CUID": os.environ["VM_API_MODEL"],
                "X-MONITORING": "False",
                "X-LIBRARY-VERSION": __version__,
                "X-DOCUMENT-TYPE": "documentation",
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

        model = os.environ.pop("VM_API_MODEL")
        with self.assertRaises(MissingModelIdError):
            api_client.init(model=None)

        os.environ["VM_API_MODEL"] = model

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
                "X-LIBRARY-VERSION": __version__,
            },
        )

    @patch("aiohttp.ClientSession.post")
    def test_log_figure_matplot(self, mock_post: MagicMock):
        mock_post.return_value = MockAsyncResponse(200, json={"cuid": "1234"})

        self.run_async(api_client.alog_figure, mock_figure())

        url = f"{os.environ['VM_API_HOST']}/log_figure"
        mock_post.assert_called_once()
        self.assertEqual(mock_post.call_args[0][0], url)
        self.assertIsInstance(mock_post.call_args[1]["data"], FormData)

    @patch("aiohttp.ClientSession.post")
    def test_log_metadata(self, mock_post: MagicMock):
        mock_post.return_value = MockAsyncResponse(200, json={"cuid": "abc1234"})

        self.run_async(
            api_client.alog_metadata,
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
    def test_log_metadata_with_section_id(self, mock_post: MagicMock):
        mock_post.return_value = MockAsyncResponse(200, json={"cuid": "abc1234"})

        self.run_async(
            api_client.alog_metadata,
            "1234",
            text="Some Text",
            section_id="intended_use",
        )

        mock_post.assert_called_with(
            f"{os.environ['VM_API_HOST']}/log_metadata?section_id=intended_use",
            data=json.dumps(
                {
                    "content_id": "1234",
                    "text": "Some Text",
                }
            ),
        )

    @patch("aiohttp.ClientSession.post")
    def test_log_test_result(self, mock_post):
        result = {
            "test_name": "test_name",
            "ref_id": "asdf",
            "params": {"a": 1},
            "inputs": ["input1"],
            "passed": True,
            "summary": [{"key": "value"}],
            "config": None,
        }

        mock_post.return_value = MockAsyncResponse(200, json={"cuid": "abc1234"})

        self.run_async(api_client.alog_test_result, result)

        url = f"{os.environ['VM_API_HOST']}/log_test_results"

        mock_post.assert_called_with(url, data=json.dumps(result))

    @patch("requests.post")
    @patch("aiohttp.ClientSession.post")
    def test_log_text_generates_text_and_logs_metadata(
        self, mock_aiohttp_post, mock_requests_post
    ):
        mock_requests_post.return_value = Mock(status_code=200)
        mock_requests_post.return_value.json.return_value = {
            "content": "## Generated Summary\nGenerated content."
        }
        mock_aiohttp_post.return_value = MockAsyncResponse(
            200,
            json={
                "content_id": "dataset_summary_text",
                "text": md_to_html("## Generated Summary\nGenerated content.", mathml=True),
            },
        )

        api_client.log_text(
            content_id="dataset_summary_text",
            prompt="Summarize the dataset.",
            context={"content_ids": ["train_dataset", "target_description_text"]},
        )

        mock_requests_post.assert_called_once_with(
            url=f"{os.environ['VM_API_HOST']}/ai/generate/qualitative_text_generation",
            headers={
                "X-API-KEY": os.environ["VM_API_KEY"],
                "X-API-SECRET": os.environ["VM_API_SECRET"],
                "X-MODEL-CUID": os.environ["VM_API_MODEL"],
                "X-MONITORING": "False",
                "X-LIBRARY-VERSION": __version__,
            },
            json={
                "content_id": "dataset_summary_text",
                "generate": True,
                "prompt": "Summarize the dataset.",
                "context": {
                    "content_ids": ["train_dataset", "target_description_text"]
                },
            },
        )
        mock_aiohttp_post.assert_called_once_with(
            f"{os.environ['VM_API_HOST']}/log_metadata",
            data=json.dumps(
                {
                    "content_id": "dataset_summary_text",
                    "text": md_to_html(
                        "## Generated Summary\nGenerated content.", mathml=True
                    ),
                }
            ),
        )

    @patch("requests.post")
    @patch("aiohttp.ClientSession.post")
    def test_log_text_logs_metadata_with_section_id(
        self, mock_aiohttp_post, mock_requests_post
    ):
        mock_requests_post.return_value = Mock(status_code=200)
        mock_requests_post.return_value.json.return_value = {
            "content": "Generated content."
        }
        mock_aiohttp_post.return_value = MockAsyncResponse(
            200,
            json={
                "content_id": "dataset_summary_text",
                "text": "Generated content.",
            },
        )

        api_client.log_text(
            content_id="dataset_summary_text",
            prompt="Summarize the dataset.",
            section_id="intended_use",
        )

        mock_requests_post.assert_called_once_with(
            url=f"{os.environ['VM_API_HOST']}/ai/generate/qualitative_text_generation",
            headers={
                "X-API-KEY": os.environ["VM_API_KEY"],
                "X-API-SECRET": os.environ["VM_API_SECRET"],
                "X-MODEL-CUID": os.environ["VM_API_MODEL"],
                "X-MONITORING": "False",
                "X-LIBRARY-VERSION": __version__,
            },
            json={
                "content_id": "dataset_summary_text",
                "generate": True,
                "prompt": "Summarize the dataset.",
                "section_id": "intended_use",
            },
        )
        mock_aiohttp_post.assert_called_once_with(
            f"{os.environ['VM_API_HOST']}/log_metadata?section_id=intended_use",
            data=json.dumps(
                {
                    "content_id": "dataset_summary_text",
                    "text": md_to_html("Generated content.", mathml=True),
                }
            ),
        )

    def test_log_text_rejects_prompt_when_text_is_provided(self):
        with self.assertRaisesRegex(
            ValueError, "`prompt` is only supported when `text` is omitted"
        ):
            api_client.log_text(
                content_id="dataset_summary_text",
                text="Hello world",
                prompt="Ignore the provided text.",
            )

    def test_log_text_rejects_invalid_context(self):
        with self.assertRaisesRegex(
            ValueError,
            "`context\\['content_ids'\\]` must contain only non-empty strings",
        ):
            api_client.log_text(
                content_id="dataset_summary_text",
                context={"content_ids": ["valid", ""]},
            )


if __name__ == "__main__":
    unittest.main()
