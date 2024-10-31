"""
Unit tests for the framewor's init() method
"""

import os
import unittest

from unittest import mock

import validmind as vm
from validmind.errors import MissingAPICredentialsError, MissingModelIdError

INVALID_CREDENTIALS_JSON_RESPONSE = {
    "code": "invalid_credentials",
    "description": "Invalid API credentials",
}

INVALID_PROJECT_JSON_RESPONSE = {
    "code": "invalid_project",
    "description": "Invalid project",
}

SUCCESSFUL_PING_JSON_RESPONSE = {
    "project": {"name": "Test Project", "cuid": "clhoavbng001p5n8he0titquj"}
}


class MockResponse:
    def __init__(self, json_data, status_code):
        self.json_data = json_data
        self.status_code = status_code
        self.text = str(json_data)

    def json(self):
        return self.json_data


# If using with side_effect, use it like this on your test case:
#
# @mock.patch("requests.Session.get", side_effect=mocked_requests_get)
#
def mocked_requests_get(*args, **kwargs):
    # if args[0] == 'http://someurl.com/test.json':
    #     return MockResponse({"key1": "value1"}, 200)
    # elif args[0] == 'http://someotherurl.com/anothertest.json':
    #     return MockResponse({"key2": "value2"}, 200)

    return MockResponse(None, 404)


class TestFrameworkInit(unittest.TestCase):
    def setUp(self):
        self.api_key = (
            os.environ.pop("VM_API_KEY") if "VM_API_KEY" in os.environ else ""
        )
        self.api_secret = (
            os.environ.pop("VM_API_SECRET") if "VM_API_SECRET" in os.environ else ""
        )
        self.api_model = (
            os.environ.pop("VM_API_MODEL") if "VM_API_MODEL" in os.environ else ""
        )

    def tearDown(self):
        os.environ["VM_API_KEY"] = self.api_key
        os.environ["VM_API_SECRET"] = self.api_secret
        os.environ["VM_API_MODEL"] = self.api_model

    def test_no_args(self):
        """
        Test that init() raises a TypeError when no arguments are passed.
        """
        with self.assertRaises(MissingModelIdError) as err:
            vm.init()

        self.assertIn("Model ID must be provided", str(err.exception))

    def test_model_id_only(self):
        """
        Test that init() raises a ValueError when only a model is passed.
        """
        with self.assertRaises(MissingAPICredentialsError) as err:
            vm.init(model="test")

        self.assertIn("API key", str(err.exception))

    @mock.patch(
        "requests.get",
        return_value=MockResponse(INVALID_CREDENTIALS_JSON_RESPONSE, 401),
    )
    def test_all_args_ok_bad_credentials(self, mock_get):
        with self.assertRaises(Exception) as err:
            vm.init(api_key="bad_api_key", api_secret="bad_api_secret", project="test")

        self.assertIn("invalid_credentials", str(err.exception))

    @mock.patch(
        "requests.get",
        return_value=MockResponse(INVALID_PROJECT_JSON_RESPONSE, 401),
    )
    def test_all_args_ok_bad_project(self, mock_get):
        with self.assertRaises(Exception) as err:
            vm.init(api_key="api_key", api_secret="api_secret", project="bad_project")

        self.assertIn("invalid_project", str(err.exception))

    @mock.patch(
        "requests.get",
        return_value=MockResponse(SUCCESSFUL_PING_JSON_RESPONSE, 200),
    )
    def test_all_args_ok(self, mock_get):
        client_ok = vm.init(
            api_key="api_key", api_secret="api_secret", project="project"
        )
        self.assertIsNone(client_ok)


if __name__ == "__main__":
    unittest.main()
