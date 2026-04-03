"""
Unit tests for high-level ValidMind client methods
"""

import unittest
from dataclasses import dataclass
from unittest import mock, TestCase

import pandas as pd
import numpy as np
import sklearn

import validmind
from validmind import (
    get_content_ids,
    init_dataset,
    init_model,
    get_test_suite,
    run_text_generation,
    run_documentation_tests,
)
from validmind.errors import UnsupportedModelError
from validmind.vm_models.result import TextGenerationResult


@dataclass
class MockedConfig:
    documentation_template = {
        "template_id": "binary_classification_v2",
        "template_name": "Binary classification",
        "description": "Template for binary classification models.",
        "version": "3.0.0",
        "sections": [
            {
                "id": "test_section_1",
                "title": "Test Section",
                "index_only": True,
                "order": 0,
            },
            {
                "id": "test_subsection_1",
                "title": "Test Subsection",
                "parent_section": "test_section_1",
                "contents": [
                    {
                        "content_type": "test",
                        "content_id": "validmind.data_validation.ClassImbalance",
                    },
                ],
            },
            {
                "id": "test_section_2",
                "title": "Test Section 2",
                "index_only": True,
                "order": 1,
            },
            {
                "id": "test_subsection_2",
                "title": "Test Subsection 2",
                "parent_section": "test_section_2",
                "contents": [
                    {
                        "content_id": "validmind.data_validation.DatasetSplit",
                        "content_type": "metric",
                    },
                ],
            },
        ],
    }


class TestInitDataset(TestCase):
    @mock.patch(
        "validmind.client.log_input",
        return_value="1234",
    )
    def test_init_dataset_pandas(self, mock_log_input):
        # Test initializing a Pandas DataFrame
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        vm_dataset = init_dataset(df, target_column="col1")
        self.assertIsInstance(vm_dataset.df, pd.DataFrame)
        # sort since order of columns may be different
        self.assertTrue(vm_dataset.df.sort_index(axis=1).equals(df.sort_index(axis=1)))

    @mock.patch(
        "validmind.client.log_input",
        return_value="1234",
    )
    def test_init_dataset_numpy(self, mock_log_input):
        # Test initializing a numpy ndarray
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        vm_dataset = init_dataset(arr, target_column=2)
        self.assertIsInstance(vm_dataset.df, pd.DataFrame)
        self.assertTrue(vm_dataset.df.equals(pd.DataFrame(arr)))

    # TODO: Test initializing a PyTorch tensor


class TestInitModel(TestCase):
    def test_init_model_invalid_model(self):
        model = dict()
        with self.assertRaises(UnsupportedModelError):
            init_model(model)

    @mock.patch(
        "validmind.client.log_input",
        return_value="1234",
    )
    def test_init_model_supported_model(self, mock_log_input):
        # Test initializing an SKLearn model
        model = sklearn.linear_model.LinearRegression()
        vm_model = init_model(model)
        self.assertEqual(vm_model.model, model)

    def test_init_model_invalid_metadata_dict(self):
        # Model metadata requires architecture and language at a minimum
        metadata = {
            "key": "value",
            "foo": "bar",
        }
        with self.assertRaises(UnsupportedModelError):
            init_model(attributes=metadata, __log=False)

    def test_init_model_metadata_dict(self):
        # Model metadata requires architecture and language at a minimum
        metadata = {
            "architecture": "Spark",
            "language": "Python",
        }
        vm_model = init_model(attributes=metadata, __log=False)

        # Model will be none but attributes will be populated
        self.assertEqual(vm_model.attributes.architecture, metadata["architecture"])
        self.assertEqual(vm_model.attributes.language, metadata["language"])


# Run methods are tested in test_full_suite_nb.py and test_full_suite.py


class TestGetTestSuite(TestCase):
    def test_get_specfic_suite(self):
        test_suite = get_test_suite("classifier_full_suite")
        self.assertIsInstance(test_suite, validmind.vm_models.TestSuite)

    @mock.patch(
        "validmind.client_config.client_config.documentation_template",
        MockedConfig.documentation_template,
    )
    def test_get_project_test_suite(self):
        test_suite = get_test_suite()
        self.assertIsInstance(test_suite, validmind.vm_models.TestSuite)
        self.assertEqual(len(test_suite.sections), 3)

    @mock.patch(
        "validmind.client_config.client_config.documentation_template",
        MockedConfig.documentation_template,
    )
    def test_get_default_config(self):
        test_suite = get_test_suite()
        default_config = test_suite.get_default_config()

        # Mock template has two tests only
        self.assertEqual(len(default_config), 2)
        for _, config in default_config.items():
            self.assertIn("inputs", config)
            self.assertIn("params", config)


class TestGetContentIds(TestCase):
    @mock.patch(
        "validmind.client_config.client_config.documentation_template",
        MockedConfig.documentation_template,
    )
    def test_get_all_content_ids(self):
        content_ids = get_content_ids()
        self.assertEqual(
            content_ids,
            [
                "validmind.data_validation.ClassImbalance",
                "validmind.data_validation.DatasetSplit",
            ],
        )

    @mock.patch(
        "validmind.client_config.client_config.documentation_template",
        MockedConfig.documentation_template,
    )
    def test_get_content_ids_for_single_section(self):
        content_ids = get_content_ids("test_section_1")
        self.assertEqual(content_ids, ["validmind.data_validation.ClassImbalance"])

    @mock.patch(
        "validmind.client_config.client_config.documentation_template",
        MockedConfig.documentation_template,
    )
    def test_get_content_ids_for_multiple_sections(self):
        content_ids = get_content_ids(["test_section_1", "test_section_2"])
        self.assertEqual(
            content_ids,
            [
                "validmind.data_validation.ClassImbalance",
                "validmind.data_validation.DatasetSplit",
            ],
        )


class TestRunTextGeneration(TestCase):
    @mock.patch(
        "validmind.client.api_client._generate_log_text",
        return_value="<p>Generated text</p>",
    )
    def test_run_text_generation(self, mock_generate_text):
        result = run_text_generation(
            content_id="dataset_summary_text",
            prompt="Summarize the dataset.",
            context={"content_ids": ["train_dataset"]},
            show=False,
        )

        self.assertIsInstance(result, TextGenerationResult)
        self.assertEqual(result.content_id, "dataset_summary_text")
        self.assertEqual(result.prompt, "Summarize the dataset.")
        self.assertEqual(result.context, {"content_ids": ["train_dataset"]})
        self.assertEqual(result.description, "<p>Generated text</p>")
        mock_generate_text.assert_called_once_with(
            "dataset_summary_text",
            "Summarize the dataset.",
            {"content_ids": ["train_dataset"]},
        )


# TODO: Fix this test
# class TestPreviewTemplate(TestCase):
#     @mock.patch(
#         "validmind.client_config.client_config.documentation_template",
#         MockedConfig.documentation_template,
#     )
#     @mock.patch("validmind.utils.is_notebook")
#     @mock.patch("IPython.display.display")
#     def test_preview_template(self, mock_is_notebook, mock_display):
#         mock_is_notebook.return_value = True
#         preview_template()
#         mock_is_notebook.assert_called_once()
#         mock_display.assert_called_once()


class TestRunDocumentationTests(TestCase):
    @mock.patch(
        "validmind.client_config.client_config.documentation_template",
        MockedConfig.documentation_template,
    )
    @mock.patch(
        "validmind.client.log_input",
        return_value="1234",
    )
    def test_run_documentation_tests(self, mock_log_input):
        # create a very simple logistic regression model
        model = sklearn.linear_model.LogisticRegression()
        dataset = pd.DataFrame([[1, 1], [1, 0], [0, 1], [0, 0]], columns=["x", "y"])
        model.fit(dataset[["x"]], dataset["y"])
        vm_dataset = init_dataset(dataset, target_column="y")
        vm_model = init_model(model)

        test_suite = run_documentation_tests(
            model=vm_model, dataset=vm_dataset, send=False
        )

        self.assertIsInstance(test_suite, validmind.vm_models.TestSuite)
        self.assertEqual(len(test_suite.sections), 3)


if __name__ == "__main__":
    unittest.main()
