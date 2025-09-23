# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Client interface for all data and model validation functions
"""

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import polars as pl

try:
    import torch  # noqa: F401
except Exception:
    torch = None  # type: ignore  # noqa: F401

from .api_client import log_input as log_input
from .client_config import client_config
from .errors import (
    GetTestSuiteError,
    InitializeTestSuiteError,
    MissingDocumentationTemplate,
    MissingRExtrasError,
    UnsupportedDatasetError,
    UnsupportedModelError,
)
from .input_registry import input_registry
from .logging import get_logger
from .models.metadata import MetadataModel
from .models.r_model import RModel
from .template import get_template_test_suite
from .template import preview_template as _preview_template
from .test_suites import get_by_id as get_test_suite_by_id
from .utils import get_dataset_info, get_model_info
from .vm_models import TestSuite, TestSuiteRunner
from .vm_models.dataset import DataFrameDataset, PolarsDataset, TorchDataset, VMDataset
from .vm_models.model import (
    ModelAttributes,
    VMModel,
    get_model_class,
    is_model_metadata,
)

pd.option_context("format.precision", 2)

logger = get_logger(__name__)


def init_dataset(
    dataset: Union[
        pd.DataFrame, pl.DataFrame, "np.ndarray", "torch.utils.data.TensorDataset"
    ],
    model: Optional[VMModel] = None,
    index: Optional[Any] = None,
    index_name: Optional[str] = None,
    date_time_index: bool = False,
    columns: Optional[List[str]] = None,
    text_column: Optional[str] = None,
    target_column: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
    extra_columns: Optional[Dict[str, Any]] = None,
    class_labels: Optional[Dict[str, Any]] = None,
    type: Optional[str] = None,
    input_id: Optional[str] = None,
    copy_data: bool = True,
    __log: bool = True,
) -> VMDataset:
    """
    Initializes a VM Dataset, which can then be passed to other functions
    that can perform additional analysis and tests on the data. This function
    also ensures we are reading a valid dataset type.

    The following dataset types are supported:
    - Pandas DataFrame
    - Polars DataFrame
    - Numpy ndarray
    - Torch TensorDataset

    Args:
        dataset: Dataset from various Python libraries.
        model (VMModel): ValidMind model object.
        index (Any, optional): Index for the dataset.
        index_name (str, optional): Name of the index column.
        date_time_index (bool): Whether the index is a datetime index.
        columns (List[str], optional): List of column names.
        text_column (str, optional): Name of the text column.
        target_column (str, optional): The name of the target column in the dataset.
        feature_columns (List[str], optional): A list of names of feature columns in the dataset.
        extra_columns (Dict[str, Any], optional): A dictionary containing the names of the
            prediction_column and group_by_columns in the dataset.
        class_labels (Dict[str, Any], optional): A list of class labels for classification problems.
        type (str, optional): The type of dataset (one of DATASET_TYPES) - DEPRECATED.
        input_id (str, optional): The input ID for the dataset (e.g. "my_dataset"). By default,
            this will be set to `dataset` but if you are passing this dataset as a
            test input using some other key than `dataset`, then you should set
            this to the same key.
        copy_data (bool, optional): Whether to copy the data. Defaults to True.
        __log (bool): Whether to log the input. Defaults to True.

    Raises:
        ValueError: If the dataset type is not supported.

    Returns:
        vm.vm.Dataset: A VM Dataset instance.
    """
    # Show deprecation notice if type is passed
    if type is not None:
        logger.info(
            "The 'type' argument to init_dataset() argument is deprecated and no longer required."
        )

    dataset_class = dataset.__class__.__name__
    input_id = input_id or "dataset"

    # Instantiate supported dataset types here
    if isinstance(dataset, pd.DataFrame):
        vm_dataset = DataFrameDataset(
            input_id=input_id,
            raw_dataset=dataset,
            model=model,
            target_column=target_column,
            feature_columns=feature_columns,
            text_column=text_column,
            extra_columns=extra_columns,
            target_class_labels=class_labels,
            date_time_index=date_time_index,
            copy_data=copy_data,
        )
    elif isinstance(dataset, pl.DataFrame):
        vm_dataset = PolarsDataset(
            input_id=input_id,
            raw_dataset=dataset,
            model=model,
            target_column=target_column,
            feature_columns=feature_columns,
            text_column=text_column,
            extra_columns=extra_columns,
            target_class_labels=class_labels,
            date_time_index=date_time_index,
        )
    elif dataset_class == "ndarray":
        vm_dataset = VMDataset(
            input_id=input_id,
            raw_dataset=dataset,
            model=model,
            index=index,
            index_name=index_name,
            # if no columns are passed, use the index
            columns=columns or [i for i in range(dataset.shape[1])],
            target_column=target_column,
            feature_columns=feature_columns,
            text_column=text_column,
            extra_columns=extra_columns,
            target_class_labels=class_labels,
            date_time_index=date_time_index,
        )
    elif dataset_class == "TensorDataset":
        vm_dataset = TorchDataset(
            input_id=input_id,
            raw_dataset=dataset,
            model=model,
            index=index,
            index_name=index_name,
            columns=columns,
            target_column=target_column,
            feature_columns=feature_columns,
            text_column=text_column,
            extra_columns=extra_columns,
            target_class_labels=class_labels,
        )
    else:
        raise UnsupportedDatasetError(
            "Only Pandas datasets and Tensor Datasets are supported at the moment."
        )

    if __log:
        log_input(
            input_id=input_id,
            type="dataset",
            metadata=get_dataset_info(vm_dataset),
        )

    input_registry.add(key=input_id, obj=vm_dataset)

    return vm_dataset


def init_model(
    model: Optional[object] = None,
    input_id: str = "model",
    attributes: Optional[Dict[str, Any]] = None,
    predict_fn: Optional[Callable] = None,
    __log: bool = True,
    **kwargs: Any,
) -> VMModel:
    """
    Initializes a VM Model, which can then be passed to other functions
    that can perform additional analysis and tests on the data. This function
    also ensures we are creating a model supported libraries.

    Args:
        model: A trained model or VMModel instance.
        input_id (str): The input ID for the model (e.g. "my_model"). By default,
            this will be set to `model` but if you are passing this model as a
            test input using some other key than `model`, then you should set
            this to the same key.
        attributes (dict): A dictionary of model attributes.
        predict_fn (callable): A function that takes an input and returns a prediction.
        **kwargs: Additional arguments to pass to the model.

    Raises:
        ValueError: If the model type is not supported.

    Returns:
        vm.VMModel: A VM Model instance.
    """
    vm_model = model if isinstance(model, VMModel) else None
    class_obj = get_model_class(model=model, predict_fn=predict_fn)

    if not vm_model and not class_obj:
        if not attributes:
            raise UnsupportedModelError(
                f"Model class {str(model.__class__)} is not supported at the moment."
            )

        if not is_model_metadata(attributes):
            raise UnsupportedModelError(
                f"Model attributes {str(attributes)} are missing required keys 'architecture' and 'language'."
            )

    if isinstance(vm_model, VMModel):
        vm_model.input_id = (
            input_id if input_id != "model" else (vm_model.input_id or input_id)
        )
        metadata = get_model_info(vm_model)
    elif hasattr(class_obj, "__name__") and class_obj.__name__ == "PipelineModel":
        vm_model = class_obj(
            pipeline=model,
            input_id=input_id,
            attributes=(
                ModelAttributes.from_dict(attributes)
                if attributes
                else ModelAttributes()
            ),
        )
        # TODO: Add metadata for pipeline model
        metadata = get_model_info(vm_model)
    elif class_obj:
        vm_model = class_obj(
            input_id=input_id,
            model=model,  # Trained model instance
            predict_fn=predict_fn,
            attributes=ModelAttributes.from_dict(attributes) if attributes else None,
            **kwargs,
        )
        metadata = get_model_info(vm_model)
    else:
        vm_model = MetadataModel(
            input_id=input_id, attributes=ModelAttributes.from_dict(attributes)
        )
        metadata = attributes

    if __log:
        log_input(
            input_id=input_id,
            type="model",
            metadata=metadata,
        )

    input_registry.add(key=input_id, obj=vm_model)

    return vm_model


def init_r_model(
    model_path: str,
    input_id: str = "model",
) -> VMModel:
    """
    Initialize a VM Model from an R model.

    LogisticRegression and LinearRegression models are converted to sklearn models by extracting
    the coefficients and intercept from the R model. XGB models are loaded using the xgboost
    since xgb models saved in .json or .bin format can be loaded directly with either Python or R.

    Args:
        model_path (str): The path to the R model saved as an RDS or XGB file.
        input_id (str): The input ID for the model. Defaults to "model".

    Returns:
        VMModel: A VM Model instance.
    """

    # TODO: proper check for supported models
    #
    # if model.get("method") not in R_MODEL_METHODS:
    #     raise UnsupportedRModelError(
    #         "R model method must be one of {}. Got {}".format(
    #             R_MODEL_METHODS, model.get("method")
    #         )
    #     )

    # first we need to load the model using rpy2
    # since rpy2 is an extra we need to conditionally import it
    try:
        import rpy2.robjects as robjects
    except ImportError:
        raise MissingRExtrasError()

    r = robjects.r
    loaded_objects = r.load(model_path)
    model_name = loaded_objects[0]
    model = r[model_name]

    vm_model = RModel(
        r=r,
        model=model,
        input_id=input_id,
    )

    return vm_model


def get_test_suite(
    test_suite_id: Optional[str] = None,
    section: Optional[str] = None,
    *args: Any,
    **kwargs: Any,
) -> TestSuite:
    """Gets a TestSuite object for the current project or a specific test suite.

    This function provides an interface to retrieve the TestSuite instance for the
    current project or a specific TestSuite instance identified by test_suite_id.
    The project Test Suite will contain sections for every section in the project's
    documentation template and these Test Suite Sections will contain all the tests
    associated with that template section.

    Args:
        test_suite_id (str, optional): The test suite name. If not passed, then the
            project's test suite will be returned. Defaults to None.
        section (str, optional): The section of the documentation template from which
            to retrieve the test suite. This only applies if test_suite_id is None.
            Defaults to None.
        args: Additional arguments to pass to the TestSuite.
        kwargs: Additional keyword arguments to pass to the TestSuite.

    Returns:
        TestSuite: The TestSuite instance.
    """
    if test_suite_id is None:
        if client_config.documentation_template is None:
            raise MissingDocumentationTemplate(
                "No documentation template found. Please run `vm.init()`"
            )

        return get_template_test_suite(
            client_config.documentation_template, section=section
        )

    return get_test_suite_by_id(test_suite_id)(*args, **kwargs)


def run_test_suite(
    test_suite_id: str,
    send: bool = True,
    fail_fast: bool = False,
    config: Optional[Dict[str, Any]] = None,
    inputs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> TestSuite:
    """High Level function for running a test suite.

    This function provides a high level interface for running a test suite. A test suite is
    a collection of tests. This function will automatically find the correct test suite
    class based on the test_suite_id, initialize each of the tests, and run them.

    Args:
        test_suite_id (str): The test suite name. For example, 'classifier_full_suite'.
        config (dict, optional): A dictionary of parameters to pass to the tests in the
            test suite. Defaults to None.
        send (bool, optional): Whether to post the test results to the API. send=False
            is useful for testing. Defaults to True.
        fail_fast (bool, optional): Whether to stop running tests after the first failure. Defaults to False.
        inputs (dict, optional): A dictionary of test inputs to pass to the TestSuite, such as `model`, `dataset`
            `models`, etc. These inputs will be accessible by any test in the test suite. See the test
            documentation or `vm.describe_test()` for more details on the inputs required for each. Defaults to None.
        **kwargs: backwards compatibility for passing in test inputs using keyword arguments.

    Raises:
        ValueError: If the test suite name is not found or if there is an error initializing the test suite.

    Returns:
        TestSuite: The TestSuite instance.
    """
    try:
        Suite: TestSuite = get_test_suite_by_id(test_suite_id)
    except ValueError as exc:
        raise GetTestSuiteError(
            "Error retrieving test suite {}. {}".format(test_suite_id, str(exc))
        )

    try:
        suite = Suite()
    except ValueError as exc:
        raise InitializeTestSuiteError(
            "Error initializing test suite {}. {}".format(test_suite_id, str(exc))
        )

    TestSuiteRunner(
        suite=suite,
        inputs={**kwargs, **(inputs or {})},
        config=config or {},
    ).run(fail_fast=fail_fast, send=send)

    return suite


def preview_template() -> None:
    """Preview the documentation template for the current project.

    This function will display the documentation template for the current project. If
    the project has not been initialized, then an error will be raised.

    Raises:
        ValueError: If the project has not been initialized.
    """
    if client_config.documentation_template is None:
        raise MissingDocumentationTemplate(
            "No documentation template found. Please run `vm.init()`"
        )

    _preview_template(client_config.documentation_template)


def run_documentation_tests(
    section: Optional[str] = None,
    send: bool = True,
    fail_fast: bool = False,
    inputs: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Union[TestSuite, Dict[str, TestSuite]]:
    """Collect and run all the tests associated with a template.

    This function will analyze the current project's documentation template and collect
    all the tests associated with it into a test suite. It will then run the test
    suite, log the results to the ValidMind API, and display them to the user.

    Args:
        section (str or list, optional): The section(s) to preview. Defaults to None.
        send (bool, optional): Whether to send the results to the ValidMind API. Defaults to True.
        fail_fast (bool, optional): Whether to stop running tests after the first failure. Defaults to False.
        inputs (dict, optional): A dictionary of test inputs to pass to the TestSuite.
        config: A dictionary of test parameters to override the defaults.
        **kwargs: backwards compatibility for passing in test inputs using keyword arguments.

    Returns:
        TestSuite or dict: The completed TestSuite instance or a dictionary of TestSuites if section is a list.

    Raises:
        ValueError: If the project has not been initialized.
    """
    if client_config.documentation_template is None:
        raise MissingDocumentationTemplate(
            "No documentation template found. Please run `vm.init()`"
        )

    if section is None:
        section = [None]  # Convert None to a list containing None for consistency

    if isinstance(section, str):
        section = [section]  # Convert a single section string to a list

    test_suites = {}

    for _section in section:
        test_suite = _run_documentation_section(
            template=client_config.documentation_template,
            section=_section,
            send=send,
            fail_fast=fail_fast,
            inputs=inputs,
            config=config,
            **kwargs,
        )
        test_suites[_section] = test_suite

    if len(test_suites) == 1:
        return list(test_suites.values())[0]  # Return the only TestSuite

    else:
        return test_suites  # If there are multiple entries, return the dictionary of TestSuites


def _run_documentation_section(
    template: str,
    section: str,
    send: bool = True,
    fail_fast: bool = False,
    config: Optional[Dict[str, Any]] = None,
    inputs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> TestSuite:
    """Run all tests in a template section.

    This function will collect all tests used in a template section into a TestSuite and then
    run the TestSuite as usual.

    Args:
        template: A valid flat template.
        section: The section of the template to run (if not provided, run all sections).
        send: Whether to send the results to the ValidMind API.
        fail_fast (bool, optional): Whether to stop running tests after the first failure. Defaults to False.
        config: A dictionary of test parameters to override the defaults.
        inputs: A dictionary of test inputs to pass to the TestSuite.
        **kwargs: backwards compatibility for passing in test inputs using keyword arguments.

    Returns:
        The completed TestSuite instance.
    """
    test_suite = get_template_test_suite(template, section)

    TestSuiteRunner(
        suite=test_suite,
        inputs={**kwargs, **(inputs or {})},
        config=config,
    ).run(send=send, fail_fast=fail_fast)

    return test_suite
