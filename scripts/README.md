# Creating a new test

All ValidMind tests are in the `validmind/tests/` directory. Each file should be named using Camel Case and should have a single test class that matches the file name. For example, `MyNewTest.py` should have the test `class MyNewTest`. This class should inherit from `validmind.vm_models.Metric` or `validmind.vm_models.ThresholdTest` depending on the type of test you are creating.

The tests are separated into subdirectories based on the category and type of test. For example, `validmind/tests/model_validation/sklearn` contains all of the model validation tests for sklearn-compatible models. There are two subdirectories in this folder: `metrics/` and `threshold_tests/` that contain the different types of tests. Any sub category can be used here and the `__init__.py` file will automatically pick up the tests.

Please see the notebook `listing-and-loading-tests.ipynb` for more information and examples and to learn about how the directory relates to the test's ID which is used across the ValidMind Platform.

New tests should currently be created manually by adding a new file in the appropriate `validmind/tests/` subdirectory and implementing the matching test class or function in that file. Use the surrounding tests in the target directory as templates for naming, structure, and registration patterns.
