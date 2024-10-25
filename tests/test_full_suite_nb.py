# this will run a test suite while simulating the event loop that runs in a jupyter notebook
# between this file and test_full_suite.py, we test running in a notebook and running in a script

import asyncio
import unittest
from unittest.mock import patch

import xgboost as xgb

from validmind.datasets.classification import customer_churn


class TestFullTestSuiteNB(unittest.TestCase):
    @patch("validmind.utils.is_notebook")
    @patch.multiple(
        "validmind.api_client",
        log_figure=unittest.mock.DEFAULT,
        log_metadata=unittest.mock.DEFAULT,
        log_metric_result=unittest.mock.DEFAULT,
        log_test_result=unittest.mock.DEFAULT,
        log_input=unittest.mock.DEFAULT,
    )
    @patch("validmind.client.log_input", return_value="1234")
    def test_run_full_suite(self, mock_ipython, *mocks, **mocks2):
        mock_ipython.return_value = True

        import validmind as vm
        from validmind.vm_models.dataset.dataset import VMDataset
        from validmind.vm_models.model import VMModel
        from validmind.vm_models.test_suite.test_suite import TestSuite

        raw_df = customer_churn.load_data()

        train_df, validation_df, test_df = customer_churn.preprocess(raw_df)

        x_train = train_df.drop(customer_churn.target_column, axis=1)
        y_train = train_df[customer_churn.target_column]
        x_val = validation_df.drop(customer_churn.target_column, axis=1)
        y_val = validation_df[customer_churn.target_column]

        model = xgb.XGBClassifier(early_stopping_rounds=10)
        model.set_params(
            eval_metric=["error", "logloss", "auc"],
        )
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            verbose=False,
        )

        vm_raw_dataset = vm.init_dataset(
            dataset=raw_df,
            input_id="raw_dataset",
            target_column=customer_churn.target_column,
            class_labels=customer_churn.class_labels,
        )
        self.assertIsInstance(vm_raw_dataset, VMDataset)

        vm_train_ds = vm.init_dataset(
            dataset=train_df,
            input_id="train_dataset",
            target_column=customer_churn.target_column,
        )
        self.assertIsInstance(vm_train_ds, VMDataset)

        vm_test_ds = vm.init_dataset(
            dataset=test_df,
            input_id="test_dataset",
            target_column=customer_churn.target_column,
        )
        self.assertIsInstance(vm_test_ds, VMDataset)

        vm_model = vm.init_model(
            model,
            input_id="model",
        )
        self.assertIsInstance(vm_model, VMModel)

        vm_train_ds.assign_predictions(
            model=vm_model,
        )

        vm_test_ds.assign_predictions(
            model=vm_model,
        )

        config = customer_churn.get_demo_test_config(
            vm.get_test_suite("classifier_full_suite")
        )

        async def run_test_suite():
            return vm.run_test_suite(
                "classifier_full_suite",
                config=config,
                fail_fast=True,
            )

        suite = asyncio.run(run_test_suite())

        self.assertIsInstance(suite, TestSuite)


if __name__ == "__main__":
    unittest.main()
