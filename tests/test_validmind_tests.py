# """This is a test harness to run unit tests against the ValidMind tests"""

# import unittest

# import xgboost as xgb
# from tqdm import tqdm

# import validmind as vm
# from validmind.datasets.classification import customer_churn as demo_dataset
# from validmind.datasets.classification import taiwan_credit as demo_dataset
# from validmind.models import FoundationModel, Prompt
# from validmind.tests import list_tests, load_test


# class TestValidMindTests(unittest.TestCase):
#     pass


# test_inputs = {}


# def _setup_test_inputs():
#     df = demo_dataset.load_data()

#     train_df, validation_df, test_df = demo_dataset.preprocess(df)
#     x_train = train_df.drop(demo_dataset.target_column, axis=1)
#     y_train = train_df[demo_dataset.target_column]
#     x_val = validation_df.drop(demo_dataset.target_column, axis=1)
#     y_val = validation_df[demo_dataset.target_column]

#     classifier = xgb.XGBClassifier(early_stopping_rounds=10)
#     classifier.set_params(eval_metric=["error", "logloss", "auc"])
#     classifier.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

#     vm_classifier_model = vm.init_model(
#         classifier,
#         __log=False,
#     )
#     vm_train_ds = vm.init_dataset(
#         model=vm_classifier_model,
#         dataset=train_df,
#         target_column=demo_dataset.target_column,
#         __log=False,
#     )
#     test_inputs["classification"] = TestInput(
#         {
#             "dataset": vm_train_ds,
#             "datasets": (vm_train_ds, vm_train_ds),
#             "model": vm_classifier_model,
#         }
#     )

#     vm_foundation_model = FoundationModel(
#         predict_fn=lambda x: x,
#         prompt=Prompt(
#             template="hello, {name}",
#             variables=["name"],
#         ),
#     )
#     test_inputs["llm"] = TestInput({"model": vm_foundation_model})


# def create_unit_test_func(vm_test):
#     def unit_test_func(self):
#         vm_test.run()
#         vm_test.test()

#     return unit_test_func


# def create_unit_test_funcs_from_vm_tests():
#     _setup_test_inputs()

#     for vm_test_id in tqdm(list_tests(pretty=False)):
#         # load the test class
#         vm_test_class = load_test(vm_test_id)

#         # check if test class has `test` method
#         if not hasattr(vm_test_class, "test"):
#             continue

#         # initialize with the right test context
#         # TODO: we need to better handle the test context based on the test metadata
#         if getattr(vm_test_class, "category", None) == "prompt_validation":
#             test_input = test_inputs["llm"]
#         else:
#             test_input = test_inputs["classification"]

#         vm_test = vm_test_class(
#             test_id=vm_test_id,
#             context=TestContext(),
#             inputs=test_input,
#             params={},
#         )

#         # create a unit test function for the test class
#         unit_test_func = create_unit_test_func(vm_test)
#         unit_test_func_name = f'test_{vm_test_id.replace(".", "_")}'

#         # add the unit test function to the unit test class
#         setattr(TestValidMindTests, f"test_{unit_test_func_name}", unit_test_func)


# create_unit_test_funcs_from_vm_tests()


# if __name__ == "__main__":
#     unittest.main()
