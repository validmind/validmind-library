#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quickstart for model documentation

Welcome! Let's get you started with the basic process of documenting models with ValidMind.

You will learn how to initialize the ValidMind Library, load a sample dataset to train a simple classification model, 
and then run a ValidMind test suite to quickly generate documentation about the data and model.

This script uses the Bank Customer Churn Prediction sample dataset from Kaggle to train the classification model.
"""

# Install required packages
# %pip install -q validmind

# Import required libraries
import validmind as vm
import xgboost as xgb
from validmind.datasets.classification import customer_churn
from validmind.utils import preview_test_config

# Initialize the ValidMind Library
# Note: Replace these values with your actual credentials from ValidMind Platform
vm.init(
    # api_host="...",
    # api_key="...",
    # api_secret="...",
    # model="...",
)

# Preview the documentation template
vm.preview_template()

# Load the sample dataset
print(
    f"Loaded demo dataset with: \n\n\t• Target column: '{customer_churn.target_column}' \n\t• Class labels: {customer_churn.class_labels}"
)

raw_df = customer_churn.load_data()
print("\nFirst few rows of the dataset:")
print(raw_df.head())

# Preprocess the raw dataset
train_df, validation_df, test_df = customer_churn.preprocess(raw_df)

x_train = train_df.drop(customer_churn.target_column, axis=1)
y_train = train_df[customer_churn.target_column]
x_val = validation_df.drop(customer_churn.target_column, axis=1)
y_val = validation_df[customer_churn.target_column]

# Initialize and train the XGBoost model
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

# Initialize ValidMind datasets
vm_raw_dataset = vm.init_dataset(
    dataset=raw_df,
    input_id="raw_dataset",
    target_column=customer_churn.target_column,
    class_labels=customer_churn.class_labels,
)

vm_train_ds = vm.init_dataset(
    dataset=train_df,
    input_id="train_dataset",
    target_column=customer_churn.target_column,
)

vm_test_ds = vm.init_dataset(
    dataset=test_df, 
    input_id="test_dataset", 
    target_column=customer_churn.target_column
)

# Initialize ValidMind model
vm_model = vm.init_model(
    model,
    input_id="model",
)

# Assign predictions to the datasets
vm_train_ds.assign_predictions(
    model=vm_model,
)

vm_test_ds.assign_predictions(
    model=vm_model,
)

# Get test configuration and preview it
test_config = customer_churn.get_demo_test_config()
preview_test_config(test_config)

# Run the full suite of tests
full_suite = vm.run_documentation_tests(config=test_config)

# Note: After running this script, you can view the results in the ValidMind Platform
# by going to the Model Inventory and selecting your model.

if __name__ == "__main__":
    print("\nScript execution completed. Check the ValidMind Platform for results.") 