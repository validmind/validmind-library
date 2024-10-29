"""
Utility functions for running integration tests for run_test().
"""

import ast
import os

import numpy as np
import pandas as pd
import validmind as vm
import xgboost as xgb

from transformers import pipeline
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from validmind.datasets.classification import customer_churn
from validmind.datasets.cluster import digits as digits_dataset
from validmind.datasets.nlp import cnn_dailymail
from validmind.datasets.regression import fred_timeseries
from validmind.models import FoundationModel, Prompt

os.environ["OPENAI_API_KEY"] = "123"


def assign_features_pairs_config(test_config, raw_df, train_df=None, test_df=None):
    """
    Generates pairs of features for tests such as BivariateFeaturesBarPlots test
    """
    test_config["features_pairs_raw"] = {"features_pairs": {}}

    for i in range(0, len(raw_df.columns)):
        for j in range(i + 1, len(raw_df.columns)):
            test_config["features_pairs_raw"]["features_pairs"][raw_df.columns[i]] = (
                raw_df.columns[j]
            )


def string_to_numpy_array(array_string):
    """
    This is needed to convert string to numpy array for embedding tests
    """
    try:
        array_list = ast.literal_eval(array_string)
        return np.array(array_list)
    except ValueError as e:
        print(f"Error converting string to array: {e}")
        return None


def setup_tabular_test_inputs(test_inputs={}, test_config={}):
    """
    Setup test inputs for tabular datasets

    Args:
        test_inputs (dict): Glogal test inputs to be updated
        test_config (dict): Global test config to be updated
    """
    df = customer_churn.load_data()
    df = df.sample(1000, random_state=42)

    train_df, validation_df, test_df = customer_churn.preprocess(df)
    x_train = train_df.drop(customer_churn.target_column, axis=1)
    y_train = train_df[customer_churn.target_column]
    x_val = validation_df.drop(customer_churn.target_column, axis=1)
    y_val = validation_df[customer_churn.target_column]

    classifier = xgb.XGBClassifier(early_stopping_rounds=10)
    classifier.set_params(eval_metric=["error", "logloss", "auc"])
    classifier.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

    classifier_rf = RandomForestClassifier()
    classifier_rf.fit(x_train, y_train)

    # Models
    vm_classifier_model = vm.init_model(
        classifier,
        input_id="xgb_classifier",
        __log=False,
    )
    vm_classifier_rf_model = vm.init_model(
        classifier_rf,
        input_id="rf_classifier",
        __log=False,
    )

    # Datasets
    vm_raw_dataset = vm.init_dataset(
        dataset=df,
        input_id="raw_dataset",
        target_column=customer_churn.target_column,
        class_labels=customer_churn.class_labels,
        __log=False,
    )
    vm_train_ds = vm.init_dataset(
        dataset=train_df,
        input_id="train_dataset",
        target_column=customer_churn.target_column,
        __log=False,
    )
    vm_test_ds = vm.init_dataset(
        dataset=test_df,
        input_id="test_dataset",
        target_column=customer_churn.target_column,
        __log=False,
    )

    # Assign predictions for each model
    vm_train_ds.assign_predictions(vm_classifier_model)
    vm_train_ds.assign_predictions(vm_classifier_rf_model)

    vm_test_ds.assign_predictions(vm_classifier_model)
    vm_test_ds.assign_predictions(vm_classifier_rf_model)

    assign_features_pairs_config(test_config, df)
    test_config["kmeans_config"] = {
        "n_clusters": [2, 3],
    }

    # Usage:
    #
    # For 1 dataset tests use the raw dataset (i.e. data quality tests)
    # For 2 dataset/model tests, use model and test dataset
    # For 3 two dataset tests, use both the training and test datasets (comparison tests)
    test_inputs["classification"] = {
        "single_dataset": {
            "dataset": vm_raw_dataset,
        },
        "two_datasets": {
            "datasets": [vm_train_ds, vm_test_ds],
        },
        "single_model": {
            "model": vm_classifier_model,
        },
        "dataset_and_two_models": {
            "dataset": vm_test_ds,
            "models": [vm_classifier_model, vm_classifier_rf_model],
        },
        "model_and_dataset": {
            "dataset": vm_test_ds,
            "model": vm_classifier_model,
        },
        "model_and_two_datasets": {
            "model": vm_classifier_model,
            "datasets": [vm_train_ds, vm_test_ds],
        },
        "two_models": {
            "models": [vm_classifier_model, vm_classifier_rf_model],
        },
    }


def setup_time_series_test_inputs(test_inputs={}, test_config={}):
    """
    Setup test inputs for time series datasets

    Args:
        test_inputs (dict): Glogal test inputs to be updated
        test_config (dict): Global test config to be updated
    """
    df = fred_timeseries.load_data()

    target_column = fred_timeseries.target_column

    # Models
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

    # Take the first difference of the training and test sets
    train_diff_df = train_df.diff().dropna()
    test_diff_df = test_df.diff().dropna()

    # Extract the features and target variable from the training set
    X_diff_train = train_diff_df.drop(target_column, axis=1)
    y_diff_train = train_diff_df[target_column]

    # Extract the features and target variable from the test set
    X_diff_test = test_diff_df.drop(target_column, axis=1)

    # Fit the random forest model
    model_rf = RandomForestRegressor(n_estimators=1500, random_state=0)
    model_rf.fit(X_diff_train, y_diff_train)

    # Make predictions on the training and test sets
    y_diff_train_pred = model_rf.predict(X_diff_train)
    y_diff_test_pred = model_rf.predict(X_diff_test)

    # Transform the predictions back to the original scale
    y_train_rf_pred = _transform_to_levels(
        y_diff_train_pred, first_value=train_df[target_column].iloc[0]
    )
    y_test_rf_pred = _transform_to_levels(
        y_diff_test_pred, first_value=test_df[target_column].iloc[0]
    )

    vm_model_rf = vm.init_model(
        model_rf,
        input_id="random_forests_model",
        __log=False,
    )

    # Datasets
    vm_raw_dataset = vm.init_dataset(
        input_id="raw_dataset",
        dataset=df,
        target_column=target_column,
        __log=False,
    )

    vm_train_ds = vm.init_dataset(
        input_id="train_ds",
        dataset=train_df,
        target_column=target_column,
        __log=False,
    )

    vm_test_ds = vm.init_dataset(
        input_id="test_ds",
        dataset=test_df,
        target_column=target_column,
        __log=False,
    )

    # Assign predictions for each model
    vm_train_ds.assign_predictions(
        model=vm_model_rf,
        prediction_values=y_train_rf_pred,
    )

    vm_test_ds.assign_predictions(
        model=vm_model_rf,
        prediction_values=y_test_rf_pred,
    )

    test_inputs["time_series"] = {
        "single_dataset": {
            "dataset": vm_raw_dataset,
        },
        "two_datasets": {
            "datasets": [vm_train_ds, vm_test_ds],
        },
        "single_model": {
            "model": vm_model_rf,
        },
        "model_and_dataset": {
            "dataset": vm_test_ds,
            "model": vm_model_rf,
        },
    }


def setup_summarization_test_inputs(test_inputs={}, test_config={}):
    """
    Setup test inputs for summarization tests

    Args:
        test_inputs (dict): Glogal test inputs to be updated
        test_config (dict): Global test config to be updated
    """
    _, test_df = cnn_dailymail.load_data(source="offline", dataset_size="100")

    # Dataset
    vm_test_ds = vm.init_dataset(
        dataset=test_df,
        input_id="test_dataset",
        text_column="article",
        target_column="highlights",
        __log=False,
    )

    # Prompt and model
    prompt_template = """
    You are an AI with expertise in summarizing financial news.
    Your task is to provide a concise summary of the specific news article provided below.
    Before proceeding, take a moment to understand the context and nuances of the financial terminology used in the article.

    Article to Summarize:

    ```
    {article}
    ```

    Please respond with a concise summary of the article's main points.
    Ensure that your summary is based on the content of the article and not on external information or assumptions.
    """.strip()

    prompt_variables = ["article"]

    model = OpenAI()

    # This should not call the model since we're using precomputed predictions
    def call_model(prompt):
        return (
            model.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            .choices[0]
            .message.content
        )

    vm_model = vm.init_model(
        model=FoundationModel(
            predict_fn=call_model,
            prompt=Prompt(
                template=prompt_template,
                variables=prompt_variables,
            ),
        ),
        input_id="gpt_35",
        __log=False,
    )

    # Assign precomputed predictions
    vm_test_ds.assign_predictions(vm_model, prediction_column="gpt_35_prediction")

    test_inputs["text_summarization"] = {
        "single_dataset": {
            "dataset": vm_test_ds,
        },
        "model_and_dataset": {
            "dataset": vm_test_ds,
            "model": vm_model,
        },
    }


def setup_embeddings_test_inputs(test_inputs={}, test_config={}):
    """
    Setup test inputs for embeddings

    Args:
        test_inputs (dict): Glogal test inputs to be updated
        test_config (dict): Global test config to be updated
    """
    _, test_df = cnn_dailymail.load_data(source="offline", dataset_size="100")
    # Use only 5 rows to speed up the test
    test_df = test_df.head(5)

    test_df["bert_embedding_model_prediction"] = test_df[
        "bert_embedding_model_prediction"
    ].apply(string_to_numpy_array)

    # Dataset
    vm_test_ds = vm.init_dataset(
        dataset=test_df,
        input_id="test_dataset",
        text_column="article",
        target_column="highlights",
        __log=False,
    )

    # Model

    embedding_model = pipeline(
        "feature-extraction",
        model="bert-base-uncased",
        tokenizer="bert-base-uncased",
        truncation=True,
    )

    vm_embedding_model = vm.init_model(
        model=embedding_model,
        input_id="bert_embedding_model",
        __log=False,
    )

    # Assign precomputed predictions
    vm_test_ds.assign_predictions(
        vm_embedding_model, prediction_column="bert_embedding_model_prediction"
    )

    test_config["t_sne_config"] = {
        "n_components": 2,
        "perplexity": 1,
    }
    test_config["stability_analysis_keyword_config"] = {
        "keyword_dict": {"finance": "financial"},
    }

    test_inputs["embeddings"] = {
        "single_dataset": {
            "dataset": vm_test_ds,
        },
        "model_and_dataset": {
            "dataset": vm_test_ds,
            "model": vm_embedding_model,
        },
    }


def setup_clustering_test_inputs(test_inputs={}, test_config={}):
    """
    Setup test inputs for clustering

    Args:
        test_inputs (dict): Glogal test inputs to be updated
        test_config (dict): Global test config to be updated
    """
    df = digits_dataset.load_data()
    target_column = digits_dataset.target_column
    train_df, validation_df, test_df = digits_dataset.preprocess(df)

    x_train = train_df.drop(target_column, axis=1)
    y_train = train_df[target_column]
    x_val = validation_df.drop(target_column, axis=1)
    y_val = validation_df[target_column]
    x_test = test_df.drop(target_column, axis=1)
    x_train = pd.concat([x_train, x_val], axis=0)
    y_train = pd.concat([y_train, y_val], axis=0)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.fit_transform(x_val)
    x_test = scaler.fit_transform(x_test)

    model = KMeans(init="k-means++", n_clusters=10, n_init=4, random_state=0)
    model = model.fit(x_train)

    # Datasets
    vm_train_ds = vm.init_dataset(
        dataset=train_df,
        input_id="train_dataset",
        target_column=target_column,
        __log=False,
    )

    vm_test_ds = vm.init_dataset(
        dataset=test_df,
        input_id="test_dataset",
        target_column=target_column,
        __log=False,
    )

    # Model
    vm_model = vm.init_model(
        model=model,
        input_id="kmeans",
        __log=False,
    )

    # Assign precomputed predictions
    vm_train_ds.assign_predictions(vm_model)
    vm_test_ds.assign_predictions(vm_model)

    test_config["hyperparameter_tuning_config"] = {
        "param_grid": {"n_clusters": range(40, 60)}
    }

    test_inputs["clustering"] = {
        "single_dataset": {
            "dataset": vm_test_ds,
        },
        "model_and_dataset": {
            "dataset": vm_test_ds,
            "model": vm_model,
        },
        "model_and_two_datasets": {
            "model": vm_model,
            "datasets": [vm_train_ds, vm_test_ds],
        },
    }


def _transform_to_levels(y_diff_pred, first_value=0):
    y_pred = [first_value]
    for pred in y_diff_pred:
        y_pred.append(y_pred[-1] + pred)
    return y_pred
