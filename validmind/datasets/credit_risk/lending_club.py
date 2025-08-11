# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import logging
import os
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import validmind as vm
from validmind.errors import MissingDependencyError

try:
    import scorecardpy as sc
except ImportError as e:
    if "scorecardpy" in str(e):
        raise MissingDependencyError(
            "Missing required package `scorecardpy` for credit risk demos. "
            "Please run `pip install validmind[credit_risk]` or `pip install scorecardpy`.",
            required_dependencies=["scorecardpy"],
            extra="credit_risk",
        ) from e
    raise e

current_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_path, "datasets")

# URLs or file paths for online and offline data
online_data_file = "https://vmai.s3.us-west-1.amazonaws.com/datasets/lending_club_loan_data_2007_2014.csv"
offline_data_file = os.path.join(
    dataset_path, "lending_club_loan_data_2007_2014_clean.csv.gz"
)

target_column = "loan_status"

drop_columns = [
    "Unnamed: 0",
    "id",
    "member_id",
    "funded_amnt",
    "emp_title",
    "url",
    "desc",
    "application_type",
    "title",
    "zip_code",
    "delinq_2yrs",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "mths_since_last_major_derog",
    "revol_bal",
    "total_rec_prncp",
    "total_rec_late_fee",
    "recoveries",
    "out_prncp_inv",
    "out_prncp",
    "collection_recovery_fee",
    "next_pymnt_d",
    "initial_list_status",
    "pub_rec",
    "collections_12_mths_ex_med",
    "policy_code",
    "acc_now_delinq",
    "pymnt_plan",
    "tot_coll_amt",
    "tot_cur_bal",
    "total_rev_hi_lim",
    "last_pymnt_d",
    "last_credit_pull_d",
    "earliest_cr_line",
    "issue_d",
    "addr_state",
    "dti",
    "revol_util",
    "total_pymnt_inv",
    "inq_last_6mths",
    "total_rec_int",
    "last_pymnt_amnt",
]

drop_features = [
    "loan_amnt",
    "funded_amnt_inv",
    "total_pymnt",
]

categorical_variables = [
    "term",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
]

breaks_adj = {
    "loan_amnt": [5000, 10000, 15000, 20000, 25000],
    "int_rate": [10, 15, 20],
    "annual_inc": [50000, 100000, 150000],
}

score_params = {
    "target_score": 600,
    "target_odds": 50,
    "pdo": 20,
}


def load_data(source: str = "online", verbose: bool = True) -> pd.DataFrame:
    """
    Load data from either an online source or offline files, automatically dropping specified columns for offline data.

    Args:
        source: 'online' for online data, 'offline' for offline files. Defaults to 'online'.

    Returns:
        DataFrame: DataFrame containing the loaded data.
    """

    if source == "online":
        if verbose:
            print(f"Loading data from an online source: {online_data_file}")
        df = pd.read_csv(online_data_file)
        df = _clean_data(df, verbose=verbose)

    elif source == "offline":
        if verbose:
            print(f"Loading data from an offline .gz file: {offline_data_file}")
        # Since we know the offline_data_file path ends with '.zip', we replace it with '.csv.gz'
        gzip_file_path = offline_data_file.replace(".zip", ".csv.gz")
        if verbose:
            print(f"Attempting to read from .gz file: {gzip_file_path}")
        # Read the CSV file directly from the .gz archive
        df = pd.read_csv(gzip_file_path, compression="gzip")
        if verbose:
            print("Data loaded successfully.")
    else:
        raise ValueError("Invalid source specified. Choose 'online' or 'offline'.")

    if verbose:
        print(
            f"Rows: {df.shape[0]}, Columns: {df.shape[1]}, Missing values: {df.isnull().sum().sum()}"
        )
    return df


def _clean_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    df = df.copy()

    # Drop columns not relevant for application scorecards
    df = df.drop(columns=drop_columns)

    # Drop rows with missing target values
    df.dropna(subset=[target_column], inplace=True)
    if verbose:
        print("Dropping rows with missing target values:")
        print(
            f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}\nMissing values: {df.isnull().sum().sum()}\n"
        )

    # Drop columns with more than N percent missing values
    missing_values = df.isnull().mean()
    df = df.loc[:, missing_values < 0.7]
    if verbose:
        print("Dropping columns with more than 70% missing values:")
        print(
            f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}\nMissing values: {df.isnull().sum().sum()}\n"
        )

    # Drop columns with only one unique value
    unique_values = df.nunique()
    df = df.loc[:, unique_values > 1]
    if verbose:
        print("Dropping columns with only one unique value:")
        print(
            f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}\nMissing values: {df.isnull().sum().sum()}\n"
        )

    # Define the target variable for the model, representing loan default status.
    df[target_column] = df[target_column].map({"Fully Paid": 0, "Charged Off": 1})

    # Drop rows with NaN in target_column after mapping
    df.dropna(subset=[target_column], inplace=True)
    if verbose:
        print("Dropping rows with missing target values:")
        print(
            f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}\nMissing values: {df.isnull().sum().sum()}\n"
        )

    return df


def preprocess(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    df = df.copy()

    # Convert the target variable to integer type for modeling.
    df[target_column] = df[target_column].astype(int)

    # Keep rows where purpose is 'debt_consolidation' or 'credit_card'
    df = df[df["purpose"].isin(["debt_consolidation", "credit_card"])]
    if verbose:
        print("Filtering 'purpose' to 'debt_consolidation' and 'credit_card':")
        print(
            f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}\nMissing values: {df.isnull().sum().sum()}\n"
        )

    # Remove rows where grade is 'F' or 'G'
    df = df[~df["grade"].isin(["F", "G"])]
    if verbose:
        print("Filtering out 'grade' F and G:")
        print(
            f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}\nMissing values: {df.isnull().sum().sum()}\n"
        )

    # Remove rows where sub_grade starts with 'F' or 'G'
    df = df[~df["sub_grade"].str.startswith(("F", "G"))]
    if verbose:
        print("Filtering out 'sub_grade' F and G:")
        print(
            f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}\nMissing values: {df.isnull().sum().sum()}\n"
        )

    # Remove rows where home_ownership is 'OTHER', 'NONE', or 'ANY'
    df = df[~df["home_ownership"].isin(["OTHER", "NONE", "ANY"])]
    if verbose:
        print("Filtering out 'home_ownership' OTHER, NONE, ANY:")
        print(
            f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}\nMissing values: {df.isnull().sum().sum()}\n"
        )

    # Drop features that are not useful for modeling
    df.drop(drop_features, axis=1, inplace=True)
    if verbose:
        print("Dropping specified features:")
        print(
            f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}\nMissing values: {df.isnull().sum().sum()}\n"
        )

    # Drop rows with missing values
    df.dropna(inplace=True)
    if verbose:
        print("Dropping rows with any missing values:")
        print(
            f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}\nMissing values: {df.isnull().sum().sum()}\n"
        )

    # Preprocess emp_length column
    df = _preprocess_emp_length(df)

    # Preprocess term column
    df = _preprocess_term(df)

    return df


def _preprocess_term(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove ' months' and convert to integer
    df["term"] = df["term"].str.replace(" months", "").astype(object)

    return df


def _preprocess_emp_length(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Mapping string values to numbers
    emp_length_map = {
        "10+ years": 10,
        "< 1 year": 0,
        "1 year": 1,
        "2 years": 2,
        "3 years": 3,
        "4 years": 4,
        "5 years": 5,
        "6 years": 6,
        "7 years": 7,
        "8 years": 8,
        "9 years": 9,
    }

    # Apply the mapping to the emp_length column
    df["emp_length"] = df["emp_length"].map(emp_length_map).astype(object)

    # Drop rows where emp_length is NaN after mapping
    # df.dropna(subset=["emp_length"], inplace=True)

    return df


def feature_engineering(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    df = df.copy()

    # WoE encoding of numerical and categorical features
    df = woe_encoding(df, verbose=verbose)

    if verbose:
        print(
            f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}\nMissing values: {df.isnull().sum().sum()}\n"
        )

    return df


def woe_encoding(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    df = df.copy()

    woe = _woebin(df, verbose=verbose)
    bins = _woe_to_bins(woe)

    # Make sure we don't transform the target column
    if target_column in bins:
        del bins[target_column]
        if verbose:
            print(f"Excluded {target_column} from WoE transformation.")

    # Apply the WoE transformation
    df = sc.woebin_ply(df, bins=bins)

    if verbose:
        print("Successfully converted features to WoE values.")

    return df


def _woe_to_bins(woe: Dict[str, Any]) -> Dict[str, Any]:
    # Select and rename columns
    transformed_df = woe[
        [
            "variable",
            "bin",
            "count",
            "count_distr",
            "good",
            "bad",
            "badprob",
            "woe",
            "bin_iv",
            "total_iv",
        ]
    ].copy()
    transformed_df.rename(columns={"bin_iv": "total_iv"}, inplace=True)

    # Create 'is_special_values' column (assuming there are no special values)
    transformed_df["is_special_values"] = False

    # Transform 'bin' column into interval format and store it in 'breaks' column
    transformed_df["breaks"] = transformed_df["bin"].apply(
        lambda x: "[-inf, %s)" % x if isinstance(x, float) else "[%s, inf)" % x
    )

    # Group by 'variable' to create bins dictionary
    bins = {}
    for variable, group in transformed_df.groupby("variable"):
        bins[variable] = group

    return bins


def _woebin(df: pd.DataFrame, verbose: bool = True) -> Dict[str, Any]:
    """
    This function performs automatic binning using WoE.
    df: A pandas dataframe
    target_column: The target variable in quotes, e.g. 'loan_status'
    """

    non_numeric_cols = df.select_dtypes(exclude=["int64", "float64"]).columns
    df[non_numeric_cols] = df[non_numeric_cols].astype(str)

    try:
        if verbose:
            print(
                f"Performing binning with breaks_adj: {breaks_adj}"
            )  # print the breaks_adj being used
        bins = sc.woebin(df, target_column, breaks_list=breaks_adj)
    except Exception as e:
        print("Error during binning: ")
        print(e)
    else:
        bins_df = pd.concat(bins.values(), keys=bins.keys())
        bins_df.reset_index(inplace=True)
        bins_df.drop(columns=["variable"], inplace=True)
        bins_df.rename(columns={"level_0": "variable"}, inplace=True)

        bins_df["bin_number"] = bins_df.groupby("variable").cumcount()

        return bins_df


def split(
    df: pd.DataFrame,
    validation_split: Optional[float] = None,
    test_size: float = 0.2,
    add_constant: bool = False,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into train, validation (optional), and test sets.

    Args:
        df: Input DataFrame
        validation_split: If None, returns train/test split. If float, returns train/val/test split
        test_size: Proportion of data for test set (default: 0.2)
        add_constant: Whether to add constant column for statsmodels (default: False)

    Returns:
        If validation_size is None:
            train_df, test_df
        If validation_size is float:
            train_df, validation_df, test_df
    """
    df = df.copy()

    # First split off the test set
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    if add_constant:
        test_df = sm.add_constant(test_df)

    if validation_split is None:
        if add_constant:
            train_val_df = sm.add_constant(train_val_df)

        # Print details for two-way split
        if verbose:
            print("After splitting the dataset into training and test sets:")
            print(
                f"Training Dataset:\nRows: {train_val_df.shape[0]}\nColumns: {train_val_df.shape[1]}\n"
                f"Missing values: {train_val_df.isnull().sum().sum()}\n"
            )
            print(
                f"Test Dataset:\nRows: {test_df.shape[0]}\nColumns: {test_df.shape[1]}\n"
                f"Missing values: {test_df.isnull().sum().sum()}\n"
            )

        return train_val_df, test_df

    # Calculate validation size as proportion of remaining data
    val_size = validation_split / (1 - test_size)
    train_df, validation_df = train_test_split(
        train_val_df, test_size=val_size, random_state=42
    )

    if add_constant:
        train_df = sm.add_constant(train_df)
        validation_df = sm.add_constant(validation_df)

    # Print details for three-way split
    if verbose:
        print("After splitting the dataset into training, validation, and test sets:")
        print(
            f"Training Dataset:\nRows: {train_df.shape[0]}\nColumns: {train_df.shape[1]}\n"
            f"Missing values: {train_df.isnull().sum().sum()}\n"
        )
        print(
            f"Validation Dataset:\nRows: {validation_df.shape[0]}\nColumns: {validation_df.shape[1]}\n"
            f"Missing values: {validation_df.isnull().sum().sum()}\n"
        )
        print(
            f"Test Dataset:\nRows: {test_df.shape[0]}\nColumns: {test_df.shape[1]}\n"
            f"Missing values: {test_df.isnull().sum().sum()}\n"
        )

    return train_df, validation_df, test_df


def compute_scores(probabilities: np.ndarray) -> np.ndarray:
    target_score = score_params["target_score"]
    target_odds = score_params["target_odds"]
    pdo = score_params["pdo"]

    factor = pdo / np.log(2)
    offset = target_score - (factor * np.log(target_odds))

    # Add negative sign to reverse the relationship
    scores = offset - factor * np.log(probabilities / (1 - probabilities))

    return scores


def get_demo_test_config(
    x_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Get demo test configuration.

    Args:
        x_test: Test features DataFrame
        y_test: Test target Series

    Returns:
        dict: Test configuration dictionary
    """
    default_config = {}

    # RAW DATA TESTS
    default_config["validmind.data_validation.DatasetDescription:raw_data"] = {
        "inputs": {
            "dataset": "raw_dataset",
        }
    }
    default_config["validmind.data_validation.DescriptiveStatistics:raw_data"] = {
        "inputs": {
            "dataset": "raw_dataset",
        }
    }
    default_config["validmind.data_validation.MissingValues:raw_data"] = {
        "inputs": {
            "dataset": "raw_dataset",
        },
        "params": {"min_threshold": 1},
    }
    default_config["validmind.data_validation.ClassImbalance:raw_data"] = {
        "inputs": {
            "dataset": "raw_dataset",
        },
        "params": {"min_percent_threshold": 10},
    }
    default_config["validmind.data_validation.Duplicates:raw_data"] = {
        "inputs": {
            "dataset": "raw_dataset",
        },
        "params": {"min_threshold": 1},
    }
    default_config["validmind.data_validation.HighCardinality:raw_data"] = {
        "inputs": {
            "dataset": "raw_dataset",
        },
        "params": {
            "num_threshold": 100,
            "percent_threshold": 0.1,
            "threshold_type": "percent",
        },
    }
    default_config["validmind.data_validation.Skewness:raw_data"] = {
        "inputs": {
            "dataset": "raw_dataset",
        },
        "params": {"max_threshold": 1},
    }
    default_config["validmind.data_validation.UniqueRows:raw_data"] = {
        "inputs": {
            "dataset": "raw_dataset",
        },
        "params": {"min_percent_threshold": 1},
    }
    default_config["validmind.data_validation.TooManyZeroValues:raw_data"] = {
        "inputs": {
            "dataset": "raw_dataset",
        },
        "params": {"max_percent_threshold": 0.03},
    }
    default_config["validmind.data_validation.IQROutliersTable:raw_data"] = {
        "inputs": {
            "dataset": "raw_dataset",
        },
        "params": {"threshold": 5},
    }

    # PREPROCESSED DATA TESTS
    default_config[
        "validmind.data_validation.DescriptiveStatistics:preprocessed_data"
    ] = {
        "inputs": {
            "dataset": "preprocess_dataset",
        }
    }
    default_config[
        "validmind.data_validation.TabularDescriptionTables:preprocessed_data"
    ] = {
        "inputs": {
            "dataset": "preprocess_dataset",
        }
    }
    default_config["validmind.data_validation.MissingValues:preprocessed_data"] = {
        "inputs": {
            "dataset": "preprocess_dataset",
        },
        "params": {"min_threshold": 1},
    }
    default_config[
        "validmind.data_validation.TabularNumericalHistograms:preprocessed_data"
    ] = {
        "inputs": {
            "dataset": "preprocess_dataset",
        }
    }
    default_config[
        "validmind.data_validation.TabularCategoricalBarPlots:preprocessed_data"
    ] = {
        "inputs": {
            "dataset": "preprocess_dataset",
        }
    }
    default_config["validmind.data_validation.TargetRateBarPlots:preprocessed_data"] = {
        "inputs": {
            "dataset": "preprocess_dataset",
        },
        "params": {"default_column": "loan_status"},
    }

    # DEVELOPMENT DATA TESTS
    default_config[
        "validmind.data_validation.DescriptiveStatistics:development_data"
    ] = {"input_grid": {"dataset": ["train_dataset", "test_dataset"]}}

    default_config[
        "validmind.data_validation.TabularDescriptionTables:development_data"
    ] = {"input_grid": {"dataset": ["train_dataset", "test_dataset"]}}

    default_config["validmind.data_validation.ClassImbalance:development_data"] = {
        "input_grid": {"dataset": ["train_dataset", "test_dataset"]},
        "params": {"min_percent_threshold": 10},
    }

    default_config["validmind.data_validation.UniqueRows:development_data"] = {
        "input_grid": {"dataset": ["train_dataset", "test_dataset"]},
        "params": {"min_percent_threshold": 1},
    }

    default_config[
        "validmind.data_validation.TabularNumericalHistograms:development_data"
    ] = {"input_grid": {"dataset": ["train_dataset", "test_dataset"]}}

    # FEATURE SELECTION TESTS
    default_config["validmind.data_validation.MutualInformation:development_data"] = {
        "input_grid": {"dataset": ["train_dataset", "test_dataset"]},
        "params": {"min_threshold": 0.01},
    }

    default_config[
        "validmind.data_validation.PearsonCorrelationMatrix:development_data"
    ] = {"input_grid": {"dataset": ["train_dataset", "test_dataset"]}}

    default_config[
        "validmind.data_validation.HighPearsonCorrelation:development_data"
    ] = {
        "input_grid": {"dataset": ["train_dataset", "test_dataset"]},
        "params": {"max_threshold": 0.3, "top_n_correlations": 10},
    }

    default_config["validmind.data_validation.WOEBinTable"] = {
        "input_grid": {"dataset": ["preprocess_dataset"]},
        "params": {"breaks_adj": breaks_adj},
    }

    default_config["validmind.data_validation.WOEBinPlots"] = {
        "input_grid": {"dataset": ["preprocess_dataset"]},
        "params": {"breaks_adj": breaks_adj},
    }

    # MODEL TRAINING TESTS
    default_config["validmind.data_validation.DatasetSplit"] = {
        "inputs": {"datasets": ["train_dataset", "test_dataset"]}
    }

    default_config["validmind.model_validation.ModelMetadata"] = {
        "input_grid": {"model": ["xgb_model", "rf_model"]}
    }

    default_config["validmind.model_validation.sklearn.ModelParameters"] = {
        "input_grid": {"model": ["xgb_model", "rf_model"]}
    }

    # MODEL SELECTION TESTS
    default_config["validmind.model_validation.statsmodels.GINITable"] = {
        "input_grid": {
            "dataset": ["train_dataset", "test_dataset"],
            "model": ["xgb_model", "rf_model"],
        }
    }

    default_config["validmind.model_validation.sklearn.ClassifierPerformance"] = {
        "input_grid": {
            "dataset": ["train_dataset", "test_dataset"],
            "model": ["xgb_model", "rf_model"],
        }
    }

    default_config[
        "validmind.model_validation.sklearn.TrainingTestDegradation:XGBoost"
    ] = {
        "inputs": {"datasets": ["train_dataset", "test_dataset"], "model": "xgb_model"},
        "params": {"max_threshold": 0.1},
    }

    default_config[
        "validmind.model_validation.sklearn.TrainingTestDegradation:RandomForest"
    ] = {
        "inputs": {"datasets": ["train_dataset", "test_dataset"], "model": "rf_model"},
        "params": {"max_threshold": 0.1},
    }

    default_config["validmind.model_validation.sklearn.HyperParametersTuning"] = {
        "inputs": {"model": "xgb_model", "dataset": "train_dataset"},
        "params": {
            "param_grid": {"n_estimators": [50, 100]},
            "scoring": ["roc_auc", "recall"],
            "fit_params": {
                "eval_set": [(x_test, y_test)],
                "verbose": False,
            },
            "thresholds": [0.3, 0.5],
        },
    }

    # MODEL PERFORMANCE - DISCRIMINATION TESTS
    default_config["validmind.model_validation.sklearn.ROCCurve"] = {
        "input_grid": {
            "dataset": ["train_dataset", "test_dataset"],
            "model": ["xgb_model"],
        }
    }

    default_config["validmind.model_validation.sklearn.MinimumROCAUCScore"] = {
        "input_grid": {
            "dataset": ["train_dataset", "test_dataset"],
            "model": ["xgb_model"],
        },
        "params": {"min_threshold": 0.5},
    }

    default_config[
        "validmind.model_validation.statsmodels.PredictionProbabilitiesHistogram"
    ] = {
        "input_grid": {
            "dataset": ["train_dataset", "test_dataset"],
            "model": ["xgb_model"],
        }
    }

    default_config[
        "validmind.model_validation.statsmodels.CumulativePredictionProbabilities"
    ] = {
        "input_grid": {
            "model": ["xgb_model"],
            "dataset": ["train_dataset", "test_dataset"],
        }
    }

    default_config["validmind.model_validation.sklearn.PopulationStabilityIndex"] = {
        "inputs": {"datasets": ["train_dataset", "test_dataset"], "model": "xgb_model"},
        "params": {"num_bins": 10, "mode": "fixed"},
    }

    # MODEL PERFORMANCE - ACCURACY TESTS
    default_config["validmind.model_validation.sklearn.ConfusionMatrix"] = {
        "input_grid": {
            "dataset": ["train_dataset", "test_dataset"],
            "model": ["xgb_model"],
        }
    }

    default_config["validmind.model_validation.sklearn.MinimumAccuracy"] = {
        "input_grid": {
            "dataset": ["train_dataset", "test_dataset"],
            "model": ["xgb_model"],
        },
        "params": {"min_threshold": 0.7},
    }

    default_config["validmind.model_validation.sklearn.MinimumF1Score"] = {
        "input_grid": {
            "dataset": ["train_dataset", "test_dataset"],
            "model": ["xgb_model"],
        },
        "params": {"min_threshold": 0.5},
    }

    default_config["validmind.model_validation.sklearn.PrecisionRecallCurve"] = {
        "input_grid": {
            "dataset": ["train_dataset", "test_dataset"],
            "model": ["xgb_model"],
        }
    }

    default_config["validmind.model_validation.sklearn.CalibrationCurve"] = {
        "input_grid": {
            "dataset": ["train_dataset", "test_dataset"],
            "model": ["xgb_model"],
        }
    }

    default_config[
        "validmind.model_validation.sklearn.ClassifierThresholdOptimization"
    ] = {
        "inputs": {"dataset": "train_dataset", "model": "xgb_model"},
        "params": {
            "target_recall": 0.8  # Find a threshold that achieves a recall of 80%
        },
    }

    # MODEL PERFORMANCE - SCORING TESTS
    default_config["validmind.model_validation.statsmodels.ScorecardHistogram"] = {
        "input_grid": {"dataset": ["train_dataset", "test_dataset"]},
        "params": {"score_column": "xgb_scores"},
    }

    default_config["validmind.data_validation.ScoreBandDefaultRates"] = {
        "input_grid": {"dataset": ["train_dataset"], "model": ["xgb_model"]},
        "params": {
            "score_column": "xgb_scores",
            "score_bands": [504, 537, 570],  # Creates four score bands
        },
    }

    default_config["validmind.model_validation.sklearn.ScoreProbabilityAlignment"] = {
        "input_grid": {"dataset": ["train_dataset"], "model": ["xgb_model"]},
        "params": {"score_column": "xgb_scores"},
    }

    # MODEL DIAGNOSIS TESTS
    default_config["validmind.model_validation.sklearn.WeakspotsDiagnosis"] = {
        "inputs": {
            "datasets": ["train_dataset", "test_dataset"],
            "model": "xgb_model",
        },
    }

    default_config["validmind.model_validation.sklearn.OverfitDiagnosis"] = {
        "inputs": {
            "model": "xgb_model",
            "datasets": ["train_dataset", "test_dataset"],
        },
        "params": {"cut_off_threshold": 0.04},
    }

    default_config["validmind.model_validation.sklearn.RobustnessDiagnosis"] = {
        "inputs": {
            "datasets": ["train_dataset", "test_dataset"],
            "model": "xgb_model",
        },
        "params": {
            "scaling_factor_std_dev_list": [0.1, 0.2, 0.3, 0.4, 0.5],
            "performance_decay_threshold": 0.05,
        },
    }

    # EXPLAINABILITY TESTS
    default_config[
        "validmind.model_validation.sklearn.PermutationFeatureImportance"
    ] = {
        "input_grid": {
            "dataset": ["train_dataset", "test_dataset"],
            "model": ["xgb_model"],
        }
    }

    default_config["validmind.model_validation.FeaturesAUC"] = {
        "input_grid": {
            "model": ["xgb_model"],
            "dataset": ["train_dataset", "test_dataset"],
        },
    }

    default_config["validmind.model_validation.sklearn.SHAPGlobalImportance"] = {
        "input_grid": {
            "model": ["xgb_model"],
            "dataset": ["train_dataset", "test_dataset"],
        },
        "params": {
            "kernel_explainer_samples": 10,
            "tree_or_linear_explainer_samples": 200,
        },
    }

    return default_config


def load_scorecard():

    warnings.filterwarnings("ignore")
    logging.getLogger("scorecardpy").setLevel(logging.ERROR)

    os.environ["VALIDMIND_LLM_DESCRIPTIONS_CONTEXT_ENABLED"] = "1"

    context = """
    FORMAT FOR THE LLM DESCRIPTIONS:
    **<Test Name>** is designed to <begin with a concise overview of what the test does and its primary purpose, extracted from the test description>.

    The test operates by <write a paragraph about the test mechanism, explaining how it works and what it measures. Include any relevant formulas or methodologies mentioned in the test description.>

    The primary advantages of this test include <write a paragraph about the test's strengths and capabilities, highlighting what makes it particularly useful for specific scenarios.>

    Users should be aware that <write a paragraph about the test's limitations and potential risks. Include both technical limitations and interpretation challenges. If the test description includes specific signs of high risk, incorporate these here.>

    **Key Insights:**

    The test results reveal:

    - **<insight title>**: <comprehensive description of one aspect of the results>
    - **<insight title>**: <comprehensive description of another aspect>
    ...

    Based on these results, <conclude with a brief paragraph that ties together the test results with the test's purpose and provides any final recommendations or considerations.>

    ADDITIONAL INSTRUCTIONS:
        Present insights in order from general to specific, with each insight as a single bullet point with bold title.

        For each metric in the test results, include in the test overview:
        - The metric's purpose and what it measures
        - Its mathematical formula
        - The range of possible values
        - What constitutes good/bad performance
        - How to interpret different values

        Each insight should progressively cover:
        1. Overall scope and distribution
        2. Complete breakdown of all elements with specific values
        3. Natural groupings and patterns
        4. Comparative analysis between datasets/categories
        5. Stability and variations
        6. Notable relationships or dependencies

        Remember:
        - Keep all insights at the same level (no sub-bullets or nested structures)
        - Make each insight complete and self-contained
        - Include specific numerical values and ranges
        - Cover all elements in the results comprehensively
        - Maintain clear, concise language
        - Use only "- **Title**: Description" format for insights
        - Progress naturally from general to specific observations

    """.strip()

    os.environ["VALIDMIND_LLM_DESCRIPTIONS_CONTEXT"] = context

    # Load the data
    df = load_data(source="offline", verbose=False)
    preprocess_df = preprocess(df, verbose=False)
    fe_df = feature_engineering(preprocess_df, verbose=False)

    # Split the data
    train_df, test_df = split(fe_df, test_size=0.2, verbose=False)

    x_train = train_df.drop(target_column, axis=1)
    y_train = train_df[target_column]

    x_test = test_df.drop(target_column, axis=1)
    y_test = test_df[target_column]

    # Define the XGBoost model
    xgb_model = xgb.XGBClassifier(
        n_estimators=50, random_state=42, early_stopping_rounds=10
    )
    xgb_model.set_params(
        eval_metric=["error", "logloss", "auc"],
    )

    # Fit the model
    xgb_model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)

    # Define the Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
    )

    # Fit the model
    rf_model.fit(x_train, y_train)

    # Compute the probabilities
    train_xgb_prob = xgb_model.predict_proba(x_train)[:, 1]
    test_xgb_prob = xgb_model.predict_proba(x_test)[:, 1]

    train_rf_prob = rf_model.predict_proba(x_train)[:, 1]
    test_rf_prob = rf_model.predict_proba(x_test)[:, 1]

    # Compute binary predictions
    cut_off_threshold = 0.3

    train_xgb_binary_predictions = (train_xgb_prob > cut_off_threshold).astype(int)
    test_xgb_binary_predictions = (test_xgb_prob > cut_off_threshold).astype(int)

    train_rf_binary_predictions = (train_rf_prob > cut_off_threshold).astype(int)
    test_rf_binary_predictions = (test_rf_prob > cut_off_threshold).astype(int)

    # Compute credit risk scores
    train_xgb_scores = compute_scores(train_xgb_prob)
    test_xgb_scores = compute_scores(test_xgb_prob)

    scorecard = {
        "df": df,
        "preprocess_df": preprocess_df,
        "fe_df": fe_df,
        "train_df": train_df,
        "test_df": test_df,
        "x_test": x_test,
        "y_test": y_test,
        "xgb_model": xgb_model,
        "rf_model": rf_model,
        "train_xgb_binary_predictions": train_xgb_binary_predictions,
        "test_xgb_binary_predictions": test_xgb_binary_predictions,
        "train_xgb_prob": train_xgb_prob,
        "test_xgb_prob": test_xgb_prob,
        "train_xgb_scores": train_xgb_scores,
        "test_xgb_scores": test_xgb_scores,
        "train_rf_binary_predictions": train_rf_binary_predictions,
        "test_rf_binary_predictions": test_rf_binary_predictions,
        "train_rf_prob": train_rf_prob,
        "test_rf_prob": test_rf_prob,
    }

    return scorecard


def init_vm_objects(scorecard):

    df = scorecard["df"]
    preprocess_df = scorecard["preprocess_df"]
    fe_df = scorecard["fe_df"]
    train_df = scorecard["train_df"]
    test_df = scorecard["test_df"]
    xgb_model = scorecard["xgb_model"]
    rf_model = scorecard["rf_model"]
    train_xgb_binary_predictions = scorecard["train_xgb_binary_predictions"]
    test_xgb_binary_predictions = scorecard["test_xgb_binary_predictions"]
    train_xgb_prob = scorecard["train_xgb_prob"]
    test_xgb_prob = scorecard["test_xgb_prob"]
    train_rf_binary_predictions = scorecard["train_rf_binary_predictions"]
    test_rf_binary_predictions = scorecard["test_rf_binary_predictions"]
    train_rf_prob = scorecard["train_rf_prob"]
    test_rf_prob = scorecard["test_rf_prob"]
    train_xgb_scores = scorecard["train_xgb_scores"]
    test_xgb_scores = scorecard["test_xgb_scores"]

    vm.init_dataset(
        dataset=df,
        input_id="raw_dataset",
        target_column=target_column,
    )

    vm.init_dataset(
        dataset=preprocess_df,
        input_id="preprocess_dataset",
        target_column=target_column,
    )

    vm.init_dataset(
        dataset=fe_df,
        input_id="fe_dataset",
        target_column=target_column,
    )

    vm_train_ds = vm.init_dataset(
        dataset=train_df,
        input_id="train_dataset",
        target_column=target_column,
    )

    vm_test_ds = vm.init_dataset(
        dataset=test_df,
        input_id="test_dataset",
        target_column=target_column,
    )

    vm_xgb_model = vm.init_model(
        xgb_model,
        input_id="xgb_model",
    )

    vm_rf_model = vm.init_model(
        rf_model,
        input_id="rf_model",
    )

    # Assign predictions
    vm_train_ds.assign_predictions(
        model=vm_xgb_model,
        prediction_values=train_xgb_binary_predictions,
        prediction_probabilities=train_xgb_prob,
    )

    vm_test_ds.assign_predictions(
        model=vm_xgb_model,
        prediction_values=test_xgb_binary_predictions,
        prediction_probabilities=test_xgb_prob,
    )

    vm_train_ds.assign_predictions(
        model=vm_rf_model,
        prediction_values=train_rf_binary_predictions,
        prediction_probabilities=train_rf_prob,
    )

    vm_test_ds.assign_predictions(
        model=vm_rf_model,
        prediction_values=test_rf_binary_predictions,
        prediction_probabilities=test_rf_prob,
    )

    # Assign scores to the datasets
    vm_train_ds.add_extra_column("xgb_scores", train_xgb_scores)
    vm_test_ds.add_extra_column("xgb_scores", test_xgb_scores)


def load_test_config(scorecard):

    x_test = scorecard["x_test"]
    y_test = scorecard["y_test"]

    # Get the test config
    test_config = get_demo_test_config(x_test, y_test)

    return test_config
