# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Quickstart for model documentation

Welcome! Let's get you started with the basic process of documenting models with ValidMind.

You will learn how to initialize the ValidMind Library, load a sample dataset to train a simple classification model, 
and then run a ValidMind test suite to quickly generate documentation about the data and model.

This script uses the Bank Customer Churn Prediction sample dataset from Kaggle to train the classification model.
"""

# Import required libraries
import validmind as vm
import xgboost as xgb
import logging
import os
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from validmind.datasets.classification import customer_churn


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_development.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Configuration management
class ConfigManager:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def save_config(self, config: Dict[str, Any]):
        """Save configuration to YAML file"""
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
        self.config = config


# Initialize configuration
config_manager = ConfigManager()
model_config = {
    "model_name": "customer_churn_xgboost",
    "version": "1.0.0",
    "parameters": {
        "early_stopping_rounds": 10,
        "eval_metric": ["error", "logloss", "auc"]
    },
    "security": {
        "api_key_rotation_days": 30,
        "data_encryption": True
    }
}
config_manager.save_config(model_config)

# Initialize the ValidMind Library
vm.init(
    api_host="...",
    api_key="...",
    api_secret="...",
    model="...",
)

# Preview the documentation template
vm.preview_template()

# Load the sample dataset
logger.info("Loading demo dataset")
print(
    f"Loaded demo dataset with: \n\n\t• Target column: '{customer_churn.target_column}' \n\t• Class labels: {customer_churn.class_labels}"
)

raw_df = customer_churn.load_data()
print("\nFirst few rows of the dataset:")
print(raw_df.head())

# Preprocess the raw dataset
logger.info("Preprocessing dataset")
train_df, validation_df, test_df = customer_churn.preprocess(raw_df)

x_train = train_df.drop(customer_churn.target_column, axis=1)
y_train = train_df[customer_churn.target_column]
x_val = validation_df.drop(customer_churn.target_column, axis=1)
y_val = validation_df[customer_churn.target_column]
x_test = test_df.drop(customer_churn.target_column, axis=1)
y_test = test_df[customer_churn.target_column]

# Initialize and train the XGBoost model
logger.info("Training XGBoost model")
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


# Model versioning and artifact management
class ModelVersioning:
    def __init__(self, model_dir: str = "model_artifacts"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

    def save_model(self, model: Any, version: str):
        """Save model artifacts with versioning"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_dir / f"model_v{version}_{timestamp}.json"
        model.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")

    def load_model(self, version: str) -> Any:
        """Load model artifacts by version"""
        model_files = list(self.model_dir.glob(f"model_v{version}_*.json"))
        if not model_files:
            raise ValueError(f"No model found for version {version}")
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        model = xgb.XGBClassifier()
        model.load_model(str(latest_model))
        return model


# Initialize model versioning
model_versioning = ModelVersioning()
model_versioning.save_model(model, model_config["version"])

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

# Define test configuration
test_config = customer_churn.get_demo_test_config()

# Run the full suite of tests
logger.info("Running documentation tests")
full_suite = vm.run_documentation_tests(config=test_config)


# Model inference and scoring
class ModelInference:
    def __init__(self, model: Any):
        self.model = model

    def predict(self, data: Any) -> Any:
        """Make predictions on new data"""
        return self.model.predict(data)

    def predict_proba(self, data: Any) -> Any:
        """Get prediction probabilities"""
        return self.model.predict_proba(data)

    def score(self, data: Any, target: Any) -> float:
        """Calculate model score"""
        return self.model.score(data, target)


# Initialize inference
inference = ModelInference(model)


# Example usage
def example_usage():
    """Example usage of the model"""
    # Load new data
    new_data = x_test[:5]  # Example with first 5 test samples

    # Make predictions
    predictions = inference.predict(new_data)
    probabilities = inference.predict_proba(new_data)

    # Calculate score
    score = inference.score(x_test, y_test)

    logger.info(f"Model score: {score:.4f}")
    logger.info(f"Predictions: {predictions}")
    logger.info(f"Probabilities: {probabilities}")


# Run example
example_usage()

# Note: After running this script, you can view the results in the ValidMind Platform
# by going to the Model Inventory and selecting your model.

if __name__ == "__main__":
    print("\nScript execution completed. Check the ValidMind Platform for results.")
