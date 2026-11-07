# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""Samples and models for the CECL residential mortgage notebooks.

Two model sets share one interface — default and prepayment classifiers and an LGD
regressor, all taking ``model_features`` frames — so either runs through the lifetime
loss engine in ``cecl_model.py``:

- **champion**: gradient-boosted trees with monotonic constraints.
- **challenger**: logistic hazards and a linear LGD on piecewise-linear shapes, the
  transparent benchmark a bank would hold the champion against.

Both are fitted on the train sample only. ``load_models`` reads them from
``models/cecl_models.joblib`` and refits when the file is missing, was saved with
another scikit-learn version, or predates this code. Run ``python cecl_train.py`` to
refit and save explicitly.
"""

from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

from cecl_data import simulate_portfolio
from cecl_model import model_features

HERE = Path(__file__).parent
MODEL_FILE = HERE / "models" / "cecl_models.joblib"
SEED = 42
DEV_END = pd.Period("2019Q4")
TEST_SHARE = 0.3

GBM_PARAMS = dict(
    learning_rate=0.05,
    max_iter=500,
    max_leaf_nodes=15,
    min_samples_leaf=500,
    l2_regularization=1.0,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=30,
    categorical_features="from_dtype",
    random_state=SEED,
)
DEFAULT_DIRECTION = {"fico": -1, "current_ltv": 1, "dti": 1, "unemployment_rate": 1}
PREPAY_DIRECTION = {"refi_incentive": 1, "current_ltv": -1}
SHAPED = ["fico", "current_ltv", "dti", "loan_age", "unemployment_rate", "refi_incentive"]
DUMMIES = ["occupancy", "purpose", "term_months"]


def split_samples(panel):
    """Train, test and out-of-time loan-quarter samples, plus their defaults.

    Quarters through ``DEV_END`` are split by loan, so no loan has quarters on both
    sides; later quarters are out of time. Quarters in which a loan reached contractual
    maturity are dropped, since that payoff is scheduled, not a prepayment decision.
    """
    panel = panel.assign(
        defaulted=(panel["outcome"] == 1).astype(int),
        prepaid=(panel["outcome"] == 2).astype(int),
    )
    modelling = panel[3 * panel["loan_age"] < panel["term_months"]]
    dev = modelling[modelling["quarter"] <= DEV_END]
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SHARE, random_state=SEED)
    train_idx, test_idx = next(splitter.split(dev, groups=dev["loan_id"]))
    samples = {
        "train": dev.iloc[train_idx],
        "test": dev.iloc[test_idx],
        "oot": modelling[modelling["quarter"] > DEV_END],
    }
    for name in list(samples):
        df = samples[name]
        samples[f"{name}_defaults"] = df[df["defaulted"] == 1]
    return samples


def linear_shapes(estimator):
    """``estimator`` on hinge-function shapes of the numeric drivers plus dummies.

    Degree-1 splines with quantile knots are piecewise-linear terms — the seasoning
    curve, the refinance S-curve and the LTV kinks a logit needs built by hand. Beyond
    the training range they extrapolate linearly, where the trees stay flat.
    """
    shapes = make_column_transformer(
        (
            SplineTransformer(
                degree=1, n_knots=5, knots="quantile", extrapolation="linear"
            ),
            SHAPED,
        ),
        (OneHotEncoder(handle_unknown="ignore"), DUMMIES),
    )
    return make_pipeline(shapes, estimator)


def fit_models(train, train_defaults):
    """Fit the champion and challenger model sets on the train sample."""
    X, X_lgd = model_features(train), model_features(train_defaults)
    champion = {
        "default": HistGradientBoostingClassifier(
            monotonic_cst=DEFAULT_DIRECTION, **GBM_PARAMS
        ).fit(X, train["defaulted"]),
        "prepay": HistGradientBoostingClassifier(
            monotonic_cst=PREPAY_DIRECTION, **GBM_PARAMS
        ).fit(X, train["prepaid"]),
        # With only a few hundred defaults the LGD trees are kept shallow.
        "lgd": HistGradientBoostingRegressor(
            monotonic_cst={"current_ltv": 1},
            learning_rate=0.05,
            max_iter=200,
            max_depth=3,
            min_samples_leaf=50,
            categorical_features="from_dtype",
            random_state=SEED,
        ).fit(X_lgd, train_defaults["lgd"]),
    }
    challenger = {
        "default": linear_shapes(LogisticRegression(max_iter=1000)).fit(
            X, train["defaulted"]
        ),
        "prepay": linear_shapes(LogisticRegression(max_iter=1000)).fit(
            X, train["prepaid"]
        ),
        "lgd": linear_shapes(Ridge(alpha=1.0)).fit(X_lgd, train_defaults["lgd"]),
    }
    return champion, challenger


def train_and_save():
    _, panel = simulate_portfolio(seed=SEED)
    samples = split_samples(panel)
    champion, challenger = fit_models(samples["train"], samples["train_defaults"])
    MODEL_FILE.parent.mkdir(exist_ok=True)
    joblib.dump(
        {
            "champion": champion,
            "challenger": challenger,
            "sklearn_version": sklearn.__version__,
        },
        MODEL_FILE,
    )
    return champion, challenger


def load_models():
    """The ``(champion, challenger)`` model sets, each a dict of default, prepay and lgd."""
    sources = [HERE / f for f in ["cecl_train.py", "cecl_data.py", "cecl_model.py"]]
    if MODEL_FILE.exists() and MODEL_FILE.stat().st_mtime > max(
        f.stat().st_mtime for f in sources
    ):
        saved = joblib.load(MODEL_FILE)
        # Pickled scikit-learn models are only reliable on the version that saved them.
        if saved["sklearn_version"] == sklearn.__version__:
            return saved["champion"], saved["challenger"]
    print(f"Fitting the CECL models and saving them to {MODEL_FILE.name} ...")
    return train_and_save()


if __name__ == "__main__":
    train_and_save()
    print(f"Saved {MODEL_FILE}")
