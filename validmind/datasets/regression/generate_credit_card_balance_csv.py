# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""
Helper script to generate the bundled synthetic credit card balance CSV file.

The data is synthetic: every value is drawn from the distributions below with a
fixed seed. Each row is a cardholder, and the target is their average monthly
revolving balance. The relationships are chosen so a linear regression
validation has something real to find:

- CreditLimit is driven by CreditScore, so the two are almost collinear (high VIF).
- TenureYears and Region have no effect on Balance (not statistically significant;
  Region is a control segment where segment tests should find nothing).
- RecentInquiries has a diminishing effect on Balance, so a straight-line term
  misspecifies it.
- Customers who pay in full have a Balance of 0, which breaks the normality and
  constant-variance assumptions of ordinary least squares.
- Subprime customers (CreditScore below 580) carry more than their other
  characteristics explain. A model without CreditScore under-predicts this segment,
  which only a segment-level check reveals.
- A few incomes were keyed with a misplaced decimal point (x10), creating
  influential observations.
- Customers are observed over 12 quarters (2023Q1-2025Q4). From OOT_START a new
  acquisition campaign brings in lower-income customers (population drift), and
  higher rates make every customer carry a larger balance (behaviour drift), so a
  model developed on earlier quarters under-predicts the out-of-time sample.

Run this script to regenerate the bundled dataset file:
    python validmind/datasets/regression/generate_credit_card_balance_csv.py
"""

import os

import numpy as np
import pandas as pd

current_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_path, "datasets")
output_file = os.path.join(dataset_path, "credit_card_balance.csv")

SEED = 43
N_CUSTOMERS = 5000
N_INCOME_ENTRY_ERRORS = 5
REGIONS = ["Northeast", "Midwest", "South", "West"]
QUARTERS = [f"{year}Q{q}" for year in (2023, 2024, 2025) for q in (1, 2, 3, 4)]
OOT_START = "2025Q3"


def generate(n=N_CUSTOMERS, seed=SEED):
    rng = np.random.default_rng(seed)

    quarter = np.sort(rng.choice(QUARTERS, n))
    oot = quarter >= OOT_START

    median_income = np.where(oot, 30, 40)  # acquisition campaign shifts the mix
    income = np.clip(rng.lognormal(np.log(median_income), 0.6), 10, 200).round(1)
    dti = np.clip(rng.normal(0.30, 0.10, n) - 0.002 * (income - 40), 0.0, 0.8)
    dti = dti.round(3)
    score = rng.normal(650, 80, n) + 0.2 * (income - 40) - 100 * (dti - 0.3)
    score = np.clip(score, 300, 850).round()
    limit = (15 * (score - 300) + 10 * income + rng.normal(0, 120, n)).clip(500)
    limit = limit.round(-1)
    inquiries = np.clip(rng.poisson(1.5, n), 0, 10)
    cards = rng.integers(1, 8, n)
    tenure = rng.integers(0, 31, n)
    student = (rng.random(n) < 0.1).astype(int)
    region = rng.choice(REGIONS, n)

    balance = (
        -1250
        + 0.25 * limit
        - 8 * income
        + 600 * dti
        + 250 * np.log1p(inquiries)  # diminishing effect
        + 20 * cards
        + 400 * student
        + 250 * (score < 580)  # subprime customers revolve more
        + 150 * oot  # higher rates: everyone revolves more
        + rng.normal(0, 120, n)
    )
    balance = balance.clip(0).round()

    # Income keyed with a misplaced decimal point for a few customers. Balance was
    # generated from the true income, so these rows are data errors, not signal.
    errors = rng.choice(np.where(~oot)[0], N_INCOME_ENTRY_ERRORS, replace=False)
    income[errors] = income[errors] * 10

    return pd.DataFrame(
        {
            "Quarter": quarter,
            "Income": income,
            "DebtToIncome": dti,
            "CreditScore": score.astype(int),
            "CreditLimit": limit.astype(int),
            "RecentInquiries": inquiries,
            "NumCards": cards,
            "TenureYears": tenure,
            "IsStudent": student,
            "Region": region,
            "Balance": balance.astype(int),
        }
    )


if __name__ == "__main__":
    df = generate()

    os.makedirs(dataset_path, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"✓ Saved {len(df)} rows and {len(df.columns)} columns to {output_file}")
