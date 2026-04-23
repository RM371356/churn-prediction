"""BB-01 to BB-06: Black Box Tests -- business scenario validation."""

import numpy as np
import pandas as pd
import pytest


def _make_single_customer(overrides: dict, X_template: pd.DataFrame) -> pd.DataFrame:
    """Create a single-row DataFrame matching X's schema, with overrides."""
    row = X_template.iloc[[0]].copy()
    for col, val in overrides.items():
        if col in row.columns:
            row[col] = val
    return row


class TestBlackBox:

    # BB-01 -- High churn risk profile
    def test_monthly_high_charge_low_tenure_churns(self, fitted_log_pipeline, X_y):
        X, _ = X_y
        customer = _make_single_customer({
            "Contract": "Month-to-month",
            "Monthly Charges": 100.0,
            "Tenure Months": 2,
            "Internet Service": "Fiber optic",
            "Online Security": "No",
            "Tech Support": "No",
        }, X)
        proba = fitted_log_pipeline.predict_proba(customer)[0, 1]
        assert proba > 0.5, f"Expected high churn probability, got {proba:.4f}"

    # BB-02 -- Low churn risk profile
    def test_two_year_low_charge_high_tenure_stays(self, fitted_log_pipeline, X_y):
        X, _ = X_y
        customer = _make_single_customer({
            "Contract": "Two year",
            "Monthly Charges": 30.0,
            "Tenure Months": 60,
            "Internet Service": "DSL",
            "Online Security": "Yes",
            "Tech Support": "Yes",
        }, X)
        pred = fitted_log_pipeline.predict(customer)[0]
        assert pred == 0, "Expected non-churn for stable customer"

    # BB-03 -- Senior without support
    def test_senior_no_support_high_risk(self, fitted_log_pipeline, X_y):
        X, _ = X_y
        customer = _make_single_customer({
            "Senior Citizen": "Yes",
            "Tech Support": "No",
            "Online Security": "No",
            "Contract": "Month-to-month",
            "Internet Service": "Fiber optic",
            "Tenure Months": 5,
            "Monthly Charges": 90.0,
        }, X)
        proba = fitted_log_pipeline.predict_proba(customer)[0, 1]
        assert proba > 0.4, f"Expected elevated churn prob for senior, got {proba:.4f}"

    # BB-04 -- Fiber optic + electronic check
    def test_fiber_electronic_check_elevated_risk(self, fitted_log_pipeline, X_y):
        X, _ = X_y
        customer = _make_single_customer({
            "Internet Service": "Fiber optic",
            "Payment Method": "Electronic check",
            "Contract": "Month-to-month",
            "Monthly Charges": 85.0,
            "Tenure Months": 6,
        }, X)
        proba = fitted_log_pipeline.predict_proba(customer)[0, 1]
        assert proba > 0.4, f"Expected elevated churn prob, got {proba:.4f}"

    # BB-05 -- Batch prediction distribution
    def test_batch_prediction_distribution(self, fitted_log_pipeline, X_y):
        X, _ = X_y
        rng = np.random.default_rng(42)
        indices = rng.choice(len(X), 100, replace=False)
        sample = X.iloc[indices]
        preds = fitted_log_pipeline.predict(sample)
        churn_rate = preds.mean()
        assert 0.10 < churn_rate < 0.60, (
            f"Churn rate {churn_rate:.1%} outside plausible range"
        )

    # BB-06 -- Determinism
    def test_identical_input_same_output(self, fitted_log_pipeline, X_y):
        X, _ = X_y
        sample = X.head(50)
        preds1 = fitted_log_pipeline.predict(sample)
        proba1 = fitted_log_pipeline.predict_proba(sample)
        preds2 = fitted_log_pipeline.predict(sample)
        proba2 = fitted_log_pipeline.predict_proba(sample)
        np.testing.assert_array_equal(preds1, preds2)
        np.testing.assert_array_almost_equal(proba1, proba2)
