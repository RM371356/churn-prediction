"""DQ-01 to DQ-09: Data Quality / Validation Tests."""

import numpy as np
import pandas as pd
import pytest

EXPECTED_COLUMNS = [
    "CustomerID", "Count", "Country", "State", "City", "Zip Code",
    "Lat Long", "Latitude", "Longitude", "Gender", "Senior Citizen",
    "Partner", "Dependents", "Tenure Months", "Phone Service",
    "Multiple Lines", "Internet Service", "Online Security",
    "Online Backup", "Device Protection", "Tech Support", "Streaming TV",
    "Streaming Movies", "Contract", "Paperless Billing", "Payment Method",
    "Monthly Charges", "Total Charges", "Churn Label", "Churn Value",
    "Churn Score", "CLTV", "Churn Reason",
]


class TestDataQuality:
    """Validate the integrity and quality of the raw dataset."""

    # DQ-01
    def test_schema_columns(self, raw_df):
        assert list(raw_df.columns) == EXPECTED_COLUMNS

    def test_schema_column_count(self, raw_df):
        assert raw_df.shape[1] == 33

    # DQ-02
    def test_row_count(self, raw_df):
        assert len(raw_df) == 7043

    def test_no_duplicate_customer_ids(self, raw_df):
        assert raw_df["CustomerID"].is_unique

    # DQ-03
    def test_total_charges_nulls(self, raw_df):
        total_charges = pd.to_numeric(raw_df["Total Charges"], errors="coerce")
        null_count = total_charges.isna().sum()
        assert null_count <= 11, f"Expected <= 11 nulls, got {null_count}"

    # DQ-04
    def test_target_distribution(self, raw_df):
        churn_rate = raw_df["Churn Value"].mean()
        assert 0.255 <= churn_rate <= 0.275, (
            f"Expected churn rate ~26.5%, got {churn_rate:.1%}"
        )

    # DQ-05
    def test_contract_categories(self, raw_df):
        expected = {"Month-to-month", "One year", "Two year"}
        actual = set(raw_df["Contract"].dropna().unique())
        assert actual == expected, f"Unexpected contract values: {actual - expected}"

    # DQ-06
    def test_monthly_charges_non_negative(self, raw_df):
        assert (raw_df["Monthly Charges"] >= 0).all()

    def test_total_charges_non_negative(self, raw_df):
        total = pd.to_numeric(raw_df["Total Charges"], errors="coerce")
        assert (total.dropna() >= 0).all()

    # DQ-07
    def test_tenure_months_range(self, raw_df):
        tenure = raw_df["Tenure Months"]
        assert tenure.min() >= 0, f"Min tenure {tenure.min()} < 0"
        assert tenure.max() <= 72, f"Max tenure {tenure.max()} > 72"

    # DQ-08 -- CSV vs XLSX consistency (skipped when xlsx is absent)
    def test_csv_xlsx_consistency(self):
        from tests.conftest import CSV_PATH, XLSX_PATH

        if not XLSX_PATH.exists():
            pytest.skip("XLSX file not present")
        if not CSV_PATH.exists():
            pytest.skip("CSV file not present")

        df_xlsx = pd.read_excel(XLSX_PATH)
        df_csv = pd.read_csv(CSV_PATH)
        assert len(df_xlsx) == len(df_csv), (
            f"Row mismatch: xlsx={len(df_xlsx)}, csv={len(df_csv)}"
        )

    # DQ-09 -- Data drift detection placeholder
    def test_data_drift_ks_test(self, X_y):
        """Kolmogorov-Smirnov test between random train/test split of the
        existing dataset.  Data is shuffled first to avoid geographic ordering
        bias. Excludes geo-coordinates. In production this would compare
        against new incoming data."""
        from scipy import stats

        X, _ = X_y
        geo_cols = {"Latitude", "Longitude"}
        num_cols = [
            c for c in X.select_dtypes(include=np.number).columns
            if c not in geo_cols
        ]

        X_shuffled = X.sample(frac=1, random_state=42)
        n = len(X_shuffled)
        split = n // 2

        for col in num_cols:
            stat, p_value = stats.ks_2samp(
                X_shuffled[col].iloc[:split].dropna(),
                X_shuffled[col].iloc[split:].dropna(),
            )
            assert p_value > 0.01, (
                f"Possible drift in '{col}': KS stat={stat:.4f}, p={p_value:.4f}"
            )
