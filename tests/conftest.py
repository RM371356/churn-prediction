"""Shared fixtures for the churn prediction test suite."""

import pathlib

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
XLSX_PATH = DATA_DIR / "Telco_customer_churn.xlsx"
CSV_PATH = DATA_DIR / "Telco_customer_churn.csv"

DROP_COLS = [
    "CustomerID", "Count", "Lat Long",
    "Churn Label", "Churn Score", "Churn Reason",
    "City", "State", "Zip Code",
]
TARGET_COL = "Churn Value"


def _load_dataframe() -> pd.DataFrame:
    """Load the Telco churn dataset, preferring xlsx over csv."""
    if XLSX_PATH.exists():
        return pd.read_excel(XLSX_PATH)
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH)
    pytest.skip("Dataset file not found (xlsx or csv)")


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """Full raw dataframe as loaded from disk."""
    return _load_dataframe()


@pytest.fixture(scope="session")
def clean_df(raw_df: pd.DataFrame):
    """DataFrame after dropping leakage / irrelevant columns."""
    return raw_df.drop(columns=DROP_COLS, errors="ignore")


@pytest.fixture(scope="session")
def X_y(clean_df: pd.DataFrame):
    """Feature matrix X and target vector y, with dtype fixes applied."""
    X = clean_df.drop(columns=[TARGET_COL])
    y = clean_df[TARGET_COL]

    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns
    X[cat_cols] = X[cat_cols].astype("string")
    X[cat_cols] = X[cat_cols].replace(" ", np.nan)
    X[cat_cols] = X[cat_cols].fillna("missing")
    return X, y


@pytest.fixture(scope="session")
def num_cols(X_y):
    X, _ = X_y
    return X.select_dtypes(include=np.number).columns


@pytest.fixture(scope="session")
def cat_cols(X_y):
    X, _ = X_y
    return X.select_dtypes(include=["object", "string"]).columns


def build_preprocessor(num_cols, cat_cols) -> ColumnTransformer:
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
    ])


@pytest.fixture(scope="session")
def preprocessor(num_cols, cat_cols) -> ColumnTransformer:
    return build_preprocessor(num_cols, cat_cols)


@pytest.fixture(scope="session")
def log_pipeline(preprocessor) -> Pipeline:
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


@pytest.fixture(scope="session")
def dummy_pipeline(num_cols, cat_cols) -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor(num_cols, cat_cols)),
        ("model", DummyClassifier(strategy="most_frequent")),
    ])


@pytest.fixture(scope="session")
def fitted_log_pipeline(log_pipeline, X_y):
    X, y = X_y
    log_pipeline.fit(X, y)
    return log_pipeline


@pytest.fixture(scope="session")
def fitted_dummy_pipeline(dummy_pipeline, X_y):
    X, y = X_y
    dummy_pipeline.fit(X, y)
    return dummy_pipeline


# ---------------------------------------------------------------------------
# Synthetic data helpers (for tests that don't need the real dataset)
# ---------------------------------------------------------------------------

def make_synthetic_df(n_rows: int = 200, random_state: int = 42) -> pd.DataFrame:
    """Build a synthetic DataFrame matching the expected schema."""
    rng = np.random.default_rng(random_state)
    contracts = rng.choice(["Month-to-month", "One year", "Two year"], n_rows)
    internet = rng.choice(["DSL", "Fiber optic", "No"], n_rows)
    def yes_no():
        return rng.choice(["Yes", "No"], n_rows)

    return pd.DataFrame({
        "CustomerID": [f"CUST-{i:04d}" for i in range(n_rows)],
        "Count": np.ones(n_rows, dtype=int),
        "Country": ["United States"] * n_rows,
        "State": ["California"] * n_rows,
        "City": ["Los Angeles"] * n_rows,
        "Zip Code": rng.integers(90000, 90999, n_rows),
        "Lat Long": ["34.05, -118.25"] * n_rows,
        "Latitude": rng.uniform(33.5, 34.5, n_rows),
        "Longitude": rng.uniform(-118.5, -117.5, n_rows),
        "Gender": rng.choice(["Male", "Female"], n_rows),
        "Senior Citizen": rng.choice(["Yes", "No"], n_rows),
        "Partner": yes_no(),
        "Dependents": yes_no(),
        "Tenure Months": rng.integers(0, 73, n_rows),
        "Phone Service": yes_no(),
        "Multiple Lines": rng.choice(["Yes", "No", "No phone service"], n_rows),
        "Internet Service": internet,
        "Online Security": rng.choice(["Yes", "No", "No internet service"], n_rows),
        "Online Backup": rng.choice(["Yes", "No", "No internet service"], n_rows),
        "Device Protection": rng.choice(["Yes", "No", "No internet service"], n_rows),
        "Tech Support": rng.choice(["Yes", "No", "No internet service"], n_rows),
        "Streaming TV": rng.choice(["Yes", "No", "No internet service"], n_rows),
        "Streaming Movies": rng.choice(["Yes", "No", "No internet service"], n_rows),
        "Contract": contracts,
        "Paperless Billing": yes_no(),
        "Payment Method": rng.choice([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ], n_rows),
        "Monthly Charges": rng.uniform(18.0, 120.0, n_rows).round(2),
        "Total Charges": rng.uniform(18.0, 9000.0, n_rows).round(2),
        "Churn Label": rng.choice(["Yes", "No"], n_rows),
        "Churn Value": rng.choice([0, 1], n_rows, p=[0.735, 0.265]),
        "Churn Score": rng.integers(0, 100, n_rows),
        "CLTV": rng.integers(2000, 7000, n_rows),
        "Churn Reason": rng.choice(["Moved", "Competitor", "Price", ""], n_rows),
    })

@pytest.fixture
def synthetic_df():
    return make_synthetic_df()
