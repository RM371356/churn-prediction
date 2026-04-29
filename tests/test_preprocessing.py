"""UT-01 to UT-10: Unit Tests for data preprocessing steps."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class TestTotalChargesConversion:
    """UT-01: pd.to_numeric coercion for Total Charges."""

    def test_spaces_become_nan(self):
        s = pd.Series(["100.5", " ", "200.3", ""])
        result = pd.to_numeric(s, errors="coerce")
        assert result.isna().sum() == 2
        assert result.dtype == np.float64

    def test_valid_values_preserved(self):
        s = pd.Series(["100.5", "200.3", "0"])
        result = pd.to_numeric(s, errors="coerce")
        assert result.isna().sum() == 0
        np.testing.assert_array_almost_equal(result.values, [100.5, 200.3, 0.0])


class TestNumericImputer:
    """UT-02: SimpleImputer with median strategy."""

    def test_nan_filled_with_median(self):
        data = np.array([[1.0], [2.0], [np.nan], [4.0], [5.0]])
        imputer = SimpleImputer(strategy="median")
        result = imputer.fit_transform(data)
        assert not np.isnan(result).any()
        assert result[2, 0] == 3.0  # median of [1, 2, 4, 5]

    def test_no_nan_remaining(self):
        data = np.array([[np.nan, 1.0], [2.0, np.nan], [3.0, 3.0]])
        imputer = SimpleImputer(strategy="median")
        result = imputer.fit_transform(data)
        assert not np.isnan(result).any()


class TestCategoricalImputer:
    """UT-03: SimpleImputer with most_frequent strategy on categoricals."""

    def test_missing_replaced_by_mode(self):
        data = pd.DataFrame({"col": ["A", "B", "A", np.nan, "A"]}, dtype="object")
        imputer = SimpleImputer(strategy="most_frequent")
        result = imputer.fit_transform(data)
        assert (result == "A").sum() == 4  # NaN -> "A" (mode)

    def test_nan_replaced_by_mode(self):
        data = pd.DataFrame({"col": ["Yes", "No", "Yes", np.nan, "Yes"]}, dtype="object")
        imputer = SimpleImputer(strategy="most_frequent")
        result = imputer.fit_transform(data)
        assert not pd.isna(result).any()


class TestStandardScaler:
    """UT-04: StandardScaler normalization."""

    def test_mean_near_zero(self):
        data = np.array([[10.0], [20.0], [30.0], [40.0], [50.0]])
        scaler = StandardScaler()
        result = scaler.fit_transform(data)
        assert abs(result.mean()) < 1e-10

    def test_std_near_one(self):
        data = np.random.RandomState(42).randn(1000, 3) * 10 + 50
        scaler = StandardScaler()
        result = scaler.fit_transform(data)
        np.testing.assert_allclose(result.std(axis=0), 1.0, atol=0.01)


class TestOneHotEncoder:
    """UT-05: OneHotEncoder with handle_unknown='ignore'."""

    def test_unknown_categories_produce_zeros(self):
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(np.array([["A"], ["B"], ["C"]]))
        result = encoder.transform(np.array([["D"]]))
        assert result.sum() == 0
        assert result.shape == (1, 3)

    def test_known_categories_encoded(self):
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(np.array([["A"], ["B"], ["C"]]))
        result = encoder.transform(np.array([["A"]]))
        assert result.sum() == 1


class TestLeakageColumnsDropped:
    """UT-06: Leakage columns removed."""

    def test_leakage_columns_absent(self, clean_df):
        leakage = {"Churn Label", "Churn Score", "Churn Reason"}
        remaining = set(clean_df.columns) & leakage
        assert remaining == set(), f"Leakage columns still present: {remaining}"


class TestIrrelevantColumnsDropped:
    """UT-07: Irrelevant columns removed."""

    def test_irrelevant_columns_absent(self, clean_df):
        irrelevant = {"CustomerID", "Count", "Lat Long", "City", "State", "Zip Code"}
        remaining = set(clean_df.columns) & irrelevant
        assert remaining == set(), f"Irrelevant columns present: {remaining}"


class TestSpaceToNanReplacement:
    """UT-08: Spaces replaced with NaN in categorical columns."""

    def test_space_strings_become_nan(self):
        df = pd.DataFrame({"col": ["Yes", " ", "No", " "]})
        df["col"] = df["col"].astype("string")
        df["col"] = df["col"].replace(" ", np.nan)
        assert df["col"].isna().sum() == 2

    def test_non_space_values_unchanged(self):
        df = pd.DataFrame({"col": ["Yes", "No", "Maybe"]})
        df["col"] = df["col"].astype("string")
        df["col"] = df["col"].replace(" ", np.nan)
        assert df["col"].isna().sum() == 0


class TestTenureGroup:
    """UT-09: tenure_group via pd.cut."""

    def test_bins_correct(self):
        tenure = pd.Series([6, 18, 36, 60])
        groups = pd.cut(tenure, bins=[0, 12, 24, 48, 72])
        expected_labels = ["(0, 12]", "(12, 24]", "(24, 48]", "(48, 72]"]
        assert [str(g) for g in groups] == expected_labels

    def test_out_of_range_is_nan(self):
        tenure = pd.Series([0, 80])
        groups = pd.cut(tenure, bins=[0, 12, 24, 48, 72])
        assert pd.isna(groups.iloc[0])  # 0 is not in (0, 12]
        assert pd.isna(groups.iloc[1])


class TestAvgTicket:
    """UT-10: avg_ticket = Total Charges / (Tenure Months + 1)."""

    def test_calculation_correct(self):
        df = pd.DataFrame({
            "Total Charges": [100.0, 500.0, 0.0],
            "Tenure Months": [9, 49, 0],
        })
        avg_ticket = df["Total Charges"] / (df["Tenure Months"] + 1)
        np.testing.assert_array_almost_equal(avg_ticket.values, [10.0, 10.0, 0.0])

    def test_no_division_by_zero(self):
        df = pd.DataFrame({
            "Total Charges": [100.0],
            "Tenure Months": [0],
        })
        avg_ticket = df["Total Charges"] / (df["Tenure Months"] + 1)
        assert np.isfinite(avg_ticket.iloc[0])
