"""UT-11 to UT-14 (Unit) + MV-01 to MV-10 (Model Validation) Tests."""

import warnings

import numpy as np
import pytest
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate

SCORING = {
    "f1": "f1",
    "roc_auc": "roc_auc",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
}
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ---------------------------------------------------------------------------
# Unit Tests (UT-11 to UT-14)
# ---------------------------------------------------------------------------

class TestModelUnit:

    # UT-11
    def test_logistic_regression_converges(self, log_pipeline, X_y):
        X, y = X_y
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=UserWarning)
            try:
                log_pipeline.fit(X, y)
            except UserWarning as e:
                if "ConvergenceWarning" in str(type(e).__name__):
                    pytest.fail(f"LogisticRegression did not converge: {e}")

    # UT-12
    def test_dummy_classifier_baseline(self, dummy_pipeline, X_y):
        X, y = X_y
        results = cross_validate(
            dummy_pipeline, X, y, cv=CV, scoring=SCORING, return_train_score=False,
        )
        assert np.mean(results["test_f1"]) == pytest.approx(0.0, abs=0.01)
        assert np.mean(results["test_recall"]) == pytest.approx(0.0, abs=0.01)
        assert np.mean(results["test_roc_auc"]) == pytest.approx(0.5, abs=0.05)

    # UT-13
    def test_column_transformer_output_shape(self, preprocessor, X_y):
        X, y = X_y
        transformed = preprocessor.fit_transform(X)
        n_num = len(X.select_dtypes(include=np.number).columns)
        assert transformed.shape[0] == len(X)
        assert transformed.shape[1] >= n_num  # at least numeric + some OHE cols

    # UT-14
    def test_pipeline_accepts_raw_dataframe(self, fitted_log_pipeline, X_y):
        X, _ = X_y
        preds = fitted_log_pipeline.predict(X.head(10))
        assert len(preds) == 10
        assert set(preds).issubset({0, 1})


# ---------------------------------------------------------------------------
# Model Validation Tests (MV-01 to MV-10)
# ---------------------------------------------------------------------------

class TestModelValidation:

    @pytest.fixture(scope="class")
    def cv_results(self, log_pipeline, X_y):
        X, y = X_y
        return cross_validate(
            log_pipeline, X, y, cv=CV, scoring=SCORING, return_train_score=True,
        )

    # MV-01
    def test_recall_minimum(self, cv_results):
        recalls = cv_results["test_recall"]
        for i, r in enumerate(recalls):
            assert r >= 0.73, f"Fold {i}: recall {r:.4f} < 0.73"

    # MV-02
    def test_roc_auc_minimum(self, cv_results):
        aucs = cv_results["test_roc_auc"]
        for i, a in enumerate(aucs):
            assert a >= 0.84, f"Fold {i}: ROC AUC {a:.4f} < 0.84"
        assert np.std(aucs) < 0.02

    # MV-03
    def test_f1_minimum(self, cv_results):
        f1s = cv_results["test_f1"]
        for i, f in enumerate(f1s):
            assert f >= 0.62, f"Fold {i}: F1 {f:.4f} < 0.62"

    # MV-04
    def test_beats_dummy(self, dummy_pipeline, log_pipeline, X_y):
        X, y = X_y
        dummy_res = cross_validate(
            dummy_pipeline, X, y, cv=CV, scoring=SCORING, return_train_score=False,
        )
        log_res = cross_validate(
            log_pipeline, X, y, cv=CV, scoring=SCORING, return_train_score=False,
        )
        for metric in SCORING:
            key = f"test_{metric}"
            assert np.mean(log_res[key]) > np.mean(dummy_res[key]), (
                f"Logistic <= Dummy on {metric}"
            )

    # MV-05
    def test_recall_stability(self, cv_results):
        assert np.std(cv_results["test_recall"]) < 0.03

    # MV-06
    def test_no_overfitting(self, cv_results):
        train_recall = np.mean(cv_results["train_recall"])
        test_recall = np.mean(cv_results["test_recall"])
        gap = train_recall - test_recall
        assert gap < 0.15, f"Overfitting detected: train-test gap = {gap:.4f}"

    # MV-07
    def test_feature_importance_alignment(self, fitted_log_pipeline, X_y):
        X, _ = X_y
        model = fitted_log_pipeline.named_steps["model"]
        preprocessor = fitted_log_pipeline.named_steps["preprocess"]

        try:
            all_features = list(preprocessor.get_feature_names_out())
        except AttributeError:
            num_features = list(X.select_dtypes(include=np.number).columns)
            cat_features = list(
                preprocessor.named_transformers_["cat"]
                .named_steps["encoder"]
                .get_feature_names_out()
            )
            all_features = num_features + cat_features

        importances = np.abs(model.coef_[0])
        top_indices = np.argsort(importances)[-20:]
        top_features = {all_features[i] for i in top_indices}

        key_patterns = [
            "Contract", "Tenure", "Monthly", "Internet",
            "Total Charges", "Tech Support", "Online Security",
        ]
        matched = [
            p for p in key_patterns
            if any(p.lower() in f.lower() for f in top_features)
        ]
        assert len(matched) >= 1, (
            f"None of the key EDA drivers found in top-20 features: {top_features}"
        )

    # MV-08 -- Fairness: gender invariance
    def test_gender_invariance(self, fitted_log_pipeline, X_y):
        X, _ = X_y
        sample = X.head(100).copy()

        sample_male = sample.copy()
        sample_female = sample.copy()
        if "Gender" in sample.columns:
            sample_male["Gender"] = "Male"
            sample_female["Gender"] = "Female"

            preds_m = fitted_log_pipeline.predict_proba(sample_male)[:, 1]
            preds_f = fitted_log_pipeline.predict_proba(sample_female)[:, 1]
            mean_diff = abs(preds_m.mean() - preds_f.mean())
            assert mean_diff < 0.10, (
                f"Gender bias detected: mean prob diff = {mean_diff:.4f}"
            )

    # MV-09 -- Directionality: higher tenure -> lower churn
    def test_tenure_directionality(self, fitted_log_pipeline):
        model = fitted_log_pipeline.named_steps["model"]
        preprocessor = fitted_log_pipeline.named_steps["preprocess"]
        num_features = list(preprocessor.transformers_[0][2])

        if "Tenure Months" in num_features:
            idx = num_features.index("Tenure Months")
            coef = model.coef_[0][idx]
            assert coef < 0, (
                f"Expected negative coefficient for Tenure Months, got {coef:.4f}"
            )

    # MV-10 -- Reproducibility
    def test_reproducibility(self, log_pipeline, X_y):
        X, y = X_y
        cv1 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        r1 = cross_validate(log_pipeline, X, y, cv=cv1, scoring={"recall": "recall"})
        r2 = cross_validate(log_pipeline, X, y, cv=cv2, scoring={"recall": "recall"})

        np.testing.assert_array_almost_equal(
            r1["test_recall"], r2["test_recall"], decimal=10,
        )
