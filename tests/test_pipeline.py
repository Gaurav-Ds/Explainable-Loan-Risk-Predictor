"""Testing & validation: predictions, explanations and fairness metrics.

Run with:  python -m pytest tests -q   (after prepare_data.py and train.py)
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import DATA_CSV, FEATURES, MODELS_DIR, PROCESSED_DIR, TARGET  # noqa: E402
from explain import RiskExplainer, plain_language_summary  # noqa: E402
from fairness import fairness_report  # noqa: E402


@pytest.fixture(scope="module")
def artifacts():
    models = joblib.load(MODELS_DIR / "all_models.joblib")
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    return models, train, test


def test_dataset_integrity():
    df = pd.read_csv(DATA_CSV)
    assert len(df) == 1000
    assert df[FEATURES].isna().sum().sum() == 0
    assert set(df[TARGET].unique()) == {0, 1}
    assert abs(df[TARGET].mean() - 0.30) < 1e-9
    assert set(df["gender"].unique()) == {"Male", "Female"}


def test_no_leakage_between_splits(artifacts):
    _, train, test = artifacts
    assert set(train["applicant_id"]).isdisjoint(test["applicant_id"])
    assert "gender" not in FEATURES and "personal_status" not in FEATURES


def test_predictions_are_valid_probabilities(artifacts):
    models, _, test = artifacts
    for name, pipe in models.items():
        proba = pipe.predict_proba(test[FEATURES])[:, 1]
        assert proba.shape == (len(test),), name
        assert np.all((proba >= 0) & (proba <= 1)), name


@pytest.mark.parametrize("model_name", ["Logistic Regression", "Random Forest", "XGBoost"])
def test_shap_additivity(artifacts, model_name):
    """Base value + sum of SHAP contributions must reconstruct the model output."""
    models, train, test = artifacts
    pipe = models[model_name]
    explainer = RiskExplainer(pipe, train.sample(100, random_state=0))
    sample = test.head(5)
    contrib, base = explainer.shap_by_feature(sample)
    reconstructed = base + contrib.sum(axis=1).values
    proba = pipe.predict_proba(sample[FEATURES])[:, 1]
    expected = proba if explainer.units == "probability" else np.log(proba / (1 - proba))
    assert np.allclose(reconstructed, expected, atol=1e-3), model_name


def test_local_explanation_and_summary(artifacts):
    models, train, test = artifacts
    pipe = models["Random Forest"]
    explainer = RiskExplainer(pipe, train.sample(100, random_state=0))
    applicant = test[FEATURES].head(1)
    exp = explainer.explain_applicant(applicant)
    assert list(exp.columns) == ["feature", "label", "value", "contribution", "direction"]
    assert len(exp) == len(FEATURES)
    assert exp["contribution"].abs().is_monotonic_decreasing
    text = plain_language_summary(float(pipe.predict_proba(applicant)[0, 1]), exp, 0.5)
    assert "probability" in text and "risk" in text


def test_fairness_metrics_on_known_values():
    # Group A: 4 applicants, all good, 3 approved -> approval 0.75, TPR 0.75
    # Group B: 4 applicants, all good, 1 approved -> approval 0.25, TPR 0.25
    y_true = np.zeros(8, dtype=int)
    y_pred_default = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    groups = np.array(["A"] * 4 + ["B"] * 4)
    rep = fairness_report(y_true, y_pred_default, groups, "toy")
    assert rep["demographic_parity_difference"] == pytest.approx(0.5)
    assert rep["equal_opportunity_difference"] == pytest.approx(0.5)
    assert rep["disparate_impact_ratio"] == pytest.approx(1 / 3, abs=1e-3)
    assert rep["least_favoured_group"] == "B"
    assert rep["verdict"].startswith("Potential disparity")


def test_fairness_when_nobody_is_approved():
    y_true = np.array([0, 1, 0, 1])
    rep = fairness_report(y_true, np.ones(4, dtype=int), np.array(["A", "A", "B", "B"]), "toy")
    assert rep["verdict"].startswith("Cannot assess")


def test_fairness_no_disparity():
    y_true = np.array([0, 0, 1, 0, 0, 1])
    y_pred = np.array([0, 0, 1, 0, 0, 1])
    rep = fairness_report(y_true, y_pred, np.array(["A"] * 3 + ["B"] * 3), "toy")
    assert rep["demographic_parity_difference"] == 0
    assert rep["verdict"].startswith("No significant")
