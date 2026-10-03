"""End-to-end UI tests for the Streamlit dashboard (headless, via streamlit.testing).

Run with:  python -m pytest tests -q   (after prepare_data.py and train.py)
"""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")
sys.path.insert(0, str(ROOT / "src"))

from config import FEATURES, INPUT_RANGES  # noqa: E402

TIMEOUT = 300


def selectbox(at, label):
    return next(s for s in at.selectbox if s.label == label)


def slider(at, label):
    return next(s for s in at.slider if s.label == label)


def predict(at):
    next(b for b in at.button if "Predict" in b.label).click().run()


def result_text(at):
    """The plain-language explanation card shown with every prediction."""
    return [m.value for m in at.markdown if 'class="why"' in m.value and "probability" in m.value]


@pytest.fixture
def app():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert not at.exception
    return at


def test_app_loads_dark_by_default(app):
    assert app.session_state.theme == "dark"
    assert len(app.tabs) == 6


def test_theme_toggle_round_trip(app):
    next(b for b in app.button if "Light" in b.label).click().run()
    assert not app.exception and app.session_state.theme == "light"
    next(b for b in app.button if "Dark" in b.label).click().run()
    assert not app.exception and app.session_state.theme == "dark"


@pytest.mark.parametrize("model", ["Random Forest", "XGBoost", "Logistic Regression"])
def test_each_model_predicts_sample_applicant(app, model):
    selectbox(app, "Classifier").set_value(model).run()
    sample = selectbox(app, "Load a sample applicant")
    sample.set_value(sample.options[1]).run()
    assert not app.exception
    assert result_text(app), "plain-language summary should be shown"
    assert any("RISK" in m.value for m in app.markdown)


def test_custom_result_survives_other_widget_changes(app):
    predict(app)
    first = result_text(app)
    assert first
    slider(app, "Number of factors to show").set_value(15).run()
    slider(app, "Decision threshold").set_value(0.3).run()
    assert not app.exception
    assert result_text(app), "result must not disappear after changing a slider"


@pytest.mark.parametrize("model", ["Random Forest", "Logistic Regression"])
def test_waterfall_view_and_report_download(app, model):
    selectbox(app, "Classifier").set_value(model).run()
    sample = selectbox(app, "Load a sample applicant")
    sample.set_value(sample.options[1]).run()
    app.session_state["shap_view"] = "Waterfall"
    app.run()
    assert not app.exception
    assert app.get("download_button"), "download report button should be shown with a result"


def test_input_bounds_are_realistic(app):
    bounds = {n.label: (n.proto.min, n.proto.max) for n in app.number_input}
    assert bounds["Age"] == INPUT_RANGES["age"]
    assert bounds["Installment rate (% of income)"] == INPUT_RANGES["installment_rate"]


def test_extreme_inputs_warn_and_do_not_crash(app):
    for n in app.number_input:
        n.set_value(n.proto.max)
    predict(app)
    assert not app.exception
    assert any("outside the range" in w.value for w in app.warning)
    for n in app.number_input:
        n.set_value(n.proto.min)
    predict(app)
    assert not app.exception and result_text(app)


@pytest.mark.parametrize("threshold", [0.1, 0.9])
def test_extreme_thresholds(app, threshold):
    slider(app, "Decision threshold").set_value(threshold).run()
    assert not app.exception


def test_dataset_explorer_all_features(app):
    for f in FEATURES:
        selectbox(app, "Explore default rate by").set_value(f).run()
        assert not app.exception, f
