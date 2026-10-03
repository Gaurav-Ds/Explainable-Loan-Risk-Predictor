"""SHAP-based global and local explanations, plus plain-language summaries."""
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import FEATURE_LABELS, FEATURES
from modeling import feature_groups


class RiskExplainer:
    """Wraps a fitted pipeline and explains predictions in terms of the original features.

    One-hot encoded columns are summed back into their source feature, so an
    explanation reads "Checking account status" rather than 4 dummy columns.
    """

    def __init__(self, pipeline: Pipeline, background: pd.DataFrame):
        self.pipeline = pipeline
        self.preprocess = pipeline.named_steps["preprocess"]
        self.model = pipeline.named_steps["model"]
        self.groups = np.array(feature_groups(self.preprocess))

        bg = self.preprocess.transform(background[FEATURES])
        if isinstance(self.model, LogisticRegression):
            self.explainer = shap.LinearExplainer(self.model, bg)
            self.units = "log-odds"
        else:
            self.explainer = shap.TreeExplainer(self.model)
            self.units = "probability" if hasattr(self.model, "estimators_") else "log-odds"

    def _raw_shap(self, X: pd.DataFrame) -> tuple[np.ndarray, float]:
        Xt = self.preprocess.transform(X[FEATURES])
        values = self.explainer.shap_values(Xt)
        base = self.explainer.expected_value
        values = np.asarray(values)
        if values.ndim == 3:  # (n, features, classes) -> default class
            values = values[:, :, 1]
        base = np.atleast_1d(base)
        base = float(base[1] if base.size > 1 else base[0])
        return values, base

    def shap_by_feature(self, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """SHAP values aggregated to original features: rows = applicants, cols = features."""
        values, base = self._raw_shap(X)
        df = pd.DataFrame(values, columns=self.groups, index=X.index)
        df = df.T.groupby(level=0).sum().T[FEATURES]
        return df, base

    def global_importance(self, X: pd.DataFrame) -> pd.Series:
        """Mean absolute SHAP value per feature (global explanation)."""
        df, _ = self.shap_by_feature(X)
        return df.abs().mean().sort_values(ascending=False)

    def explain_applicant(self, applicant: pd.DataFrame) -> pd.DataFrame:
        """Local explanation for a single applicant, sorted by absolute impact."""
        contrib, base = self.shap_by_feature(applicant)
        row = contrib.iloc[0]
        out = pd.DataFrame({
            "feature": row.index,
            "label": [FEATURE_LABELS[f] for f in row.index],
            "value": [applicant.iloc[0][f] for f in row.index],
            "contribution": row.values,
        })
        out["direction"] = np.where(out["contribution"] > 0, "increases risk", "decreases risk")
        out = out.reindex(out["contribution"].abs().sort_values(ascending=False).index)
        out.attrs["base_value"] = base
        out.attrs["units"] = self.units
        return out.reset_index(drop=True)


def plain_language_summary(probability: float, explanation: pd.DataFrame, threshold: float, top_n: int = 3) -> str:
    """Turn a SHAP explanation into a short, non-technical paragraph."""
    level = "HIGH" if probability >= threshold else "LOW"
    lines = [
        f"The model estimates a {probability:.1%} probability that this applicant will default, "
        f"which is classed as {level} risk (threshold {threshold:.0%})."
    ]

    def describe(rows: pd.DataFrame) -> str:
        return "; ".join(f"{r.label} = {_fmt(r.value)}" for r in rows.itertuples())

    up = explanation[explanation["contribution"] > 0].head(top_n)
    down = explanation[explanation["contribution"] < 0].head(top_n)
    if not up.empty:
        lines.append(f"The main factors that INCREASED the risk were: {describe(up)}.")
    if not down.empty:
        lines.append(f"The main factors that DECREASED the risk were: {describe(down)}.")
    top = explanation.iloc[0]
    lines.append(f"The single most influential factor was {top.label.lower()} ({_fmt(top.value)}), which {top.direction}.")
    return " ".join(lines)


def _fmt(value) -> str:
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"
    return str(value)
