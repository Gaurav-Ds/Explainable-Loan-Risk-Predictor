"""Group-level fairness audit: Demographic Parity and Equal Opportunity.

Favourable outcome = the model predicts the applicant will NOT default
(i.e. the loan would be approved).
"""
import numpy as np
import pandas as pd


def group_metrics(y_true, y_pred_default, groups) -> pd.DataFrame:
    """Per-group approval rate, true positive rate on good applicants, and error rates."""
    df = pd.DataFrame({
        "group": np.asarray(groups),
        "actual_good": 1 - np.asarray(y_true),
        "approved": 1 - np.asarray(y_pred_default),
    })
    rows = []
    for name, g in df.groupby("group"):
        good = g[g["actual_good"] == 1]
        bad = g[g["actual_good"] == 0]
        rows.append({
            "group": name,
            "count": len(g),
            "actual_good_rate": g["actual_good"].mean(),
            "approval_rate": g["approved"].mean(),
            # Equal opportunity: among genuinely creditworthy applicants, share approved.
            "true_approval_rate": good["approved"].mean() if len(good) else np.nan,
            # Share of actual defaulters that the model wrongly approves.
            "false_approval_rate": bad["approved"].mean() if len(bad) else np.nan,
        })
    return pd.DataFrame(rows).set_index("group")


def fairness_report(y_true, y_pred_default, groups, attribute: str) -> dict:
    metrics = group_metrics(y_true, y_pred_default, groups)
    approval = metrics["approval_rate"]
    tpr = metrics["true_approval_rate"]
    dp_diff = float(approval.max() - approval.min())
    eo_diff = float(tpr.max() - tpr.min())
    di_ratio = float(approval.min() / approval.max()) if approval.max() > 0 else float("nan")
    return {
        "attribute": attribute,
        "groups": metrics.round(4).reset_index().to_dict(orient="records"),
        "demographic_parity_difference": round(dp_diff, 4),
        "equal_opportunity_difference": round(eo_diff, 4),
        "disparate_impact_ratio": round(di_ratio, 4),
        "least_favoured_group": str(approval.idxmin()),
        "verdict": verdict(dp_diff, eo_diff, di_ratio),
    }


def verdict(dp_diff: float, eo_diff: float, di_ratio: float, tolerance: float = 0.1) -> str:
    """Rule of thumb: |difference| <= 0.10 and the 'four-fifths' disparate-impact rule."""
    if np.isnan(di_ratio):
        return "Cannot assess: the model approves no applicants at this threshold."
    issues = []
    if dp_diff > tolerance:
        issues.append(f"approval rates differ by {dp_diff:.0%}")
    if eo_diff > tolerance:
        issues.append(f"creditworthy applicants are approved at rates differing by {eo_diff:.0%}")
    if di_ratio < 0.8:
        issues.append(f"disparate impact ratio {di_ratio:.2f} is below the 0.80 four-fifths rule")
    if not issues:
        return "No significant disparity detected (all differences within 10% and DI ratio >= 0.80)."
    return "Potential disparity: " + "; ".join(issues) + "."
