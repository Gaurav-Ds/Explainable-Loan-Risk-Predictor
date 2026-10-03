"""Train, compare and evaluate Logistic Regression, Random Forest and XGBoost.

Pipeline: load data -> stratified train/test split -> 5-fold CV on train ->
fit on train -> evaluate on held-out test -> select best model by CV ROC-AUC ->
SHAP global explanation -> fairness audit -> save artifacts.
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from config import (
    DATA_CSV, DECISION_THRESHOLD, FEATURE_LABELS, FEATURES, FIGURES_DIR, MODELS_DIR,
    PROCESSED_DIR, RANDOM_STATE, REPORTS_DIR, SENSITIVE_ATTRIBUTES, TARGET, TEST_SIZE,
)
from explain import RiskExplainer
from fairness import fairness_report
from modeling import build_models, build_pipeline


def evaluate(y_true, proba) -> dict:
    pred = (proba >= DECISION_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main() -> None:
    for d in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_CSV)
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE
    )
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results, fitted, probas = {}, {}, {}
    for name, model in build_models(pos_weight).items():
        pipe = build_pipeline(model)
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, proba)
        metrics["cv_roc_auc_mean"] = float(cv_auc.mean())
        metrics["cv_roc_auc_std"] = float(cv_auc.std())
        results[name], fitted[name], probas[name] = metrics, pipe, proba
        print(f"{name:20s} CV AUC {cv_auc.mean():.3f}+/-{cv_auc.std():.3f} | "
              f"test AUC {metrics['roc_auc']:.3f} acc {metrics['accuracy']:.3f} "
              f"prec {metrics['precision']:.3f} rec {metrics['recall']:.3f} f1 {metrics['f1']:.3f}")

    # Select on cross-validated AUC so the held-out test set stays unbiased.
    best_name = max(results, key=lambda n: results[n]["cv_roc_auc_mean"])
    best = fitted[best_name]
    print(f"\nBest model (by CV ROC-AUC): {best_name}")

    joblib.dump(best, MODELS_DIR / "best_model.joblib")
    joblib.dump(fitted, MODELS_DIR / "all_models.joblib")

    # ---- Explainability (global) ----
    explainer = RiskExplainer(best, X_train.sample(min(200, len(X_train)), random_state=RANDOM_STATE))
    importance = explainer.global_importance(X_test)
    shap_df, _ = explainer.shap_by_feature(X_test)

    # ---- Fairness audit on the held-out test set ----
    pred_test = (probas[best_name] >= DECISION_THRESHOLD).astype(int)
    fairness = {
        attr: fairness_report(y_test.values, pred_test, test_df[attr].values, attr)
        for attr in SENSITIVE_ATTRIBUTES
    }
    for attr, rep in fairness.items():
        print(f"Fairness [{attr}] DP diff {rep['demographic_parity_difference']:.3f} "
              f"EO diff {rep['equal_opportunity_difference']:.3f} "
              f"DI ratio {rep['disparate_impact_ratio']:.3f} -> {rep['verdict']}")

    summary = {
        "best_model": best_name,
        "selection_criterion": "5-fold cross-validated ROC-AUC on training set",
        "decision_threshold": DECISION_THRESHOLD,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "models": results,
        "global_importance": importance.round(5).to_dict(),
        "shap_units": explainer.units,
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    (REPORTS_DIR / "fairness.json").write_text(json.dumps(fairness, indent=2))

    save_figures(results, probas, y_test, importance, shap_df, X_test, fairness, best_name)
    print(f"\nArtifacts saved to {MODELS_DIR} and {REPORTS_DIR}")


def save_figures(results, probas, y_test, importance, shap_df, X_test, fairness, best_name) -> None:
    # Model comparison
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    comp = pd.DataFrame({m: {n: r[m] for n, r in results.items()} for m in metrics})
    ax = comp.plot.bar(figsize=(9, 4.5), rot=0)
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison on held-out test set")
    ax.legend(loc="lower right", ncols=5, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=150)
    plt.close()

    # ROC curves
    plt.figure(figsize=(5.5, 5))
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC {results[name]['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curves.png", dpi=150)
    plt.close()

    # SHAP global importance (bar)
    top = importance.head(12)[::-1]
    plt.figure(figsize=(7, 5))
    plt.barh([FEATURE_LABELS[f] for f in top.index], top.values)
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"Global feature importance ({best_name})")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_global_importance.png", dpi=150)
    plt.close()

    # SHAP beeswarm (numeric features only have a meaningful colour scale)
    expl = shap.Explanation(
        values=shap_df.values,
        data=X_test[shap_df.columns].apply(lambda c: c if c.dtype != object else c.astype("category").cat.codes).values,
        feature_names=[FEATURE_LABELS[f] for f in shap_df.columns],
    )
    for suffix, style, face in [("", "default", "#ffffff"), ("_dark", "dark_background", "#0f1a2e")]:
        with plt.style.context(style):
            shap.plots.beeswarm(expl, max_display=12, show=False)
            fig = plt.gcf()
            fig.set_facecolor(face)
            ink = "#e6edf7" if suffix else "#333333"
            for ax in fig.axes:
                ax.set_facecolor(face)
                ax.tick_params(colors=ink, labelcolor=ink)
                ax.xaxis.label.set_color(ink)
                ax.yaxis.label.set_color(ink)
            plt.title(f"SHAP summary ({best_name})")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"shap_beeswarm{suffix}.png", dpi=150, facecolor=face)
            plt.close()

    # Fairness comparison
    fig, axes = plt.subplots(1, len(fairness), figsize=(5 * len(fairness), 4))
    for ax, (attr, rep) in zip(np.atleast_1d(axes), fairness.items()):
        g = pd.DataFrame(rep["groups"]).set_index("group")[["approval_rate", "true_approval_rate"]]
        g.columns = ["Approval rate", "True approval rate (EO)"]
        g.plot.bar(ax=ax, rot=0, ylim=(0, 1))
        ax.set_title(f"{attr}: DP diff {rep['demographic_parity_difference']:.2f}, "
                     f"EO diff {rep['equal_opportunity_difference']:.2f}", fontsize=9)
        ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fairness_comparison.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
