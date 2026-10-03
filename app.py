"""Explainable Loan Risk Predictor - Streamlit dashboard.

Run with:  streamlit run app.py
"""
import json
import numbers
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve
from streamlit import config as st_config

sys.path.insert(0, str(Path(__file__).parent / "src"))

import joblib  # noqa: E402

from config import (  # noqa: E402
    DATA_CSV, DECISION_THRESHOLD, FEATURE_LABELS, FEATURES, FIGURES_DIR, INPUT_HELP, INPUT_RANGES, MODELS_DIR,
    PROCESSED_DIR, REPORTS_DIR, SENSITIVE_ATTRIBUTES, TARGET,
)
from explain import RiskExplainer, plain_language_summary  # noqa: E402
from fairness import fairness_report  # noqa: E402

st.set_page_config(page_title="Explainable Loan Risk Predictor", page_icon="🏦", layout="wide")

# ================================================================== theme
# "st" = Streamlit theme options (applied at runtime); the rest are tokens for custom CSS & charts.
THEMES = {
    "dark": {
        "st": {
            "base": "dark", "primaryColor": "#3987e5", "backgroundColor": "#0b1324",
            "secondaryBackgroundColor": "#14203a", "textColor": "#e6edf7", "borderColor": "#24324f",
            "sidebar.backgroundColor": "#0f1a2e",
        },
        "bg": "#0b1324", "surface": "#0f1a2e", "card": "rgba(255,255,255,0.035)", "border": "#24324f",
        "text": "#e6edf7", "muted": "#9aa8bd", "grid": "#1e2a44", "axis": "#3a4866",
        "glow": "rgba(57,135,229,0.18)", "shadow": "0 8px 28px rgba(0,0,0,0.35)",
        "hero": "linear-gradient(125deg, #0f2a5c 0%, #1c5cab 55%, #4a3aa7 100%)",
        "series": ["#3987e5", "#d95926", "#199e70"],
        "up": "#e66767", "down": "#3987e5", "good": "#0ca30c", "bad": "#e66767",
        "seq": ["#14203a", "#184f95", "#2a78d6", "#5598e7", "#9ec5f4"],
        "plotly": "plotly_dark", "beeswarm": "shap_beeswarm_dark.png",
    },
    "light": {
        "st": {
            "base": "light", "primaryColor": "#2a78d6", "backgroundColor": "#f5f7fb",
            "secondaryBackgroundColor": "#e9eef7", "textColor": "#0b1b33", "borderColor": "#d5dcea",
            "sidebar.backgroundColor": "#ffffff",
        },
        "bg": "#f5f7fb", "surface": "#ffffff", "card": "#ffffff", "border": "#dfe5f0",
        "text": "#0b1b33", "muted": "#5b6b82", "grid": "#e6ebf3", "axis": "#c3cad8",
        "glow": "rgba(42,120,214,0.10)", "shadow": "0 6px 20px rgba(16,40,80,0.08)",
        "hero": "linear-gradient(125deg, #0d366b 0%, #2a78d6 60%, #4a3aa7 100%)",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
        "up": "#d03b3b", "down": "#2a78d6", "good": "#0ca30c", "bad": "#d03b3b",
        "seq": ["#eef4fd", "#9ec5f4", "#5598e7", "#256abf", "#0d366b"],
        "plotly": "plotly_white", "beeswarm": "shap_beeswarm.png",
    },
}

if "theme" not in st.session_state:
    st.session_state.theme = "dark"


def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"


def apply_streamlit_theme(mode: str) -> bool:
    """Push the chosen palette into Streamlit's theme config. Returns True if anything changed."""
    changed = False
    for key, value in THEMES[mode]["st"].items():
        if st_config.get_option(f"theme.{key}") != value:
            st_config.set_option(f"theme.{key}", value)
            changed = True
    return changed


if apply_streamlit_theme(st.session_state.theme):
    st.rerun()  # re-send the session so the browser picks up the new theme

T = THEMES[st.session_state.theme]
SERIES = T["series"]

st.markdown(
    f"""
    <style>
    .stApp {{
        background:
          radial-gradient(1200px 500px at 10% -10%, {T['glow']}, transparent 60%),
          radial-gradient(900px 400px at 100% 0%, {T['glow']}, transparent 55%),
          {T['bg']};
        color: {T['text']};
    }}
    [data-testid="stHeader"] {{background: transparent;}}
    [data-testid="stSidebar"] {{background: {T['surface']}; border-right: 1px solid {T['border']};}}
    .block-container {{padding-top: 1.6rem; max-width: 1320px;}}
    h1, h2, h3, h4, h5 {{letter-spacing: -0.01em;}}

    /* hero */
    .hero {{position: relative; overflow: hidden; padding: 1.8rem 2rem; border-radius: 20px;
        margin-bottom: 1.3rem; color: #fff; background: {T['hero']}; box-shadow: {T['shadow']};}}
    .hero::after {{content: ""; position: absolute; right: -80px; top: -80px; width: 280px; height: 280px;
        border-radius: 50%; background: rgba(255,255,255,0.08);}}
    .hero .eyebrow {{font-size: .78rem; letter-spacing: .14em; text-transform: uppercase; opacity: .8;
        font-weight: 600;}}
    .hero h1 {{color: #fff; margin: .25rem 0 .35rem 0; font-size: 2.15rem; font-weight: 800; padding: 0;}}
    .hero p {{margin: 0; opacity: .9; font-size: 1.02rem; max-width: 760px;}}
    .hero .chips {{margin-top: 1rem; display: flex; flex-wrap: wrap; gap: .5rem; position: relative; z-index: 1;}}
    .hero .chip {{background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.25);
        padding: .25rem .75rem; border-radius: 999px; font-size: .82rem; backdrop-filter: blur(4px);}}

    /* cards */
    .card {{background: {T['card']}; border: 1px solid {T['border']}; border-radius: 16px;
        padding: 1.1rem 1.2rem; box-shadow: {T['shadow']}; height: 100%;}}
    .card .icon {{font-size: 1.6rem;}}
    .card h4 {{margin: .4rem 0 .2rem 0; padding: 0; font-size: 1.05rem;}}
    .card p {{margin: 0; color: {T['muted']}; font-size: .9rem; line-height: 1.45;}}
    .card .big {{font-size: 1.7rem; font-weight: 800; margin-top: .3rem; color: {T['text']};}}
    .pillar {{border-top: 3px solid var(--accent);}}

    .steps {{display: flex; flex-wrap: wrap; gap: .6rem; align-items: stretch;}}
    .step {{flex: 1 1 140px; background: {T['card']}; border: 1px solid {T['border']}; border-radius: 14px;
        padding: .8rem .9rem; position: relative;}}
    .step .n {{display: inline-flex; width: 1.6rem; height: 1.6rem; border-radius: 50%; align-items: center;
        justify-content: center; font-weight: 700; font-size: .8rem; color: #fff; background: {SERIES[0]};}}
    .step b {{display: block; margin-top: .45rem; font-size: .92rem;}}
    .step span {{color: {T['muted']}; font-size: .8rem;}}

    div[data-testid="stMetric"] {{background: {T['card']}; border: 1px solid {T['border']};
        border-radius: 14px; padding: .85rem 1rem; box-shadow: {T['shadow']};}}
    div[data-testid="stMetricLabel"] p {{color: {T['muted']}; font-weight: 500;}}
    [data-testid="stForm"] {{background: {T['card']}; border: 1px solid {T['border']}; border-radius: 18px;
        padding: 1.2rem 1.3rem; box-shadow: {T['shadow']};}}

    /* tabs as pills */
    .stTabs [data-baseweb="tab-list"] {{gap: .35rem; background: {T['card']}; border: 1px solid {T['border']};
        padding: .35rem; border-radius: 14px; flex-wrap: wrap;}}
    .stTabs [data-baseweb="tab"] {{padding: .45rem 1rem; border-radius: 10px; height: auto;}}
    .stTabs [aria-selected="true"] {{background: {SERIES[0]}; color: #fff !important;}}
    .stTabs [aria-selected="true"] p {{color: #fff !important;}}
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{display: none;}}

    .stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button {{
        background: linear-gradient(120deg, #1c5cab, #2a78d6 55%, #4a3aa7); border: 0; color: #fff;
        font-weight: 600; box-shadow: 0 6px 18px rgba(42,120,214,.35);}}

    .decision {{border-radius: 16px; padding: 1.1rem 1.3rem; border: 1px solid; margin-bottom: .9rem;}}
    .decision h3 {{margin: 0; padding: 0; font-size: 1.35rem;}}
    .decision p {{margin: .35rem 0 0 0; color: {T['muted']};}}
    .decision.good {{border-color: rgba(12,163,12,.5); background: rgba(12,163,12,.10);}}
    .decision.bad {{border-color: rgba(208,59,59,.55); background: rgba(208,59,59,.10);}}
    .factor {{display: flex; justify-content: space-between; gap: .8rem; padding: .5rem .1rem;
        border-bottom: 1px dashed {T['border']}; font-size: .92rem;}}
    .factor .val {{color: {T['muted']}; text-align: right;}}
    .section-label {{font-size: .74rem; text-transform: uppercase; letter-spacing: .08em;
        color: {T['muted']}; margin: .2rem 0 .3rem 0; font-weight: 700;}}
    .verdict {{border-radius: 12px; padding: .8rem 1rem; margin: .5rem 0 .9rem 0; border: 1px solid;}}
    .verdict.ok {{border-color: rgba(12,163,12,.5); background: rgba(12,163,12,.10);}}
    .verdict.warn {{border-color: rgba(236,131,90,.6); background: rgba(236,131,90,.12);}}
    .footer {{text-align: center; color: {T['muted']}; font-size: .8rem; margin-top: 2rem; padding: 1rem 0;
        border-top: 1px solid {T['border']};}}
    </style>
    """,
    unsafe_allow_html=True,
)


# ================================================================== data
@st.cache_resource
def load_artifacts():
    if not (MODELS_DIR / "best_model.joblib").exists():
        st.error("Model not found. Run `python src/prepare_data.py` and `python src/train.py` first.")
        st.stop()
    models = joblib.load(MODELS_DIR / "all_models.joblib")
    metrics = json.loads((REPORTS_DIR / "metrics.json").read_text())
    data = pd.read_csv(DATA_CSV)
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    return models, metrics, data, train, test


@st.cache_resource
def get_explainer(model_name: str):
    models, _, _, train, _ = load_artifacts()
    return RiskExplainer(models[model_name], train.sample(200, random_state=0))


@st.cache_data
def test_probabilities(model_name: str):
    models, _, _, _, test = load_artifacts()
    return models[model_name].predict_proba(test[FEATURES])[:, 1]


@st.cache_data
def global_importance(model_name: str) -> pd.Series:
    _, _, _, _, test = load_artifacts()
    return get_explainer(model_name).global_importance(test[FEATURES])


def style_fig(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template=T["plotly"], height=height, margin=dict(l=10, r=10, t=48, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=T["text"], size=13),
        title_font=dict(size=15, color=T["text"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=T["muted"])),
        hoverlabel=dict(font_size=13, bgcolor=T["surface"], bordercolor=T["border"], font_color=T["text"]),
    )
    fig.update_xaxes(gridcolor=T["grid"], zerolinecolor=T["axis"], linecolor=T["axis"], tickfont_color=T["muted"])
    fig.update_yaxes(gridcolor=T["grid"], zerolinecolor=T["axis"], linecolor=T["axis"], tickfont_color=T["muted"])
    return fig


def show(fig: go.Figure, height: int = 360, toolbar: bool = True, container=st):
    container.plotly_chart(style_fig(fig, height), width="stretch", theme=None,
                           config={"displayModeBar": toolbar, "displaylogo": False})


def fmt(v) -> str:
    if isinstance(v, numbers.Real):
        return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.2f}"
    return str(v)


def card(icon: str, title: str, body: str, big: str = "", accent: str | None = None) -> str:
    style = f' style="--accent:{accent}"' if accent else ""
    cls = "card pillar" if accent else "card"
    big_html = f'<div class="big">{big}</div>' if big else ""
    return f'<div class="{cls}"{style}><div class="icon">{icon}</div><h4>{title}</h4>{big_html}<p>{body}</p></div>'


models, metrics, data, train, test = load_artifacts()
best_name = metrics["best_model"]

FORM_SECTIONS = {
    "👤 Personal": ["age", "num_dependents", "job", "employment_since", "housing", "residence_since",
                   "telephone", "foreign_worker"],
    "💰 Finances": ["checking_status", "savings", "property", "existing_credits"],
    "📄 Loan request": ["credit_amount", "duration_months", "purpose", "installment_rate", "other_debtors"],
    "🧾 Credit history": ["credit_history", "other_installment_plans"],
}
assert sorted(sum(FORM_SECTIONS.values(), [])) == sorted(FEATURES)

# ================================================================== sidebar
with st.sidebar:
    st.markdown("## 🏦 Loan Risk AI")
    st.caption("Explainable · Fair · Auditable")
    is_dark = st.session_state.theme == "dark"
    st.button("☀️ Switch to Light mode" if is_dark else "🌙 Switch to Dark mode",
              on_click=toggle_theme, width="stretch", key="theme_btn_sidebar")
    st.divider()
    st.markdown('<div class="section-label">Model settings</div>', unsafe_allow_html=True)
    model_name = st.selectbox(
        "Classifier", list(models), index=list(models).index(best_name),
        format_func=lambda n: f"{n}  ★" if n == best_name else n,
        help=f"★ = best model by 5-fold cross-validated ROC-AUC ({best_name}).",
    )
    threshold = st.slider(
        "Decision threshold", 0.1, 0.9, DECISION_THRESHOLD, 0.05,
        help="Applicants with a default probability at or above this value are classed as HIGH risk.",
    )
    m = metrics["models"][model_name]
    c1, c2 = st.columns(2)
    c1.metric("Test AUC", f"{m['roc_auc']:.3f}")
    c2.metric("CV AUC", f"{m['cv_roc_auc_mean']:.3f}")
    st.divider()
    with st.expander("ℹ️ About this project"):
        st.markdown(
            "M.Sc. Computer Science (Part II, Sem III) — RTMNU, 2026-27.\n\n"
            "Predicts **loan default risk**, explains each decision with **SHAP**, and audits the model "
            "for **fairness** across gender and age.\n\n"
            "**Data:** UCI Statlog (German Credit), 1,000 applicants."
        )
    with st.expander("🧭 How to use"):
        st.markdown(
            "1. Open **Predict & Explain**.\n2. Load a sample applicant or fill in the form.\n"
            "3. Click **Predict risk** to see the decision and why.\n"
            "4. Check **Fairness Report** to audit the model.\n5. Move the threshold to see trade-offs."
        )

pipeline = models[model_name]
explainer = get_explainer(model_name)

# ================================================================== header
head_l, head_r = st.columns([6, 1.3], vertical_alignment="center")
head_r.button("☀️ Light" if st.session_state.theme == "dark" else "🌙 Dark", on_click=toggle_theme,
              width="stretch", key="theme_btn_header", help="Switch between dark and light mode")
head_l.markdown(
    f'<div class="section-label" style="margin:0">Machine learning · Explainable AI · Responsible AI</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="hero">
      <div class="eyebrow">Credit risk intelligence</div>
      <h1>Explainable Loan Risk Predictor</h1>
      <p>Predicts the probability of loan default, shows <b>why</b> with SHAP, and audits the model for
         <b>fairness</b> across sensitive groups — accurate, explainable and auditable.</p>
      <div class="chips">
        <span class="chip">📊 {len(data):,} applicants</span>
        <span class="chip">⚠️ {data[TARGET].mean():.0%} historical default rate</span>
        <span class="chip">🤖 {model_name}{' ★' if model_name == best_name else ''}</span>
        <span class="chip">📈 ROC-AUC {m['roc_auc']:.3f}</span>
        <span class="chip">🎯 Threshold {threshold:.0%}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_home, tab_predict, tab_models, tab_global, tab_fair, tab_data = st.tabs([
    "🏠 Overview", "🔍 Predict & Explain", "📊 Model Comparison", "🌐 Global Explanation",
    "⚖️ Fairness Report", "📁 Dataset",
])

# ================================================================== Overview
with tab_home:
    fair_all = {a: fairness_report(test[TARGET].values, (test_probabilities(best_name) >= DECISION_THRESHOLD).astype(int),
                                   test[a].values, a) for a in SENSITIVE_ATTRIBUTES}
    n_flag = sum(not r["verdict"].startswith("No") for r in fair_all.values())
    imp_best = global_importance(best_name)
    pillars = st.columns(3)
    pillars[0].markdown(card(
        "🎯", "Accurate", f"Best of 3 classifiers — <b>{best_name}</b>, selected by 5-fold cross-validated ROC-AUC.",
        f"{metrics['models'][best_name]['roc_auc']:.3f} AUC", SERIES[0]), unsafe_allow_html=True)
    pillars[1].markdown(card(
        "🔎", "Explainable", f"SHAP traces every prediction to its features. Top driver: "
        f"<b>{FEATURE_LABELS[imp_best.index[0]].lower()}</b>.", f"{len(FEATURES)} features", SERIES[1]),
        unsafe_allow_html=True)
    pillars[2].markdown(card(
        "⚖️", "Auditable", "Demographic Parity &amp; Equal Opportunity checked across gender and age.",
        f"{n_flag} of {len(fair_all)} attributes flagged", SERIES[2]), unsafe_allow_html=True)

    st.markdown("#### 🧩 How the system works")
    steps = [
        ("Data collection", "UCI German Credit, 1,000 applicants"),
        ("Preprocessing", "Impute · one-hot encode · scale"),
        ("Model building", "LogReg · Random Forest · XGBoost"),
        ("Explainability", "SHAP global &amp; local explanations"),
        ("Fairness audit", "Demographic Parity · Equal Opportunity"),
        ("Dashboard", "Interactive Streamlit interface"),
    ]
    st.markdown('<div class="steps">' + "".join(
        f'<div class="step"><span class="n">{i}</span><b>{t}</b><span>{d}</span></div>'
        for i, (t, d) in enumerate(steps, 1)) + "</div>", unsafe_allow_html=True)

    st.markdown("#### 🎯 Project objectives")
    objectives = [
        ("🤖", "Predict default risk", "Estimate each applicant's probability of default from financial and demographic data."),
        ("📊", "Compare algorithms", "Accuracy, precision, recall, F1 and ROC-AUC for three classifiers."),
        ("💡", "Explain every decision", "SHAP contributions plus a plain-language summary for each applicant."),
        ("⚖️", "Audit for fairness", "Group-level report across sensitive attributes (gender, age)."),
    ]
    oc = st.columns(4)
    for col, (i, t, d) in zip(oc, objectives):
        col.markdown(card(i, t, d), unsafe_allow_html=True)

    st.markdown("#### 📌 Key findings")
    k1, k2 = st.columns([1.2, 1])
    with k1:
        top = imp_best.head(6)[::-1]
        f = go.Figure(go.Bar(
            x=top.values, y=[FEATURE_LABELS[x] for x in top.index], orientation="h",
            marker=dict(color=SERIES[0], cornerradius=4),
            hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
        ))
        f.update_layout(title="Top 6 risk drivers (SHAP)", xaxis_title="Mean |SHAP value|")
        show(f, 330, toolbar=False)
    with k2:
        for attr, rep in fair_all.items():
            ok = rep["verdict"].startswith("No")
            st.markdown(
                f'<div class="verdict {"ok" if ok else "warn"}"><b>{"✅" if ok else "⚠️"} '
                f'{attr.replace("_", " ").title()}</b> — DP diff {rep["demographic_parity_difference"]:.2f}, '
                f'EO diff {rep["equal_opportunity_difference"]:.2f}, DI ratio {rep["disparate_impact_ratio"]:.2f}'
                f'<br><span style="color:{T["muted"]};font-size:.88rem">{rep["verdict"]}</span></div>',
                unsafe_allow_html=True,
            )
        st.caption("Computed for the selected best model at the default 50% threshold on the held-out test set.")

# ================================================================== Predict
with tab_predict:
    top_l, top_r = st.columns([2, 1], vertical_alignment="bottom")
    top_l.markdown("#### 📝 Applicant details")
    preset = top_r.selectbox(
        "Load a sample applicant",
        ["Custom"] + [f"Applicant #{i}" for i in test["applicant_id"].head(50)],
        help="Pre-fill the form with a real applicant from the held-out test set.",
    )
    preset_row = test[test["applicant_id"] == int(preset.split("#")[1])].iloc[0] if preset != "Custom" else None
    defaults = preset_row if preset_row is not None else data[FEATURES].mode().iloc[0]

    with st.form("applicant"):
        values = {}
        section_cols = st.columns(len(FORM_SECTIONS))
        for col, (section, feats) in zip(section_cols, FORM_SECTIONS.items()):
            with col:
                st.markdown(f"**{section}**")
                for feat in feats:
                    if feat in INPUT_RANGES:
                        lo, hi = INPUT_RANGES[feat]
                        values[feat] = st.number_input(
                            FEATURE_LABELS[feat], min_value=lo, max_value=hi,
                            value=min(max(int(defaults[feat]), lo), hi),
                            step=500 if feat == "credit_amount" else 1, help=INPUT_HELP.get(feat),
                        )
                    else:
                        options = sorted(data[feat].unique())
                        values[feat] = st.selectbox(FEATURE_LABELS[feat], options, index=options.index(defaults[feat]))
        submitted = st.form_submit_button("🔮 Predict risk", type="primary", width="stretch")

    # Keep the last submitted applicant so the result survives reruns (sliders, model, theme).
    if submitted:
        st.session_state.submitted = {"preset": preset, "values": values}
    saved = st.session_state.get("submitted")
    if saved and saved["preset"] != preset:
        saved = None
    show_values = saved["values"] if saved else (values if preset_row is not None else None)

    if show_values is not None:
        applicant = pd.DataFrame([show_values])[FEATURES]
        outside = [
            f"{FEATURE_LABELS[f]} = {fmt(applicant.iloc[0][f])} (training range {fmt(data[f].min())}–{fmt(data[f].max())})"
            for f in INPUT_RANGES
            if not data[f].min() <= applicant.iloc[0][f] <= data[f].max()
        ]
        if outside:
            st.warning("Some values are outside the range seen during training, so this prediction is less "
                       "reliable: " + "; ".join(outside) + ".", icon="⚠️")
        prob = float(pipeline.predict_proba(applicant)[0, 1])
        high = prob >= threshold
        explanation = explainer.explain_applicant(applicant)

        st.markdown("#### 📋 Result")
        g_col, d_col = st.columns([1, 1.4])
        with g_col:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 46, "color": T["text"]}},
                title={"text": "Probability of default", "font": {"size": 15, "color": T["muted"]}},
                gauge={
                    "axis": {"range": [0, 100], "ticksuffix": "%", "tickcolor": T["axis"],
                             "tickfont": {"color": T["muted"]}},
                    "bar": {"color": T["bad"] if high else T["good"], "thickness": 0.3},
                    "bgcolor": T["grid"], "borderwidth": 0,
                    "threshold": {"line": {"color": T["text"], "width": 3}, "thickness": 0.9,
                                  "value": threshold * 100},
                },
            ))
            show(gauge, 280, toolbar=False)
            st.caption(f"Marker = decision threshold ({threshold:.0%}).")

        with d_col:
            if high:
                st.markdown(
                    f'<div class="decision bad"><h3>⛔ HIGH RISK — Reject / refer for review</h3>'
                    f'<p>Estimated default probability <b>{prob:.1%}</b> is at or above the {threshold:.0%} threshold.</p></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="decision good"><h3>✅ LOW RISK — Approve</h3>'
                    f'<p>Estimated default probability <b>{prob:.1%}</b> is below the {threshold:.0%} threshold.</p></div>',
                    unsafe_allow_html=True,
                )
            if preset_row is not None:
                actual = preset_row[TARGET]
                correct = bool(actual) == high
                st.markdown(
                    f"**Actual outcome:** {'🔴 Defaulted' if actual else '🟢 Repaid'} &nbsp;·&nbsp; "
                    f"Model was **{'correct ✔' if correct else 'incorrect ✘'}**"
                )
            up = explanation[explanation["contribution"] > 0].head(3)
            down = explanation[explanation["contribution"] < 0].head(3)
            f1, f2 = st.columns(2)
            with f1:
                st.markdown('<div class="section-label">▲ Raised the risk</div>', unsafe_allow_html=True)
                for r in up.itertuples():
                    st.markdown(f'<div class="factor"><span>{r.label}</span><span class="val">{fmt(r.value)}</span></div>',
                                unsafe_allow_html=True)
            with f2:
                st.markdown('<div class="section-label">▼ Lowered the risk</div>', unsafe_allow_html=True)
                for r in down.itertuples():
                    st.markdown(f'<div class="factor"><span>{r.label}</span><span class="val">{fmt(r.value)}</span></div>',
                                unsafe_allow_html=True)

        st.markdown("#### 💬 Why? In plain language")
        st.info(plain_language_summary(prob, explanation, threshold), icon="💡")

        st.markdown("#### 📊 Feature contributions (SHAP)")
        n_show = st.slider("Number of factors to show", 5, len(FEATURES), 10, key="n_factors")
        top = explanation.head(n_show)[::-1]
        bar = go.Figure(go.Bar(
            x=top["contribution"], y=[f"{l} = {fmt(v)}" for l, v in zip(top["label"], top["value"])],
            orientation="h",
            marker=dict(color=[T["up"] if c > 0 else T["down"] for c in top["contribution"]], cornerradius=4),
            customdata=top["direction"],
            hovertemplate="<b>%{y}</b><br>%{customdata}<br>SHAP contribution: %{x:+.4f}<extra></extra>",
        ))
        bar.add_vline(x=0, line_color=T["axis"], line_width=1)
        bar.update_layout(
            title=dict(text=f"<span style='color:{T['up']}'>■</span> increases risk &nbsp; "
                            f"<span style='color:{T['down']}'>■</span> decreases risk"),
            xaxis_title=f"Contribution to default risk ({explanation.attrs['units']})", bargap=0.3,
        )
        show(bar, 130 + 32 * n_show)

        with st.expander("📋 Full contribution table"):
            st.dataframe(
                explanation[["label", "value", "contribution", "direction"]].astype({"value": str})
                .rename(columns={"label": "Feature", "value": "Value", "contribution": "SHAP contribution",
                                 "direction": "Effect"}),
                width="stretch", hide_index=True,
                column_config={"SHAP contribution": st.column_config.NumberColumn(format="%+.4f")},
            )
            st.caption(f"Base value (average model output): {explanation.attrs['base_value']:.3f} "
                       f"{explanation.attrs['units']}. Base value + sum of contributions = this applicant's model output.")
    else:
        st.info("Fill in the form and click **Predict risk**, or load a sample applicant.", icon="👆")

# ================================================================== Models
with tab_models:
    st.markdown("#### 📊 Classifier comparison (held-out test set)")
    cards = st.columns(len(metrics["models"]))
    for i, (col, (name, mm)) in enumerate(zip(cards, metrics["models"].items())):
        badge = " &nbsp;<span style='font-size:.75rem;background:%s;color:#fff;padding:.1rem .5rem;border-radius:999px'>★ BEST</span>" % SERIES[0] if name == best_name else ""
        col.markdown(
            f'<div class="card pillar" style="--accent:{SERIES[i]}"><h4>{name}{badge}</h4>'
            f'<div class="big">{mm["roc_auc"]:.3f} <span style="font-size:.85rem;color:{T["muted"]};font-weight:500">ROC-AUC</span></div>'
            f'<p>F1 {mm["f1"]:.3f} · Accuracy {mm["accuracy"]:.3f}<br>Precision {mm["precision"]:.3f} · Recall {mm["recall"]:.3f}'
            f'<br>5-fold CV AUC {mm["cv_roc_auc_mean"]:.3f} ± {mm["cv_roc_auc_std"]:.3f}</p></div>',
            unsafe_allow_html=True,
        )
    st.write("")
    c1, c2 = st.columns(2)
    labels = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC"}
    comp = go.Figure()
    for i, (name, mm) in enumerate(metrics["models"].items()):
        comp.add_bar(
            name=name, x=list(labels.values()), y=[mm[k] for k in labels],
            marker=dict(color=SERIES[i], cornerradius=4, line=dict(width=2, color=T["bg"])),
            hovertemplate=f"<b>{name}</b><br>%{{x}}: %{{y:.3f}}<extra></extra>",
        )
    comp.update_layout(title="Metrics by model", barmode="group", bargap=0.25, bargroupgap=0.05,
                       yaxis=dict(range=[0, 1], title="Score"))
    show(comp, 400, container=c1)

    roc = go.Figure()
    for i, name in enumerate(metrics["models"]):
        fpr, tpr, _ = roc_curve(test[TARGET], test_probabilities(name))
        roc.add_scatter(
            x=fpr, y=tpr, mode="lines", name=f"{name} ({metrics['models'][name]['roc_auc']:.3f})",
            line=dict(color=SERIES[i], width=2.2),
            hovertemplate=f"<b>{name}</b><br>FPR %{{x:.2f}} · TPR %{{y:.2f}}<extra></extra>",
        )
    roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                    line=dict(color=T["muted"], dash="dash", width=1), hoverinfo="skip")
    roc.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate")
    show(roc, 400, container=c2)

    st.markdown("#### 🧮 Confusion matrices")
    cm_cols = st.columns(len(metrics["models"]))
    for col, (name, mm) in zip(cm_cols, metrics["models"].items()):
        cm = mm["confusion_matrix"]
        z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
        heat = go.Figure(go.Heatmap(
            z=z, x=["Pred: repaid", "Pred: default"], y=["Actual: repaid", "Actual: default"],
            colorscale=[[i / 4, c] for i, c in enumerate(T["seq"])], showscale=False,
            text=z, texttemplate="<b>%{text}</b>", textfont=dict(size=20), xgap=3, ygap=3,
            hovertemplate="%{y} · %{x}: %{z}<extra></extra>",
        ))
        heat.update_layout(title=name)
        heat.update_yaxes(autorange="reversed", showgrid=False)
        heat.update_xaxes(showgrid=False)
        show(heat, 290, toolbar=False, container=col)
    st.caption(f"Best model selected by {metrics['selection_criterion']}. Train {metrics['n_train']} / "
               f"test {metrics['n_test']} applicants, threshold {metrics['decision_threshold']}.")

# ================================================================== Global SHAP
with tab_global:
    st.markdown(f"#### 🌐 What drives the {model_name} model overall?")
    importance = global_importance(model_name)
    c1, c2 = st.columns([1.1, 1])
    top = importance.head(12)[::-1]
    imp = go.Figure(go.Bar(
        x=top.values, y=[FEATURE_LABELS[f] for f in top.index], orientation="h",
        marker=dict(color=SERIES[0], cornerradius=4),
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
    ))
    imp.update_layout(title="Global feature importance (mean |SHAP|)", xaxis_title="Mean |SHAP value|")
    show(imp, 500, container=c1)
    with c2:
        if model_name == best_name:
            st.image(str(FIGURES_DIR / T["beeswarm"]),
                     caption="SHAP beeswarm: each dot is an applicant; right = pushes towards default.")
        else:
            st.info(f"The SHAP beeswarm plot is pre-computed for the best model ({best_name}).", icon="ℹ️")
    top3 = [FEATURE_LABELS[f].lower() for f in importance.index[:3]]
    st.success(f"**Key insight:** the three most influential factors are {top3[0]}, {top3[1]} and {top3[2]}.", icon="💡")
    st.caption("Global importance = average absolute SHAP contribution of each feature across the test set. "
               "One-hot encoded categories are summed back into their original feature.")

# ================================================================== Fairness
with tab_fair:
    st.markdown(f"#### ⚖️ Fairness audit — {model_name} at threshold {threshold:.0%}")
    st.caption("Favourable outcome = predicted to repay (approved). Gender is **not** a model input; it is only "
               "audited. Computed live on the held-out test set — move the threshold in the sidebar to explore.")
    pred = (test_probabilities(model_name) >= threshold).astype(int)
    for attr in SENSITIVE_ATTRIBUTES:
        rep = fairness_report(test[TARGET].values, pred, test[attr].values, attr)
        ok = rep["verdict"].startswith("No")
        with st.container(border=True):
            st.markdown(f"##### {'✅' if ok else '⚠️'} {attr.replace('_', ' ').title()}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Demographic parity diff.", f"{rep['demographic_parity_difference']:.3f}",
                      help="Gap between highest and lowest group approval rate. Ideal: 0 (≤ 0.10 acceptable).")
            m2.metric("Equal opportunity diff.", f"{rep['equal_opportunity_difference']:.3f}",
                      help="Gap in approval rate among applicants who actually repaid. Ideal: 0 (≤ 0.10 acceptable).")
            m3.metric("Disparate impact ratio", f"{rep['disparate_impact_ratio']:.3f}",
                      help="Lowest / highest approval rate. Should be ≥ 0.80 (four-fifths rule).")
            st.markdown(f'<div class="verdict {"ok" if ok else "warn"}">{rep["verdict"]}</div>', unsafe_allow_html=True)

            groups = pd.DataFrame(rep["groups"]).set_index("group")
            if groups["count"].min() < 50:
                st.caption(f"ℹ️ Smallest group ({groups['count'].idxmin()}) has only {groups['count'].min()} test "
                           "applicants, so differences should be interpreted with caution.")
            c1, c2 = st.columns([1.3, 1])
            fb = go.Figure()
            for i, (colname, label) in enumerate([("approval_rate", "Approval rate"),
                                                  ("true_approval_rate", "Approval rate of actual repayers")]):
                fb.add_bar(
                    name=label, x=groups.index, y=groups[colname],
                    marker=dict(color=SERIES[i], cornerradius=4, line=dict(width=2, color=T["bg"])),
                    text=[f"{v:.0%}" for v in groups[colname]], textposition="outside",
                    textfont=dict(color=T["text"]),
                    hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.1%}}<extra></extra>",
                )
            fb.update_layout(barmode="group", bargap=0.35, bargroupgap=0.05,
                             yaxis=dict(range=[0, 1.12], tickformat=".0%"))
            show(fb, 320, toolbar=False, container=c1)
            with c2:
                st.dataframe(
                    groups.rename(columns={
                        "count": "Applicants", "actual_good_rate": "Actually repaid", "approval_rate": "Approved",
                        "true_approval_rate": "Repayers approved", "false_approval_rate": "Defaulters approved",
                    }),
                    width="stretch",
                    column_config={c: st.column_config.NumberColumn(format="percent") for c in
                                   ["Actually repaid", "Approved", "Repayers approved", "Defaulters approved"]},
                )
                st.caption(f"Least favoured group: **{rep['least_favoured_group']}**")
    with st.expander("📘 How to read these metrics"):
        st.markdown(
            "- **Demographic parity difference** — gap between the highest and lowest group approval rates.\n"
            "- **Equal opportunity difference** — gap in approval rates among applicants who actually repaid.\n"
            "- **Disparate impact ratio** — lowest / highest approval rate; below 0.80 fails the four-fifths rule.\n"
            "- These metrics can conflict; improving one does not guarantee overall fairness."
        )

# ================================================================== Data
with tab_data:
    st.markdown("#### 📁 UCI Statlog (German Credit) dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Applicants", f"{len(data):,}")
    c2.metric("Default rate", f"{data[TARGET].mean():.1%}")
    c3.metric("Model features", len(FEATURES))
    c4.metric("Train / test", f"{len(train)} / {len(test)}")

    feat = st.selectbox("Explore default rate by", FEATURES, format_func=FEATURE_LABELS.get,
                        index=FEATURES.index("checking_status"))
    if feat in data.select_dtypes("number").columns and data[feat].nunique() > 10:
        bucket = pd.qcut(data[feat], 5, duplicates="drop").astype(str)
    else:
        bucket = data[feat].astype(str)
    grp = data.groupby(bucket)[TARGET].agg(["mean", "count"]).sort_values("mean")
    dist = go.Figure(go.Bar(
        x=grp["mean"], y=grp.index, orientation="h", marker=dict(color=SERIES[0], cornerradius=4),
        customdata=grp["count"], text=[f"{v:.0%}" for v in grp["mean"]], textposition="outside",
        textfont=dict(color=T["text"]),
        hovertemplate="<b>%{y}</b><br>Default rate: %{x:.1%}<br>Applicants: %{customdata}<extra></extra>",
    ))
    dist.add_vline(x=data[TARGET].mean(), line_dash="dash", line_color=T["muted"],
                   annotation_text=f"Overall {data[TARGET].mean():.0%}", annotation_position="top",
                   annotation_font_color=T["muted"])
    dist.update_layout(title=f"Default rate by {FEATURE_LABELS[feat].lower()}",
                       xaxis=dict(tickformat=".0%", range=[0, max(0.8, grp["mean"].max() * 1.2)]))
    show(dist, 150 + 40 * len(grp))

    with st.expander("🗂️ Browse raw data"):
        st.dataframe(data, width="stretch", hide_index=True)

st.markdown(
    '<div class="footer">Explainable Loan Risk Predictor · M.Sc. Computer Science, RTMNU 2026-27 · '
    "Built with scikit-learn, XGBoost, SHAP &amp; Streamlit</div>",
    unsafe_allow_html=True,
)
