# Explainable Loan Risk Predictor

M.Sc. Computer Science (Part II, Sem III) project — RTMNU, 2026-27.

Predicts the probability of loan default, explains every prediction with SHAP, audits the
model for fairness across sensitive attributes, and presents everything in a Streamlit dashboard.

## Dataset
**UCI Statlog (German Credit Data)** — 1,000 applicants, 20 attributes, 30% default rate.
`src/prepare_data.py` decodes the raw UCI codes into readable values and derives:
- `default` — target (1 = bad credit / default, 0 = good credit)
- `gender` — from the `personal_status` attribute (sensitive, audited only — **not** a model input)
- `age_group` — `Under 25` / `25 and over` (sensitive, audited)

## Project structure
```
data/raw/german.data          original UCI file
data/german_credit.csv        decoded dataset
data/processed/               train / test split (80/20, stratified)
src/config.py                 paths, feature lists, settings
src/prepare_data.py           1. data collection & decoding
src/modeling.py               2-3. preprocessing (impute, one-hot, scale) + 3 classifiers
src/train.py                  3. training, 5-fold CV, evaluation, model selection
src/explain.py                4. SHAP global / local explanations + plain-language summary
src/fairness.py               5. Demographic Parity, Equal Opportunity, Disparate Impact
app.py                        6. Streamlit dashboard
tests/test_pipeline.py        7. testing & validation (data, models, SHAP, fairness)
tests/test_app.py             7. end-to-end dashboard tests (headless)
TEST_REPORT.md                QA test report
.streamlit/config.toml        dashboard theme (dark by default)
models/                       saved pipelines
reports/metrics.json          model comparison results
reports/fairness.json         fairness audit
reports/figures/              comparison, ROC, SHAP and fairness charts
```

## Requirements
- **Python 3.12 or 3.13** (required — numpy, xgboost and shap in `requirements.txt` do not support 3.11 or older).
  Check with `python --version`; download from https://www.python.org/downloads/ if needed.
- Git, and about 1 GB of free disk space for the libraries.
- No internet is needed after installation (the dataset is included in `data/raw/`).

## How to run
The repository already contains the trained models, so after installing you can go straight to
`streamlit run app.py`. The prepare/train steps rebuild everything from the raw data (≈1 minute).

```bash
git clone https://github.com/Gaurav-Ds/Explainable-Loan-Risk-Predictor.git
cd Explainable-Loan-Risk-Predictor
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt

python src/prepare_data.py        # build data/german_credit.csv
python src/train.py               # train, compare, explain, audit -> models/ & reports/
python -m pytest tests -q         # 21 automated tests (~1-2 min)
streamlit run app.py              # open http://localhost:8501
```

### Troubleshooting
| Problem | Fix |
|---|---|
| `No matching distribution found for numpy==...` | Your Python is older than 3.12 — install Python 3.12/3.13 and recreate `.venv`. |
| `streamlit: command not found` / not recognized | Activate the virtual environment first, or run `python -m streamlit run app.py`. |
| `activate` blocked in PowerShell | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt. |
| "Model not found" in the app | Run `python src/prepare_data.py` then `python src/train.py`. |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |

## Results (held-out test set, 200 applicants, threshold 0.5)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | 5-fold CV ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.760 | 0.570 | 0.817 | 0.671 | 0.814 | 0.768 ± 0.058 |
| **Random Forest** (selected) | 0.750 | 0.562 | 0.750 | 0.643 | 0.811 | **0.795 ± 0.046** |
| XGBoost | 0.750 | 0.566 | 0.717 | 0.632 | 0.796 | 0.788 ± 0.038 |

The best model is chosen by cross-validated ROC-AUC on the training set, so the test set stays unbiased.
Class imbalance is handled with balanced class weights / `scale_pos_weight`.

**Top global risk factors (SHAP):** checking account status, loan duration, savings, credit history,
other installment plans.

**Fairness audit (Random Forest):**
| Attribute | Demographic parity diff. | Equal opportunity diff. | Disparate impact | Verdict |
|---|---|---|---|---|
| Gender | 0.061 | 0.060 | 0.90 | No significant disparity |
| Age group | 0.236 | 0.233 | 0.64 | Disparity — applicants under 25 approved less often |

## Dashboard
Opens in **dark mode**; use the ☀️ / 🌙 button (top-right or sidebar) to switch theme.

0. **Overview** — accurate / explainable / auditable summary, system workflow, objectives, key findings.
1. **Predict & Explain** — enter applicant details (or load a test applicant), get default probability,
   risk level, a plain-language summary and a SHAP contribution chart.
2. **Model Comparison** — metrics table, comparison chart, ROC curves, confusion matrices.
3. **Global Explanation** — mean |SHAP| importance and SHAP beeswarm.
4. **Fairness Report** — group metrics for gender and age, recomputed live for the chosen model and threshold.
5. **Dataset** — default rate by any feature, plus the raw data.

Inputs are limited to realistic ranges; values outside the training range trigger a reliability warning.

## References
1. Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions", NeurIPS 2017.
2. UCI ML Repository, "Statlog (German Credit Data)".
3. Barocas, Hardt & Narayanan, *Fairness and Machine Learning*, fairmlbook.org, 2019.
4. Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System", KDD 2016.
5. Streamlit documentation — docs.streamlit.io
6. Scikit-learn documentation — scikit-learn.org
