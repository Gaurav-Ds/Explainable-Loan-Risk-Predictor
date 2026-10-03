# Test Report — Explainable Loan Risk Predictor

**Environment:** Windows 11, Python 3.13, versions pinned in `requirements.txt`
**Result:** 21 / 21 automated tests passed · clean-install reproduction passed · 150+ UI states exercised with 0 errors

## 1. How to reproduce
```bash
pip install -r requirements.txt
python src/prepare_data.py
python src/train.py
python -m pytest tests -v
```

## 2. Automated tests (`tests/`)
| # | Test | What it verifies | Result |
|---|---|---|---|
| 1 | `test_dataset_integrity` | 1,000 rows, no missing values, binary target, 30% default rate, gender derived | Pass |
| 2 | `test_no_leakage_between_splits` | Train/test sets disjoint; gender/personal status not used as model inputs | Pass |
| 3 | `test_predictions_are_valid_probabilities` | All 3 models output probabilities in [0, 1] | Pass |
| 4-6 | `test_shap_additivity` (×3 models) | Base value + Σ SHAP contributions reproduces model output (tolerance 1e-3) | Pass |
| 7 | `test_local_explanation_and_summary` | Per-applicant explanation sorted by impact; plain-language summary generated | Pass |
| 8 | `test_fairness_metrics_on_known_values` | DP difference, EO difference, DI ratio match hand-computed values | Pass |
| 9 | `test_fairness_when_nobody_is_approved` | No divide-by-zero; reports "Cannot assess" instead of a false "fair" verdict | Pass |
| 10 | `test_fairness_no_disparity` | Equal outcomes → no disparity reported | Pass |
| 11 | `test_app_loads_dark_by_default` | Dashboard loads without errors, dark theme, 6 tabs | Pass |
| 12 | `test_theme_toggle_round_trip` | Dark → Light → Dark switching | Pass |
| 13-15 | `test_each_model_predicts_sample_applicant` (×3) | Prediction, decision card and summary for every model | Pass |
| 16 | `test_custom_result_survives_other_widget_changes` | Result stays visible after changing sliders | Pass |
| 17 | `test_input_bounds_are_realistic` | Form limits (e.g. age 18-100, instalment rate 1-4) | Pass |
| 18 | `test_extreme_inputs_warn_and_do_not_crash` | Min/max inputs predict safely; out-of-training-range warning shown | Pass |
| 19-20 | `test_extreme_thresholds` (0.1, 0.9) | Fairness and decisions at extreme thresholds | Pass |
| 21 | `test_dataset_explorer_all_features` | Default-rate explorer works for all 19 features | Pass |

## 3. Exploratory / stress testing
| Scenario | Coverage | Result |
|---|---|---|
| All 50 sample applicants × 3 models | 150 predictions with SHAP explanation | 0 errors |
| Thresholds 0.1 / 0.5 / 0.9 × 3 models | Decision + fairness recomputation | 0 errors |
| Theme toggled 4× in a row | Both buttons (header + sidebar) | 0 errors |
| Clean install in a fresh folder and virtualenv | Install → prepare → train → test | Identical metrics, 21/21 pass |

## 4. Defects found and fixed during QA
| ID | Severity | Defect | Fix |
|---|---|---|---|
| D1 | High | UCI category label "None" was read by pandas as a missing value, silently corrupting 2 features | Relabelled to "No other debtors" / "No other plans"; integrity test added |
| D2 | High | Result for a custom applicant disappeared when any slider/model/theme changed | Last submitted applicant kept in session state; regression test added |
| D3 | Medium | Form accepted unrealistic values (age 150, 1-4 scales up to 8) | Realistic bounds + help text; warning when outside training range |
| D4 | Medium | Fairness verdict could say "no disparity" when nobody was approved (0/0) | Explicit "Cannot assess" verdict; unit test added |
| D5 | Low | Dark SHAP beeswarm had unreadable dark-grey labels | Light label colour in dark figure |

## 5. Known limitations (by design — see synopsis §5)
- **Small subgroups:** only 39 test applicants are under 25; the dashboard shows a caution note for groups under 50.
- **SHAP values are approximations** of model behaviour, not a literal trace of the algorithm.
- **Fairness metrics can conflict;** improving one does not guarantee overall fairness.
- **Dataset:** German Credit (1994) is small and historical; amounts are in Deutsche Mark.
- **Theme setting** is applied through Streamlit's server config. It is designed for a single-user local demo;
  if several people use one shared server at once, a theme switch may briefly affect other open sessions.
