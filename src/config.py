"""Shared paths and column definitions for the Explainable Loan Risk Predictor."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = ROOT / "data" / "raw" / "german.data"
DATA_CSV = ROOT / "data" / "german_credit.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2
DECISION_THRESHOLD = 0.5  # predicted default probability >= threshold -> high risk / reject

TARGET = "default"  # 1 = bad credit (default), 0 = good credit

NUMERIC_FEATURES = [
    "duration_months",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
]

CATEGORICAL_FEATURES = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings",
    "employment_since",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Sensitive attributes audited for fairness. Gender (and personal_status, which
# encodes it) is deliberately NOT used as a model input; it is only audited.
SENSITIVE_ATTRIBUTES = ["gender", "age_group"]

# Allowed form inputs (min, max). Ordinal UCI attributes keep their original 1-4 / 1-2 scales;
# age, duration and amount allow realistic values beyond the training range (the app warns).
INPUT_RANGES = {
    "age": (18, 100),
    "duration_months": (1, 120),
    "credit_amount": (100, 100_000),
    "installment_rate": (1, 4),
    "residence_since": (1, 4),
    "existing_credits": (1, 4),
    "num_dependents": (1, 2),
}

INPUT_HELP = {
    "installment_rate": "Instalment as % of disposable income, UCI scale: 1 (< 20%) to 4 (>= 35%).",
    "residence_since": "Years at present residence, UCI scale 1-4 (4 = 4 years or more).",
    "existing_credits": "Number of existing credits at this bank (1-4, 4 = four or more).",
    "num_dependents": "Number of people the applicant is liable to maintain (1 or 2+).",
    "credit_amount": "Loan amount in Deutsche Mark (DM).",
}

FEATURE_LABELS = {
    "duration_months": "Loan duration (months)",
    "credit_amount": "Credit amount (DM)",
    "installment_rate": "Installment rate (% of income)",
    "residence_since": "Years at current residence",
    "age": "Age",
    "existing_credits": "Existing credits at bank",
    "num_dependents": "Number of dependents",
    "checking_status": "Checking account status",
    "credit_history": "Credit history",
    "purpose": "Loan purpose",
    "savings": "Savings account",
    "employment_since": "Employment duration",
    "other_debtors": "Other debtors / guarantors",
    "property": "Property",
    "other_installment_plans": "Other installment plans",
    "housing": "Housing",
    "job": "Job type",
    "telephone": "Telephone",
    "foreign_worker": "Foreign worker",
}
