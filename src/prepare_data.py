"""Convert the raw UCI Statlog (German Credit) file into a readable CSV.

Source: UCI Machine Learning Repository, "Statlog (German Credit Data)".
The raw file uses coded values (A11, A12, ...); this script decodes them,
derives the sensitive attributes `gender` and `age_group`, and maps the
target so that 1 = default (bad credit) and 0 = no default (good credit).
"""
import pandas as pd

from config import DATA_CSV, RAW_DATA

RAW_COLUMNS = [
    "checking_status", "duration_months", "credit_history", "purpose", "credit_amount",
    "savings", "employment_since", "installment_rate", "personal_status", "other_debtors",
    "residence_since", "property", "age", "other_installment_plans", "housing",
    "existing_credits", "job", "num_dependents", "telephone", "foreign_worker", "class",
]

CODES = {
    "checking_status": {"A11": "< 0 DM", "A12": "0-200 DM", "A13": ">= 200 DM", "A14": "No checking account"},
    "credit_history": {
        "A30": "No credits / all paid duly",
        "A31": "All credits at this bank paid duly",
        "A32": "Existing credits paid duly",
        "A33": "Delay in paying in the past",
        "A34": "Critical account / other credits",
    },
    "purpose": {
        "A40": "Car (new)", "A41": "Car (used)", "A42": "Furniture / equipment",
        "A43": "Radio / television", "A44": "Domestic appliances", "A45": "Repairs",
        "A46": "Education", "A47": "Vacation", "A48": "Retraining", "A49": "Business",
        "A410": "Others",
    },
    "savings": {
        "A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM",
        "A64": ">= 1000 DM", "A65": "Unknown / no savings",
    },
    "employment_since": {
        "A71": "Unemployed", "A72": "< 1 year", "A73": "1-4 years",
        "A74": "4-7 years", "A75": ">= 7 years",
    },
    "personal_status": {
        "A91": "Male: divorced/separated",
        "A92": "Female: divorced/separated/married",
        "A93": "Male: single",
        "A94": "Male: married/widowed",
        "A95": "Female: single",
    },
    "other_debtors": {"A101": "No other debtors", "A102": "Co-applicant", "A103": "Guarantor"},
    "property": {
        "A121": "Real estate",
        "A122": "Building society savings / life insurance",
        "A123": "Car or other",
        "A124": "Unknown / no property",
    },
    "other_installment_plans": {"A141": "Bank", "A142": "Stores", "A143": "No other plans"},
    "housing": {"A151": "Rent", "A152": "Own", "A153": "For free"},
    "job": {
        "A171": "Unemployed / unskilled non-resident",
        "A172": "Unskilled resident",
        "A173": "Skilled employee",
        "A174": "Management / self-employed / highly qualified",
    },
    "telephone": {"A191": "No", "A192": "Yes"},
    "foreign_worker": {"A201": "Yes", "A202": "No"},
}


def build_dataset() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA, sep=" ", header=None, names=RAW_COLUMNS)
    for col, mapping in CODES.items():
        df[col] = df[col].map(mapping)
        if df[col].isna().any():
            raise ValueError(f"Unmapped code in column {col!r}")

    df["gender"] = df["personal_status"].str.split(":").str[0]
    df["age_group"] = pd.cut(df["age"], bins=[0, 24, 200], labels=["Under 25", "25 and over"]).astype(str)
    df["default"] = (df.pop("class") == 2).astype(int)
    df.insert(0, "applicant_id", range(1, len(df) + 1))
    return df


def main() -> None:
    df = build_dataset()
    DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_CSV, index=False)
    print(f"Saved {len(df)} rows x {df.shape[1]} columns to {DATA_CSV}")
    print(f"Default rate: {df['default'].mean():.1%}")
    print(df["gender"].value_counts().to_string())
    print(df["age_group"].value_counts().to_string())


if __name__ == "__main__":
    main()
