from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

print("EXERCISE 3 - Missing values (employees.csv)")


def make_demo_employees(n=200, seed=0):
    """Demo data with the usual employees.csv structure (8 columns)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "First Name": rng.choice(["Anna", "Karim", "Lina", "Omar", "Sara", "Yacine"], n),
        "Gender": rng.choice(["Male", "Female"], n),
        "Start Date": rng.choice(["1/15/2005", "3/2/2010", "7/21/2015", "11/5/2019"], n),
        "Last Login Time": rng.choice(["9:15 AM", "1:30 PM", "4:45 PM", "11:00 AM"], n),
        "Salary": rng.integers(40000, 140000, n).astype(float),
        "Bonus %": rng.uniform(1, 20, n).round(3),
        "Senior Management": rng.choice([True, False], n),
        "Team": rng.choice(["Sales", "HR", "Finance", "Engineering"], n),
    })
    for col in df.columns:                        # inject ~5% missing values
        df[col] = df[col].astype(object).mask(rng.random(n) < 0.05)
    df[["Salary", "Bonus %"]] = df[["Salary", "Bonus %"]].astype(float)
    return df


# 1. Load and inspect
emp_path = DATA_DIR / "employees.csv"
if emp_path.exists():
    emp = pd.read_csv(emp_path)
else:
    print("[employees.csv not found -> using DEMO data with the same 8 columns]")
    emp = make_demo_employees()
print("--- Columns and dtypes ---")
print(emp.dtypes)
print("\n--- First rows ---")
print(emp.head())

# 2. Missing values per column
print("\n--- Missing values per column (original) ---")
print(emp.isna().sum())

# 3. Heatmap of missing-value patterns
plt.figure(figsize=(9, 5))
sns.heatmap(emp.isna(), cbar=False, yticklabels=False, cmap="viridis")
plt.title("Missing-value pattern in employees.csv (yellow = missing)")
plt.xlabel("Columns")
plt.ylabel("Rows (observations)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "missing_values_heatmap.png", dpi=150)
plt.show()

# 4. Copy + one synthetic row with 8 fields: First Name='-', Gender='na', rest = NaN
emp_copy = emp.copy()
new_row = pd.DataFrame([["-", "na"] + [np.nan] * (emp.shape[1] - 2)], columns=emp.columns)
emp_copy = pd.concat([emp_copy, new_row], ignore_index=True)
print("\nSynthetic row added:\n", emp_copy.tail(1))

# 5. Recount: '-' and 'na' are NOT counted
print("\n--- Missing values after adding the row (before conversion) ---")
print(emp_copy.isna().sum())
print("""
WHY? pandas only treats NaN/None/NaT as missing in a DataFrame. read_csv also
parses a default list of strings (such as '', 'NA', 'N/A', 'null') as NaN, but
only while reading a file, and lowercase 'na' and '-' are not in that list.
Our row was appended AFTER loading, so '-' and 'na' are just ordinary strings
(valid values) until we convert them ourselves.""")

# 6. Convert placeholders to NaN with a documented, case-insensitive rule
PLACEHOLDERS = {"-", "na", "n/a", "nan", "null", "none", ""}   # compared in lower case, stripped


def placeholders_to_nan(df):
    """Replace any text cell whose stripped, lower-cased value is a placeholder by NaN."""
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        s = out[col]
        is_placeholder = s.astype("string").str.strip().str.lower().isin(PLACEHOLDERS)
        out[col] = s.mask(is_placeholder.fillna(False))
    return out


emp_clean = placeholders_to_nan(emp_copy)
print("\n--- Missing values after converting placeholders to NaN ---")
print(emp_clean.isna().sum())

# 7. SimpleImputer (fit on TRAIN only, then transform train and test)
num_cols = ["Salary", "Bonus %"]                                   # median
cat_cols = ["Gender", "Senior Management", "Team",
            "Start Date", "Last Login Time"]                       # most_frequent
# 'First Name' is an identifier -> NOT imputed (inventing a name has no justification)

emp_model = emp_clean.copy()
emp_model[cat_cols] = emp_model[cat_cols].astype(object).where(emp_model[cat_cols].notna(), np.nan)
emp_train, emp_test = train_test_split(emp_model, test_size=0.2, random_state=42)

imputer = ColumnTransformer(
    [("num", SimpleImputer(strategy="median"), num_cols),
     ("cat", SimpleImputer(strategy="most_frequent"), cat_cols)],
    remainder="passthrough", verbose_feature_names_out=False)
imputer.set_output(transform="pandas")

emp_train_imp = imputer.fit_transform(emp_train)   # statistics learned on train only
emp_test_imp = imputer.transform(emp_test)         # reused on test (no leakage)

# 8. Before / after comparison
after = pd.concat([emp_train_imp, emp_test_imp]).isna().sum()
comparison = pd.DataFrame({"original file": emp.isna().sum(),
                           "after placeholders->NaN": emp_clean.isna().sum(),
                           "after imputation": after}).loc[emp.columns]
print("\n--- Missing values: before vs after ---")
print(comparison)
print("\nLearned medians (train):", dict(zip(num_cols, imputer.named_transformers_["num"].statistics_)))
print("Learned modes   (train):", dict(zip(cat_cols, imputer.named_transformers_["cat"].statistics_)))

print("""
LIMITATIONS OF IMPUTATION: it invents values, so it reduces variance and can
distort distributions and correlations; the median/mode ignores relations
between columns; it hides the fact that data were missing (a 'was_missing'
indicator can help); it assumes values are missing at random; a wrong choice
can bias the model. Fitting it on all data (not only train) leaks information.""")
