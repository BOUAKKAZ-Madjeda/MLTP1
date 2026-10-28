import io
from pathlib import Path
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, MaxAbsScaler,
                                   RobustScaler, normalize, Binarizer,
                                   OrdinalEncoder, LabelEncoder, OneHotEncoder)
 
BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
pd.set_option("display.width", 120)
 
 
def title(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)
 
 
print("EXERCISE 1 - Iris dataset")
 
# 1. Load the CSV into a DataFrame named D
D = pd.read_csv(BASE / "Iris_Dataset.csv")
 
# 2. Inspect the dataset
print("--- D.info() ---")
D.info()          # column names, non-null counts, dtypes, memory -> detects missing values / wrong types
print("\n--- D.describe() ---")
print(D.describe())  # count, mean, std, min, quartiles, max of numeric columns -> scale and spread
print("\n--- D.shape ---", D.shape)          # (rows, columns) = (150, 5)
print("--- D.columns ---", list(D.columns))  # variable names
print("\n--- D.head() ---")
print(D.head())   # first 5 rows: quick look at the real values
 
# 3. Select two features and store them in a new DataFrame
two_features = D[["sepal_length", "petal_width"]]
 
# 4. First 20 observations
print("\n--- First 20 observations (sepal_length, petal_width) ---")
print(two_features.head(20))
 
# 5. Copy of the original DataFrame
D_copy = D.copy()   # independent copy: changes to D_copy do NOT affect D
D_copy.loc[0, "sepal_length"] = 999
print("\nOriginal D unchanged after editing the copy:", D.loc[0, "sepal_length"])
 
# 6-7. Train/test split (80/20, random_state=42, stratified on the class label)
feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = D[feature_cols]     # input features
y = D["species"]        # target (class label)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print("\nInput features:", feature_cols, "| Target: species")
 
# 8. Verify sizes and proportions
print("\n--- Split verification ---")
print(f"Total: {len(D)} | Train: {len(X_train)} ({len(X_train)/len(D):.0%}) "
      f"| Test: {len(X_test)} ({len(X_test)/len(D):.0%})")
print("Class proportions (full):\n", y.value_counts(normalize=True).round(3))
print("Class proportions (train):\n", y_train.value_counts(normalize=True).round(3))
print("Class proportions (test):\n", y_test.value_counts(normalize=True).round(3))
 
print("""
DISCUSSION (Ex.1): The test set simulates unseen data. If it influences
preprocessing (e.g. scaler fitted on all data, imputation values, feature
selection) or model selection (tuning hyper-parameters on it), information
leaks from test to training: the score becomes optimistic and no longer
estimates real-world performance (data leakage / overfitting to the test set).
Keep it untouched until the final evaluation; tune with a validation set or
cross-validation on the training data.""")
 