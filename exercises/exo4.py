import numpy as np
import pandas as pd

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    normalize,
    Binarizer,
    OrdinalEncoder,
    LabelEncoder,
    OneHotEncoder,
)
print("EXERCISE 4 - Part A: scaling and binarization")
 
A = np.array([[1, -1, 2],
              
              [2, 0, 0],
              [0, 1, -1]], dtype=float)
cols = ["Feature 1", "Feature 2", "Feature 3"]
show = lambda M: pd.DataFrame(M, columns=cols).round(4)
 
print("Original matrix:\n", show(A))
print("\nStandardScaler (per feature: (x-mean)/std):\n", show(StandardScaler().fit_transform(A)))
print("\nMinMaxScaler (per feature: to [0,1]):\n", show(MinMaxScaler().fit_transform(A)))
print("\nMaxAbsScaler (per feature: x / max|x|, range [-1,1]):\n", show(MaxAbsScaler().fit_transform(A)))
print("\nRobustScaler (per feature: (x-median)/IQR):\n", show(RobustScaler().fit_transform(A)))
A_norm = normalize(A, norm="l2")      # function API, works ROW by ROW (no fit)
print("\nnormalize (per SAMPLE: each row divided by its L2 norm):\n", show(A_norm))
print("Row norms after normalize:", np.linalg.norm(A_norm, axis=1).round(4))
 
print("\nBinarizer, threshold = 0  (x > 0 -> 1, else 0):\n",
      show(Binarizer(threshold=0.0).fit_transform(A)))
print("\nBinarizer, threshold = 1  (x > 1 -> 1, else 0):\n",
      show(Binarizer(threshold=1.0).fit_transform(A)))
 
print("""
METHODS:
 - StandardScaler: mean 0, std 1 for each FEATURE.            (feature scaling)
 - MinMaxScaler  : maps each feature to [0,1] with min/max.   (feature scaling)
 - MaxAbsScaler  : divides each feature by its max |value|; keeps zeros/sparsity.
 - RobustScaler  : uses median and IQR -> resistant to outliers.
 - normalize     : rescales each SAMPLE (row) to unit norm.   (sample normalization)
 - Binarizer     : value > threshold -> 1, otherwise 0; a higher threshold gives
                   fewer 1s, a lower one gives more.
 
DISCUSSION: k-NN and SVM rely on distances/kernels, so a feature with a large
range dominates the distance. Gradient-based models converge faster and more
stably when features have similar scales (and regularization treats weights
fairly). Decision trees split on one feature at a time with thresholds, which
are invariant to monotonic rescaling, so scaling is NOT necessary for them.""")
 
print("EXERCISE 4 - Part B: categorical encoding")
 
B = pd.DataFrame({
    "Feature 1": ["male", "female", "female", "male"],
    "Feature 2": ["from US", "from Europe", "from Asia", "from Africa"],
    "Feature 3": ["uses Safari", "uses Safari", "uses Firefox", "uses Chrome"],
    "Target": ["US", "France", "Japan", "Algeria"],
})
print(B)
 
# 1. OrdinalEncoder on the 3 input features
oe = OrdinalEncoder()
X_ord = oe.fit_transform(B[["Feature 1", "Feature 2", "Feature 3"]])
print("\nOrdinalEncoder output:\n", pd.DataFrame(X_ord, columns=B.columns[:3]))
for name, cats in zip(B.columns[:3], oe.categories_):
    print(f"categories_ {name}: {list(cats)}  -> codes 0..{len(cats)-1}")
print("(codes follow alphabetical order, so they are labels, NOT a meaningful ranking)")
 
# 2. Encode the target with LabelEncoder
le = LabelEncoder()
y_enc = le.fit_transform(B["Target"])
print("\nLabelEncoder target:", y_enc)
print("Mapping:", {c: int(i) for i, c in enumerate(le.classes_)})
 
# 3. OneHotEncoder on the target
ohe = OneHotEncoder(sparse_output=False)
y_ohe = ohe.fit_transform(B[["Target"]])
print("\nOneHotEncoder on target (columns =", list(ohe.categories_[0]), "):")
print(pd.DataFrame(y_ohe, columns=ohe.categories_[0], dtype=int))
 
print("""
LABEL vs ONE-HOT: label encoding gives one integer column (Algeria=0, France=1,
Japan=2, US=3), which implies an order and distances (Japan is 'between' France
and US). One-hot gives one binary column per class, with no order and every
class equally distant from the others.
 
WHY ONE-HOT FOR NOMINAL INPUTS: categories like Safari/Firefox/Chrome or
countries have no natural order. Ordinal codes would let distance-based and
linear models treat Chrome(0) < Firefox(1) < Safari(2) as meaningful, creating
fake relations. One-hot avoids this (cost: more columns). Trees are less
sensitive to this issue.""")
 
print("CONCLUSION")
print("""Numerical data : handle missing values (median/mean), scale (Standard/MinMax/
                 Robust) for distance- or gradient-based models, binarize if needed.
Categorical    : impute with the mode, then encode: one-hot for nominal inputs,
                 ordinal only when an order exists, LabelEncoder for targets.
Text data      : load with the right encoding, inspect lengths/words, then
                 vectorize (e.g. bag-of-words, TF-IDF) before modelling.
Always         : split first, fit every preprocessing step on train only, and
                 work on copies to keep the original data unchanged.""")
 

