# Machine Learning – Practical Session 1

Python for data exploration and preprocessing.

## Files

- `TP1_Solutions.ipynb`: the notebook (Exercises 1 to 4)
- Data files (in the notebook folder or in a `data` folder): `Iris_Dataset.csv`, `txt_file.txt`, `employees.csv`

## Requirements

```bash
python -m pip install numpy pandas matplotlib scikit-learn seaborn ipykernel
```

## Content of the notebook

- **Exercise 1:** load and inspect the Iris dataset, select features, copy the DataFrame, train/test split (80/20, `random_state=42`, `stratify=y`).
- **Exercise 2:** read a text file, load and explore the 20 Newsgroups dataset (internet needed the first time).
- **Exercise 3:** detect missing values (`isna`, seaborn heatmap), convert placeholders to `np.nan`, impute with `SimpleImputer`.
- **Exercise 4:** `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, `normalize`, `Binarizer`, `OrdinalEncoder`, `LabelEncoder`, `OneHotEncoder`.

The notebook also contains the answers to the discussion questions and a short conclusion.
