AMEDJKOUH Darine 

## Discussion

**Exercise 1 – Why keep the test set separate?**
The test set must stay unseen to measure the model fairly. If it is used in preprocessing or model selection, information leaks (data leakage) and the score becomes too optimistic.

**Exercise 4A – Why does scaling matter? Is it needed for decision trees?**
KNN, SVM and gradient-based models depend on distances or gradients, so features with large values dominate. Decision trees do not need scaling, because they split one feature at a time.

## Explanations

**Exercise 1**
- Q2: `info()` gives types and missing values, `describe()` gives statistics, `shape` gives the size (150, 5), `columns` gives the names, `head()` shows the first rows.
- Q5: A copy lets us modify data without changing the original.
- Q7: Features: sepal_length, sepal_width, petal_length, petal_width. Target: species.
- Q8: Train = 120 rows (80%), test = 30 rows (20%).

**Exercise 2**
- Q4: 11314 documents and 20 categories.
- Q6: Each target number is the index of its name in `target_names` (7 = rec.autos).

**Exercise 3**
- Q5: pandas only counts NaN as missing; '-' and 'na' are normal text.
- Q6: Values are lowercased and stripped; '-' and 'na' become NaN.
- Q7: Median for numbers, most frequent for categories. First Name and dates are not imputed.
- Q8: Missing values drop to 0 in imputed columns. Limitation: imputed values are guesses and can bias the data.

**Exercise 4A**
- Q3: StandardScaler (mean 0, std 1), MinMaxScaler ([0, 1]), MaxAbsScaler ([-1, 1]), RobustScaler (median and IQR) scale each feature. `normalize` scales each sample (row).
- Q4: Threshold = 0: values above 0 become 1, others 0. A higher threshold gives more zeros.

**Exercise 4B**
- Q1: The numbers are category codes, not rankings.
- Q2: Algeria = 0, France = 1, Japan = 2, US = 3.
- Q3: Label encoding gives one number per class; one-hot gives one 0/1 column per class.
- Q4: One-hot avoids creating a false order between categories.

## Conclusion
- Numerical: impute with median, then scale.
- Categorical: impute with most frequent, then one-hot encode.
- Text: convert to numbers (word counts, TF-IDF).
- Always fit preprocessing on training data only.
