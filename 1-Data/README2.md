# Lecture: Building Good Training Datasets — Data Preprocessing

I this lesson we are diving into what is arguably the most critical stage of any machine learning pipeline: **Data Preprocessing**. While building fancy models is exciting, the quality of your data and the amount of useful information it contains determine how well your algorithm can actually learn. We must examine and preprocess our datasets before feeding them to any algorithm.

![data](./images/data.jpg)

---

## 1. Dealing with Missing Data

Real-world data is often missing values due to collection errors, non-applicable measurements, or blank survey fields. These appear as blank spaces, `NaN` (Not a Number), or `NULL` indicators. 

### Identifying Missing Values
Using `pandas`, we can scan for these gaps efficiently.For large datasets, manual inspection is tedious, so we use the `isnull()` method to find missing cells and `sum()` to count them per column.



### Strategies for Handling Gaps
1.  **Elimination:** You can drop rows (`axis=0`) or columns (`axis=1`) containing missing data using the `dropna` method. 
    * **Pros:** Quick and easy.
    * **Cons:** You risk losing too much data, which can make reliable analysis impossible or hinder a classifier's ability to discriminate between classes.
2.  **Imputation:** If removal isn't feasible, we use **interpolation techniques**. 
    * **Mean Imputation:** Replacing a missing value with the mean of the entire feature column.
    * **Median/Most Frequent:** Alternatives for the `strategy` parameter. "Most frequent" is particularly useful for categorical data like color names.

---

## 2. The Scikit-Learn Estimator API

To perform transformations like imputation, scikit-learn provides a consistent **Transformer API**. 
* **`fit`:** Used to learn parameters (like the mean) from the training data.
* **`transform`:** Uses those learned parameters to actually change the data.



> **Note:** We always `fit` on the training data *only*, then use that fitted instance to `transform` both the training and the test sets to ensure consistency.

---

## 3. Handling Categorical Data

We must distinguish between two types of categorical features:
* **Ordinal Features:** Values that can be sorted or ordered (e.g., T-shirt size: $XL > L > M$).
* **Nominal Features:** Values that do not imply any order (e.g., T-shirt color: red is not "larger" than blue).

### Encoding Strategies
* **Ordinal Mapping:** We manually define a mapping (e.g., $\{'XL': 3, 'L': 2, 'M': 1\}$) so the algorithm interprets the order correctly.
* **One-Hot Encoding:** A common mistake is encoding nominal features as ordered integers (0, 1, 2), which leads models to assume a mathematical hierarchy that doesn't exist. Instead, we use **one-hot encoding** to create a "dummy" feature for each unique category.
    * **Multi-collinearity:** To avoid numerically unstable estimates, it is best practice to remove one redundant column from the encoded array (e.g., using `drop_first=True` in pandas).

---

## 4. Partitioning the Dataset

To evaluate how well our model generalizes, we split data into a **training set** and a **test set**.

* **Common Splits:** 60:40, 70:30, or 80:20 are standard, though 90:10 is fine for very large datasets.
* **Stratification:** Using `stratify=y` in `train_test_split` ensures both sets have the same class proportions as the original dataset.

---

## 5. Feature Scaling

Most algorithms (except scale-invariant ones like decision trees) perform better when features are on the same scale.

### Normalization vs. Standardization

| Technique | Description | Formula |
| :--- | :--- | :--- |
| **Normalization** | Rescales data to a range of $[0, 1]$ (min-max scaling). | $$x_{norm}^{(i)}=\frac{x^{(i)}-x_{min}}{x_{max}-x_{min}}$$  |
| **Standardization** | Centers features at mean 0 with standard deviation 1. | $$x_{std}^{(i)}=\frac{x^{(i)}-\mu_{x}}{\sigma_{x}}$$  |

**Why Standardization?** It is often more practical for optimization algorithms like gradient descent because it centers the data, making it easier to learn weights while maintaining information about outliers.

---

## 6. Overfitting and Regularization

If a model performs significantly better on training data than test data, it is **overfitting** (high variance).

### Solutions for Overfitting:
* Collect more training data.
* Choose a simpler model with fewer parameters.
* **Regularization:** Introducing a penalty for complexity (e.g., **L2 regularization**, which penalizes large weights).

$$L2:||w||_{2}^{2}=\sum_{j=1}^{m}w_{j}^{2}$$

---

**Next Step:** Would you like me to create a summary table comparing the different scikit-learn classes (like `SimpleImputer`, `StandardScaler`, and `OneHotEncoder`) we discussed today?