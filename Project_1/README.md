# 🧠 Vertebral Column Disorder Detection using K-Nearest Neighbors


This project applies K-Nearest Neighbors (KNN) classification on a biomedical dataset to distinguish between **normal** and **abnormal** vertebral column conditions using various distance metrics and voting strategies.

---

## 📊 Dataset Description

- **Source**: [UCI Vertebral Column Data Set](https://archive.ics.uci.edu/ml/datasets/Vertebral+Column)
- **Features**: 6 biomechanical attributes derived from pelvis and lumbar spine shape
- **Classes**:
  - `0` → Normal
  - `1` → Abnormal (includes Disk Hernia, Spondylolisthesis)

---

## 🔬 Tasks Performed

### 📌 Exploratory Data Analysis
- Scatterplots and boxplots of all independent variables colored by class label

### 📌 Dataset Preparation
- Selected:
  - First 70 samples from Class 0
  - First 140 samples from Class 1 → **Training set**
  - Remaining data → **Test set**

---

## 🔁 Classification Tasks

### 1. KNN with Euclidean Distance
- Evaluated train/test error vs. `k ∈ {208, 205, ..., 1}`
- Selected best `k*` based on F1-score
- Reported:
  - Confusion Matrix
  - Precision, Recall, F1-score, True Positive/Negative Rates

### 2. Learning Curve
- Varied training set size `N ∈ {10, 20, ..., 210}`
- For each `N`, found best `k` from `{1, 6, ..., N-4}` and plotted **test error vs. N**

---

## 📐 Distance Metrics Compared

| Metric Type     | Parameter Setting |
|-----------------|-------------------|
| Euclidean       | –                 |
| Manhattan       | Minkowski (p = 1) |
| General p       | `log10(p) ∈ {0.1, ..., 1}` |
| Chebyshev       | `p → ∞`           |
| Mahalanobis     | Covariance-based  |

✅ Best test errors from each distance function were compared in tabular form.

---

## 📊 Weighted KNN

- Implemented weighted voting where weights = `1/distance`
- Applied to:
  - Euclidean
  - Manhattan
  - Chebyshev
- Reported best test errors for each with `k ∈ {1, 6, ..., 196}`

---

## 🏁 Performance Summary

- **Best training error achieved**: _<insert your value here>_
- **Optimal k**: _<insert k*>_ from unweighted voting

---

## 🛠️ Technologies Used

- Python (NumPy, pandas, matplotlib)
- scikit-learn (`neighbors.KNeighborsClassifier`, `DistanceMetric`)
- Jupyter Notebook for visualization

---

## 📌 Future Work

- Apply PCA before distance-based learning
- Use more advanced classifiers (e.g., SVM, ensemble trees)
- Hyperparameter tuning using cross-validation
