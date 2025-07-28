# ⏱️ Advanced Time Series Classification with Logistic Regression
 
> Dataset: AReM - Activity Recognition System Based on Multisensor Data Fusion

This project builds on time-domain feature extraction to perform both **binary** and **multiclass** classification of human activities based on sensor time series. The classification pipeline uses **logistic regression**, **L1-penalized models**, and **Naïve Bayes**, with evaluation through ROC, AUC, and confusion matrices.

---

## 📦 Dataset Overview

- **Source**: [UCI AReM Dataset](https://archive.ics.uci.edu/ml/datasets/Activity+Recognition+system+based+on+Multisensor+data+fusion)
- **Activities**: 7 total (bending, walking, standing, etc.)
- **Data Format**: 88 instances × 6 time series (each 480 samples)

---

## 🧠 Feature Engineering

- Time-domain features: min, max, mean, median, std, Q1, Q3
- Generated 42 features per instance
- Estimated std dev and constructed 90% bootstrap confidence intervals
- Selected top 3 informative features for visualization

---

## 🔁 Binary Classification: Bending vs Others

### 🔹 Logistic Regression (Unpenalized)
- Visualized 3 selected features across TS1, TS2, and TS6
- Repeated classification after splitting time series into:
  - 2 parts (→ 12 time series)
  - Up to 20 parts (higher-dimensional features)
- Feature pruning via:
  - p-values
  - Recursive Feature Elimination (RFE)
- Used 5-fold **stratified cross-validation** to handle class imbalance
- Evaluation:
  - Confusion Matrix
  - ROC Curve and AUC
  - β coefficients and associated p-values

### 🔹 L1-Penalized Logistic Regression
- Simultaneously performed classification + feature selection
- Tuned hyperparameter λ (regularization strength) via cross-validation
- Compared accuracy and implementation simplicity with p-value-based selection

---

## 🧪 Multiclass Classification

### 🔸 L1-Penalized Multinomial Logistic Regression
- Selected best `l` (number of splits) via cross-validation
- Evaluated using:
  - Test accuracy
  - Confusion matrix
  - ROC curves (OvA)

### 🔸 Naïve Bayes
- Gaussian and Multinomial variants
- Compared against L1 multinomial regression on performance and interpretability

---

## 🛠️ Technologies Used

- Python (NumPy, pandas, scikit-learn, matplotlib, seaborn)
- Bootstrap CI (custom sampling)
- Feature selection: `RFE`, `LogisticRegressionCV` with L1 penalty
- ROC/AUC: `roc_auc_score`, `StratifiedKFold`, `confusion_matrix`

---

## 📌 Key Takeaways

- Feature explosion from time-splitting increases accuracy but risks overfitting
- Stratified cross-validation is crucial with imbalanced classes (e.g., bending)
- L1-penalized logistic regression offers built-in feature selection
- Naïve Bayes is efficient and interpretable for multiclass tasks



---

## 🚀 Future Work

- Extend to RNN/CNN models using raw time series
- Apply dimensionality reduction (e.g., PCA) before classification
- Use SMOTE or upsampling to balance rare classes more effectively
