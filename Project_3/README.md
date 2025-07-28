# 📈 Human Activity Classification Using Time Series Features
 
> Dataset: AReM (Activity Recognition system based on Multisensor data fusion)

This project focuses on the **feature extraction and classification** of time series data for human activity recognition using wireless sensor network signals.

---

## 📊 Dataset Overview

- **Source**: [UCI AReM Dataset](https://archive.ics.uci.edu/ml/datasets/Activity+Recognition+system+based+on+Multisensor+data+fusion)
- **Activities**: 7 types (e.g., bending, walking, cycling)
- **Instances**: 88 total (each with 6 time series of 480 values)
- **Sensors**: avg_rss12, var_rss12, avg_rss13, var_rss13, vg_rss23, ar_rss23

---

## 🧠 Feature Extraction

Extracted **time-domain features** from each time series:
- Minimum
- Maximum
- Mean
- Median
- Standard Deviation
- 1st Quartile
- 3rd Quartile

Each instance was converted to a 42-dimensional feature vector (7 features × 6 time series).

Also:
- Estimated standard deviation for each feature
- Built 90% bootstrap confidence intervals

---

## 🔁 Binary Classification (Bending vs Others)

- **Logistic Regression** using:
  - Raw features
  - Split time series (2 parts, 20 parts, etc.)
- **Feature Selection**:
  - p-value based pruning
  - Recursive Feature Elimination (RFE)
  - L1-penalized logistic regression

### Evaluation:
- Confusion Matrix
- ROC Curves and AUC Scores
- Cross-validation with stratified folds
- Case-control sampling to address class imbalance

---

## 🧪 Multiclass Classification

- Built **L1-penalized multinomial logistic regression**
- Compared with:
  - **Naive Bayes Classifier** (Gaussian and Multinomial priors)
- Measured:
  - Test Accuracy
  - Multiclass ROC (One-vs-All)
  - Confusion Matrices

---

## 📌 Key Takeaways

- Feature extraction was essential to reduce noise and dimensionality.
- L1-penalized logistic regression outperformed manual p-value-based pruning in robustness.
- Class imbalance and linear separability were major concerns for bending detection.
- Naive Bayes showed competitive performance for multiclass classification with less training overhead.

---

## 🛠️ Technologies Used

- Python (NumPy, pandas, scikit-learn, matplotlib, seaborn)
- Feature selection: RFE, L1-penalty
- Bootstrap CI: `scipy.stats.bootstrap`, custom sampling
- Cross-validation: `StratifiedKFold`


## 🚀 Future Directions

- Explore RNNs or 1D CNNs for direct time series classification
- Apply temporal feature engineering (e.g., trend, autocorrelation)
- Incorporate external sensor fusion data if available
