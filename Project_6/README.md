# 🌲 APS Failure Prediction Using Tree-Based Models & SMOTE

> Topics: Random Forests, XGBoost, Model Trees, SMOTE, Class Imbalance

This project tackles the challenge of predicting rare failures in automotive systems using a highly imbalanced dataset. We applied decision trees, ensemble methods, and class rebalancing techniques (like SMOTE) to build interpretable and high-performance models.

---

## 📦 Dataset

- **Source**: [APS Failure at Scania Trucks – UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks)
- **Training Set**: ~60,000 rows, 1,000 positive (failure) cases
- **Features**: 170 numeric columns + 1 binary label (`class`)

---

## 🧪 Part 1: Tree-Based Classification

### 🔧 Data Preparation
- Imputed missing values using statistical methods (e.g., median/mode)
- Computed **coefficient of variation (CV)** for each feature
- Selected top √170 ≈ 13 features with highest CV for visualization
- Generated scatter plots and box plots
- Verified **serious class imbalance** (1:59 ratio)

---

### 🌳 Random Forest (Uncompensated)
- Trained using default class distribution
- Evaluated on:
  - Confusion Matrix (Train/Test)
  - ROC & AUC
  - Misclassification Rate
  - Out-of-Bag (OOB) error

### ⚖️ Random Forest with Class Weight Compensation
- Applied class weighting to balance decision thresholds
- Compared ROC, AUC, OOB vs test set metrics
- Observed improved recall for minority class

---

## 🚀 Part 2: XGBoost with Model Trees

- Used **L1-penalized logistic regression** at each split node
- Determined regularization parameter `α` using 5-fold CV
- Trained on unbalanced data
- Evaluated:
  - ROC, AUC (Train/Test)
  - Confusion Matrices
  - Cross-validation vs actual test error

---

## 🧬 Part 3: XGBoost with SMOTE

- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to rebalance classes
- Ensured correct usage of SMOTE with cross-validation (resampling within train folds)
- Retrained XGBoost model
- Compared performance with uncompensated version:
  - Significant improvement in **recall**
  - ROC/AUC curves more balanced

---

## 📌 Summary of Models

| Model                         | Imbalance Handling | AUC    | Recall | OOB Error |
|------------------------------|--------------------|--------|--------|-----------|
| Random Forest (baseline)     | ❌                 | XX.XX  | XX.X%  | XX.X%     |
| Random Forest (weighted)     | ✅ (Class Weights) | XX.XX  | XX.X%  | XX.X%     |
| XGBoost (unbalanced)         | ❌                 | XX.XX  | XX.X%  | XX.X%     |
| XGBoost + SMOTE              | ✅ (SMOTE)         | XX.XX  | XX.X%  | XX.X%     |

---

## 🛠️ Tools & Libraries

- `scikit-learn` (RandomForestClassifier, ROC, confusion_matrix)
- `xgboost` (L1 regularization, logistic loss)
- `imblearn` (SMOTE)
- `pandas`, `NumPy`, `matplotlib`, `seaborn`

---


## 🔮 Future Enhancements

- Try **LightGBM** or **CatBoost** for better class imbalance handling
- Use **SHAP** values for model interpretability
- Benchmark SMOTE against **ADASYN**, **BorderlineSMOTE**, and **Random Oversampling**
