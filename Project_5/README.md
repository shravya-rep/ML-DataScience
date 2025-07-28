# 🌲 Interpretable Models and Regularized Regression

> Topics: Decision Trees, LASSO, Ridge, PCR, Boosting

This project explores interpretable and regularized models for both classification and regression tasks using two UCI datasets: **Acute Inflammations** and **Communities and Crime**. The focus is on extracting interpretable rules, selecting features, handling missing data, and evaluating model generalization.

---

## 📁 Project Tasks

### 🩺 1. Decision Trees for Multi-Label Classification

- **Dataset**: [Acute Inflammations (UCI)](https://archive.ics.uci.edu/ml/datasets/Acute+Inflammations)
- **Objective**: Predict diagnosis of urinary tract diseases based on symptoms
- **Steps**:
  - Built a decision tree classifier using scikit-learn (label powerset for multi-label handling)
  - Visualized the decision tree
  - Extracted **IF-THEN rules** using rule simplification techniques
  - Applied **cost-complexity pruning** to simplify the model and improve interpretability

---

### 🏙️ 2. LASSO, Ridge, PCR, and Boosting for Regression

- **Dataset**: [Communities and Crime (UCI)](https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime)
- **Objective**: Predict the per capita violent crime rate
- **Challenges**:
  - Imputed missing values
  - Removed non-predictive features
  - Selected √128 ≈ 11 features using **Coefficient of Variation (CV)**

#### 🔍 Data Analysis
- Computed correlation matrix
- Created scatter and box plots for top CV features
- Evaluated feature importance visually

#### 📉 Regression Models

| Model                  | Details                                 | Test Error |
|-----------------------|------------------------------------------|------------|
| Linear Regression     | Least squares with selected features     | TBD        |
| Ridge Regression      | α tuned with cross-validation            | TBD        |
| LASSO (Standardized)  | Feature selection via L1 penalty         | TBD        |
| PCR                   | Dimension reduction + linear regression | TBD        |
| XGBoost Regression    | L1-penalized gradient boosting tree      | TBD        |

- Reported variables selected by **LASSO**
- Compared standardized vs. unstandardized performance

---

## 🛠️ Technologies Used

- Python (NumPy, pandas, matplotlib, seaborn)
- scikit-learn (DecisionTreeClassifier, RidgeCV, LassoCV, PCA)
- XGBoost (for penalized boosting)
- Visualization: `plot_tree`, `sns.heatmap`, boxplots

---

## 📌 Highlights

- Used **label powerset transformation** for multi-label classification
- Derived **interpretable decision rules** from complex trees
- Performed **feature selection** via Coefficient of Variation
- Compared **regularization techniques** for regression generalization
- Implemented **L1-penalized gradient boosting** using XGBoost



## 🚀 Future Extensions

- Integrate SHAP for tree interpretability
- Apply ElasticNet regression for hybrid L1/L2 control
- Use LightGBM/CatBoost as alternative boosting algorithms
