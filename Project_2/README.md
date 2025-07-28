# ⚡ Energy Output Prediction from Power Plant Sensor Data

This project applies **regression analysis and KNN regression** on the Combined Cycle Power Plant dataset to predict hourly energy output (EP) from ambient conditions. The goal is to compare model accuracy, interpret coefficients, assess nonlinearity, and tune performance.

---

## 📊 Dataset Description

- **Source**: [UCI Combined Cycle Power Plant Data](https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant)
- **Features (predictors)**:
  - Temperature (T)
  - Ambient Pressure (AP)
  - Relative Humidity (RH)
  - Exhaust Vacuum (V)
- **Target (response)**: Net hourly electrical energy output (EP)
- **Data Size**: 9568 rows × 5 columns (Sheet 1 only)

---

## 🔍 Tasks Performed

### 1. Exploratory Data Analysis
- Summary statistics: mean, median, range, quartiles, IQR
- Pairwise scatterplots for all variable combinations
- Initial observations of correlation strength and distribution

### 2. Simple Linear Regression (SLR)
- One predictor at a time to estimate EP
- Statistical significance evaluated using p-values
- Residual plots checked for outliers or influential data points

### 3. Multiple Linear Regression (MLR)
- Combined all four predictors to predict EP
- Tested hypotheses: H₀: βⱼ = 0
- Compared coefficients from SLR vs. MLR using a 2D scatterplot

---

## 🔁 Model Enhancements

### 4. Polynomial Regression
- Fit cubic model: `Y = β₀ + β₁X + β₂X² + β₃X³ + ε`
- Explored nonlinear relationships for each predictor
- Detected curvature in temperature and vacuum effects

### 5. Interaction Effects
- Constructed full pairwise interaction model
- Identified statistically significant interaction terms
- Dropped variables based on p-value pruning

### 6. Final Models
- Trained two models on 70% split:
  - Basic linear regression
  - Full interaction + quadratic model (reduced via p-values)
- Compared **Train/Test MSE** for both

---

## 🧠 KNN Regression

- Applied with both:
  - **Raw features**
  - **Normalized features**
- Explored `k ∈ [1, 100]`
- Plotted error curves for train and test sets vs. 1/k
- Identified optimal `k` for lowest test error

---

## 📈 Results Summary

| Model Type                    | Best Test MSE |
|------------------------------|---------------|
| Basic Linear Regression       | TBD           |
| Full Poly+Interaction Model   | TBD           |
| KNN Regression (Normalized)   | TBD           |

> ✏️ *KNN performed competitively but required careful normalization and tuning. Linear models offered better interpretability.*

---

## 🛠️ Technologies Used

- Python 3
- pandas, NumPy, matplotlib, seaborn
- scikit-learn (`LinearRegression`, `PolynomialFeatures`, `KNeighborsRegressor`)
- statsmodels for p-value analysis

---

## 📌 Future Work

- Incorporate regularization (Ridge/Lasso)
- Apply PCA to reduce feature interaction explosion
- Try ensemble regression methods (Random Forests, Gradient Boosting)
