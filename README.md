# Machine Learning & Data Science Projects

9 end-to-end projects spanning classical ML, ensemble methods, deep learning, and transfer learning — from KNN to fine-tuned CNNs.

---

## Highlights

### Final Project — Waste Image Classification (Transfer Learning)
> `Waste_Classification/`

Classifies 9 types of waste from images using ResNet50, ResNet101, VGG16, and EfficientNetB0. Includes a **live Gradio demo** — run `python demo.py` to try it interactively.

- EfficientNetB0 achieved 87% training / 66% validation accuracy
- Data augmentation pipeline with early stopping and model checkpointing
- Full train/val/test evaluation with Precision, Recall, F1, AUC

### Project 6 — APS Failure Prediction
> `Truck_APS_Failure_Prediction/`

Predicts rare Air Pressure System failures in Scania trucks. Severe class imbalance (1:59 ratio) handled with SMOTE and cost-sensitive learning. 60,000 training samples, 170 features.

- Models: Random Forest, XGBoost
- Cost-aware evaluation: false negatives cost 50× more than false positives
- Feature selection via coefficient of variation

### Project 8 — Active & Semi-Supervised Learning
> `Cancer_and_Fraud_Detection/`

Compares four learning paradigms — supervised, semi-supervised (self-training), unsupervised (K-Means, Spectral Clustering), and active learning (uncertainty sampling) — on medical and financial datasets.

- Active learning reaches supervised performance with ~50% fewer labels
- 30 Monte Carlo runs for robust comparison

---

## All Projects

| # | Project | Techniques |
|---|---------|------------|
| 1 | Vertebral Column Classification | KNN, distance metrics, learning curves |
| 2 | Energy Output Prediction | Linear/Polynomial/KNN regression, feature engineering |
| 3 | Human Activity Recognition | Time series feature extraction, Logistic Regression, Naive Bayes |
| 4 | Logistic Regression on Time Series | L1/L2 regularization, RFE, multiclass ROC/AUC |
| 5 | Interpretable Models & Regularization | Decision Trees, Ridge, LASSO, PCR, XGBoost |
| 6 | APS Failure Prediction | Random Forest, XGBoost, SMOTE, cost-sensitive evaluation |
| 7 | SVM & Clustering on Anuran Calls | Gaussian/Linear SVM, K-Means, Hamming metrics, multi-label |
| 8 | Active & Semi-Supervised Learning | Self-training, Spectral Clustering, uncertainty sampling |
| Final | Waste Classification | ResNet50/101, VGG16, EfficientNetB0, Gradio demo |

---

## Stack

`Python` · `scikit-learn` · `XGBoost` · `imbalanced-learn` · `TensorFlow/Keras` · `OpenCV` · `Gradio` · `NumPy` · `Pandas` · `Matplotlib` · `Seaborn`

---

**Shravya Shashidhar**
