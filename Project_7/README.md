# 🐸 Multi-Label SVM Classification & Clustering on Anuran Calls Dataset

> Dataset: [Anuran Calls (MFCCs) – UCI](https://archive.ics.uci.edu/ml/datasets/Anuran+Calls+%28MFCCs%29)

This project explores supervised and unsupervised learning techniques for **multi-label** and **multi-class** classification using **Support Vector Machines (SVM)** and **K-Means Clustering**. The dataset includes MFCC features of frog calls labeled at three taxonomic levels: Family, Genus, and Species.

---

## 🎯 Objectives

### Part 1: Multi-Class & Multi-Label Classification Using SVMs

- **Binary Relevance (One-vs-Rest)** approach for each label
- **Gaussian Kernel SVMs** with 10-fold cross-validation:
  - Hyperparameters tuned: penalty (λ), kernel width (σ)
- **L1-Penalized Linear SVMs** with feature sparsity
- Applied **SMOTE** for class imbalance correction
- Evaluation Metrics:
  - Exact Match Score
  - Hamming Score & Hamming Loss
  - ROC, AUC, Confusion Matrix (Multi-label setting)
- Extra Practice: Implemented **Classifier Chain Method**

---

## 🔁 Part 2: K-Means Clustering with Multi-Label Ground Truth

- Performed clustering directly on raw MFCCs (no supervision)
- Automatically selected **optimal number of clusters (k)** using:
  - Gap Statistic, CH Index, Scree Plot, or Silhouette Score
- For each cluster:
  - Assigned the **majority label triplet** (Family, Genus, Species)
  - Compared against true labels using:
    - **Hamming Distance**
    - **Hamming Score**
    - **Hamming Loss**
- Used **Monte Carlo Simulation**: repeated the procedure 50 times
  - Reported mean and standard deviation of Hamming metrics

---

## 🛠️ Tools & Libraries

- `scikit-learn` (SVC, LinearSVC, OneVsRestClassifier, KMeans)
- `imbalanced-learn` (SMOTE)
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- Multi-label metrics: custom or `sklearn.metrics`

---


## 📌 Key Takeaways

- Binary relevance SVMs are effective for decomposing multi-label tasks
- SMOTE significantly improved classifier performance for rare classes
- K-means clustering can discover useful label structure, but performance varies
- Exact Match Score is very strict; Hamming metrics are more forgiving

---

## 📚 References

- Sorower, M. S. *A Literature Survey on Algorithms for Multi-label Learning*
- scikit-learn documentation for multi-label classification and metrics
