# 🧠 Supervised, Semi-Supervised, Unsupervised & Active Learning

> Topics: L1-SVM, Self-Training, Clustering, Active vs Passive SVM Learning  
> Datasets: Breast Cancer Wisconsin (Diagnostic), Banknote Authentication (UCI)

This project applies a range of learning paradigms—**supervised**, **semi-supervised**, **unsupervised**, and **active learning**—to real-world binary classification datasets. It demonstrates how label scarcity, cluster structure, and query strategies influence classification performance.

---

## 🧪 1. Learning Methods on Breast Cancer Dataset

### 🔹 Dataset
- **Source**: [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- **Features**: 30 normalized numeric features
- **Labels**: Benign (B), Malignant (M)

### 🔸 Monte Carlo Simulation
All methods below were evaluated over **30 randomized runs** with 20% of each class as test set.

---

### ✅ Supervised Learning
- Trained **L1-penalized SVM** with 5-fold cross-validation
- Evaluated metrics:
  - Accuracy, Precision, Recall, F1-score, AUC
  - ROC Curve & Confusion Matrix (sample run)

---

### 🧩 Semi-Supervised Self-Training
- 50% of each class in training set used as labeled data
- Trained L1-SVM, iteratively labeled most confident unlabeled point (farthest from decision boundary)
- Re-trained after each pseudo-label
- Reported same metrics as supervised setup

---

### 🌀 Unsupervised Learning: K-Means (k = 2)
- Ran K-Means clustering on training set without labels
- Repeated with random initializations to avoid local minima
- **Cluster Labeling Strategy**:
  - Chose 30 nearest points to each cluster center
  - Applied majority voting based on true labels
- Evaluated predictions on training and test sets
- Metrics: Accuracy, F1, AUC, ROC, Confusion Matrix

---

### 🔍 Unsupervised Learning: Spectral Clustering
- Used **RBF kernel** (γ = 1 or γ tuned for cluster balance)
- Assigned cluster labels via **majority vote** within each cluster
- Compared with true labels (train/test)
- Reported all performance metrics

---

### 📊 Comparative Evaluation
- Compared all four learning methods (Supervised, Semi-Supervised, K-Means, Spectral)
- Highlighted expected degradation from supervised → semi → unsupervised

---

## 🚀 2. Active Learning on Banknote Dataset

### 🔹 Dataset
- [Banknote Authentication](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)
- 900 training, 472 test samples
- Binary classification

### 💤 Passive Learning
- Randomly selected 10, 20, ..., 900 points incrementally
- Trained L1-SVM at each step
- Repeated for **50 Monte Carlo runs**
- Collected average test error across runs

### 🧠 Active Learning
- Initial 10 random points for L1-SVM
- At each step, added 10 closest points to the hyperplane (max uncertainty)
- Repeated for 90 steps
- Averaged errors over 50 runs

### 📈 Learning Curve
- Plotted **average test error vs. # of training instances** for:
  - Active learning
  - Passive learning
- Analyzed comparative label efficiency

---

## 🛠️ Tools & Libraries
- `scikit-learn`: SVM, KMeans, Spectral Clustering, metrics
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `imbalanced-learn`: For potential SMOTE usage
- Monte Carlo logic implemented with `random.seed` for reproducibility

---

## 🔮 Future Work

- Explore more sophisticated semi-supervised techniques (e.g., label propagation)
- Use margin sampling or entropy-based acquisition in active learning
- Apply t-SNE/UMAP for cluster visualization
- Integrate class imbalance solutions with clustering
