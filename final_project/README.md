# 🧠 Transfer Learning for Waste Image Classification


This project applies **transfer learning** using pre-trained deep learning models to classify images into **nine types of waste**. The focus is on adapting large-scale image models to smaller datasets via feature extraction and lightweight retraining.

---

## 📁 Dataset & Preprocessing

- Images categorized into 9 waste classes
- Used the **first 80%** of images in each class for training, remaining 20% for test
- Resized or zero-padded images for consistent input shape
- Applied **one-hot encoding** for multi-class labels

---

## 🔄 Transfer Learning Strategy

- Models Used:
  - `ResNet50`
  - `ResNet100`
  - `EfficientNetB0`
  - `VGG16`
- Layers below the final fully connected layer were **frozen**
- Replaced top layers with:
  - `ReLU` activation
  - `BatchNormalization`
  - `Dropout (20%)`
  - `Softmax` for output
- Used **ADAM** optimizer and **multinomial cross-entropy loss**

---

## 🧪 Regularization & Augmentation

- Applied image augmentation via:
  - Random crop, zoom, rotation, horizontal flip, translation, contrast adjustment
- Tools used: `OpenCV`, `Keras ImageDataGenerator`
- Included **L2 regularization** to control overfitting

---

## 🧼 Training Procedure

- Batch size: 5 (experimentally optimal)
- Epochs: 50–100 (with **early stopping** on validation error)
- Validation Set: Random 20% subset of each class
- Metrics:
  - **Precision**
  - **Recall**
  - **F1 Score**
  - **AUC**
- Plotted **training & validation loss vs. epochs**  
- Saved model weights at lowest validation error

---

## 📊 Evaluation

- Performed evaluation on **training, validation, and test** sets
- Compared model performance across architectures
- Identified the best-performing model by average **F1 score & AUC**
- Discussed overfitting risks and model robustness

---

## 🛠️ Tools & Libraries

- `TensorFlow`, `Keras`
- `OpenCV`, `NumPy`, `Matplotlib`, `Scikit-learn`

## 🔮 Future Work

- Try **fine-tuning more layers** for each model to boost generalization
- Use **Grad-CAM** to visualize important image regions per class
- Extend to **object detection** or **segmentation** tasks
