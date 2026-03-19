# Transfer Learning for Waste Image Classification

This project applies transfer learning using pre-trained deep learning models to classify images into nine types of waste. The focus is on adapting large-scale image models to a smaller domain-specific dataset via feature extraction and lightweight retraining.

---

## Live Demo

A Gradio web app is included for interactive inference.

```bash
cd final_project
pip install -r requirements.txt
python demo.py
```

Upload any waste image and the model returns the top predicted categories with confidence scores.

---

## Dataset & Preprocessing

- **Dataset**: RealWaste — 9 waste categories, ~4,750 images
- **Classes**: Cardboard, Food Organics, Glass, Metal, Miscellaneous Trash, Paper, Plastic, Textile Trash, Vegetation
- 80% of images per class used for training, 20% for test
- Applied one-hot encoding for multi-class labels
- Image augmentation: rotation, zoom, horizontal flip, brightness shift, translation

---

## Transfer Learning Strategy

- **Models compared**: ResNet50, ResNet101, EfficientNetB0, VGG16
- Pre-trained ImageNet weights; all base layers frozen (feature extraction mode)
- Custom classifier head added on top:
  - GlobalAveragePooling → BatchNorm → Dense(256, ReLU) + L2 regularization → Dropout(20%) → Softmax(9)
- Optimizer: Adam (lr=1e-4), loss: categorical cross-entropy
- Batch size: 5 | Up to 100 epochs with early stopping on val loss

---

## Results (Training History)

| Model | Train Accuracy | Val Accuracy |
|-------|---------------|--------------|
| EfficientNetB0 | 87% | 66% |
| ResNet101 | 84% | 60% |
| ResNet50 | 82% | 62% |
| VGG16 | 80% | 60% |

EfficientNetB0 achieved the best validation accuracy. All models show some overfitting — further improvement possible via fine-tuning deeper layers or larger datasets.

---

## Tools & Libraries

- `TensorFlow`, `Keras`
- `OpenCV`, `NumPy`, `Matplotlib`, `Scikit-learn`
- `Gradio`, `HuggingFace Transformers` (demo)

---

## Future Work

- Fine-tune deeper layers of each model to boost generalization
- Apply Grad-CAM to visualize which image regions drive predictions
- Extend to object detection or segmentation tasks
