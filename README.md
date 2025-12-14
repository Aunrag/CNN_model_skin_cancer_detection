# Melanoma Skin Cancer Detection using CNN

This project uses a **Convolutional Neural Network (CNN)** built with **TensorFlow and Keras** to classify skin lesion images for **melanoma cancer detection**. The model is trained on image data and evaluated using accuracy and a confusion matrix.

---

## Project Overview

Melanoma is one of the most dangerous types of skin cancer. Early detection plays a crucial role in increasing survival rates.  
This project aims to automatically classify skin lesion images into two categories:

- Melanoma (Positive)
- Benign (Negative)

The project includes image preprocessing, data augmentation, CNN model training, evaluation, and visualization.

---

## Dataset Structure

The dataset must be organized in the following format:

melanoma_cancer_dataset/
      train/
          - class_0/
          - class_1/
      test/
          - class_0/
          - class_1/

Each class folder should contain corresponding image files.

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## Workflow

1. Load image datasets from directory
2. Split training data into training and validation sets (80/20)
3. Visualize sample images
4. Apply data augmentation
5. Normalize images
6. Build CNN model
7. Train model with early stopping
8. Evaluate model on test data
9. Generate confusion matrix
10. Save trained model

---

## Model Architecture

- Data Augmentation (Flip, Rotation)
- Convolutional Layers with ReLU activation
- Max Pooling Layers
- Flatten Layer
- Fully Connected Dense Layer (64 units)
- Output Layer (Softmax – 2 classes)

---

## Model Training Details

- Image Size: 256 × 256
- Batch Size: 32
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Metric: Accuracy
- Early Stopping: Enabled

---

## Evaluation

- Training and validation accuracy plots
- Confusion matrix on test dataset
- Heatmap visualization using Seaborn

---

## Model Saving

The trained model is saved as:


You can load it later using:

```python
from tensorflow import keras
model = keras.models.load_model('model.keras')
```
---
## How to Run the Project

Install dependencies:
```python 
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn

```
Update dataset paths in the code:
```python 
train_path = 'path_to_train_dataset'
test_path = 'path_to_test_dataset'
```
