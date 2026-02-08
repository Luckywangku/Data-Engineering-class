# Pattern Recognition – Coursework 1 (PR_CW1)  
**MNIST Digit Classification with CNN (Keras / TensorFlow)**

This repository contains my coursework submission and learning notes for the *Pattern Recognition, Neural Networks, and Deep Learning* module.  
The goal of this project is to design, train, and evaluate a neural network using **Keras (TensorFlow 2.x)** to classify handwritten digits from the **MNIST dataset**.

---

## 📌 Project Objectives

- Build a neural network using **Keras (tf.keras)**  
- Train the model on the MNIST dataset  
- Achieve high classification accuracy on unseen test data  
- Ensure compatibility with the provided **test script**  
- Document the full workflow as reproducible learning notes in Jupyter Notebook  

---

## 🧠 Methods & Model Design

### Model Architecture
- Convolutional Neural Network (CNN)
- Components:
  - Convolutional layers (Conv2D)
  - Batch Normalization
  - Max Pooling
  - Dropout (regularization)
  - Fully Connected (Dense) layers
  - Softmax output layer (10 classes)

### Data Preprocessing
- Normalization: pixel values scaled to `[0, 1]`
- Reshaping: `(28, 28, 1)` for CNN input
- One-hot encoding of labels

### Data Augmentation
To improve generalization:
- Random rotation
- Width/height shifting
- Zooming

```python
ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
