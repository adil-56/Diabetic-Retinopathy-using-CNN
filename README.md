# Diabetic Retinopathy Detection using Convolutional Neural Networks (CNN)

## 1. Overview

This project focuses on the automated detection of Diabetic Retinopathy (DR) from retinal fundus images using a Convolutional Neural Network (CNN). The goal is to classify images into five severity levels of DR (No DR, Mild, Moderate, Severe, Proliferative DR), providing a tool to assist ophthalmologists in early diagnosis and prevention of vision loss.

---

## 2. Problem Statement

Diabetic Retinopathy is a leading cause of blindness among working-aged adults. Early detection and treatment can prevent severe vision loss. However, manual diagnosis by screening retinal photographs is a time-consuming task that requires trained clinicians and is prone to human error. This project aims to develop a deep learning model to automate this process, making it faster, more accessible, and potentially more accurate.

---

## 3. Dataset

The model was trained on the **APTOS 2019 Blindness Detection** dataset, which is available on Kaggle.

* **Source:** [Kaggle APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection)
* **Content:** The dataset contains 3,662 high-resolution retinal images taken under various imaging conditions.
* **Labels:** Each image is rated for the severity of diabetic retinopathy on a scale of 0 to 4.

---

## 4. Methodology

The solution follows a standard deep learning pipeline for image classification.

#### a. Data Preprocessing & Augmentation
* **Image Resizing:** Images were resized to a uniform dimension (e.g., 224x224 pixels) to be fed into the CNN.
* **Normalization:** Pixel values were normalized to the range [0, 1] to aid in faster convergence.
* **Data Augmentation:** To prevent overfitting and increase the diversity of the training set, various augmentation techniques were applied, including random rotations, horizontal/vertical flips, and brightness adjustments.

#### b. Model Architecture
A Convolutional Neural Network was designed for this classification task. The architecture consists of:
* Multiple convolutional blocks (Conv2D + ReLU activation + Batch Normalization + MaxPooling2D) to extract hierarchical features from the images.
* A `Flatten` layer to convert the 2D feature maps into a 1D vector.
* `Dense` (fully connected) layers for high-level reasoning.
* A `Dropout` layer to reduce overfitting.
* A final `Dense` layer with a `Softmax` activation function to output the probability for each of the 5 classes.

---

## 5. Hypothetical Results

The model would be evaluated based on classification accuracy and Cohen's Kappa score, which is suitable for imbalanced multi-class problems.

* **Training Accuracy:** ~95%
* **Validation Accuracy:** ~85-90%
* **Key Metrics:** Precision, Recall, and F1-Score would also be calculated for each class to understand the model's performance in detail. A confusion matrix would be used to visualize misclassifications.

*(Note: These are representative results for this type of problem.)*

---

## 6. How to Run (Hypothetical)

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[Your-Username]/diabetic-retinopathy-cnn.git
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download the dataset** from the Kaggle link above and place it in a `data/` directory.
4.  **Run the training script:**
    ```bash
    python src/train.py
    ```

---

## 7. Technologies Used

* **Python 3.8+**
* **TensorFlow / Keras**
* **Pandas & NumPy**
* **Scikit-learn**
* **OpenCV**
* **Matplotlib**
