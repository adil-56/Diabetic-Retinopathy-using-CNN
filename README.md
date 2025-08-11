# 🩺 Diabetic Retinopathy Detection using CNN

This project uses a **Convolutional Neural Network (CNN)** to detect **Diabetic Retinopathy** from retinal fundus images.  
It was developed as part of my academic learning in **Deep Learning and Medical Image Processing** during my MCA (Data Science Specialization) at **Alliance University**.

Diabetic Retinopathy is one of the leading causes of blindness. Early detection and treatment can significantly reduce vision loss.  
This project aims to demonstrate how deep learning can assist in medical diagnosis.

---

##  Project Overview
-Model Type: Convolutional Neural Network (custom architecture)
- Libraries: TensorFlow, Keras, NumPy, Pandas, Matplotlib, OpenCV
- Dataset: [Diabetic Retinopathy Dataset (Kaggle)](https://www.kaggle.com/datasets/sachinkumar413/diabetic-retinopathy-dataset)
- Goal: Classify retinal images into different severity stages of diabetic retinopathy.

---

## 🗂 Folder Structure
```

Diabetic\_Retinopathy/
│
├── Diabetic\_retinopathy.ipynb   # Main Jupyter Notebook
├── requirements.txt             # Dependencies
├── README.md                    # Documentation
├── assets/                      # Images/screenshots (optional)
└── .gitignore                   # Ignore unnecessary files

````

---

## ⚙ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/diabetic-retinopathy-cnn.git
   cd diabetic-retinopathy-cnn
````

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download Dataset**

   * Dataset: [Kaggle Link](https://www.kaggle.com/datasets/sachinkumar413/diabetic-retinopathy-dataset)
   * Place it in a folder named `data/` inside the project directory.

---

## 🚀 How to Run

1. **Open the Jupyter Notebook**

   ```bash
   jupyter notebook Diabetic_retinopathy.ipynb
   ```
2. Run all cells to:

   * Preprocess and augment images
   * Train the CNN model
   * Evaluate the performance
   * Test predictions on sample images

---

## 📊 Results (Example)

> *You can update this section with your actual metrics once you rerun the model.*

* Training Accuracy:85%
* Validation Accuracy:82%
* Successfully identifies multiple severity levels of diabetic retinopathy.


## 💡 Learning Outcomes

* Learned the complete process of building a deep learning model from dataset loading to evaluation
* Understood image preprocessing techniques for medical datasets
* Gained hands-on experience in CNN architecture building and tuning

---

## 🔮 Future Improvements

* Experiment with transfer learning (EfficientNet, ResNet)
* Deploy as a web application for real-time use
* Add explainability features (Grad-CAM) to highlight affected regions

---

## 📝 Acknowledgments

* Dataset from [Kaggle](https://www.kaggle.com/)
* TensorFlow & Keras documentation
* Alliance University MCA Faculty for guidance

---

Author: Adil Khan
Course: MCA - Data Science Specialization
Institution: Alliance University

