# 🩺 COVID-19 Detection from Chest X-ray Images using ResNet18 (PyTorch)

## 💡 Project Overview
This project presents a solution for the automated classification of lung conditions from **chest X-ray images** using a deep learning approach. The primary goal was to develop a Convolutional Neural Network (CNN) capable of distinguishing between **four critical classes**: Normal lungs, **COVID-19**, **Viral Pneumonia**, and **Pulmonary Opacity**.

The methodology leverages **Transfer Learning** with a modified **ResNet18** architecture, which was successfully trained to demonstrate considerably good accuracy in classifying input images.

---

## 🧠 Model and Methodology

### Transfer Learning with Modified ResNet18
The core of the system is a modified **ResNet18** CNN. We utilized **Transfer Learning**, taking advantage of the robust feature extraction capabilities of the model pre-trained on the ImageNet dataset. The final classification layer was adapted for our 4-class task.

* **Classification Task:** Multi-class classification of chest X-ray images.
* **Classes:**
    1.  **COVID-19**
    2.  **Viral Pneumonia**
    3.  **Pulmonary Opacity**
    4.  **Normal** lungs
* **Training Framework:** Implementation was carried out using **PyTorch**.
* **Training Goal:** The model was trained until a target training accuracy of **0.95** was obtained.

### Data Acquisition and Pre-processing
A 4-class dataset of chest X-ray images was used.

#### Dataset Distribution (Training Set)
| Class | Approximate Image Count |
| :--- | :--- |
| **Normal** | $\approx 10,000$ |
| **Viral Pneumonia** | $\approx 1,200$ |
| **COVID-19** | $\approx 3,600$ |
| **Lung Opacity** | $\approx 4,000$ |
| **Total Training Images** | **21,045** |

#### Test Set
A dedicated test set of **120 images** (30 images per class) was held back for final, unbiased evaluation.

#### Pre-processing and Data Augmentation
To enhance feature visualization and model robustness, the following steps were applied:
* **Contrast Stretch:** Applied to all images to improve the visibility of features, especially in low-contrast images.
* **Transformations:**
    * **Resize**
    * **RandomHorizontalFlip** (for data augmentation)
    * **Normalization**

---

## 📈 Results and Evaluation
The model's classification efficiency was validated using a range of evaluation metrics on the reserved test set.

### Overall Accuracy
| Metric | Value | Details |
| :--- | :--- | :--- |
| **Test Accuracy** | **0.86 (86%)** | 103 out of 120 images were correctly classified. |
| **Train Accuracy** | **0.86 (86%)** | |

### Classification Report (Test Set)
The per-class performance is detailed below:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Normal** | 0.90 | 0.75 | 0.82 | 24 |
| **Lung Opacity** | **0.90** | **1.00** | **0.95** | 36 |
| **Pneumonia** | **0.96** | 0.69 | 0.80 | 32 |
| **COVID-19** | 0.73 | **0.96** | 0.83 | 28 |
| **Average/Total** | - | - | **0.86 (Accuracy)** | 120 |

### Visualization and Localization
* **Confusion Matrix:** Matrices for both train and test datasets showed a strong clustering along the diagonal, confirming effective classification.
* **GradCAM (Gradient-based Class Activation Maps):** Used for localization, the GradCAM results confirmed that the model was focusing on the correct, pathology-relevant regions of the X-ray images. 

---

## 🚀 Future Improvements
To prepare the model for optimal real-world application, the following improvements are recommended:

1.  **Dataset Balancing and Expansion:**
    * **Balance:** Address the imbalance in the current dataset (e.g., Pneumonia vs. Normal).
    * **Expansion:** Significantly increase the overall dataset size (e.g., target 50,000 images per class) to cover a wider variety of cases and effectively reduce **False Positives (FP)** and **False Negatives (FN)**.

2.  **Model Optimization (Quantization):**
    * **Size Reduction:** Quantize the exported model file (change data type from Float to Uint).
    * **Efficiency:** This step is critical for efficient deployment, reducing the model's memory size (currently 40MB). The resulting slight loss in accuracy is expected to be insignificant given the already high performance.
