# Breast Cancer Diagnostic Classification Pipeline (WDBC)

## a. Problem Statement
Breast cancer is a leading cause of oncological mortality globally. Early and precise classification of breast mass biopsies is vital for survival rates. This project establishes an automated, end-to-end Machine Learning classification pipeline using digitized fine needle aspirate (FNA) image metrics. The primary objective is to evaluate, benchmark, and deploy five distinct machine learning models to accurately classify breast mass biopsies into binary clinical diagnostic targets: **Malignant** (Cancerous) or **Benign** (Non-Cancerous).

---

## b. Dataset Description
The model pipeline is trained on the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset. 
* **Instance Count**: 569 patient biopsy samples.
* **Target Distribution**: Balanced class layout consisting of 357 Benign (62.74%) and 212 Malignant (37.26%) instances.
* **Features Matrix**: 32 total columns containing 1 unique Patient ID, 1 categorical Target Label (`M` = Malignant, `B` = Benign), and 30 continuous numerical features.
* **Feature Engineering Structure**: 10 core geometric attributes of the cell nuclei (Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave Points, Symmetry, and Fractal Dimension) are extracted. Each attribute features three statistical metrics capturing its **Mean** values (Features 1-10), **Standard Error** (Features 11-20), and **Worst/Largest** values (Features 21-30).

---

## c. GitHub Repository Link
* **Remote Repository Workspace**: [https://github.com/vinayaka198711/ml_assigment_2.git]

---

## d. Models Used & Quantitative Evaluation Summary
The classification architectures are evaluated using an 80/20 train/test partition stratified by the baseline class distribution.

### Metric Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9825 | 0.9974 | 0.9762 | 0.9762 | 0.9762 | 0.9623 |
| **Decision Tree** | 0.9298 | 0.9255 | 0.9070 | 0.9048 | 0.9059 | 0.8504 |
| **kNN** | 0.9561 | 0.9782 | 0.9524 | 0.9302 | 0.9412 | 0.9069 |
| **Naive Bayes** | 0.9474 | 0.9881 | 0.9302 | 0.9302 | 0.9302 | 0.8885 |
| **Random Forest (Ensemble)** | 0.9649 | 0.9911 | 0.9535 | 0.9524 | 0.9524 | 0.9248 |

---

## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | **Exceptional Regularization**: Achieves the highest baseline performance across all 6 tracking metrics. Linearly separable continuous boundaries excel on scaled numerical dimensions, maximizing stability and minimizing variance. |
| **Decision Tree** | **High Variance / Overfitting Risk**: Lower metric results suggest structural vulnerabilities to local variations. Prone to creating brittle, overfit decision boundaries compared to the smoother multi-feature evaluation models. |
| **kNN** | **Distance-Dependent Baseline**: Delivers strong, reliable geometric classification when backed by standardization. Performance declines slightly due to noise vulnerabilities across the wider 30-dimensional feature space. |
| **Naive Bayes** | **Robust Probabilistic Model**: Performs consistently despite its assumption of feature independence. High AUC shows excellent boundary separation, though extreme correlated structural parameters slightly limit absolute precision. |
| **Random Forest (Ensemble)** | **Robust Structural Generalization**: Strong ensemble variance reduction across randomized sub-trees. Marginally lags behind Logistic Regression on absolute recall, but provides robust defense against overfitting. |

### Overall Winner for your dataset?
The **Overall Winner** for this dataset is **Logistic Regression**. 

* **Why it won**: In medical diagnoses, minimizing **False Negatives** (failing to identify malignancy) is paramount. Logistic Regression achieves top marks in both **Recall (0.9762)** and **MCC (0.9623)**. This indicates a high level of prediction confidence and near-perfect correlation with actual diagnostic results, outperforming complex ensemble and non-linear options on standard scaled numerical features.
