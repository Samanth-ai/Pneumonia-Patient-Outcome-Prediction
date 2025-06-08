# Multi-Task Deep Learning for Pneumonia Patient Outcome Prediction

This project uses a multi-task deep learning model to predict ten different clinical outcomes for pneumonia patients from the MIMIC-IV dataset. The goal is to build a tool that can aid clinicians by providing simultaneous predictions for patient mortality, length of stay, and risk of developing critical conditions like sepsis and acute kidney injury (AKI).

![Knowledge Graph Subgraph](https://i.imgur.com/gY9gL2d.png)
*A knowledge graph visualization of the relationships between two patients and their various diagnoses and outcomes.*

---

## 📋 Table of Contents
* [Project Objective](#-project-objective)
* [Dataset](#-dataset)
* [Methodology](#-methodology)
* [Model Performance](#-model-performance)
* [How to Run](#-how-to-run)
* [Future Work](#-future-work)

---

## 🎯 Project Objective

Pneumonia is a serious respiratory infection with highly variable patient outcomes. This project aims to predict these outcomes accurately to assist with:

* **Clinical Decision-Making:** Identifying high-risk patients who may require more immediate attention.
* **Resource Allocation:** Optimizing the management of hospital beds and ICU resources.
* **Personalized Care:** Developing tailored treatment strategies for individual patients.

The core of this project is a multi-task neural network built with PyTorch that simultaneously predicts **10 patient outcomes**.

---

## 💾 Dataset

The project utilizes the **MIMIC-IV** dataset, a large, freely-available database comprising de-identified health-related data from patients who stayed in the intensive care units of the Beth Israel Deaconess Medical Center.

A comprehensive cohort of pneumonia patients was extracted using detailed **SQL queries in Google BigQuery**. The extracted data includes:
* Patient demographics
* Admission and ICU details
* Vital signs (e.g., heart rate, blood pressure)
* Lab results (e.g., creatinine, lactate)
* Comorbidities (e.g., diabetes, cancer)

---

## 🛠️ Methodology

The project followed a multi-stage process, from data extraction and preprocessing to model training and evaluation.

### 1. Data Preprocessing & Feature Engineering

* **Categorical & Numerical Features:** Handled missing values through mean imputation for numerical columns and label encoding for categorical data like gender.
* **Feature Scaling:** Standardized all numerical features using `StandardScaler` from scikit-learn to ensure they have zero mean and unit variance, which helps the model train more effectively.
* **Target Transformation:** Applied a log transformation (`np.log1p`) to the `hospital_los` and `icu_los` targets to normalize their skewed distributions, improving regression performance.

### 2. Multi-Task Model Architecture

A multi-task neural network was built in PyTorch to handle the ten prediction tasks concurrently.

* **Shared Layers:** The model starts with shared layers consisting of `Linear` layers, `ReLU` activation, `BatchNorm1d`, and `Dropout`. These layers learn a common, useful representation of the patient data.
* **Task-Specific Output Layers:** Following the shared layers, the model branches into ten separate linear output layers—one for each target variable. This allows the model to make specialized predictions for each outcome.
* **Loss Functions:** A combination of loss functions was used to train the model:
    * `BCEWithLogitsLoss`: For the eight binary classification tasks (e.g., mortality, sepsis).
    * `MSELoss`: For the two regression tasks (hospital and ICU length of stay).
    * **Weighted Loss:** To address class imbalance in the mortality and myocardial infarction tasks, a weighted BCE loss was used to give more importance to the minority class.

### 3. Knowledge Graph

A knowledge graph was constructed using the `networkx` library to visualize the complex relationships between patients, their diagnoses, and outcomes. This provides an intuitive way to explore patient data and verify connections within the dataset.

---

## 📊 Model Performance

The model was evaluated on a held-out test set, demonstrating strong predictive power across multiple tasks.

| Task | Metric | Score |
| :--- | :--- | :--- |
| **Mortality** | AUC | 0.8651 |
| | Accuracy | 0.8309 |
| **High-Risk Sepsis Shock**| F1-Score | 0.9593 |
| **High-Risk AKI** | AUC | 0.8428 |
| **Myocardial Infarction**| AUC | 0.8942 |
| **Recovery** | Accuracy | 0.9899 |
| **Hospital LOS** | MAE | 0.4662 |
| **ICU LOS** | MAE | 2.0716 |

---

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/Pneumonia-Patient-Outcome-Prediction.git](https://github.com/your-username/Pneumonia-Patient-Outcome-Prediction.git)
    cd Pneumonia-Patient-Outcome-Prediction
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Launch the Jupyter Notebook:**
    ```bash
    jupyter notebook pneumonia_outcome_prediction.ipynb
    ```
4.  **Run the cells:** Execute the cells in the notebook sequentially to preprocess the data, train the model, and evaluate its performance.

---

## 🔮 Future Work

The model shows promising results, but there are several avenues for future improvement:

* **Advanced Architectures:** Explore more complex neural network architectures like Transformers or attention-based models.
* **Feature Engineering:** Create new, more informative features from the existing data to improve predictive accuracy.
* **Hyperparameter Tuning:** Conduct a more extensive hyperparameter search to further optimize the model.
* **Temporal Information:** Incorporate time-series models (like LSTMs or GRUs) to better utilize the time-based features in the dataset.

