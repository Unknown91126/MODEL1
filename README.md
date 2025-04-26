#  Credit Card Fraud Detection

##  Project Summary

This repository showcases a machine learning model crafted to identify fraudulent transactions within credit card data. The dataset used is anonymized and preprocessed to reflect real-world scenarios. The core objective is to create a model that is both **effective and interpretable**, with a strong emphasis on **minimizing false alarms** while **accurately detecting fraudulent activity**.

---

##  Dataset Details

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- **Total Records**: 284,807  
- **Fraud Cases**: 492 (~0.17%)  
- **Attributes**:
  - `V1` through `V28`: PCA-anonymized features
  - `Time`: Seconds elapsed from the first recorded transaction
  - `Amount`: Monetary value of the transaction
  - `Class`: Target label (0 = normal, 1 = fraud)

---

##  Tackling Class Imbalance

Since fraudulent transactions are rare compared to legitimate ones, the dataset is highly imbalanced. To overcome this, the project uses:

###  **SMOTE – Synthetic Minority Oversampling Technique**

- SMOTE is implemented to **synthesize new samples** of the minority class (fraud).
- This ensures a **balanced training dataset**, enhancing the model's learning ability.
- The test set remains untouched to simulate real deployment scenarios.

>  Other techniques like random oversampling and undersampling were considered, but SMOTE was chosen for its balance between **data diversity** and **risk of overfitting**.

---

##  Feature Engineering Insights

Because the dataset features are anonymized, traditional behavior-based attributes (like customer location or ID) are unavailable. Nonetheless, we performed meaningful transformations:

###  Features Applied:
- `scaled_amount`: Standardized version of the `Amount` column
- `scaled_time`: Standardized version of the `Time` column

###  Hypothetical Feature Extensions (if raw data were available):
- **Transaction frequency**: Number of user transactions in recent intervals
- **Location mismatch**: Binary flag if the current location differs from past patterns
- **Spending pattern**: User’s recent average spend, to detect outliers

---

##  Model Building and Assessment

###  Model Used:
- **Random Forest Classifier**
  - Ideal for handling high-dimensional, noisy data
  - Provides probability-based predictions
  - Naturally resistant to overfitting

###  Evaluation Strategy:

| Metric               | Purpose                                        |
|----------------------|------------------------------------------------|
| **Confusion Matrix** | Visualizes actual vs. predicted outcomes       |
| **Precision**        | Measures correctness of fraud predictions      |
| **Recall**           | Reflects how well fraud cases are captured     |
| **F1 Score**         | Combines precision and recall for balance      |
| **AUC-ROC Score**    | Evaluates how well the model separates classes |

> The evaluation prioritizes **reducing false positives** to avoid inconveniencing genuine users, and **maximizing recall** to ensure fraudulent behavior is not missed.

---

##  Project Goals

-  Accurately detect fraud with minimal false positives  
-  Achieve high **recall** and **AUC-ROC** performance  
-  Provide a **real-time prediction interface** for incoming transactions  
-  Deliver clean, maintainable code that can be extended or deployed

---

##  Running the Project

To get started:

1. Clone the repo:
   ```bash
   [https://github.com/Unknown91126/MODEL1.git]
