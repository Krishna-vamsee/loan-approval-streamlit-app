# Loan Approval Prediction System

This project provides an end-to-end solution for predicting whether a loan application will be approved or rejected. It includes a comprehensive data science notebook for model training and a Streamlit-based web interface for easy user interaction.

## 🚀 Overview

The system uses a **Logistic Regression** model embedded within a Scikit-Learn `Pipeline`. The pipeline automates the following steps:
1.  **Preprocessing**: Handling missing values (median/most frequent) and feature scaling/encoding.
2.  **Feature Selection**: Identifying the top 10 most influential features using `SelectKBest`.
3.  **Deployment**: A user-friendly web app built with Streamlit for real-time inference.

## 🛠️ Project Structure

* `Feature_selection.ipynb`: Jupyter Notebook containing data analysis, pipeline construction, hyperparameter tuning (`GridSearchCV`), and model evaluation.
* `app.py`: Streamlit application script.
* `model.pkl`: The serialized, trained pipeline (preprocessor + model).
* `loan_approved.csv`: The dataset used for training and testing.

## 📊 Model Performance

The final model was selected using `GridSearchCV` to optimize for accuracy.
* **Best Parameters**: `{'model__C': 0.1, 'feature_selection__k': 10}`
* **Mean Test Accuracy**: ~80.9% (via 5-fold Cross-Validation)
* **Final Test Accuracy**: ~84.4%

## 💻 Installation & Usage

### 1. Clone the repository
```bash
git clone [https://github.com/Krishna-vamsee/loan-approval-prediction.git](https://github.com/Krishna-vamsee/loan-approval-prediction.git)
cd loan-approval-prediction
